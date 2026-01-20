//! A stateful reader for efficient, sequential access into an [`IntVec`].
//!
//! This module provides [`IntVecSeqReader`], a reusable reader that is
//! specifically optimized for access patterns that are sequential or have a high
//! degree of locality (i.e., indices are strictly increasing or close to each other).
//!
//! # Performance
//!
//! [`IntVecSeqReader`] maintains an internal state of the current decoding position.
//! When a new [`get`](super::IntVec::get) request is made, it decides whether to:
//!
//! 1.  **Decode Forward (Fast Path):** If the requested index is at or after the
//!     current position and within the same sample block, the reader decodes
//!     forward from its last position. This avoids a costly seek operation and
//!     is the primary optimization.
//!
//! 2.  **Seek and Decode (Fallback):** If the requested index is far away or
//!     requires moving backward, the reader falls back to seeking to the
//!     nearest sample point and decoding from there, just like [`IntVecReader`].
//!
//! This makes it very efficient for iterating through indices that are
//! sorted or clustered together.
//!
//! [`IntVec`]: crate::variable::IntVec
//! [`IntVecReader`]: crate::variable::reader::IntVecReader

use super::{traits::Storable, IntVec, IntVecError};
use crate::common::codec_reader::{CodecReader, IntVecBitReader};
use dsi_bitstream::{
    dispatch::{CodesRead, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};

/// A stateful, sequential reader for an [`IntVec`] optimized for forward access.
///
/// This reader is created by the [`seq_reader`](super::IntVec::seq_reader)
/// method. It maintains an internal cursor corresponding to the last-read
/// element's position, making it highly efficient for sequential or
/// mostly-forward access patterns.
///
/// It is a more specialized tool than [`IntVecReader`](super::reader::IntVecReader).
///
/// # Performance
///
/// [`IntVecSeqReader`] maintains an internal state of the current decoding position.
/// When a new [`get`](super::IntVec::get) request is made, it decides whether to:
///
/// 1.  **Decode Forward (Fast Path):** If the requested index is at or after the
///     current position and within the same sample block, the reader decodes
///     forward from its last position. This avoids a costly seek operation and
///     is the primary optimization.
///
/// 2.  **Seek and Decode (Fallback):** If the requested index is far away or
///     requires moving backward, the reader falls back to seeking to the
///     nearest sample point and decoding from there, just like [`IntVecReader`](super::reader::IntVecReader).
///
/// # Examples
///
/// ```
/// use compressed_intvec::variable::{IntVec, UIntVec};
///
/// let data: Vec<u32> = (0..100).collect();
/// let vec: UIntVec<u32> = IntVec::from_slice(&data).unwrap();
///
/// // Create a reader optimized for sequential access
/// let mut seq_reader = vec.seq_reader();
///
/// // Accessing indices in increasing order is very efficient
/// assert_eq!(seq_reader.get(10).unwrap(), Some(10));
/// assert_eq!(seq_reader.get(15).unwrap(), Some(15)); // Decodes forward from index 10
/// assert_eq!(seq_reader.get(90).unwrap(), Some(90)); // Jumps to a new sample
/// ```
pub struct IntVecSeqReader<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Immutable reference to the parent IntVec.
    intvec: &'a IntVec<T, E, B>,
    /// The stateful, reusable bitstream reader.
    reader: IntVecBitReader<'a, E>,
    /// The hybrid dispatcher that handles codec reading robustly.
    code_reader: CodecReader<'a, E>,
    /// The index of the element *after* the one most recently read. This acts
    /// as a cursor for the current decoding position.
    current_index: usize,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> IntVecSeqReader<'a, T, E, B>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new `IntVecSeqReader`.
    pub(super) fn new(intvec: &'a IntVec<T, E, B>) -> Self {
        // Instantiate the hybrid dispatcher. This will not panic, as it falls
        // back to a slower but universally compatible method if the codec's
        // parameters are not supported by the fast path.
        let code_reader = CodecReader::new(intvec.encoding);
        Self {
            intvec,
            reader: IntVecBitReader::new(dsi_bitstream::impls::MemWordReader::new(
                intvec.data.as_ref(),
            )),
            code_reader,
            current_index: 0,
        }
    }

    /// Retrieves the element at `index`, or `None` if out of bounds.
    ///
    /// This method leverages the reader's internal state to optimize access,
    /// especially for sequential reads.
    pub fn get(&mut self, index: usize) -> Result<Option<T>, IntVecError> {
        if index >= self.intvec.len {
            return Ok(None);
        }
        // SAFETY: The bounds check has been performed.
        Ok(Some(unsafe { self.get_unchecked(index) }))
    }

    /// Retrieves the element at `index` without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    /// The caller must ensure that `index < self.intvec.len()`.
    pub unsafe fn get_unchecked(&mut self, index: usize) -> T {
        debug_assert!(
            index < self.intvec.len,
            "Index out of bounds: index was {} but length was {}",
            index,
            self.intvec.len
        );

        let k = self.intvec.k;
        let (target_sample_block, current_sample_block, block_start) = if k.is_power_of_two() {
            let k_exp = k.trailing_zeros();
            let tsb = index >> k_exp;
            let csb = if self.current_index == 0 {
                // A special case for the very first access.
                // We treat it as being in a "different" block to force a seek.
                usize::MAX
            } else {
                (self.current_index - 1) >> k_exp
            };
            (tsb, csb, tsb << k_exp)
        } else {
            let tsb = index / k;
            let csb = if self.current_index == 0 {
                // A special case for the very first access.
                // We treat it as being in a "different" block to force a seek.
                usize::MAX
            } else {
                (self.current_index - 1) / k
            };
            (tsb, csb, tsb * k)
        };

        // This is the core optimization: if the target index is ahead of the
        // current position and within the same sample block, we can just
        // decode forward instead of performing a full seek.
        if index < self.current_index || target_sample_block != current_sample_block {
            // Slow Path: A seek is required because we are moving backward or
            // jumping to a different sample block.
            // SAFETY: The public-facing get() performs bounds checks, and
            // internal callers are expected to uphold the same contract.
            let start_bit = self.intvec.samples.get_unchecked(target_sample_block);
            self.reader.set_bit_pos(start_bit).unwrap();
            self.current_index = block_start;
        }

        // Decode and discard intermediate elements. The hybrid dispatcher
        // will use the fastest available method.
        for _ in self.current_index..index {
            self.code_reader.read(&mut self.reader).unwrap();
        }
        // Read the target value.
        let word = self.code_reader.read(&mut self.reader).unwrap();
        // Update the current index to the next position.
        self.current_index = index + 1;
        Storable::from_word(word)
    }
}
