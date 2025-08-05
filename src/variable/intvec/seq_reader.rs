//! # `IntVec` Stateful Sequential Reader
//!
//! This module provides [`IntVecSeqReader`], a stateful, reusable reader for an
//! [`IntVec`]. It is specifically designed and optimized for access patterns that
//! are sequential or have a high degree of locality, where subsequent lookups
//! are often near previous ones.
//!
//! ## Purpose and Design
//!
//! While [`IntVec::get`] is convenient for single lookups, and [`IntVecReader`]
//! is effective for completely random access patterns, `IntVecSeqReader` fills a
//! critical performance niche. It maintains an internal state of the current
//! decoding position (`current_index`). When a new `get` request is made, it
//! intelligently decides whether to:
//!
//! 1.  **Decode Forward (Fast Path):** If the requested index is close to the
//!     current position and within the same sample block, the reader decodes
//!     forward from its last position, avoiding a costly seek operation.
//!
//! 2.  **Seek and Decode (Fallback Path):** If the requested index is far away,
//!     in a different sample block, or requires moving backward, the reader
//!     falls back to the standard strategy of seeking to the nearest sample
//!     point and decoding from there.
//!
//! This stateful approach makes it exceptionally efficient for iterating through
//! indices that are sorted or clustered together.
//!
//! [`IntVec`]: super::IntVec
//! [`IntVec::get`]: super::IntVec::get
//! [`IntVecReader`]: super::IntVecReader

use super::{IntVec, IntVecBitReader, IntVecError};
use dsi_bitstream::{
    dispatch::{CodesRead, FuncCodeReader, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};

/// A stateful, sequential reader for an `IntVec` optimized for forward access.
///
/// This reader is created by the [`IntVec::seq_reader`] method. It maintains an
/// internal state corresponding to the last-read element's position, making it
/// highly efficient for sequential or mostly-forward access patterns. By avoiding
/// redundant bitstream seeks when accessing consecutive or nearby elements, it
/// can significantly outperform other random-access methods in these scenarios.
pub struct IntVecSeqReader<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Immutable reference to the parent IntVec.
    intvec: &'a IntVec<E, B>,
    /// The stateful, reusable bitstream reader.
    reader: IntVecBitReader<'a, E>,
    /// The pre-configured code reader, created once to avoid overhead.
    code_reader: FuncCodeReader<E, IntVecBitReader<'a, E>>,
    /// The index of the element *after* the one most recently read.
    /// This tracks our logical position in the decoded sequence.
    current_index: usize,
}

impl<'a, E, B> IntVecSeqReader<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new `IntVecSeqReader`.
    ///
    /// This is `pub(super)` and is called by [`IntVec::seq_reader`].
    pub(super) fn new(intvec: &'a IntVec<E, B>) -> Self {
        let code_reader = FuncCodeReader::new(intvec.encoding)
            .expect("Failed to create code reader for DSI encoding.");
        Self {
            intvec,
            reader: IntVecBitReader::new(dsi_bitstream::impls::MemWordReader::new(
                intvec.data.as_ref(),
            )),
            code_reader,
            // The reader is initialized at bit position 0, which corresponds
            // to the start of the logical element sequence at index 0.
            current_index: 0,
        }
    }

    /// Retrieves the element at the specified index using the stateful reader.
    ///
    /// This method leverages the reader's internal state to optimize access. If
    /// `index` is near the reader's current position, it decodes forward. If `index`
    /// requires moving backward or jumping to a different sample block, it performs
    /// a full seek.
    ///
    /// # Returns
    /// - `Ok(Some(u64))` if the `index` is within bounds.
    /// - `Ok(None)` if the `index` is out of bounds.
    /// - `Err(IntVecError)` if a decoding error occurs.
    pub fn get(&mut self, index: usize) -> Result<Option<u64>, IntVecError> {
        if index >= self.intvec.len {
            return Ok(None);
        }
        // SAFETY: The bounds check has been performed.
        Ok(Some(unsafe { self.get_unchecked(index) }))
    }

    /// Retrieves the element at the specified index without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    /// The caller must ensure that `index < self.intvec.len()`.
    pub unsafe fn get_unchecked(&mut self, index: usize) -> u64 {
        debug_assert!(
            index < self.intvec.len,
            "Index out of bounds: index was {} but length was {}",
            index,
            self.intvec.len
        );

        let k = self.intvec.k;

        let target_sample_block = index / k;
        // Determine the block of the current position. Handle edge case where
        // current_index is 0.
        let current_sample_block = if self.current_index == 0 {
            0
        } else {
            (self.current_index - 1) / k
        };

        // Fast Path Condition: We can decode forward sequentially.
        // This is not met if we need to move backward or if we've jumped
        // to a different sample block entirely.
        if index < self.current_index || target_sample_block != current_sample_block {
            // Slow Path: A seek is required.
            // SAFETY: The caller guarantees that `index` is in bounds, which implies
            // that `target_sample_block` is also a valid index into the samples vector.
            let start_bit = self.intvec.samples.get_unchecked(target_sample_block);
            self.reader.set_bit_pos(start_bit).unwrap();
            // Update our logical position to the start of this sample block.
            self.current_index = target_sample_block * k;
        }

        // At this point, `self.current_index <= index` is guaranteed, and the
        // reader is correctly positioned to decode forward.

        // Decode and discard intermediate elements.
        for _ in self.current_index..index {
            self.code_reader.read(&mut self.reader).unwrap();
        }
        // Read the target value.
        let result = self.code_reader.read(&mut self.reader).unwrap();
        // Update state to reflect the new position.
        self.current_index = index + 1;
        result
    }
}