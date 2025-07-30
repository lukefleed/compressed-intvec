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
//! ## When to Use `IntVecSeqReader`
//!
//! - **Dynamic Sequential Access:** When you need to perform multiple lookups in a
//!   loop where the indices are mostly increasing (e.g., traversing a sorted
//!   list, walking a graph where neighbors have consecutive IDs).
//! - **Streaming Indices:** It is the engine behind [`IntVec::get_many_from_iter`],
//!   which processes lookups from a streaming iterator source.
//!
//! For batch lookups where all indices are known beforehand, [`IntVec::get_many`]
//! remains the most optimized choice, as it can pre-sort the indices to guarantee
//! a perfect forward scan.
//!
//! [`IntVec`]: super::IntVec
//! [`IntVec::get`]: super::IntVec::get
//! [`IntVecReader`]: super::IntVecReader
//! [`IntVec::get_many`]: super::IntVec::get_many
//! [`IntVec::get_many_from_iter`]: super::IntVec::get_many_from_iter

use super::{Encoding, IntVec, IntVecBitReader, IntVecError};
use dsi_bitstream::{
    dispatch::{CodesRead, FuncCodeReader, StaticCodeRead},
    impls::MemWordReader,
    prelude::{BitRead, BitSeek, Endianness},
};

/// A stateful, sequential reader for an `IntVec` optimized for forward access.
///
/// This reader is created by the [`IntVec::seq_reader`] method. It maintains an
/// internal state corresponding to the last-read element's position, making it
/// highly efficient for sequential or mostly-forward access patterns. By avoiding
/// redundant bitstream seeks when accessing consecutive or nearby elements, it
/// can significantly outperform other random-access methods in these scenarios.
///
/// Use this reader when your access pattern has locality (e.g., you are likely
/// to request `index + delta` after requesting `index`). For fully random
/// lookups, the standard [`IntVecReader`] is more appropriate. For batch lookups
/// from a slice, [`IntVec::get_many`] is the most performant option.
///
/// # Example
///
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[u64] = &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
/// let intvec = LEIntVec::builder(data).k(4).build().unwrap();
///
/// // Create a reusable sequential reader.
/// let mut seq_reader = intvec.seq_reader();
///
/// // Accessing elements sequentially is highly efficient.
/// assert_eq!(seq_reader.get(2).unwrap(), Some(30)); // Seeks to sample 0, decodes 3 elements.
/// assert_eq!(seq_reader.get(3).unwrap(), Some(40)); // Decodes 1 more element, no seek.
///
/// // Jumping to a new block will trigger a seek.
/// assert_eq!(seq_reader.get(8).unwrap(), Some(90)); // Seeks to sample 2, decodes 1 element.
///
/// // Moving backward also triggers a seek.
/// assert_eq!(seq_reader.get(1).unwrap(), Some(20)); // Seeks back to sample 0.
/// ```
pub struct IntVecSeqReader<'a, E: Endianness>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    // Immutable reference to the parent IntVec.
    intvec: &'a IntVec<E>,
    // The stateful, reusable bitstream reader.
    reader: IntVecBitReader<'a, E>,
    // The index of the element *after* the one most recently read.
    // This tracks our logical position in the decoded sequence.
    current_index: usize,
}

impl<'a, E: Endianness> IntVecSeqReader<'a, E>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new `IntVecSeqReader`.
    ///
    /// This is `pub(super)` and is called by [`IntVec::seq_reader`].
    pub(super) fn new(intvec: &'a IntVec<E>) -> Self {
        Self {
            intvec,
            reader: IntVecBitReader::<E>::new(MemWordReader::new(&intvec.data)),
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
    /// # Arguments
    /// * `index`: The index of the element to retrieve.
    ///
    /// # Returns
    /// - `Ok(Some(u64))` if the `index` is within bounds.
    /// - `Ok(None)` if the `index` is out of bounds.
    /// - `Err(IntVecError)` if a decoding error occurs.
    pub fn get(&mut self, index: usize) -> Result<Option<u64>, IntVecError> {
        if index >= self.intvec.len {
            return Ok(None);
        }

        let value = match self.intvec.encoding {
            Encoding::Dsi(code) => {
                // With DSI encoding, k and samples are guaranteed to be Some.
                let k = self.intvec.k.unwrap();
                let samples = self.intvec.samples.as_ref().unwrap();

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
                    // This happens when moving backward or jumping to a non-adjacent block.
                    let start_bit = samples.get(target_sample_block).unwrap();
                    self.reader.set_bit_pos(start_bit)?;
                    // Update our logical position to the start of this sample block.
                    self.current_index = target_sample_block * k;
                }

                // At this point, `self.current_index <= index` is guaranteed, and the
                // reader is correctly positioned to decode forward.
                let code_reader = FuncCodeReader::new(code)
                    .map_err(|e| IntVecError::CodecDispatch(e.to_string()))?;

                // Decode and discard intermediate elements.
                for _ in self.current_index..index {
                    code_reader.read(&mut self.reader)?;
                }
                // Read the target value.
                let result = code_reader.read(&mut self.reader)?;
                // Update state to reflect the new position.
                self.current_index = index + 1;
                result
            }
            Encoding::Fixed { num_bits } => {
                // For FixedLength, access is always O(1). We just seek and read.
                // There is no sequential decoding optimization to be had, but we
                // still update the state for logical consistency.
                let bit_pos = index as u64 * num_bits as u64;
                self.reader.set_bit_pos(bit_pos)?;
                let result = self.reader.read_bits(num_bits)?;
                // Update state. The next bit position is implicitly known.
                self.current_index = index + 1;
                result
            }
        };

        Ok(Some(value))
    }
}
