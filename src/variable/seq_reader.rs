//! # `IntVec` Stateful Sequential Reader
//!
//! This module provides [`IntVecSeqReader`], a stateful, reusable reader for a
//! generic [`IntVec`]. It is specifically designed and optimized for access
//! patterns that are sequential or have a high degree of locality.
//!
//! ## Purpose and Design
//!
//! `IntVecSeqReader` maintains an internal state of the current decoding position.
//! When a new `get` request is made, it intelligently decides whether to:
//!
//! 1.  **Decode Forward (Fast Path):** If the requested index is near the
//!     current position and within the same sample block, the reader decodes
//!     forward from its last position, avoiding a costly seek operation.
//!
//! 2.  **Seek and Decode (Fallback Path):** If the requested index is far away
//!     or requires moving backward, the reader falls back to seeking to the
//!     nearest sample point and decoding from there.
//!
//! This makes it exceptionally efficient for iterating through indices that are
//! sorted or clustered together.
//!
//! [`IntVec`]: crate::variable::IntVec

use super::{traits::Storable, IntVec, IntVecBitReader, IntVecError};
use dsi_bitstream::{
    dispatch::{CodesRead, FuncCodeReader, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};

/// A stateful, sequential reader for a generic `IntVec` optimized for forward access.
///
/// This reader is created by the [`IntVec::seq_reader`] method. It maintains an
/// internal state corresponding to the last-read element's position, making it
/// highly efficient for sequential or mostly-forward access patterns.
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
    /// The pre-configured code reader, created once to avoid overhead.
    code_reader: FuncCodeReader<E, IntVecBitReader<'a, E>>,
    /// The index of the element *after* the one most recently read.
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
        let code_reader = FuncCodeReader::new(intvec.encoding)
            .expect("Failed to create code reader for DSI encoding.");
        Self {
            intvec,
            reader: IntVecBitReader::new(dsi_bitstream::impls::MemWordReader::new(
                intvec.data.as_ref(),
            )),
            code_reader,
            current_index: 0,
        }
    }

    /// Retrieves the element at the specified index using the stateful reader.
    ///
    /// This method leverages the reader's internal state to optimize access.
    ///
    /// # Returns
    /// - `Ok(Some(T))` if the `index` is within bounds.
    /// - `Ok(None)` if the `index` is out of bounds.
    /// - `Err(IntVecError)` if a decoding error occurs.
    pub fn get(&mut self, index: usize) -> Result<Option<T>, IntVecError> {
        if index >= self.intvec.len {
            return Ok(None);
        }
        // SAFETY: The bounds check has been performed.
        Ok(Some(unsafe { self.get_unchecked(index) }))
    }

    /// Retrieves the element at the specified index without bounds checking.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    pub unsafe fn get_unchecked(&mut self, index: usize) -> T {
        debug_assert!(
            index < self.intvec.len,
            "Index out of bounds: index was {} but length was {}",
            index,
            self.intvec.len
        );

        let k = self.intvec.k;
        let target_sample_block = index / k;
        let current_sample_block = if self.current_index == 0 {
            0
        } else {
            (self.current_index - 1) / k
        };

        // Fast Path Condition: We can decode forward sequentially.
        if index < self.current_index || target_sample_block != current_sample_block {
            // Slow Path: A seek is required.
            let start_bit = unsafe{ self.intvec.samples.get_unchecked(target_sample_block) };
            self.reader.set_bit_pos(start_bit).unwrap();
            self.current_index = target_sample_block * k;
        }

        // Decode and discard intermediate elements.
        for _ in self.current_index..index {
            self.code_reader.read(&mut self.reader).unwrap();
        }
        // Read the target value.
        let word = self.code_reader.read(&mut self.reader).unwrap();
        self.current_index = index + 1;
        Storable::from_word(word)
    }
}