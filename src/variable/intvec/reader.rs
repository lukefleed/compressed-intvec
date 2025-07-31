//! # `IntVec` Stateful Reader
//!
//! This module provides [`IntVecReader`], a stateful, reusable reader for an
//! [`IntVec`]. It is designed to optimize random access performance in scenarios
//! where lookup indices are determined dynamically.
//!
//! A standard call to [`IntVec::get`] is convenient but inefficient for repeated
//! lookups, as each call creates and discards an internal bitstream reader and
//! code dispatcher. The `IntVecReader` solves this by maintaining a persistent
//! reader instance and a pre-configured code reader, amortizing the setup cost
//! across multiple `get` operations.
//!
//! This pattern is ideal for traversals where the next lookup index depends on
//! the result of the previous one. For predefined batch lookups, [`IntVec::get_many`]
//! and [`IntVec::par_get_many`] remain the preferred, more optimized alternatives.
//!
//! [`IntVec`]: super::IntVec
//! [`IntVec::get`]: super::IntVec::get
//! [`IntVec::get_many`]: super::IntVec::get_many
//! [`IntVec::par_get_many`]: super::IntVec::par_get_many

use super::{IntVec, IntVecBitReader, IntVecError};
use dsi_bitstream::{
    dispatch::{CodesRead, FuncCodeReader},
    prelude::{BitRead, BitSeek, Endianness, StaticCodeRead},
};

/// A stateful reader for an `IntVec` that provides fast random access.
///
/// This reader is created by the [`IntVec::reader`] method. It maintains an
/// internal, reusable bitstream reader and a pre-configured code reader, which
/// makes it highly efficient for performing multiple random lookups.
///
/// By reusing the same underlying components, it amortizes setup costs across
/// many `get` operations, making it ideal for access patterns that are not known
/// in advance (e.g., indices generated on-the-fly in a loop).
///
/// # Example
///
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[u64] = &;
/// let intvec = LEIntVec::builder(data).build().unwrap();
///
/// // Create a reusable reader from the IntVec.
/// let mut reader = intvec.reader();
///
/// // Perform multiple lookups. The reader efficiently handles seeks.
/// assert_eq!(reader.get(3).unwrap(), Some(40));
/// assert_eq!(reader.get(1).unwrap(), Some(20));
/// assert_eq!(reader.get(0).unwrap(), Some(10));
/// ```
pub struct IntVecReader<'a, E: Endianness>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// A reference to the parent `IntVec`.
    pub(super) intvec: &'a IntVec<E>,
    /// The stateful, reusable bitstream reader.
    pub(super) reader: IntVecBitReader<'a, E>,
    /// The pre-configured code reader, created once to avoid overhead.
    pub(super) code_reader: FuncCodeReader<E, IntVecBitReader<'a, E>>,
}

impl<'a, E: Endianness> IntVecReader<'a, E>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new `IntVecReader`.
    ///
    /// This is `pub(super)` and is called by [`IntVec::reader`].
    pub(super) fn new(intvec: &'a IntVec<E>) -> Self {
        let bit_reader =
            IntVecBitReader::new(dsi_bitstream::impls::MemWordReader::new(&intvec.data));
        let code_reader = FuncCodeReader::new(intvec.encoding)
            .expect("Failed to create code reader for DSI encoding.");
        Self {
            intvec,
            reader: bit_reader,
            code_reader,
        }
    }

    /// Retrieves the element at the specified index using the reusable reader.
    ///
    /// This method leverages the stateful nature of the reader to perform efficient
    /// random access by seeking to the nearest preceding sample point and decoding
    /// sequentially from there.
    ///
    /// # Returns
    /// - `Ok(Some(u64))` if the `index` is within bounds.
    /// - `Ok(None)` if the `index` is out of bounds.
    /// - `Err(IntVecError)` if a bitstream error occurs.
    #[inline]
    pub fn get(&mut self, index: usize) -> Result<Option<u64>, IntVecError> {
        if index >= self.intvec.len {
            return Ok(None);
        }
        // SAFETY: The bounds check has been performed.
        let value = unsafe { self.get_unchecked(index) };
        Ok(Some(value))
    }

    /// Retrieves the element at the specified index without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    /// The caller must ensure that `index < self.intvec.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&mut self, index: usize) -> u64 {
        debug_assert!(
            index < self.intvec.len(),
            "Index out of bounds: index was {} but length was {}",
            index,
            self.intvec.len()
        );
        let k = self.intvec.k.unwrap();
        let samples = self.intvec.samples.as_ref().unwrap();
        let sample_index = index / k;
        let start_bit = samples.get(sample_index).unwrap();
        let start_index = sample_index * k;

        // The underlying bitstream operations are infallible, so unwrap is safe.
        self.reader.set_bit_pos(start_bit).unwrap();

        for _ in start_index..index {
            self.code_reader.read(&mut self.reader).unwrap();
        }
        self.code_reader.read(&mut self.reader).unwrap()
    }
}
