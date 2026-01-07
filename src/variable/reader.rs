//! A reader for efficient, repeated random access into an [`IntVec`].
//!
//! This module provides [`IntVecReader`], a reusable reader that is designed to
//! optimize random access performance.
//!
//! # Performance
//!
//! A standard call to [`get`](super::IntVec::get) is convenient, but it
//! creates and discards an internal bitstream reader for each call. When
//! performing many random lookups, this can introduce significant overhead.
//!
//! [`IntVecReader`] avoids this by maintaining a persistent, reusable
//! reader instance. This amortizes the setup cost across multiple `get` operations,
//! making it ideal for access patterns where lookup indices are not known in
//! advance (e.g., graph traversals, pointer chasing).
//!
//! For reading a predefined list of indices, [`get_many`](super::IntVec::get_many)
//! is generally more efficient, as it can pre-sort the indices for a single
//! sequential scan.
//!
//! [`IntVec`]: crate::variable::IntVec

use super::{traits::Storable, IntVec, IntVecError};
use crate::common::codec_reader::{CodecReader, IntVecBitReader};
use dsi_bitstream::{
    dispatch::{CodesRead, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};

/// A stateful reader for an `IntVec` that provides fast random access.
///
/// This reader is created by the [`IntVec::reader`](super::IntVec::reader)
/// method. It maintains an internal, reusable bitstream reader, making it highly
/// efficient for performing multiple random lookups where the access pattern is
/// not known ahead of time.
///
/// # Examples
///
/// ```
/// use compressed_intvec::variable::{IntVec, UIntVec};
///
/// let data: Vec<u32> = (0..100).rev().collect(); // Data is not sequential
/// let vec: UIntVec<u32> = IntVec::from_slice(&data).unwrap();
///
/// // Create a reusable reader
/// let mut reader = vec.reader();
///
/// // Perform multiple random reads efficiently
/// assert_eq!(reader.get(99).unwrap(), Some(0));
/// assert_eq!(reader.get(0).unwrap(), Some(99));
/// assert_eq!(reader.get(50).unwrap(), Some(49));
/// ```
pub struct IntVecReader<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// A reference to the parent `IntVec`.
    pub(super) intvec: &'a IntVec<T, E, B>,
    /// The stateful, reusable bitstream reader.
    pub(super) reader: IntVecBitReader<'a, E>,
    /// The hybrid dispatcher that handles codec reading.
    pub(super) code_reader: CodecReader<'a, E>,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> IntVecReader<'a, T, E, B>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new `IntVecReader`.
    pub(super) fn new(intvec: &'a IntVec<T, E, B>) -> Self {
        let bit_reader = IntVecBitReader::new(dsi_bitstream::impls::MemWordReader::new(
            intvec.data.as_ref(),
        ));
        // This robustly selects the best available dispatch strategy.
        let code_reader = CodecReader::new(intvec.encoding);
        Self {
            intvec,
            reader: bit_reader,
            code_reader,
        }
    }

    /// Retrieves the element at `index`, or `None` if out of bounds.
    ///
    /// This method leverages the stateful nature of the reader to perform efficient
    /// random access by seeking to the nearest preceding sample point and decoding
    /// sequentially from there.
    #[inline]
    pub fn get(&mut self, index: usize) -> Result<Option<T>, IntVecError> {
        if index >= self.intvec.len {
            return Ok(None);
        }
        // SAFETY: The bounds check has been performed.
        let value = unsafe { self.get_unchecked(index) };
        Ok(Some(value))
    }

    /// Retrieves the element at `index` without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    /// The caller must ensure that `index < self.intvec.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&mut self, index: usize) -> T {
        debug_assert!(
            index < self.intvec.len(),
            "Index out of bounds: index was {} but length was {}",
            index,
            self.intvec.len()
        );

        let k = self.intvec.k;
        let sample_index = index / k;
        // SAFETY: The caller guarantees that `index` is in bounds, which implies
        // that `sample_index` is also a valid index into the samples vector.
        let start_bit = self.intvec.samples.get_unchecked(sample_index);
        let start_index = sample_index * k;

        // The underlying bitstream operations are infallible, so unwrap is safe.
        self.reader.set_bit_pos(start_bit).unwrap();

        // Sequentially decode elements from the sample point up to the target index.
        for _ in start_index..index {
            // We use the hybrid dispatcher here. It will either call a function
            // pointer or use a match statement, depending on the codec.
            self.code_reader.read(&mut self.reader).unwrap();
        }
        // Read the target value.
        let word = self.code_reader.read(&mut self.reader).unwrap();
        Storable::from_word(word)
    }
}
