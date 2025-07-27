//! # `IntVec` Stateful Reader
//!
//! This module provides [`IntVecReader`], a stateful, reusable reader for an
//! [`IntVec`]. It is designed to optimize random access performance in scenarios
//! where lookup indices are determined dynamically.
//!
//! A standard call to [`IntVec::get`] is convenient but inefficient for repeated
//! lookups, as each call creates and discards an internal bitstream reader. The
//! `IntVecReader` solves this by maintaining a persistent reader instance,
//! amortizing the setup cost across multiple `get` operations.
//!
//! This pattern is ideal for traversals where the next lookup index depends on
//! the result of the previous one. For predefined batch lookups, [`IntVec::get_many`]
//! and [`IntVec::par_get_many`] remain the preferred, more optimized alternatives.
//!
//! [`IntVec`]: super::IntVec
//! [`IntVec::get`]: super::IntVec::get
//! [`IntVec::get_many`]: super::IntVec::get_many
//! [`IntVec::par_get_many`]: super::IntVec::par_get_many

use super::{Encoding, IntVec, IntVecBitReader, IntVecError};
use dsi_bitstream::{
    dispatch::{CodesRead, FuncCodeReader, StaticCodeRead},
    impls::MemWordReader,
    prelude::{BitRead, BitSeek, Endianness},
};

/// A stateful reader for an `IntVec` that provides fast random access.
///
/// This reader is created by the [`IntVec::reader`] method. It maintains an
/// internal, reusable bitstream reader, which makes it highly efficient for
/// performing multiple random lookups, especially when the access pattern is
/// not known in advance (e.g., indices are generated on-the-fly in a loop).
///
/// By reusing the same underlying reader, it amortizes the setup cost across
/// many `get` operations.
///
/// ## When to use `IntVecReader`
///
/// - When you need to perform multiple lookups and the indices are determined
///   dynamically.
/// - When the next lookup index might depend on the result of the previous one.
///
/// For batch lookups where all indices are known beforehand, using
/// [`IntVec::get_many`] is still preferable as it can perform global optimizations
/// like sorting the indices. There is a parallel version of this method, [`IntVec::par_get_many`].
///
/// # Example
///
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[u64] = &[10, 20, 30, 40, 50];
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
        let bit_reader = IntVecBitReader::<E>::new(MemWordReader::new(&intvec.data));
        Self {
            intvec,
            reader: bit_reader,
        }
    }

    /// Retrieves the element at the specified index using the reusable reader.
    ///
    /// This method leverages the stateful nature of the reader to perform efficient
    /// random access.
    ///
    /// # Implementation Notes
    ///
    /// The access strategy is identical to that of [`IntVec::get`], but because
    /// this method operates on a long-lived reader instance, it avoids the setup
    /// overhead associated with calling `IntVec::get` repeatedly.
    ///
    /// - **For fixed-width integer encoding**: Access is O(1). The bit position is
    ///   calculated and the reader seeks directly to it.
    /// - **For variable-length, bit-level codes**: The reader seeks to the nearest
    ///   preceding sample point and decodes sequentially from there.
    ///
    /// # Example
    ///
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: &[u64] = &[10, 20, 30, 40, 50];
    /// let intvec = LEIntVec::builder(data).build().unwrap();
    /// let mut reader = intvec.reader();
    ///
    /// assert_eq!(reader.get(2).unwrap(), Some(30));
    /// assert_eq!(reader.get(10).unwrap(), None); // Out of bounds
    /// ```
    #[inline]
    pub fn get(&mut self, index: usize) -> Result<Option<u64>, IntVecError> {
        if index >= self.intvec.len {
            return Ok(None);
        }

        let value = match self.intvec.encoding {
            Encoding::Dsi(code) => {
                // With DSI encoding, k and samples are guaranteed to be Some.
                let k = self.intvec.k.unwrap();
                let samples = self.intvec.samples.as_ref().unwrap();
                let sample_index = index / k;
                let start_bit = samples.get(sample_index).unwrap();
                let start_index = sample_index * k;

                self.reader.set_bit_pos(start_bit)?;
                let code_reader = FuncCodeReader::new(code)
                    .map_err(|e| IntVecError::CodecDispatch(e.to_string()))?;
                for _ in start_index..index {
                    code_reader.read(&mut self.reader)?;
                }
                code_reader.read(&mut self.reader)?
            }
            Encoding::Fixed { num_bits } => {
                let bit_pos = index as u64 * num_bits as u64;
                self.reader.set_bit_pos(bit_pos)?;
                self.reader.read_bits(num_bits)?
            }
        };

        Ok(Some(value))
    }
}
