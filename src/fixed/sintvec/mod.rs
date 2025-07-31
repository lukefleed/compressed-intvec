//! # A compressed, randomly accessible vector of `i64` integers with fixed-width encoding.
//!
//! This module provides [`SFixedVec`], a data structure that wraps a [`FixedVec`]
//! to efficiently store and access signed integers.
//!
//! ## ZigZag Encoding
//!
//! To compress signed integers effectively, `SFixedVec` uses **ZigZag encoding**.
//! This transformation maps signed integers to unsigned integers in a way that
//! values close to zero (both positive and negative) are represented by small
//! unsigned integers. This makes the data highly compressible with a fixed-width
//! scheme, as the required number of bits is determined by the magnitude of the
//! values, not their sign.
//!
//! This encoding is handled transparently. The API accepts and returns `i64`,
//! while the underlying storage and access logic is delegated to an inner [`FixedVec`].

pub mod builder;
pub mod iter;
#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "serde")]
mod serde;

use super::intvec::{FixedVec, FixedVecError};
pub use builder::{SFixedVecBuilder, SFixedVecFromIterBuilder};
pub use iter::SFixedVecIter;

use dsi_bitstream::prelude::{Endianness, ToInt, BE, LE};
use mem_dbg::{MemDbg, MemSize};

/// A compressed, randomly accessible vector of `i64` integers with fixed-width encoding.
///
/// `SFixedVec` acts as a wrapper around a [`FixedVec`] and handles the transparent
/// ZigZag encoding of signed integers. The performance characteristics of its
/// methods are nearly identical to their [`FixedVec`] counterparts, with only
/// the negligible overhead of the ZigZag transformation.
///
/// # Example
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[i64] = &;
///
/// // The builder will automatically determine the number of bits required
/// // after ZigZag encoding the values.
/// let s_fixed_vec = LESFixedVec::builder(&data).build().unwrap();
///
/// assert_eq!(s_fixed_vec.len(), data.len());
/// assert_eq!(s_fixed_vec.get(1), Some(-128));
/// ```
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct SFixedVec<E: Endianness> {
    /// The inner `FixedVec` that stores the ZigZag-encoded `u64` values.
    inner: FixedVec<E>,
}

impl<E: Endianness> SFixedVec<E> {
    /// Returns a builder for creating an [`SFixedVec`] from a slice of `i64`.
    ///
    /// This method is generic over `AsRef<[i64]>`, so it can accept `&[i64]`,
    /// `Vec<i64>`, etc. See [`SFixedVecBuilder`] for more details.
    pub fn builder<T: AsRef<[i64]> + ?Sized>(input: &T) -> SFixedVecBuilder<E> {
        SFixedVecBuilder::new(input.as_ref())
    }

    /// Returns a builder for creating a [`SFixedVec`] from an iterator.
    ///
    /// # Limitations
    /// This builder requires that the number of bits be specified manually.
    pub fn from_iter_builder<I: IntoIterator<Item = i64>>(
        iter: I,
        num_bits: usize,
    ) -> SFixedVecFromIterBuilder<E, I> {
        SFixedVecFromIterBuilder::new(iter, num_bits)
    }

    /// Returns the number of integers in the vector.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of bits used to encode each integer.
    pub fn num_bits(&self) -> usize {
        self.inner.num_bits()
    }

    /// Returns a clone of the underlying storage (`Vec<u64>`)
    ///
    /// # Note
    ///
    /// This values are zig-zag encoded, so they are not the original `i64` values.
    pub fn limbs(&self) -> Vec<u64> {
        self.inner.limbs()
    }

    /// Retrieves the signed integer at the specified index. Access is O(1).
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<i64> {
        self.inner.get(index).map(ToInt::to_int)
    }

    /// Retrieves the signed integer at the specified index without bounds checking.
    ///
    /// In debug builds, this method will panic if the index is out of bounds.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior in release builds.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> i64 {
        self.inner.get_unchecked(index).to_int()
    }

    /// Retrieves multiple signed integers at the specified indices.
    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<i64>, FixedVecError> {
        let unsigned_values = self.inner.get_many(indices)?;
        Ok(unsigned_values.into_iter().map(ToInt::to_int).collect())
    }

    /// Retrieves multiple signed integers at the specified indices without bounds checking.
    ///
    /// In debug builds, this method will panic if any index is out of bounds.
    ///
    /// # Safety
    /// Calling this method with any out-of-bounds index is undefined behavior in release builds.
    pub unsafe fn get_many_unchecked(&self, indices: &[usize]) -> Vec<i64> {
        self.inner
            .get_many_unchecked(indices)
            .into_iter()
            .map(ToInt::to_int)
            .collect()
    }

    /// Returns an iterator over the decompressed `i64` values.
    pub fn iter(&self) -> SFixedVecIter<E> {
        SFixedVecIter::new(self)
    }

    /// Consumes the [`SFixedVec`] and returns the underlying `Vec<i64>`.
    pub fn into_vec(self) -> Vec<i64> {
        self.inner
            .into_vec()
            .into_iter()
            .map(ToInt::to_int)
            .collect()
    }
}

/// A type alias for an [`SFixedVec`] with Big-Endian ([`BE`]) bitstream encoding.
pub type BESFixedVec = SFixedVec<BE>;

/// A type alias for an [`SFixedVec`] with Little-Endian ([`LE`]) bitstream encoding.
pub type LESFixedVec = SFixedVec<LE>;
