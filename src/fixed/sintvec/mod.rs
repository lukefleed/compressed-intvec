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
pub mod slice;

#[cfg(feature = "simd")]
use std::simd::num::SimdUint;

#[cfg(feature = "serde")]
mod serde;

use crate::fixed::sintvec::iter::SFixedVecIntoIter;

use super::intvec::{FixedVec, FixedVecError};
pub use builder::{SFixedVecBuilder, SFixedVecFromIterBuilder};
pub use iter::SFixedVecIter;
pub use slice::SFixedVecSlice;

use common_traits::SignedInt;
use dsi_bitstream::prelude::{BitWrite, Endianness, ToInt, ToNat, BE, LE};
use mem_dbg::{MemDbg, MemSize};
use std::cmp::Ordering;

/// A compressed, randomly accessible vector of `i64` integers with fixed-width encoding.
///
/// `SFixedVec` acts as a wrapper around a [`FixedVec`] and handles the transparent
/// ZigZag encoding of signed integers. The performance characteristics of its
/// methods are nearly identical to their [`FixedVec`] counterparts, with only
/// the negligible overhead of the ZigZag transformation.
///
/// It is generic over the backend storage `B`, which can be an owned `Vec<u64>`
/// or a borrowed slice `&[u64]`.
/// A compressed, randomly accessible vector of `i64` integers with fixed-width encoding.
///
/// `SFixedVec` acts as a wrapper around a [`FixedVec`] and handles the transparent
/// ZigZag encoding of signed integers. The performance characteristics of its
/// methods are nearly identical to their [`FixedVec`] counterparts, with only
/// the negligible overhead of the ZigZag transformation.
///
/// It is generic over the backend storage `B`, which can be an owned `Vec<u64>`
/// or a borrowed slice `&[u64]`.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct SFixedVec<E: Endianness, B: AsRef<[u64]> = Vec<u64>> {
    /// The inner `FixedVec` that stores the ZigZag-encoded `u64` values.
    inner: FixedVec<E, B>,
}

impl<E: Endianness> SFixedVec<E, Vec<u64>> {
    /// Returns a builder for creating an owned [`SFixedVec`] from a slice of signed integers.
    ///
    /// This method is generic and can accept slices of `i8`, `i16`, `i32`, and `i64`.
    pub fn builder<I, T>(input: &T) -> SFixedVecBuilder<E, I>
    where
        I: ToNat + Copy + SignedInt,
        <I as SignedInt>::UnsignedInt: Into<u64> + Ord + Copy + Default,
        T: AsRef<[I]> + ?Sized,
    {
        SFixedVecBuilder::new(input.as_ref())
    }

    /// Creates an owned `SFixedVec` directly from a slice of data.
    ///
    /// This is a convenient alias for `SFixedVec::builder(slice).build()`.
    /// The bit width will be automatically determined using the `BitWidth::Minimal` strategy.
    /// To specify a different strategy, use the builder directly.
    pub fn from_slice<I, T>(slice: &T) -> Result<Self, FixedVecError>
    where
        I: ToNat + Copy + SignedInt,
        <I as SignedInt>::UnsignedInt: Into<u64> + Ord + Copy + Default,
        T: AsRef<[I]> + ?Sized,
        Self: Sized,
        crate::fixed::intvec::builder::FixedVecBitWriter<E>:
            BitWrite<E, Error = core::convert::Infallible>,
    {
        Self::builder(slice).build()
    }

    /// Returns a builder for creating an owned [`SFixedVec`] from an iterator.
    ///
    /// # Limitations
    /// This builder requires that the number of bits be specified manually.
    pub fn from_iter_builder<I: IntoIterator<Item = i64>>(
        iter: I,
        num_bits: usize,
    ) -> SFixedVecFromIterBuilder<E, I> {
        SFixedVecFromIterBuilder::new(iter, num_bits)
    }
}

impl<E: Endianness, B: AsRef<[u64]>> SFixedVec<E, B> {
    /// Creates an `SFixedVec` view from an existing `FixedVec`.
    ///
    /// This is the primary constructor for creating a zero-copy view. The provided
    /// `FixedVec` is assumed to contain ZigZag-encoded data.
    pub fn from_parts(inner: FixedVec<E, B>) -> Self {
        Self { inner }
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

    /// Returns a zero-copy, read-only slice of the underlying storage (`&[u64]`).
    ///
    /// # Note
    ///
    /// The values are ZigZag encoded, so they are not the original `i64` values.
    pub fn as_limbs(&self) -> &[u64] {
        self.inner.as_limbs()
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
    ///
    /// This method is transparently accelerated by SIMD instructions when the
    /// `simd` feature is enabled.
    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<i64>, FixedVecError> {
        // The underlying get_many checks for out-of-bounds indices.
        let unsigned_values = self.inner.get_many(indices)?;
        // SAFETY: The bounds have been checked by the inner call.
        Ok(
            // Convert the unsigned values to signed integers using the ZigZag transformation.
            // This is fast and can be SIMD-accelerated if the `simd` feature is enabled.
            unsigned_values.into_iter().map(ToInt::to_int).collect(),
        )
    }

    /// Retrieves multiple signed integers at the specified indices without bounds checking.
    ///
    /// This method is transparently accelerated by SIMD instructions when the
    /// `simd` feature is enabled.
    ///
    /// In debug builds, this method will panic if any index is out of bounds.
    ///
    /// # Safety
    /// Calling this method with any out-of-bounds index is undefined behavior in release builds.
    pub unsafe fn get_many_unchecked(&self, indices: &[usize]) -> Vec<i64> {
        // First, get the raw u64 values using the (potentially SIMD-accelerated)
        // inner FixedVec's method.
        let unsigned_values = self.inner.get_many_unchecked(indices);
        // Then, apply the inverse ZigZag transformation. This part is also
        // SIMD-accelerated if the `simd` feature is enabled.
        unsigned_values.into_iter().map(ToInt::to_int).collect()
    }

    /// Creates a zero-copy slice of this vector.
    pub fn slice(&self, start: usize, len: usize) -> Option<SFixedVecSlice<E, B>> {
        self.inner.slice(start, len).map(SFixedVecSlice::new)
    }

    /// Splits the vector into two slices at a given index.
    pub fn split_at(&self, mid: usize) -> Option<(SFixedVecSlice<E, B>, SFixedVecSlice<E, B>)> {
        self.inner
            .split_at(mid)
            .map(|(left, right)| (SFixedVecSlice::new(left), SFixedVecSlice::new(right)))
    }

    /// Binary searches this vector for a given element.
    ///
    /// If the vector is not sorted, the returned result is unspecified and
    /// meaningless. For `binary_search` to work correctly, the vector must be
    /// sorted in ascending order.
    pub fn binary_search(&self, value: i64) -> Result<usize, usize> {
        self.binary_search_by(|probe| probe.cmp(&value))
    }

    /// Binary searches this vector with a comparator function.
    ///
    /// The comparator function should return an `Ordering` that indicates
    /// whether its argument is `Less`, `Equal` or `Greater` than the desired
    /// target. If the vector is not sorted or if the comparator does not
    /// reflect the vector's ordering, the returned result is unspecified.
    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(i64) -> Ordering,
    {
        self.inner
            .binary_search_by(|probe_unsigned| f(probe_unsigned.to_int()))
    }

    /// Binary searches this vector with a key extraction function.
    ///
    /// If the vector is not sorted by the key, the returned result is
    /// unspecified.
    #[inline]
    pub fn binary_search_by_key<B1, F>(&self, b: &B1, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(i64) -> B1,
        B1: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Returns an iterator over the decompressed `i64` values.
    pub fn iter(&self) -> SFixedVecIter<E, B> {
        SFixedVecIter::new(self)
    }
}

impl<E: Endianness> SFixedVec<E, Vec<u64>> {
    /// Consumes the owned [`SFixedVec`] and returns the underlying `Vec<i64>`.
    pub fn into_vec(self) -> Vec<i64> {
        self.into_iter().collect()
    }
}

// Implementations of traits from the standard library
impl<E: Endianness, B: AsRef<[u64]>, B2: AsRef<[u64]>> PartialEq<SFixedVec<E, B2>>
    for SFixedVec<E, B>
{
    /// Checks for equality between two `SFixedVec` instances, regardless of backend.
    fn eq(&self, other: &SFixedVec<E, B2>) -> bool {
        self.inner == other.inner
    }
}

impl<E: Endianness, B: AsRef<[u64]>> Eq for SFixedVec<E, B> {}

macro_rules! impl_partial_eq_for_sint_slice {
    ($($t:ty),*) => {$(
        impl<E: Endianness, B: AsRef<[u64]>> PartialEq<&[$t]> for SFixedVec<E, B> {
            fn eq(&self, other: &&[$t]) -> bool {
                self.eq(*other)
            }
        }

        impl<E: Endianness, B: AsRef<[u64]>> PartialEq<[$t]> for SFixedVec<E, B> {
            fn eq(&self, other: &[$t]) -> bool {
                if self.len() != other.len() {
                    return false;
                }
                self.iter().eq(other.iter().map(|&x| x as i64))
            }
        }
    )*};
}

impl_partial_eq_for_sint_slice!(i8, i16, i32, i64);

impl<'a, E: Endianness, B: AsRef<[u64]>> IntoIterator for &'a SFixedVec<E, B> {
    type Item = i64;
    type IntoIter = SFixedVecIter<'a, E, B>;

    /// Creates an iterator over the values of the `SFixedVec`.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<E: Endianness> IntoIterator for SFixedVec<E, Vec<u64>> {
    type Item = i64;
    type IntoIter = SFixedVecIntoIter<E>;

    /// Consumes the `SFixedVec` and creates an iterator over its values.
    fn into_iter(self) -> Self::IntoIter {
        SFixedVecIntoIter::new(self)
    }
}

impl<E: Endianness, B: AsRef<[u64]>, B2: AsRef<[u64]>> PartialEq<SFixedVecSlice<'_, E, B2>>
    for SFixedVec<E, B>
{
    fn eq(&self, other: &SFixedVecSlice<'_, E, B2>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

/// A type alias for an owned [`SFixedVec`] with Big-Endian ([`BE`]) bitstream encoding.
pub type BESFixedVec = SFixedVec<BE, Vec<u64>>;

/// A type alias for an owned [`SFixedVec`] with Little-Endian ([`LE`]) bitstream encoding.
pub type LESFixedVec = SFixedVec<LE, Vec<u64>>;
