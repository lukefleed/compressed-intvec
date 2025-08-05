//! # A compressed, randomly accessible vector of signed `i64` integers.
//!
//! This module provides [`SIntVec`], a specialized vector for compressing signed
//! integer data.

use crate::{prelude::VariableCodecSpec, variable::{
    intvec::{IntVec, IntVecBitReader, IntVecError},
    sintvec::{
        builder::SIntVecFromIterBuilder, iter::SIntVecIter, slice::SIntVecSlice
    },
}};
use dsi_bitstream::{prelude::{BitRead, BitSeek, Codes, CodesRead, CodesWrite, Endianness, ToInt, BE, LE}, traits::BitWrite};
use mem_dbg::{MemDbg, MemSize};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

// Declare and export submodules.
mod builder;
mod iter;
#[cfg(feature = "parallel")]
mod parallel;
pub mod slice; // Make the module public

pub use builder::SIntVecBuilder;

/// A compressed, randomly accessible vector of signed `i64` integers.
///
/// [`SIntVec`] acts as a wrapper around [`IntVec`] that transparently handles the
/// encoding of signed integers (`i64`) into unsigned integers (`u64`) using
/// the ZigZag transformation.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct SIntVec<E: Endianness, B: AsRef<[u64]> = Vec<u64>> {
    /// The inner `IntVec` that stores the ZigZag-encoded `u64` values.
    inner: IntVec<E, B>,
}

// Manual serde implementation to handle generics correctly.
#[cfg(feature = "serde")]
impl<E: Endianness, B: AsRef<[u64]>> Serialize for SIntVec<E, B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, E: Endianness> Deserialize<'de> for SIntVec<E, Vec<u64>> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(SIntVec {
            inner: IntVec::<E, Vec<u64>>::deserialize(deserializer)?,
        })
    }
}

impl<E: Endianness> SIntVec<E, Vec<u64>> {
    /// Returns a builder for creating an [`SIntVec`] from a slice of data.
    pub fn builder<T: AsRef<[i64]> + ?Sized>(input: &T) -> SIntVecBuilder<E> {
        SIntVecBuilder::new(input.as_ref())
    }

    /// Returns a builder for creating a [`SIntVec`] from an iterator.
    pub fn from_iter_builder<I: IntoIterator<Item = i64>>(iter: I) -> SIntVecFromIterBuilder<E, I> {
        SIntVecFromIterBuilder::new(iter)
    }

    /// Creates an owned `SIntVec` directly from a slice of data.
    ///
    /// This is a convenient alias for `SIntVec::builder(slice).build()`.
    /// `VariableCodecSpec::Delta` and a default sampling rate of `k=16` will be used.
    /// To specify different parameters, use the builder directly.
    pub fn from_slice<T>(slice: &T) -> Result<Self, IntVecError>
    where
        T: AsRef<[i64]> + ?Sized,
        for<'a> crate::variable::intvec::IntVecBitWriter<E>:
            BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        Self::builder(slice)
            .k(16)
            .codec(VariableCodecSpec::Delta)
            .build()
    }
}

impl<E: Endianness, B: AsRef<[u64]>> SIntVec<E, B> {
    /// Creates an `SIntVec` view from an existing `IntVec`.
    ///
    /// This is the primary constructor for creating a zero-copy view. The provided
    /// `IntVec` is assumed to contain ZigZag-encoded data.
    pub fn from_parts(inner: IntVec<E, B>) -> Self {
        Self { inner }
    }

    /// Returns a reference to the inner `IntVec`.
    pub fn inner_ref(&self) -> &IntVec<E, B> {
        &self.inner
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the underlying `Codes` variant used for compression.
    pub fn encoding(&self) -> Codes {
        self.inner.encoding()
    }

    /// Returns the sampling rate `k` used during encoding.
    pub fn get_sampling_rate(&self) -> usize {
        self.inner.get_sampling_rate()
    }

    /// Returns the number of sample points stored in the vector.
    pub fn get_num_samples(&self) -> usize {
        self.inner.get_num_samples()
    }

    /// Returns a clone of the underlying storage (`Vec<u64>`).
    pub fn limbs(&self) -> Vec<u64> {
        self.inner.limbs()
    }

    /// Binary searches this vector for a given element.
    pub fn binary_search(&self, value: i64) -> Result<usize, usize>
    where
        for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.binary_search_by(|probe| probe.cmp(&value))
    }

    /// Binary searches this vector with a comparator function.
    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(i64) -> std::cmp::Ordering,
        for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.inner
            .binary_search_by(|probe_unsigned| f(probe_unsigned.to_int()))
    }

    /// Binary searches this vector with a key extraction function.
    #[inline]
    pub fn binary_search_by_key<B1, F>(&self, b: &B1, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(i64) -> B1,
        B1: Ord,
        for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Creates a zero-copy slice of this vector.
    pub fn slice(&self, start: usize, len: usize) -> Option<SIntVecSlice<E, B>>
    where
        for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.inner.slice(start, len).map(SIntVecSlice::new)
    }

    /// Splits the vector into two slices at a given index.
    pub fn split_at(&self, mid: usize) -> Option<(SIntVecSlice<E, B>, SIntVecSlice<E, B>)>
    where
        for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.inner
            .split_at(mid)
            .map(|(left, right)| (SIntVecSlice::new(left), SIntVecSlice::new(right)))
    }
}

impl<E: Endianness, B: AsRef<[u64]>> SIntVec<E, B>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Retrieves the signed integer at the specified index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<i64> {
        self.inner.get(index).map(ToInt::to_int)
    }

    /// Retrieves the signed integer at the specified index without bounds checking.
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> i64 {
        self.inner.get_unchecked(index).to_int()
    }

    /// Retrieves multiple signed integers at the specified indices.
    #[inline]
    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<i64>, IntVecError> {
        let unsigned_values = self.inner.get_many(indices)?;
        Ok(unsigned_values.into_iter().map(ToInt::to_int).collect())
    }

    /// Retrieves multiple signed integers at the specified indices without bounds checking.
    /// # Safety
    /// Calling this method with any out-of-bounds index is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_many_unchecked(&self, indices: &[usize]) -> Vec<i64> {
        self.inner
            .get_many_unchecked(indices)
            .into_iter()
            .map(ToInt::to_int)
            .collect()
    }

    /// Retrieves multiple signed integers from an iterator of indices.
    pub fn get_many_from_iter<I>(&self, indices: I) -> Result<Vec<i64>, IntVecError>
    where
        I: IntoIterator<Item = usize>,
    {
        let unsigned_values = self.inner.get_many_from_iter(indices)?;
        Ok(unsigned_values.into_iter().map(ToInt::to_int).collect())
    }

    /// Returns an iterator over the decompressed `i64` values.
    pub fn iter(&self) -> SIntVecIter<E, B> {
        SIntVecIter::new(self)
    }
}

macro_rules! impl_partial_eq_for_sint_slice {
    ($($t:ty),*) => {$(
        impl<E: Endianness, B: AsRef<[u64]>> PartialEq<Vec<$t>> for SIntVec<E, B>
        where
            for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
                + CodesRead<E>
                + BitSeek<Error = core::convert::Infallible>,
        {
            fn eq(&self, other: &Vec<$t>) -> bool {
                self.eq(&other[..])
            }
        }

        impl<E: Endianness, B: AsRef<[u64]>> PartialEq<&[$t]> for SIntVec<E, B>
        where
            for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
                + CodesRead<E>
                + BitSeek<Error = core::convert::Infallible>,
        {
            fn eq(&self, other: &&[$t]) -> bool {
                self.eq(*other)
            }
        }

        impl<E: Endianness, B: AsRef<[u64]>> PartialEq<[$t]> for SIntVec<E, B>
        where
            for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
                + CodesRead<E>
                + BitSeek<Error = core::convert::Infallible>,
        {
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

/// A type alias for an [`SIntVec`] with Big-Endian ([`BE`]) bitstream encoding.
pub type BESIntVec = SIntVec<BE>;

/// A type alias for an [`SIntVec`] with Little-Endian ([`LE`]) bitstream encoding.
pub type LESIntVec = SIntVec<LE>;