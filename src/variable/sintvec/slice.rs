//! # `SIntVec` Zero-Copy Slices
//!
//! This module provides [`SIntVecSlice`], a zero-copy view into a portion of an
//! [`SIntVec`].

use dsi_bitstream::prelude::{BitRead, BitSeek, CodesRead, Endianness, ToInt};
use std::cmp::Ordering;

use crate::{prelude::SIntVec, variable::intvec::{IntVecBitReader, IntVecSlice}};

/// A zero-copy slice of an [`SIntVec`].
///
/// This struct provides a view into a contiguous portion of an [`SIntVec`]
/// without copying the underlying data. It is created by the [`slice`] or
/// [`split_at`] methods on an [`SIntVec`].
///
/// [`slice`]: SIntVec::slice
/// [`split_at`]: SIntVec::split_at
#[derive(Debug, Clone)]
pub struct SIntVecSlice<'a, E: Endianness, B: AsRef<[u64]>> {
    /// The inner slice of the ZigZag-encoded `u64` values.
    inner: IntVecSlice<'a, E, B>,
}

impl<'a, E: Endianness, B: AsRef<[u64]>> SIntVecSlice<'a, E, B> {
    /// Creates a new `SIntVecSlice` that wraps an `IntVecSlice`.
    pub(super) fn new(inner_slice: IntVecSlice<'a, E, B>) -> Self {
        Self { inner: inner_slice }
    }

    /// Returns the number of elements in the slice.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the slice contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Retrieves the signed integer at the specified index within the slice.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<i64>
    where
        for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.inner.get(index).map(ToInt::to_int)
    }

    /// Retrieves the signed integer at `index` within the slice without bounds checking.
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> i64
    where
        for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.inner.get_unchecked(index).to_int()
    }

    /// Binary searches this slice for a given element.
    pub fn binary_search(&self, value: i64) -> Result<usize, usize>
    where
        for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.binary_search_by(|probe| probe.cmp(&value))
    }

    /// Binary searches this slice with a comparator function.
    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(i64) -> Ordering,
        for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.inner
            .binary_search_by(|probe_unsigned| f(probe_unsigned.to_int()))
    }

    /// Binary searches this slice with a key extraction function.
    #[inline]
    pub fn binary_search_by_key<K, F>(&self, b: &K, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(i64) -> K,
        K: Ord,
        for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Returns an iterator over the `i64` values in the slice.
    pub fn iter(&self) -> SIntVecSliceIter<'_, E, B> {
        SIntVecSliceIter::new(self)
    }
}

/// An iterator over the decompressed `i64` values of an [`SIntVecSlice`].
pub struct SIntVecSliceIter<'a, E: Endianness, B: AsRef<[u64]>> {
    slice: &'a SIntVecSlice<'a, E, B>,
    current_index: usize,
}

impl<'a, E: Endianness, B: AsRef<[u64]>> SIntVecSliceIter<'a, E, B> {
    /// Creates a new iterator for a given `SIntVecSlice`.
    fn new(slice: &'a SIntVecSlice<'a, E, B>) -> Self {
        Self {
            slice,
            current_index: 0,
        }
    }
}

impl<'a, E, B> Iterator for SIntVecSliceIter<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.slice.len() {
            return None;
        }
        // SAFETY: The iterator's logic guarantees the index is in bounds.
        let value = unsafe { self.slice.get_unchecked(self.current_index) };
        self.current_index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len().saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

impl<'a, E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for SIntVecSliceIter<'a, E, B>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn len(&self) -> usize {
        self.slice.len().saturating_sub(self.current_index)
    }
}

// Implementations of traits from the standard library
impl<'a, E, B, B2> PartialEq<SIntVec<E, B2>> for SIntVecSlice<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    B2: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn eq(&self, other: &SIntVec<E, B2>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

macro_rules! impl_partial_eq_for_sint_slice_for_slice {
    ($($t:ty),*) => {$(
        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<Vec<$t>> for SIntVecSlice<'a, E, B>
        where
            for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
                + CodesRead<E>
                + BitSeek<Error = core::convert::Infallible>,
        {
            fn eq(&self, other: &Vec<$t>) -> bool {
                self.eq(&other[..])
            }
        }

        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<&[$t]> for SIntVecSlice<'a, E, B>
        where
            for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
                + CodesRead<E>
                + BitSeek<Error = core::convert::Infallible>,
        {
            fn eq(&self, other: &&[$t]) -> bool {
                self.eq(*other)
            }
        }

        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<[$t]> for SIntVecSlice<'a, E, B>
        where
            for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
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

impl_partial_eq_for_sint_slice_for_slice!(i8, i16, i32, i64);