//! # `SFixedVec` Zero-Copy Slices
//!
//! This module provides [`SFixedVecSlice`], a zero-copy view into a portion of an
//! [`SFixedVec`].

use super::SFixedVec;
use crate::fixed::intvec::slice::FixedVecSlice;
use dsi_bitstream::prelude::{Endianness, ToInt};
use std::cmp::Ordering;

/// A zero-copy slice of an [`SFixedVec`].
///
/// This struct provides a view into a contiguous portion of an [`SFixedVec`]
/// without copying the underlying data. It is created by the [`slice`] or
/// [`split_at`] methods on an [`SFixedVec`].
///
/// [`slice`]: SFixedVec::slice
/// [`split_at`]: SFixedVec::split_at
#[derive(Debug, Clone)]
pub struct SFixedVecSlice<'a, E: Endianness, B: AsRef<[u64]>> {
    /// The inner slice of the ZigZag-encoded `u64` values.
    inner: FixedVecSlice<'a, E, B>,
}

impl<'a, E: Endianness, B: AsRef<[u64]>> SFixedVecSlice<'a, E, B> {
    /// Creates a new `SFixedVecSlice` that wraps an `FixedVecSlice`.
    pub(super) fn new(inner_slice: FixedVecSlice<'a, E, B>) -> Self {
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
    pub fn get(&self, index: usize) -> Option<i64> {
        self.inner.get(index).map(ToInt::to_int)
    }

    /// Retrieves the signed integer at `index` within the slice without bounds checking.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> i64 {
        self.inner.get_unchecked(index).to_int()
    }

    /// Binary searches this slice for a given element.
    ///
    /// If the slice is not sorted, the returned result is unspecified.
    pub fn binary_search(&self, value: i64) -> Result<usize, usize> {
        self.binary_search_by(|probe| probe.cmp(&value))
    }

    /// Binary searches this slice with a comparator function.
    ///
    /// If the slice is not sorted or the comparator does not reflect the
    /// slice's ordering, the result is unspecified.
    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(i64) -> Ordering,
    {
        self.inner
            .binary_search_by(|probe_unsigned| f(probe_unsigned.to_int()))
    }

    /// Binary searches this slice with a key extraction function.
    ///
    /// If the slice is not sorted by key, the result is unspecified.
    #[inline]
    pub fn binary_search_by_key<B1, F>(&self, b: &B1, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(i64) -> B1,
        B1: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Returns an iterator over the `i64` values in the slice.
    pub fn iter(&self) -> SFixedVecSliceIter<'_, E, B> {
        SFixedVecSliceIter::new(self)
    }
}

/// An iterator over the decompressed `i64` values of an [`SFixedVecSlice`].
pub struct SFixedVecSliceIter<'a, E: Endianness, B: AsRef<[u64]>> {
    slice: &'a SFixedVecSlice<'a, E, B>,
    current_index: usize,
}

impl<'a, E: Endianness, B: AsRef<[u64]>> SFixedVecSliceIter<'a, E, B> {
    /// Creates a new iterator for a given `SFixedVecSlice`.
    fn new(slice: &'a SFixedVecSlice<'a, E, B>) -> Self {
        Self {
            slice,
            current_index: 0,
        }
    }
}

impl<'a, E: Endianness, B: AsRef<[u64]>> Iterator for SFixedVecSliceIter<'a, E, B> {
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

impl<'a, E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for SFixedVecSliceIter<'a, E, B> {
    fn len(&self) -> usize {
        self.slice.len().saturating_sub(self.current_index)
    }
}

// Implementations of traits from the standard library
impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq for SFixedVecSlice<'a, E, B> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<'a, E: Endianness, B: AsRef<[u64]>> Eq for SFixedVecSlice<'a, E, B> {}

impl<'a, E: Endianness, B: AsRef<[u64]>, B2: AsRef<[u64]>> PartialEq<SFixedVec<E, B2>>
    for SFixedVecSlice<'a, E, B>
{
    fn eq(&self, other: &SFixedVec<E, B2>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

macro_rules! impl_partial_eq_for_sint_slice_for_slice {
    ($($t:ty),*) => {$(
        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<Vec<$t>> for SFixedVecSlice<'a, E, B> {
            fn eq(&self, other: &Vec<$t>) -> bool {
                self.eq(&other[..])
            }
        }

        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<&[$t]> for SFixedVecSlice<'a, E, B> {
            fn eq(&self, other: &&[$t]) -> bool {
                self.eq(*other)
            }
        }

        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<[$t]> for SFixedVecSlice<'a, E, B> {
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

impl<'a, E: Endianness, B: AsRef<[u64]>> IntoIterator for &'a SFixedVecSlice<'a, E, B> {
    type Item = i64;
    type IntoIter = SFixedVecSliceIter<'a, E, B>;

    /// Creates an iterator over the values of the slice.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
