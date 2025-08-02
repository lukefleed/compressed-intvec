//! # `SFixedVec` Zero-Copy Slices
//!
//! This module provides [`SFixedVecSlice`], a zero-copy view into a portion of an
//! [`SFixedVec`].

use super::SFixedVec;
use crate::fixed::intvec::slice::FixedVecSlice;
use dsi_bitstream::prelude::{Endianness, ToInt};

/// A zero-copy slice of an [`SFixedVec`].
///
/// This struct provides a view into a contiguous portion of an [`SFixedVec`]
/// without copying the underlying data. It is created by the [`slice`] or
/// [`split_at`] methods on an [`SFixedVec`].
///
/// [`slice`]: SFixedVec::slice
/// [`split_at`]: SFixedVec::split_at
#[derive(Debug, Clone, Copy)]
pub struct SFixedVecSlice<'a, E: Endianness> {
    /// The inner slice of the ZigZag-encoded `u64` values.
    inner: FixedVecSlice<'a, E>,
}

impl<'a, E: Endianness> SFixedVecSlice<'a, E> {
    /// Creates a new `SFixedVecSlice` that wraps an `FixedVecSlice`.
    pub(super) fn new(inner_slice: FixedVecSlice<'a, E>) -> Self {
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

    /// Returns an iterator over the `i64` values in the slice.
    pub fn iter(&self) -> SFixedVecSliceIter<'a, E> {
        SFixedVecSliceIter::new(*self)
    }
}

/// An iterator over the decompressed `i64` values of an [`SFixedVecSlice`].
pub struct SFixedVecSliceIter<'a, E: Endianness> {
    slice: SFixedVecSlice<'a, E>,
    current_index: usize,
}

impl<'a, E: Endianness> SFixedVecSliceIter<'a, E> {
    /// Creates a new iterator for a given `SFixedVecSlice`.
    fn new(slice: SFixedVecSlice<'a, E>) -> Self {
        Self {
            slice,
            current_index: 0,
        }
    }
}

impl<'a, E: Endianness> Iterator for SFixedVecSliceIter<'a, E> {
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

impl<'a, E: Endianness> ExactSizeIterator for SFixedVecSliceIter<'a, E> {
    fn len(&self) -> usize {
        self.slice.len().saturating_sub(self.current_index)
    }
}

// Implementations of traits from the standard library
impl<'a, 'b, E: Endianness> PartialEq<SFixedVecSlice<'b, E>> for SFixedVecSlice<'a, E> {
    fn eq(&self, other: &SFixedVecSlice<'b, E>) -> bool {
        self.inner == other.inner
    }
}

impl<'a, E: Endianness> Eq for SFixedVecSlice<'a, E> {}

impl<'a, E: Endianness> PartialEq<SFixedVec<E>> for SFixedVecSlice<'a, E> {
    fn eq(&self, other: &SFixedVec<E>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

impl<'a, E: Endianness, T: AsRef<[i64]>> PartialEq<T> for SFixedVecSlice<'a, E> {
    fn eq(&self, other: &T) -> bool {
        let other_slice = other.as_ref();
        if self.len() != other_slice.len() {
            return false;
        }
        self.iter().eq(other_slice.iter().copied())
    }
}