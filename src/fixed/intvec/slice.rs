//! # `FixedVec` Zero-Copy Slices
//!
//! This module provides [`FixedVecSlice`], a zero-copy view into a portion of a
//! [`FixedVec`].

use super::FixedVec;
use dsi_bitstream::prelude::Endianness;
use std::ops::Range;

/// A zero-copy slice of a [`FixedVec`].
///
/// This struct provides a view into a contiguous portion of a [`FixedVec`]
/// without copying the underlying data. It has an API similar to `FixedVec`
/// for accessing elements. It is created by the [`slice`] or [`split_at`]
/// methods on a [`FixedVec`].
///
/// [`slice`]: FixedVec::slice
/// [`split_at`]: FixedVec::split_at
#[derive(Debug, Clone, Copy)]
pub struct FixedVecSlice<'a, E: Endianness> {
    /// A reference to the parent vector.
    vec: &'a FixedVec<E>,
    /// The starting index of the slice within the parent vector.
    start: usize,
    /// The number of elements in the slice.
    len: usize,
}

impl<'a, E: Endianness> FixedVecSlice<'a, E> {
    /// Creates a new `FixedVecSlice`.
    ///
    /// This is `pub(super)` and is called by [`FixedVec::slice`].
    /// It assumes that bounds have already been checked.
    pub(super) fn new(vec: &'a FixedVec<E>, range: Range<usize>) -> Self {
        Self {
            vec,
            start: range.start,
            len: range.len(),
        }
    }

    /// Returns the number of elements in the slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slice contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Retrieves the element at the specified index within the slice.
    ///
    /// The index is relative to the start of the slice.
    #[inline]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }
        // SAFETY: The bounds check has been performed.
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Retrieves the element at `index` within the slice without bounds checking.
    ///
    /// The index is relative to the start of the slice.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(index < self.len, "Index out of bounds");
        self.vec.get_unchecked(self.start + index)
    }

    /// Returns an iterator over the values in the slice.
    pub fn iter(&self) -> FixedVecSliceIter<'a, E> {
        FixedVecSliceIter::new(*self)
    }
}

/// An iterator over the decompressed `u64` values of a [`FixedVecSlice`].
pub struct FixedVecSliceIter<'a, E: Endianness> {
    slice: FixedVecSlice<'a, E>,
    current_index: usize,
}

impl<'a, E: Endianness> FixedVecSliceIter<'a, E> {
    /// Creates a new iterator for a given `FixedVecSlice`.
    fn new(slice: FixedVecSlice<'a, E>) -> Self {
        Self {
            slice,
            current_index: 0,
        }
    }
}

impl<'a, E: Endianness> Iterator for FixedVecSliceIter<'a, E> {
    type Item = u64;

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

impl<'a, E: Endianness> ExactSizeIterator for FixedVecSliceIter<'a, E> {
    fn len(&self) -> usize {
        self.slice.len().saturating_sub(self.current_index)
    }
}

// Implementations of traits from the standard library
impl<'a, 'b, E: Endianness> PartialEq<FixedVecSlice<'b, E>> for FixedVecSlice<'a, E> {
    fn eq(&self, other: &FixedVecSlice<'b, E>) -> bool {
        if self.len != other.len {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

impl<'a, E: Endianness> Eq for FixedVecSlice<'a, E> {}

impl<'a, E: Endianness> PartialEq<FixedVec<E>> for FixedVecSlice<'a, E> {
    fn eq(&self, other: &FixedVec<E>) -> bool {
        if self.len != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

impl<'a, E: Endianness, T: AsRef<[u64]>> PartialEq<T> for FixedVecSlice<'a, E> {
    fn eq(&self, other: &T) -> bool {
        let other_slice = other.as_ref();
        if self.len() != other_slice.len() {
            return false;
        }
        self.iter().eq(other_slice.iter().copied())
    }
}