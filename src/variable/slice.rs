//! # `IntVec` Zero-Copy Slices
//!
//! This module provides [`IntVecSlice`], a zero-copy view into a portion of a
//! generic [`IntVec`].

use super::{traits::Storable, IntVec, IntVecBitReader};
use dsi_bitstream::prelude::{BitRead, BitSeek, CodesRead, Endianness};
use std::cmp::Ordering;
use std::ops::Range;

/// A zero-copy slice of an [`IntVec`].
///
/// This struct provides a view into a contiguous portion of an [`IntVec`]
/// without copying the underlying data. It is created by the [`slice`] or
/// [`split_at`] methods on an [`IntVec`].
///
/// [`slice`]: crate::variable::IntVec::slice
/// [`split_at`]: crate::variable::IntVec::split_at
#[derive(Debug, Clone)]
pub struct IntVecSlice<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> {
    /// A reference to the parent vector.
    vec: &'a IntVec<T, E, B>,
    /// The starting index of the slice within the parent vector.
    start: usize,
    /// The number of elements in the slice.
    len: usize,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> IntVecSlice<'a, T, E, B> {
    /// Creates a new `IntVecSlice`.
    pub(super) fn new(vec: &'a IntVec<T, E, B>, range: Range<usize>) -> Self {
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
    #[inline]
    pub fn get(&self, index: usize) -> Option<T>
    where
        for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        if index >= self.len {
            return None;
        }
        self.vec.get(self.start + index)
    }

    /// Retrieves the element at `index` within the slice without bounds checking.
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> T
    where
        for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        debug_assert!(index < self.len, "Index out of bounds");
        self.vec.get_unchecked(self.start + index)
    }

    /// Returns an iterator over the values in the slice.
    pub fn iter(&self) -> IntVecSliceIter<'_, T, E, B>
    where
        for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        IntVecSliceIter::new(self)
    }
}

impl<'a, T, E, B> IntVecSlice<'a, T, E, B>
where
    T: Storable + Ord,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Binary searches this slice for a given element.
    pub fn binary_search(&self, value: &T) -> Result<usize, usize> {
        self.binary_search_by(|probe| probe.cmp(value))
    }

    /// Binary searches this slice with a comparator function.
    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> Ordering,
    {
        let mut low = 0;
        let mut high = self.len();
        let mut reader = self.vec.reader();

        while low < high {
            let mid = low + (high - low) / 2;
            // SAFETY: Bounds are checked by the loop and the slice's construction.
            let cmp = f(unsafe { reader.get_unchecked(self.start + mid) });
            match cmp {
                Ordering::Less => low = mid + 1,
                Ordering::Equal => return Ok(mid),
                Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }

    /// Binary searches this slice with a key extraction function.
    #[inline]
    pub fn binary_search_by_key<K, F>(&self, b: &K, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> K,
        K: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }
}

/// An iterator over the decompressed values of an [`IntVecSlice`].
pub struct IntVecSliceIter<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> {
    slice: &'a IntVecSlice<'a, T, E, B>,
    current_index: usize,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> IntVecSliceIter<'a, T, E, B> {
    /// Creates a new iterator for a given `IntVecSlice`.
    fn new(slice: &'a IntVecSlice<'a, T, E, B>) -> Self {
        Self {
            slice,
            current_index: 0,
        }
    }
}

impl<'a, T, E, B> Iterator for IntVecSliceIter<'a, T, E, B>
where
    T: Storable,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;

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

impl<'a, T, E, B> ExactSizeIterator for IntVecSliceIter<'a, T, E, B>
where
    T: Storable,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn len(&self) -> usize {
        self.slice.len().saturating_sub(self.current_index)
    }
}