//! # `IntVec` Zero-Copy Slices
//!
//! This module provides [`IntVecSlice`], a zero-copy view into a portion of an
//! [`IntVec`].

use super::{IntVec, IntVecBitReader};
use dsi_bitstream::prelude::{BitRead, BitSeek, CodesRead, Endianness};
use std::cmp::Ordering;
use std::ops::Range;

/// A zero-copy slice of an [`IntVec`].
///
/// This struct provides a view into a contiguous portion of an [`IntVec`]
/// without copying the underlying data. It is created by the [`slice`] or
/// [`split_at`] methods on an [`IntVec`].
///
/// [`slice`]: IntVec::slice
/// [`split_at`]: IntVec::split_at
#[derive(Debug, Clone)]
pub struct IntVecSlice<'a, E: Endianness, B: AsRef<[u64]>> {
    /// A reference to the parent vector.
    vec: &'a IntVec<E, B>,
    /// The starting index of the slice within the parent vector.
    start: usize,
    /// The number of elements in the slice.
    len: usize,
}

impl<'a, E: Endianness, B: AsRef<[u64]>> IntVecSlice<'a, E, B> {
    /// Creates a new `IntVecSlice`.
    ///
    /// This is `pub(super)` and is called by [`IntVec::slice`].
    /// It assumes that bounds have already been checked.
    pub(super) fn new(vec: &'a IntVec<E, B>, range: Range<usize>) -> Self {
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
    pub fn get(&self, index: usize) -> Option<u64>
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
    ///
    /// The index is relative to the start of the slice.
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> u64
    where
        for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        debug_assert!(index < self.len, "Index out of bounds");
        self.vec.get_unchecked(self.start + index)
    }

    /// Binary searches this slice for a given element.
    ///
    /// If the slice is not sorted, the returned result is unspecified.
    pub fn binary_search(&self, value: u64) -> Result<usize, usize>
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
        F: FnMut(u64) -> Ordering,
        for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
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
        F: FnMut(u64) -> K,
        K: Ord,
        for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Returns an iterator over the values in the slice.
    pub fn iter(&self) -> IntVecSliceIter<'_, E, B> {
        IntVecSliceIter::new(self)
    }
}

/// An iterator over the decompressed `u64` values of an [`IntVecSlice`].
pub struct IntVecSliceIter<'a, E: Endianness, B: AsRef<[u64]>> {
    slice: &'a IntVecSlice<'a, E, B>,
    current_index: usize,
}

impl<'a, E: Endianness, B: AsRef<[u64]>> IntVecSliceIter<'a, E, B> {
    /// Creates a new iterator for a given `IntVecSlice`.
    fn new(slice: &'a IntVecSlice<'a, E, B>) -> Self {
        Self {
            slice,
            current_index: 0,
        }
    }
}

impl<E, B> Iterator for IntVecSliceIter<'_, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
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

impl<E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for IntVecSliceIter<'_, E, B>
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
impl<E, B, B2> PartialEq<IntVec<E, B2>> for IntVecSlice<'_, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    B2: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn eq(&self, other: &IntVec<E, B2>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

macro_rules! impl_partial_eq_for_uint_slice_for_slice {
    ($($t:ty),*) => {$(
        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<Vec<$t>> for IntVecSlice<'a, E, B>
        where
            for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
                + CodesRead<E>
                + BitSeek<Error = core::convert::Infallible>,
        {
            fn eq(&self, other: &Vec<$t>) -> bool {
                self.eq(&other[..])
            }
        }

        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<&[$t]> for IntVecSlice<'a, E, B>
        where
            for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
                + CodesRead<E>
                + BitSeek<Error = core::convert::Infallible>,
        {
            fn eq(&self, other: &&[$t]) -> bool {
                self.eq(*other)
            }
        }

        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<[$t]> for IntVecSlice<'a, E, B>
        where
            for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
                + CodesRead<E>
                + BitSeek<Error = core::convert::Infallible>,
        {
            fn eq(&self, other: &[$t]) -> bool {
                if self.len() != other.len() {
                    return false;
                }
                self.iter().eq(other.iter().map(|&x| x as u64))
            }
        }
    )*};
}

impl_partial_eq_for_uint_slice_for_slice!(u8, u16, u32, u64);