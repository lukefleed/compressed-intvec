//! # `FixedVec` Zero-Copy Slices
//!
//! This module provides [`FixedVecSlice`], a zero-copy view into a portion of a
//! [`FixedVec`].

use super::FixedVec;
use dsi_bitstream::prelude::Endianness;
use std::cmp::Ordering;
use std::ops::Range;

/// A zero-copy slice of a [`FixedVec`].
///
/// This struct provides a view into a contiguous portion of a [`FixedVec`]
/// without copying the underlying data. It has an API similar to `FixedVec`
/// for accessing elements. It is created by the [`slice`] or [`split_at`]
/// methods on a [`FixedVec`].
///
/// It is generic over the backend `B` of the parent vector.
///
/// [`slice`]: FixedVec::slice
/// [`split_at`]: FixedVec::split_at
#[derive(Debug, Clone)]
pub struct FixedVecSlice<'a, E: Endianness, B: AsRef<[u64]>> {
    /// A reference to the parent vector.
    vec: &'a FixedVec<E, B>,
    /// The starting index of the slice within the parent vector.
    start: usize,
    /// The number of elements in the slice.
    len: usize,
}

impl<'a, E: Endianness, B: AsRef<[u64]>> FixedVecSlice<'a, E, B> {
    /// Creates a new `FixedVecSlice`.
    ///
    /// This is `pub(super)` and is called by [`FixedVec::slice`].
    /// It assumes that bounds have already been checked.
    pub(super) fn new(vec: &'a FixedVec<E, B>, range: Range<usize>) -> Self {
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

    /// Binary searches this slice for a given element.
    ///
    /// If the slice is not sorted, the returned result is unspecified.
    pub fn binary_search(&self, value: u64) -> Result<usize, usize> {
        self.binary_search_by(|probe| probe.cmp(&value))
    }

    /// Binary searches this slice with a comparator function.
    ///
    /// If the slice is not sorted or the comparator does not reflect the
    /// slice's ordering, the result is unspecified.
    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(u64) -> Ordering,
    {
        let mut low = 0;
        let mut high = self.len();

        while low < high {
            let mid = low + (high - low) / 2;
            // SAFETY: Bounds are checked by the loop.
            let cmp = f(unsafe { self.get_unchecked(mid) });
            match cmp {
                Ordering::Less => low = mid + 1,
                Ordering::Equal => return Ok(mid),
                Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }

    /// Binary searches this slice with a key extraction function.
    ///
    /// If the slice is not sorted by key, the result is unspecified.
    #[inline]
    pub fn binary_search_by_key<B1, F>(&self, b: &B1, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(u64) -> B1,
        B1: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Returns an iterator over the values in the slice.
    pub fn iter(&self) -> FixedVecSliceIter<'_, E, B> {
        FixedVecSliceIter::new(self)
    }
}

/// An iterator over the decompressed `u64` values of a [`FixedVecSlice`].
pub struct FixedVecSliceIter<'a, E: Endianness, B: AsRef<[u64]>> {
    slice: &'a FixedVecSlice<'a, E, B>,
    current_index: usize,
}

impl<'a, E: Endianness, B: AsRef<[u64]>> FixedVecSliceIter<'a, E, B> {
    /// Creates a new iterator for a given `FixedVecSlice`.
    fn new(slice: &'a FixedVecSlice<'a, E, B>) -> Self {
        Self {
            slice,
            current_index: 0,
        }
    }
}

impl<'a, E: Endianness, B: AsRef<[u64]>> Iterator for FixedVecSliceIter<'a, E, B> {
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

impl<'a, E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for FixedVecSliceIter<'a, E, B> {
    fn len(&self) -> usize {
        self.slice.len().saturating_sub(self.current_index)
    }
}

// Implementations of traits from the standard library
impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq for FixedVecSlice<'a, E, B> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

impl<'a, E: Endianness, B: AsRef<[u64]>> Eq for FixedVecSlice<'a, E, B> {}

impl<'a, E: Endianness, B: AsRef<[u64]>, B2: AsRef<[u64]>> PartialEq<FixedVec<E, B2>>
    for FixedVecSlice<'a, E, B>
{
    fn eq(&self, other: &FixedVec<E, B2>) -> bool {
        if self.len != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

macro_rules! impl_partial_eq_for_uint_slice_for_slice {
    ($($t:ty),*) => {$(
        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<Vec<$t>> for FixedVecSlice<'a, E, B> {
            fn eq(&self, other: &Vec<$t>) -> bool {
                self.eq(&other[..])
            }
        }

        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<&[$t]> for FixedVecSlice<'a, E, B> {
            fn eq(&self, other: &&[$t]) -> bool {
                self.eq(*other)
            }
        }

        impl<'a, E: Endianness, B: AsRef<[u64]>> PartialEq<[$t]> for FixedVecSlice<'a, E, B> {
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

impl<'a, E: Endianness, B: AsRef<[u64]>> IntoIterator for &'a FixedVecSlice<'a, E, B> {
    type Item = u64;
    type IntoIter = FixedVecSliceIter<'a, E, B>;

    /// Creates an iterator over the values of the slice.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
