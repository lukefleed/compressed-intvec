//! # Zero-Copy Slices for `FixedVec`
//!
//! This module provides `FixedVecSlice`, a zero-copy view into a portion of a
//! `FixedVec`.

use crate::fixed::{iter::FixedVecSliceIter, traits::{Storable, Word}, FixedVec};
use dsi_bitstream::prelude::Endianness;
use std::ops::Range;

/// A zero-copy, immutable view into a contiguous portion of a [`FixedVec`].
///
/// This struct is created by the `slice` or `split_at` methods on a `FixedVec`.
/// It provides a read-only API similar to `FixedVec` itself but without owning
/// the underlying data.
#[derive(Debug, Clone)]
pub struct FixedVecSlice<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    parent: &'a FixedVec<T, W, E, B>,
    range: Range<usize>,
}

impl<'a, T, W, E, B> FixedVecSlice<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Creates a new `FixedVecSlice`. Assumes the range is within bounds.
    pub(super) fn new(parent: &'a FixedVec<T, W, E, B>, range: Range<usize>) -> Self {
        debug_assert!(range.end <= parent.len());
        Self { parent, range }
    }

    /// Returns the number of elements in the slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Returns `true` if the slice contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Retrieves the element at the specified index within the slice.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len() {
            return None;
        }
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Retrieves the element at `index` within the slice without bounds checking.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        debug_assert!(index < self.len());
        self.parent.get_unchecked(self.range.start + index)
    }

    /// Returns an iterator over the values in the slice.
    // The lifetime of the returned iterator is tied to the lifetime of `&self`, not `'a`.
    pub fn iter<'s>(&'s self) -> FixedVecSliceIter<'s, 'a, T, W, E, B> {
        FixedVecSliceIter::new(self)
    }

    /// Binary searches this slice for a given element.
    pub fn binary_search(&self, value: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        let mut low = 0;
        let mut high = self.len();

        while low < high {
            let mid = low + (high - low) / 2;
            let mid_val = unsafe { self.get_unchecked(mid) };
            
            match mid_val.cmp(value) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => return Ok(mid),
                std::cmp::Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }
}


// --- PartialEq Implementations ---

impl<'a, T, W, E, B> PartialEq for FixedVecSlice<'a, T, W, E, B>
where
    T: Storable<W> + PartialEq,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

impl<'a, T, W, E, B> Eq for FixedVecSlice<'a, T, W, E, B>
where
    T: Storable<W> + Eq,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{}


// --- Cross-Type PartialEq Implementations ---

// FixedVecSlice == FixedVec
impl<'a, T, W, E, B, B2> PartialEq<FixedVec<T, W, E, B2>> for FixedVecSlice<'a, T, W, E, B>
where
    T: Storable<W> + PartialEq,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    B2: AsRef<[W]>,
{
    fn eq(&self, other: &FixedVec<T, W, E, B2>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

// FixedVec == FixedVecSlice
impl<'a, T, W, E, B, B2> PartialEq<FixedVecSlice<'a, T, W, E, B2>> for FixedVec<T, W, E, B>
where
    T: Storable<W> + PartialEq,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    B2: AsRef<[W]>,
{
    fn eq(&self, other: &FixedVecSlice<'a, T, W, E, B2>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}


/// Implements `PartialEq` for comparing a `FixedVecSlice` with a standard slice.
impl<'a, T, W, E, B, T2> PartialEq<&[T2]> for FixedVecSlice<'a, T, W, E, B>
where
    T: Storable<W> + PartialEq<T2>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    T2: Clone,
{
    fn eq(&self, other: &&[T2]) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == *b)
    }
}