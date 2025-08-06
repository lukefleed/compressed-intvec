//! # Zero-Copy Views for `FixedVec`
//!
//! This module provides structs for creating zero-copy views (slices) of a
//! `FixedVec`.

use crate::fixed::{FixedVec, traits::{Storable, Word}};
use dsi_bitstream::prelude::Endianness;
use std::ops::Range;

/// A zero-copy, immutable view into a contiguous portion of a [`FixedVec`].
///
/// This struct is created by the `slice` or `split_at` methods on a `FixedVec`.
/// It provides a read-only API similar to `FixedVec` itself but without owning
/// the underlying data.
#[derive(Debug, Clone)]
pub struct FixedVecView<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    parent: &'a FixedVec<T, W, E, B>,
    range: Range<usize>,
}

impl<'a, T, W, E, B> FixedVecView<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Creates a new `FixedVecView`. Assumes the range is within bounds.
    pub(super) fn new(parent: &'a FixedVec<T, W, E, B>, range: Range<usize>) -> Self {
        debug_assert!(range.end <= parent.len());
        Self { parent, range }
    }

    /// Returns the number of elements in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Returns `true` if the view contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Retrieves the element at the specified index within the view.
    ///
    /// The index is relative to the start of the view.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len() {
            return None;
        }
        // SAFETY: Bounds check was performed. The index is relative to the slice.
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Retrieves the element at `index` within the view without bounds checking.
    ///
    /// # Safety
    /// Calling this with an out-of-bounds index is Undefined Behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        debug_assert!(index < self.len());
        self.parent.get_unchecked(self.range.start + index)
    }

    /// Returns an iterator over the values in the view.
    pub fn iter(&self) -> crate::fixed::iter::FixedVecIter<T, W, E, B> {
        // This is a bit of a workaround to create an iterator for a slice.
        // We create a temporary FixedVec that acts as the slice view.
        // A more advanced implementation would have a dedicated slice iterator.
        unimplemented!("Iterator for slices is not yet implemented cleanly.");
    }

/// Binary searches this view for a given element.
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

    /// Binary searches this view with a comparator function.
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> std::cmp::Ordering, // Accetta T, non &T
    {
        let mut low = 0;
        let mut high = self.len();

        while low < high {
            let mid = low + (high - low) / 2;
            let mid_val = unsafe { self.get_unchecked(mid) };
            // Passa la proprietà di mid_val alla closure
            let cmp = f(mid_val); 
            match cmp {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => return Ok(mid),
                std::cmp::Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }
}