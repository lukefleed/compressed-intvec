//! # Zero-Copy Slices for `FixedVec`
//!
//! This module provides `FixedVecSlice`, a zero-copy, generic view into a
//! portion of a `FixedVec`.

use crate::fixed::{
    iter::FixedVecSliceIter,
    proxy::MutProxy,
    traits::{Storable, Word},
    FixedVec,
};
use dsi_bitstream::prelude::Endianness;
use std::ops::{Deref, DerefMut, Range};

/// A zero-copy view into a contiguous portion of a [`FixedVec`].
///
/// This struct is generic over the type of reference to the parent vector (`V`),
/// allowing it to represent both immutable slices (when `V` is `&FixedVec`)
/// and mutable slices (when `V` is `&mut FixedVec`).
///
/// It is created by the `slice`, `slice_mut`, `split_at`, etc., methods on a `FixedVec`.
#[derive(Debug)]
pub struct FixedVecSlice<V> {
    /// The parent vector reference (`&FixedVec` or `&mut FixedVec`).
    parent: V,
    /// The start index and length of the slice within the parent.
    range: Range<usize>,
}

// --- Common Implementation for both Immutable and Mutable Slices ---
impl<T, W, E, B, V> FixedVecSlice<V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
    /// Creates a new `FixedVecSlice`.
    ///
    /// This is `pub(super)` and is called by methods on `FixedVec`.
    /// It assumes the provided range is within the parent's bounds.
    pub(super) fn new(parent: V, range: Range<usize>) -> Self {
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
        // SAFETY: The bounds check has been performed.
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Retrieves the element at `index` within the slice without bounds checking.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is Undefined Behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        debug_assert!(index < self.len());
        self.parent.get_unchecked(self.range.start + index)
    }

    /// Returns an iterator over the values in the slice.
    pub fn iter(&self) -> FixedVecSliceIter<'_, T, W, E, B, V> {
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
            // SAFETY: The loop invariants ensure `mid` is always in bounds.
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

// --- Mutable-Only Implementation ---
impl<T, W, E, B, V> FixedVecSlice<V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>> + DerefMut,
{
    /// Returns a mutable proxy for an element at a given index within the slice.
    ///
    /// This allows for syntax like `*slice.at_mut(i).unwrap() = new_value;`.
    ///
    /// Returns `None` if the index is out of bounds.
    pub fn at_mut(&mut self, index: usize) -> Option<MutProxy<T, W, E, B>> {
        if index >= self.len() {
            return None;
        }
        // The proxy gets a mutable reference to the *parent* vector, but uses
        // the global index from the slice's perspective.
        Some(MutProxy::new(&mut self.parent, self.range.start + index))
    }
}

// --- PartialEq Implementations ---

// FixedVecSlice == FixedVecSlice
impl<T, W, E, B, V1, V2> PartialEq<FixedVecSlice<V2>> for FixedVecSlice<V1>
where
    T: Storable<W> + PartialEq,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V1: Deref<Target = FixedVec<T, W, E, B>>,
    V2: Deref<Target = FixedVec<T, W, E, B>>,
{
    fn eq(&self, other: &FixedVecSlice<V2>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

impl<T, W, E, B, V> Eq for FixedVecSlice<V>
where
    T: Storable<W> + Eq,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
}

// FixedVecSlice == FixedVec
impl<T, W, E, B, B2, V> PartialEq<FixedVec<T, W, E, B2>> for FixedVecSlice<V>
where
    T: Storable<W> + PartialEq,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    B2: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
    fn eq(&self, other: &FixedVec<T, W, E, B2>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

// FixedVec == FixedVecSlice
impl<T, W, E, B, B2, V> PartialEq<FixedVecSlice<V>> for FixedVec<T, W, E, B>
where
    T: Storable<W> + PartialEq,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    B2: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B2>>,
{
    fn eq(&self, other: &FixedVecSlice<V>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

/// Implements `PartialEq` for comparing a `FixedVecSlice` with a standard slice.
impl<T, W, E, B, T2, V> PartialEq<&[T2]> for FixedVecSlice<V>
where
    T: Storable<W> + PartialEq<T2>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    T2: Clone,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
    fn eq(&self, other: &&[T2]) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == *b)
    }
}


// FixedVecSlice == &FixedVec
impl<T, W, E, B, B2, V> PartialEq<&FixedVec<T, W, E, B2>> for FixedVecSlice<V>
where
    T: Storable<W> + PartialEq,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    B2: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
    fn eq(&self, other: &&FixedVec<T, W, E, B2>) -> bool {
        self.eq(*other)
    }
}