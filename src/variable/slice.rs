//! An immutable, zero-copy slice of an [`IntVec`].
//!
//! This module provides [`IntVecSlice`], a view into a contiguous portion of an
//! [`IntVec`]. Slices are immutable and do not own their data; they borrow it
//! from the parent `IntVec`.
//!
//! [`IntVec`]: crate::variable::IntVec

use super::{traits::Storable, IntVec, IntVecBitReader};
use dsi_bitstream::prelude::{BitRead, BitSeek, CodesRead, Endianness};
use std::cmp::Ordering;
use std::ops::Range;

/// An immutable, zero-copy slice of an [`IntVec`].
///
/// This struct provides a view into a contiguous portion of an [`IntVec`]
/// without copying the underlying compressed data. It is created by the
/// [`slice`](super::IntVec::slice) or [`split_at`](super::IntVec::split_at)
/// methods on an [`IntVec`].
///
/// All operations on an [`IntVecSlice`] are relative to the start of the slice,
/// not the parent vector.
///
/// # Examples
///
/// ```
/// use compressed_intvec::variable::{IntVec, UIntVec};
///
/// let data: Vec<u32> = (0..100).collect();
/// let vec: UIntVec<u32> = IntVec::from_slice(&data).unwrap();
///
/// // Create a slice of the elements from index 20 to 49
/// let slice = vec.slice(20, 30).unwrap();
///
/// assert_eq!(slice.len(), 30);
///
/// // Accessing an element of the slice
/// // Index 5 of the slice corresponds to index 25 of the original vector
/// assert_eq!(slice.get(5), Some(25));
///
/// // Iterating over the slice
/// let mut slice_sum = 0;
/// for value in slice.iter() {
///     slice_sum += value;
/// }
/// assert_eq!(slice_sum, (20..50).sum());
/// ```
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
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slice contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the element at the specified index within the slice, or `None` if
    /// the index is out of bounds.
    ///
    /// The index is relative to the start of the slice.
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
        // The actual index into the parent vector is `self.start + index`.
        self.vec.get(self.start + index)
    }

    /// Returns the element at `index` within the slice without bounds checking.
    ///
    /// The index is relative to the start of the slice.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    /// The caller must ensure that `index < self.len()`.
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
    ///
    /// If the value is found, returns `Ok(usize)` with the index of the
    /// matching element within the slice. If not found, returns `Err(usize)`
    /// with the insertion point to maintain order.
    pub fn binary_search(&self, value: &T) -> Result<usize, usize> {
        self.binary_search_by(|probe| probe.cmp(value))
    }

    /// Binary searches this slice with a custom comparison function.
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
            // SAFETY: Bounds are checked by the loop invariants and the slice's
            // construction, so the index into the parent vector is always valid.
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
///
/// This struct is created by the [`iter`](IntVecSlice::iter) method.
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
