//! # Zero-Copy Slices
//!
//! This module provides [`FixedVecSlice`], a zero-copy view into a portion of a
//! [`FixedVec`]. Slices can be created from both immutable and mutable vectors
//! and provide a way to operate on a sub-region of the data without copying.
//!
//! # Examples
//!
//! ## Creating and using an immutable slice
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::fixed::{FixedVec, UFixedVec};
//!
//! let data: Vec<u32> = (0..10).collect();
//! let vec: UFixedVec<u32> = FixedVec::builder().build(&data)?;
//!
//! // Create a slice of the elements from index 2 to 6.
//! let slice = vec.slice(2, 5).ok_or("slice failed")?;
//!
//! assert_eq!(slice.len(), 5);
//! assert_eq!(slice.get(0), Some(2));
//! assert_eq!(slice.get(4), Some(6));
//!
//! // The slice can be iterated over.
//! let sum: u32 = slice.iter().sum();
//! assert_eq!(sum, 2 + 3 + 4 + 5 + 6);
//! # Ok(())
//! # }
//! ```
//!
//! ## Creating and using a mutable slice
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::fixed::{FixedVec, UFixedVec, BitWidth};
//!
//! let data: Vec<u32> = (0..10).collect();
//! let mut vec: UFixedVec<u32> = FixedVec::builder().bit_width(BitWidth::Explicit(7)).build(&data)?;
//!
//! // `split_at_mut` is one way to get mutable slices.
//! let (_, mut slice2) = vec.split_at_mut(5);
//!
//! // Mutate an element within the second slice.
//! if let Some(mut proxy) = slice2.at_mut(0) { // index 0 of slice2 is index 5 of vec
//!     *proxy = 99;
//! }
//!
//! assert_eq!(vec.get(5), Some(99));
//! # Ok(())
//! # }
//! ```

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
/// A slice is a view that allows for operations on a sub-region of a [`FixedVec`]
/// without copying the underlying data. It can be created from both immutable
/// and mutable vectors.
///
/// This struct is generic over `V`, the type of reference to the parent vector,
/// which can be `&FixedVec` (for an immutable slice) or `&mut FixedVec` (for a
/// mutable slice).
#[derive(Debug)]
pub struct FixedVecSlice<V> {
    pub(super) parent: V,
    pub(super) range: Range<usize>,
}

// Common implementation for both immutable and mutable slices.
impl<T, W, E, B, V> FixedVecSlice<V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
    /// Creates a new `FixedVecSlice`.
    pub(super) fn new(parent: V, range: Range<usize>) -> Self {
        // The range is asserted to be within the parent's bounds upon creation.
        debug_assert!(range.end <= parent.len());
        Self { parent, range }
    }

    /// Returns the number of elements in the slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Returns `true` if the slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Returns the element at `index` relative to the start of the slice.
    ///
    /// Returns `None` if `index` is out of bounds of the slice.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len() {
            return None;
        }
        // SAFETY: The bounds check above ensures `index` is valid.
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Returns the element at `index` without bounds checking.
    ///
    /// The index is relative to the start of the slice.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        debug_assert!(index < self.len());
        // The index is relative to the slice, so we add the slice's start offset
        // to get the correct index in the parent vector.
        unsafe { self.parent.get_unchecked(self.range.start + index) }
    }

    /// Returns an iterator over the elements in the slice.
    pub fn iter(&self) -> FixedVecSliceIter<'_, T, W, E, B, V> {
        FixedVecSliceIter::new(self)
    }

    /// Binary searches this slice for a given element.
    ///
    /// If the value is found, returns `Ok(usize)` with the index of the
    /// matching element within the slice. If the value is not found, returns
    /// `Err(usize)` with the index where the value could be inserted to
    /// maintain order.
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

// Mutable-only implementation.
impl<T, W, E, B, V> FixedVecSlice<V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>> + DerefMut,
{
    /// Returns a mutable proxy for an element at `index` within the slice.
    ///
    /// This allows for syntax like `*slice.at_mut(i).unwrap() = new_value;`.
    ///
    /// Returns `None` if the index is out of bounds of the slice.
    pub fn at_mut(&mut self, index: usize) -> Option<MutProxy<'_, T, W, E, B>> {
        if index >= self.len() {
            return None;
        }
        // The index is translated to the parent vector's coordinate space.
        Some(MutProxy::new(&mut self.parent, self.range.start + index))
    }
}

// --- PartialEq Implementations ---

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
