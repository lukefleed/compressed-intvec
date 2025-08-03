//! # `FixedVec` Iterators
//!
//! This module provides iterators for performing efficient, sequential
//! decompression of a [`FixedVec`]. It includes:
//! - [`FixedVecIter`]: An iterator over a borrowed `FixedVec`.
//! - [`FixedVecIntoIter`]: An iterator that consumes an owned `FixedVec`.

use super::FixedVec;
use dsi_bitstream::prelude::Endianness;

/// An iterator over the decompressed `u64` values of a borrowed [`FixedVec`].
///
/// This struct is created by the [`iter`](FixedVec::iter) method. It provides
/// a sequential, forward-only scan over the compressed data, decompressing
/// values on the fly without taking ownership.
///
/// # Example
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[u64] = &;
/// let fixed_vec = LEFixedVec::builder(data).build().unwrap();
///
/// // The iterator decompresses values as it is consumed.
/// let mut iter = fixed_vec.iter();
/// assert_eq!(iter.next(), Some(10));
/// assert_eq!(iter.next(), Some(20));
/// ```
pub struct FixedVecIter<'a, E: Endianness, B: AsRef<[u64]>> {
    vec: &'a FixedVec<E, B>,
    current_index: usize,
}

impl<'a, E: Endianness, B: AsRef<[u64]>> FixedVecIter<'a, E, B> {
    /// Creates a new iterator for a given `FixedVec`.
    ///
    /// This is `pub(super)` and is called by [`FixedVec::iter`].
    pub(super) fn new(vec: &'a FixedVec<E, B>) -> Self {
        Self {
            vec,
            current_index: 0,
        }
    }
}

impl<E: Endianness, B: AsRef<[u64]>> Iterator for FixedVecIter<'_, E, B> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.vec.len() {
            return None;
        }
        // SAFETY: The iterator's logic guarantees the index is in bounds.
        // This now calls the highly optimized, direct-access get_unchecked method.
        let value = unsafe { self.vec.get_unchecked(self.current_index) };
        self.current_index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len().saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

impl<E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for FixedVecIter<'_, E, B> {
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        self.vec.len().saturating_sub(self.current_index)
    }
}

/// An iterator that consumes an owned [`FixedVec`] and yields its decompressed values.
///
/// This struct is created by the [`into_iter`](FixedVec::into_iter) method on an
/// owned `FixedVec`. It provides an efficient, "lazy" scan over the compressed
/// data by taking ownership of the vector and decompressing values on the fly.
/// This avoids the overhead of allocating an intermediate `Vec<u64>`.
///
/// # Example
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[u64] = &;
/// let fixed_vec = LEFixedVec::builder(data).build().unwrap();
///
/// // The iterator consumes the vector and decompresses values lazily.
/// let mut count = 0;
/// for value in fixed_vec {
///     assert_eq!(value, data[count]);
///     count += 1;
/// }
/// ```
pub struct FixedVecIntoIter<E: Endianness> {
    /// The owned `FixedVec` being iterated over.
    vec: FixedVec<E, Vec<u64>>,
    /// The index of the next element to be returned.
    current_index: usize,
}

impl<E: Endianness> FixedVecIntoIter<E> {
    /// Creates a new consuming iterator from an owned `FixedVec`.
    ///
    /// This is `pub(super)` and is called by [`FixedVec::into_iter`].
    pub(super) fn new(vec: FixedVec<E, Vec<u64>>) -> Self {
        Self {
            vec,
            current_index: 0,
        }
    }
}

impl<E: Endianness> Iterator for FixedVecIntoIter<E> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.vec.len() {
            return None;
        }
        // SAFETY: The iterator's logic guarantees the index is in bounds.
        let value = unsafe { self.vec.get_unchecked(self.current_index) };
        self.current_index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len().saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

impl<E: Endianness> ExactSizeIterator for FixedVecIntoIter<E> {
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        self.vec.len().saturating_sub(self.current_index)
    }
}
