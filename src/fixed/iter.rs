//! # `FixedVec` Iterators
//!
//! This module provides iterators for performing efficient, sequential
//! decompression of a [`FixedVec`].

use crate::fixed::{FixedVec, traits::{Storable, Word}};
use dsi_bitstream::prelude::Endianness;

/// An iterator over the decompressed values of a borrowed [`FixedVec`].
///
/// This struct is created by the [`iter`](FixedVec::iter) method. It provides
/// a sequential, forward-only scan over the compressed data, decompressing
/// values on the fly without taking ownership.
pub struct FixedVecIter<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    vec: &'a FixedVec<T, W, E, B>,
    current_index: usize,
}

impl<'a, T, W, E, B> FixedVecIter<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Creates a new iterator for a given `FixedVec`.
    pub(super) fn new(vec: &'a FixedVec<T, W, E, B>) -> Self {
        Self {
            vec,
            current_index: 0,
        }
    }
}

impl<T, W, E, B> Iterator for FixedVecIter<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    type Item = T;

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

impl<T, W, E, B> ExactSizeIterator for FixedVecIter<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        self.vec.len().saturating_sub(self.current_index)
    }
}


/// An iterator that consumes an owned [`FixedVec`] and yields its decompressed values.
///
/// This struct is created by the [`into_iter`](FixedVec::into_iter) method on an
/// owned `FixedVec`.
pub struct FixedVecIntoIter<T, W, E>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
{
    vec: FixedVec<T, W, E, Vec<W>>,
    current_index: usize,
}

impl<T, W, E> FixedVecIntoIter<T, W, E>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
{
    /// Creates a new consuming iterator from an owned `FixedVec`.
    pub(super) fn new(vec: FixedVec<T, W, E, Vec<W>>) -> Self {
        Self {
            vec,
            current_index: 0,
        }
    }
}

impl<T, W, E> Iterator for FixedVecIntoIter<T, W, E>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
{
    type Item = T;

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

impl<T, W, E> ExactSizeIterator for FixedVecIntoIter<T, W, E>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
{
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        self.vec.len().saturating_sub(self.current_index)
    }
}

/// An iterator over the decompressed values of a [`FixedVec`] that does not
/// perform bounds checking.
///
/// This struct is created by the `iter_unchecked` method on `FixedVec`.
///
/// # Safety
/// The caller must ensure that `next_unchecked` is not called more than `len` times.
/// Calling it after the iterator is exhausted is **Undefined Behavior**.
pub struct FixedVecUncheckedIter<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    vec: &'a FixedVec<T, W, E, B>,
    current_index: usize,
}

impl<'a, T, W, E, B> FixedVecUncheckedIter<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Creates a new unchecked iterator.
    pub(super) unsafe fn new(vec: &'a FixedVec<T, W, E, B>) -> Self {
        Self {
            vec,
            current_index: 0,
        }
    }

    /// Returns the next element in the iterator without performing bounds checks.
    ///
    /// # Safety
    /// This method must not be called if the iterator is exhausted.
    #[inline]
    pub unsafe fn next_unchecked(&mut self) -> T {
        debug_assert!(self.current_index < self.vec.len());
        let value = self.vec.get_unchecked(self.current_index);
        self.current_index += 1;
        value
    }
}

/// An iterator over the decompressed values of a [`FixedVecSlice`].
///
/// This struct is created by the `iter` method on `FixedVecSlice`.
// The lifetime `'s` is for the slice, which may be shorter than the original vec's lifetime.
pub struct FixedVecSliceIter<'s, 'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    slice: &'s crate::fixed::slice::FixedVecSlice<'a, T, W, E, B>,
    current_index: usize,
}

impl<'s, 'a, T, W, E, B> FixedVecSliceIter<'s, 'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Creates a new iterator for a given `FixedVecSlice`.
    pub(super) fn new(slice: &'s crate::fixed::slice::FixedVecSlice<'a, T, W, E, B>) -> Self {
        Self {
            slice,
            current_index: 0,
        }
    }
}

impl<'s, 'a, T, W, E, B> Iterator for FixedVecSliceIter<'s, 'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
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

impl<'s, 'a, T, W, E, B> ExactSizeIterator for FixedVecSliceIter<'s, 'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    fn len(&self) -> usize {
        self.slice.len().saturating_sub(self.current_index)
    }
}