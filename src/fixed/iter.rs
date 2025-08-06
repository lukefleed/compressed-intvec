//! # `FixedVec` Iterators
//!
//! This module provides iterators for performing efficient, sequential
//! decompression of a [`FixedVec`].

use crate::fixed::{
    slice::FixedVecSlice,
    traits::{Storable, Word},
    FixedVec,
};
use dsi_bitstream::prelude::Endianness;
use std::ops::Deref;

use std::cmp::min;

/// An iterator over the decompressed values of a borrowed [`FixedVec`].
///
/// This struct is created by the [`iter`](FixedVec::iter) method. It provides
/// a sequential, forward-only scan over the compressed data, decompressing
/// values on the fly without taking ownership.
///
/// It also implements `DoubleEndedIterator`, allowing for efficient reverse
/// iteration using `.rev()`.
pub struct FixedVecIter<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    vec: &'a FixedVec<T, W, E, B>,
    /// The index of the next element to be returned from the front.
    front_index: usize,
    /// The index of the next element to be returned from the back.
    back_index: usize,
}

/// An iterator over immutable, non-overlapping chunks of a `FixedVec`.
///
/// This struct is created by the [`chunks`](super::FixedVec::chunks) method.
#[derive(Debug)]
pub struct Chunks<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    vec: &'a FixedVec<T, W, E, B>,
    chunk_size: usize,
    current_pos: usize,
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
            front_index: 0,
            back_index: vec.len(),
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
        if self.front_index >= self.back_index {
            return None;
        }
        // SAFETY: The iterator's logic guarantees the index is in bounds.
        let value = unsafe { self.vec.get_unchecked(self.front_index) };
        self.front_index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back_index.saturating_sub(self.front_index);
        (remaining, Some(remaining))
    }
}

impl<T, W, E, B> DoubleEndedIterator for FixedVecIter<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_index >= self.back_index {
            return None;
        }
        self.back_index -= 1;
        // SAFETY: The iterator's logic guarantees the index is in bounds.
        let value = unsafe { self.vec.get_unchecked(self.back_index) };
        Some(value)
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
        self.back_index.saturating_sub(self.front_index)
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

/// A reverse iterator over the decompressed values of a [`FixedVec`] that does not
/// perform bounds checking.
///
/// This struct is created by the `iter_rev_unchecked` method on `FixedVec`.
///
/// # Safety
/// The caller must ensure that `next_unchecked` is not called more than `len` times.
/// Calling it after the iterator is exhausted is **Undefined Behavior**.
pub struct FixedVecReverseUncheckedIter<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    vec: &'a FixedVec<T, W, E, B>,
    /// The index of the next element to be yielded. It moves from `len` down to `0`.
    current_back_index: usize,
}

impl<'a, T, W, E, B> FixedVecReverseUncheckedIter<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Creates a new unchecked reverse iterator.
    pub(super) unsafe fn new(vec: &'a FixedVec<T, W, E, B>) -> Self {
        Self {
            vec,
            current_back_index: vec.len(),
        }
    }

    /// Returns the next element from the back of the iterator without bounds checks.
    ///
    /// # Safety
    /// This method must not be called if the iterator is exhausted.
    #[inline]
    pub unsafe fn next_unchecked(&mut self) -> T {
        debug_assert!(self.current_back_index > 0);
        self.current_back_index -= 1;
        let value = self.vec.get_unchecked(self.current_back_index);
        value
    }
}

/// An iterator over the decompressed values of a [`FixedVecSlice`].
///
/// This struct is created by the `iter` method on `FixedVecSlice`.
pub struct FixedVecSliceIter<'s, T, W, E, B, V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
    slice: &'s FixedVecSlice<V>,
    current_index: usize,
    _phantom: std::marker::PhantomData<(T, W, E, B)>,
}

impl<'s, T, W, E, B, V> FixedVecSliceIter<'s, T, W, E, B, V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
    /// Creates a new iterator for a given `FixedVecSlice`.
    pub(super) fn new(slice: &'s FixedVecSlice<V>) -> Self {
        Self {
            slice,
            current_index: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, W, E, B, V> Iterator for FixedVecSliceIter<'_, T, W, E, B, V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
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

impl<T, W, E, B, V> ExactSizeIterator for FixedVecSliceIter<'_, T, W, E, B, V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
    fn len(&self) -> usize {
        self.slice.len().saturating_sub(self.current_index)
    }
}

impl<'a, T, W, E, B> Chunks<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Creates a new `Chunks` iterator.
    pub(super) fn new(vec: &'a FixedVec<T, W, E, B>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size cannot be zero");
        Self {
            vec,
            chunk_size,
            current_pos: 0,
        }
    }
}

impl<'a, T, W, E, B> Iterator for Chunks<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    type Item = FixedVecSlice<&'a FixedVec<T, W, E, B>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos >= self.vec.len() {
            return None;
        }

        let len = min(self.chunk_size, self.vec.len() - self.current_pos);
        let slice = FixedVecSlice::new(self.vec, self.current_pos..self.current_pos + len);
        self.current_pos += len;

        Some(slice)
    }
}