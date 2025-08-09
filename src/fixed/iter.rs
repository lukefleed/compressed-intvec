//! # `FixedVec` Iterators
//!
//! This module provides iterators for sequential access to the elements of a
//! [`FixedVec`]. The iterators decode values on the fly without allocating an
//! intermediate buffer, making them efficient for processing large datasets.
//!
//! # Examples
//!
//! ## Iterating over elements
//!
//! ```rust
//! use compressed_intvec::fixed::{FixedVec, SFixedVec};
//!
//! let data: &[i16] = &[-100, 0, 100, 200];
//! let vec: SFixedVec<i16> = FixedVec::builder().build(data).unwrap();
//!
//! let mut sum = 0;
//! for value in vec.iter() {
//!     sum += value;
//! }
//!
//! assert_eq!(sum, 200);
//! ```
//!
//! ## Iterating over chunks
//!
//! ```rust
//! use compressed_intvec::fixed::{FixedVec, BEFixedVec};
//!
//! let data: Vec<u64> = (0..10).collect();
//! let vec: BEFixedVec = FixedVec::builder().build(&data).unwrap();
//!
//! let mut chunks_iter = vec.chunks(3);
//!
//! let first_chunk = chunks_iter.next().unwrap();
//! assert_eq!(first_chunk.len(), 3);
//! assert_eq!(first_chunk.get(0), Some(0));
//! assert_eq!(first_chunk.get(2), Some(2));
//!
//! let last_chunk = chunks_iter.last().unwrap();
//! assert_eq!(last_chunk.len(), 1);
//! assert_eq!(last_chunk.get(0), Some(9));
//! ```

use crate::fixed::{
    slice::FixedVecSlice,
    traits::{Storable, Word},
    FixedVec,
};
use dsi_bitstream::prelude::Endianness;
use std::{marker::PhantomData, ops::Deref};

use std::cmp::min;

/// An iterator over the elements of a borrowed [`FixedVec`].
///
/// This struct is created by the [`iter`](FixedVec::iter) method. It is a
/// stateful bitstream reader that decodes values on the fly for both forward
/// and reverse iteration.
///
/// # Examples
///
/// ## Forward iteration
///
/// ```rust
/// use compressed_intvec::fixed::{FixedVec, UFixedVec};
///
/// let data: &[u8] = &[1, 2, 3, 4, 5];
/// let vec: UFixedVec<u8> = FixedVec::builder().build(data).unwrap();
/// let mut iter = vec.iter();
///
/// assert_eq!(iter.next(), Some(1));
/// assert_eq!(iter.next(), Some(2));
/// ```
///
/// ## Reverse iteration
///
/// ```rust
/// use compressed_intvec::fixed::{FixedVec, UFixedVec};
///
/// let data: &[u8] = &[1, 2, 3, 4, 5];
/// let vec: UFixedVec<u8> = FixedVec::builder().build(data).unwrap();
/// let mut iter = vec.iter();
///
/// assert_eq!(iter.next_back(), Some(5));
/// assert_eq!(iter.next_back(), Some(4));
/// ```
pub struct FixedVecIter<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    vec: &'a FixedVec<T, W, E, B>,
    front_index: usize,
    back_index: usize,

    // --- Forward iteration state ---
    front_window: W,
    front_bits_in_window: usize,
    front_word_index: usize,

    // --- Backward iteration state ---
    back_window: W,
    back_bits_in_window: usize,
    back_word_index: usize,
    _phantom: PhantomData<T>,
}

/// An iterator over non-overlapping, immutable chunks of a [`FixedVec`].
///
/// This struct is created by the [`chunks`](super::FixedVec::chunks) method.
/// Each item in the iterator is a [`FixedVecSlice`].
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
    /// Creates a new stateful, bidirectional iterator for a given `FixedVec`.
    pub(super) fn new(vec: &'a FixedVec<T, W, E, B>) -> Self {
        if vec.is_empty() {
            return Self {
                vec,
                front_index: 0,
                back_index: 0,
                front_window: W::ZERO,
                front_bits_in_window: 0,
                front_word_index: 0,
                back_window: W::ZERO,
                back_bits_in_window: 0,
                back_word_index: 0,
                _phantom: PhantomData,
            };
        }

        let limbs = vec.as_limbs();
        let bits_per_word = <W as Word>::BITS;

        // --- Setup forward state ---
        let front_word_index = 1;
        // SAFETY: The `is_empty` check ensures at least one word exists.
        let front_window = unsafe { *limbs.get_unchecked(0) };
        let front_bits_in_window = bits_per_word;

        // --- Setup backward state ---
        let total_bits = vec.len() * vec.bit_width();
        let back_word_index = (total_bits.saturating_sub(1)) / bits_per_word;
        // SAFETY: `is_empty` check ensures this is a valid index.
        let back_window = unsafe { *limbs.get_unchecked(back_word_index) };
        let back_bits_in_window = total_bits % bits_per_word;
        let back_bits_in_window = if back_bits_in_window == 0 {
            bits_per_word
        } else {
            back_bits_in_window
        };

        Self {
            vec,
            front_index: 0,
            back_index: vec.len(),
            front_window,
            front_bits_in_window,
            front_word_index,
            back_window,
            back_bits_in_window,
            back_word_index,
            _phantom: PhantomData,
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
        let index = self.front_index;
        self.front_index += 1;

        let bit_width = self.vec.bit_width();
        if bit_width == <W as Word>::BITS {
            let val = unsafe { *self.vec.as_limbs().get_unchecked(index) };
            let final_val = if E::IS_BIG { W::from_be(val) } else { val };
            return Some(<T as Storable<W>>::from_word(final_val));
        }

        if E::IS_BIG {
            return Some(unsafe { self.vec.get_unchecked(index) });
        }

        let mask = self.vec.mask;
        if self.front_bits_in_window >= bit_width {
            let value = self.front_window & mask;
            self.front_window >>= bit_width;
            self.front_bits_in_window -= bit_width;
            return Some(<T as Storable<W>>::from_word(value));
        }

        unsafe {
            let limbs = self.vec.as_limbs();
            let bits_from_old = self.front_bits_in_window;
            let mut result = self.front_window;

            self.front_window = *limbs.get_unchecked(self.front_word_index);
            self.front_word_index += 1;
            result |= self.front_window << bits_from_old;
            let value = result & mask;

            let bits_from_new = bit_width - bits_from_old;
            self.front_window >>= bits_from_new;
            self.front_bits_in_window = <W as Word>::BITS - bits_from_new;

            Some(<T as Storable<W>>::from_word(value))
        }
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
        let index = self.back_index;

        if E::IS_BIG || self.vec.bit_width() == <W as Word>::BITS {
            return Some(unsafe { self.vec.get_unchecked(index) });
        }

        let bit_width = self.vec.bit_width();
        let bits_per_word = <W as Word>::BITS;

        if self.back_bits_in_window >= bit_width {
            self.back_bits_in_window -= bit_width;
            let value = (self.back_window >> self.back_bits_in_window) & self.vec.mask;
            return Some(<T as Storable<W>>::from_word(value));
        }

        unsafe {
            let limbs = self.vec.as_limbs();
            let bits_from_old = self.back_bits_in_window;
            let mut result = self.back_window;

            self.back_word_index -= 1;
            self.back_window = *limbs.get_unchecked(self.back_word_index);

            result &= (W::ONE << bits_from_old).wrapping_sub(W::ONE);
            let bits_from_new = bit_width - bits_from_old;
            result <<= bits_from_new;
            result |= self.back_window >> (bits_per_word - bits_from_new);

            self.back_bits_in_window = bits_per_word - bits_from_new;
            Some(<T as Storable<W>>::from_word(result))
        }
    }
}

impl<T, W, E, B> ExactSizeIterator for FixedVecIter<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    fn len(&self) -> usize {
        self.back_index.saturating_sub(self.front_index)
    }
}

/// An iterator that consumes an owned [`FixedVec`] and yields its elements.
///
/// This struct is created by the `into_iter` method on `FixedVec` (which is
/// part of the [`IntoIterator`] trait).
pub struct FixedVecIntoIter<'a, T, W, E, B = Vec<W>>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    _vec_owner: B,
    iter: FixedVecIter<'a, T, W, E, B>,
    _phantom: PhantomData<T>,
}

impl<T, W, E> FixedVecIntoIter<'static, T, W, E, Vec<W>>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
{
    /// Creates a new consuming iterator from an owned `FixedVec`.
    pub(super) fn new(vec: FixedVec<T, W, E, Vec<W>>) -> Self {
        let iter = unsafe {
            let vec_ref: &'static FixedVec<T, W, E, Vec<W>> =
                std::mem::transmute(&vec as &FixedVec<T, W, E, Vec<W>>);
            FixedVecIter::new(vec_ref)
        };
        Self {
            _vec_owner: vec.bits,
            iter,
            _phantom: PhantomData,
        }
    }
}

impl<T, W, E, B> Iterator for FixedVecIntoIter<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, W, E, B> ExactSizeIterator for FixedVecIntoIter<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

/// An iterator over the elements of a [`FixedVecSlice`].
///
/// This struct is created by the [`iter`](super::slice::FixedVecSlice::iter)
/// method on [`FixedVecSlice`].
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
    _phantom: PhantomData<(T, W, E, B)>,
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
            _phantom: PhantomData,
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

/// An iterator over overlapping sub-slices of a [`FixedVec`].
///
/// This struct is created by the [`windows`](super::FixedVec::windows) method.
pub struct Windows<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    vec: &'a FixedVec<T, W, E, B>,
    size: usize,
    current_pos: usize,
}

impl<'a, T, W, E, B> Windows<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Creates a new `Windows` iterator.
    pub(super) fn new(vec: &'a FixedVec<T, W, E, B>, size: usize) -> Self {
        Self {
            vec,
            size,
            current_pos: 0,
        }
    }
}

impl<'a, T, W, E, B> Iterator for Windows<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    type Item = FixedVecSlice<&'a FixedVec<T, W, E, B>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos + self.size > self.vec.len() {
            return None;
        }

        let slice = FixedVecSlice::new(self.vec, self.current_pos..self.current_pos + self.size);
        self.current_pos += 1;

        Some(slice)
    }
}

/// An unchecked iterator over the elements of a [`FixedVec`].
///
/// This struct is created by the [`iter_unchecked`](super::FixedVec::iter_unchecked)
/// method. It does not perform any bounds checking.
///
/// # Safety
///
/// The iterator is safe to use only if it is guaranteed that it will not
/// be advanced beyond the end of the vector.
pub struct FixedVecUncheckedIter<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    iter: FixedVecIter<'a, T, W, E, B>,
}

impl<'a, T, W, E, B> FixedVecUncheckedIter<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Creates a new `FixedVecUncheckedIter`.
    pub(super) fn new(vec: &'a FixedVec<T, W, E, B>) -> Self {
        Self {
            iter: FixedVecIter::new(vec),
        }
    }

    /// Returns the next element without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method when the iterator is exhausted is undefined behavior.
    #[inline]
    pub unsafe fn next_unchecked(&mut self) -> T {
        // The underlying FixedVecIter is already highly optimized.
        // The primary gain here is removing the check in `next()`.
        self.iter.next().unwrap_unchecked()
    }
}
