//! # [`FixedVec`] Iterators
//!
//! This module provides iterators for sequential access to the elements of a
//! [`FixedVec`]. The iterators decode values on the fly without allocating an
//! intermediate buffer, making them efficient for processing large datasets.
//!
//! The core iterators ([`FixedVecIter`], [`FixedVecSliceIter`]) are stateful and
//! employ a bit-windowing technique. They decode values on the fly without
//! allocating an intermediate buffer. They support both forward and reverse
//! iteration.
//!
//! # Provided Iterators
//!
//! - [`FixedVecIter`]: A stateful, bidirectional iterator over the elements of a borrowed [`FixedVec`].
//! - [`FixedVecIntoIter`]: A consuming, bidirectional iterator that takes ownership of a [`FixedVec`].
//! - [`FixedVecSliceIter`]: A stateful, bidirectional iterator over the elements of a [`FixedVecSlice`].
//! - [`Chunks`]: An iterator that yields non-overlapping, immutable slices ([`FixedVecSlice`]).
//! - [`Windows`]: An iterator that yields overlapping, immutable slices ([`FixedVecSlice`]).
//! - [`FixedVecUncheckedIter`]: An `unsafe` bidirectional iterator that omits bounds checks for maximum performance.
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
//! ## Bidirectional Iteration
//!
//! The main iterators implement [`DoubleEndedIterator`], allowing for
//! traversal from both ends of the sequence.
//!
//! ```rust
//! use compressed_intvec::fixed::{FixedVec, UFixedVec};
//!
//! let vec: UFixedVec<u8> = (0..=5u8).collect();
//! let mut iter = vec.iter();
//!
//! assert_eq!(iter.next(), Some(0));
//! assert_eq!(iter.next_back(), Some(5));
//! assert_eq!(iter.next(), Some(1));
//! assert_eq!(iter.next_back(), Some(4));
//!
//! // The iterator correctly tracks its remaining length.
//! assert_eq!(iter.len(), 2);
//!
//! assert_eq!(iter.next(), Some(2));
//! assert_eq!(iter.next(), Some(3));
//! assert_eq!(iter.next(), None);
//! ```
//!
//! ## Iterating over a Slice
//!
//! You can also create an iterator over a zero-copy slice of a vector.
//!
//! ```rust
//! use compressed_intvec::fixed::{FixedVec, UFixedVec};
//!
//! let vec: UFixedVec<u32> = (0..100u32).collect();
//!
//! // Create a slice containing elements from index 20 to 29.
//! let slice = vec.slice(20, 10).unwrap();
//! assert_eq!(slice.len(), 10);
//!
//! // The slice iterator will yield the elements from the slice.
//! let collected: Vec<u32> = slice.iter().collect();
//! let expected: Vec<u32> = (20..30).collect();
//!
//! assert_eq!(collected, expected);
//! ```

use crate::fixed::{
    FixedVec,
    slice::FixedVecSlice,
    traits::{Storable, Word},
};
use dsi_bitstream::prelude::Endianness;
use std::iter::FusedIterator;
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
#[derive(Debug)]
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

    // Cached from vec to avoid repeated indirection in hot loops.
    bit_width: usize,
    mask: W,

    // State for forward iteration.
    front_window: W,
    front_bits_in_window: usize,
    front_word_index: usize,

    // State for backward iteration.
    back_window: W,
    back_bits_in_window: usize,
    back_word_index: usize,
    _phantom: PhantomData<T>,
}

/// An iterator over non-overlapping, immutable chunks of a [`FixedVec`].
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
    /// Creates a new stateful, bidirectional iterator for a given [`FixedVec`].
    pub(super) fn new(vec: &'a FixedVec<T, W, E, B>) -> Self {
        let bit_width = vec.bit_width();
        let mask = vec.mask;

        if vec.is_empty() {
            return Self {
                vec,
                front_index: 0,
                back_index: 0,
                bit_width,
                mask,
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
        let bits_per_word: usize = <W as Word>::BITS;

        // --- Setup forward state ---
        let front_word_index = 1;
        // Pre-load the first word into the forward window.
        // SAFETY: The vector is non-empty, so limbs[0] exists (guaranteed by
        // the padding word invariant).
        let front_window = unsafe { *limbs.get_unchecked(0) };
        let front_bits_in_window = bits_per_word;

        // --- Setup backward state ---
        let total_bits = vec.len() * bit_width;
        let back_word_index = (total_bits.saturating_sub(1)) / bits_per_word;
        // Pre-load the last word into the backward window.
        // SAFETY: back_word_index is within bounds because total_bits > 0
        // and the buffer has num_words + 1 words (padding invariant).
        let back_window = unsafe { *limbs.get_unchecked(back_word_index) };
        // Calculate how many bits in the last word are valid data.
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
            bit_width,
            mask,
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

        let bit_width = self.bit_width;
        let mask = self.mask;

        // Path for word-sized elements.
        if bit_width == <W as Word>::BITS {
            // SAFETY: index < back_index <= vec.len(), and for word-sized
            // elements limbs[index] is valid.
            let val = unsafe { *self.vec.as_limbs().get_unchecked(index) };
            let final_val = if E::IS_BIG { W::from_be(val) } else { val };
            return Some(<T as Storable<W>>::from_word(final_val));
        }

        // Fallback to the standard `get_unchecked` for Big-Endian.
        if E::IS_BIG {
            // SAFETY: index < back_index <= vec.len().
            return Some(unsafe { self.vec.get_unchecked(index) });
        }

        // If the current window has enough bits for the next element.
        if self.front_bits_in_window >= bit_width {
            let value = self.front_window & mask;
            self.front_window >>= bit_width;
            self.front_bits_in_window -= bit_width;
            return Some(<T as Storable<W>>::from_word(value));
        }

        // Otherwise, load the next word to replenish the window.
        // SAFETY: front_word_index is valid because we advance it only when
        // there are remaining elements, and the padding word invariant
        // guarantees at least one extra word beyond the data.
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

    fn fold<Acc, F>(mut self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut acc = init;
        // Drain all remaining elements without per-element bounds checking.
        while self.front_index < self.back_index {
            // SAFETY: We have verified front_index < back_index, so there is
            // at least one element remaining. unwrap_unchecked is safe because
            // next() only returns None when front_index >= back_index.
            let item = unsafe { self.next().unwrap_unchecked() };
            acc = f(acc, item);
        }
        acc
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

        let bit_width = self.bit_width;
        let mask = self.mask;

        if E::IS_BIG || bit_width == <W as Word>::BITS {
            // SAFETY: index < back_index (before decrement) <= vec.len().
            return Some(unsafe { self.vec.get_unchecked(index) });
        }

        let bits_per_word: usize = <W as Word>::BITS;

        if self.back_bits_in_window >= bit_width {
            self.back_bits_in_window -= bit_width;
            let value = (self.back_window >> self.back_bits_in_window) & mask;
            return Some(<T as Storable<W>>::from_word(value));
        }

        // SAFETY: back_word_index is valid because we only decrement it when
        // there are remaining elements, and the buffer has sufficient words.
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

impl<T, W, E, B> FusedIterator for FixedVecIter<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
}

/// An iterator that consumes an owned [`FixedVec`] and yields its elements.
///
/// This struct is created by the `into_iter` method on `FixedVec`.
///
/// # Implementation
///
/// This is a self-referential struct: the [`FixedVecIter`] borrows from the
/// heap-allocated [`FixedVec`] stored in `_vec_owner`. The `Box` ensures a
/// stable address that won't move, making the `'static` transmute sound.
/// The `iter` field is declared before `_vec_owner` so it is dropped first,
/// though neither type has a custom `Drop` that accesses borrowed data.
pub struct FixedVecIntoIter<T, W, E>
where
    T: Storable<W> + 'static,
    W: Word,
    E: Endianness,
{
    iter: FixedVecIter<'static, T, W, E, Vec<W>>,
    /// Owns the `FixedVec` on the heap. Must outlive `iter`.
    _vec_owner: Box<FixedVec<T, W, E, Vec<W>>>,
}

impl<T, W, E> FixedVecIntoIter<T, W, E>
where
    T: Storable<W> + 'static,
    W: Word,
    E: Endianness,
{
    /// Creates a new consuming iterator from an owned `FixedVec`.
    pub(super) fn new(vec: FixedVec<T, W, E, Vec<W>>) -> Self {
        // Move the FixedVec to the heap so it has a stable address.
        let boxed = Box::new(vec);
        // SAFETY: `boxed` is heap-allocated and will not move. The reference
        // is valid for the lifetime of this struct because `_vec_owner` holds
        // the Box. We transmute to 'static to express this self-referential
        // borrow, which cannot be expressed with Rust lifetimes.
        let iter = unsafe {
            let vec_ref: &'static FixedVec<T, W, E, Vec<W>> =
                std::mem::transmute(&*boxed as &FixedVec<T, W, E, Vec<W>>);
            FixedVecIter::new(vec_ref)
        };
        Self {
            iter,
            _vec_owner: boxed,
        }
    }
}

impl<T, W, E> Iterator for FixedVecIntoIter<T, W, E>
where
    T: Storable<W> + 'static,
    W: Word,
    E: Endianness,
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

impl<T, W, E> DoubleEndedIterator for FixedVecIntoIter<T, W, E>
where
    T: Storable<W> + 'static,
    W: Word,
    E: Endianness,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<T, W, E> ExactSizeIterator for FixedVecIntoIter<T, W, E>
where
    T: Storable<W> + 'static,
    W: Word,
    E: Endianness,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<T, W, E> FusedIterator for FixedVecIntoIter<T, W, E>
where
    T: Storable<W> + 'static,
    W: Word,
    E: Endianness,
{
}

/// An iterator over the elements of a [`FixedVecSlice`].
///
/// This struct is created by the [`iter`](super::slice::FixedVecSlice::iter)
/// method on [`FixedVecSlice`]. It is a stateful bitstream reader that decodes
/// values on the fly for both forward and reverse iteration.
pub struct FixedVecSliceIter<'s, T, W, E, B, V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
    slice: &'s FixedVecSlice<V>,
    front_index: usize, // index relative to slice start
    back_index: usize,  // index relative to slice start

    // State for forward iteration.
    front_window: W,
    front_bits_in_window: usize,
    front_word_index: usize,

    // State for backward iteration.
    back_window: W,
    back_bits_in_window: usize,
    back_word_index: usize,

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
    /// Creates a new stateful, bidirectional iterator for a given `FixedVecSlice`.
    pub(super) fn new(slice: &'s FixedVecSlice<V>) -> Self {
        let parent_vec = &slice.parent;
        if slice.is_empty() {
            return Self {
                slice,
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

        let bits_per_word: usize = <W as Word>::BITS;
        let bit_width: usize = parent_vec.bit_width();
        let limbs: &[W] = parent_vec.as_limbs();
        let slice_start_abs: usize = slice.range.start;
        let slice_end_abs: usize = slice.range.end;

        // --- Setup forward state ---
        let start_bit_pos: usize = slice_start_abs * bit_width;
        let start_word_index: usize = start_bit_pos / bits_per_word;
        let start_bit_offset: usize = start_bit_pos % bits_per_word;

        let front_window_raw: W = unsafe { *limbs.get_unchecked(start_word_index) };
        let front_window: W = front_window_raw >> start_bit_offset;
        let front_bits_in_window: usize = bits_per_word - start_bit_offset;
        let front_word_index: usize = start_word_index + 1;

        // --- Setup backward state ---
        let end_bit_pos: usize = slice_end_abs * bit_width;
        let back_word_index: usize = (end_bit_pos.saturating_sub(1)) / bits_per_word;
        let back_window: W = unsafe { *limbs.get_unchecked(back_word_index) };

        let back_bits_in_window = end_bit_pos % bits_per_word;
        let back_bits_in_window = if back_bits_in_window == 0 && end_bit_pos > 0 {
            bits_per_word
        } else {
            back_bits_in_window
        };

        Self {
            slice,
            front_index: 0,
            back_index: slice.len(),
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
        if self.front_index >= self.back_index {
            return None;
        }

        let parent_vec = &self.slice.parent;
        let bit_width = parent_vec.bit_width();
        let mask = parent_vec.mask;

        let abs_index = self.slice.range.start + self.front_index;
        self.front_index += 1;

        if bit_width == <W as Word>::BITS {
            let val = unsafe { *parent_vec.as_limbs().get_unchecked(abs_index) };
            let final_val = if E::IS_BIG { W::from_be(val) } else { val };
            return Some(<T as Storable<W>>::from_word(final_val));
        }

        if E::IS_BIG {
            return Some(unsafe { parent_vec.get_unchecked(abs_index) });
        }

        if self.front_bits_in_window >= bit_width {
            let value = self.front_window & mask;
            self.front_window >>= bit_width;
            self.front_bits_in_window -= bit_width;
            return Some(<T as Storable<W>>::from_word(value));
        }

        unsafe {
            let limbs = parent_vec.as_limbs();
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

impl<T, W, E, B, V> DoubleEndedIterator for FixedVecSliceIter<'_, T, W, E, B, V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_index >= self.back_index {
            return None;
        }
        self.back_index -= 1;
        let abs_index = self.slice.range.start + self.back_index;
        let parent_vec = &self.slice.parent;

        if E::IS_BIG || parent_vec.bit_width() == <W as Word>::BITS {
            return Some(unsafe { parent_vec.get_unchecked(abs_index) });
        }

        let bit_width = parent_vec.bit_width();
        let bits_per_word: usize = <W as Word>::BITS;

        if self.back_bits_in_window >= bit_width {
            self.back_bits_in_window -= bit_width;
            let value = (self.back_window >> self.back_bits_in_window) & parent_vec.mask;
            return Some(<T as Storable<W>>::from_word(value));
        }

        unsafe {
            let limbs = parent_vec.as_limbs();
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

impl<T, W, E, B, V> ExactSizeIterator for FixedVecSliceIter<'_, T, W, E, B, V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
    fn len(&self) -> usize {
        self.back_index.saturating_sub(self.front_index)
    }
}

impl<T, W, E, B, V> FusedIterator for FixedVecSliceIter<'_, T, W, E, B, V>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    V: Deref<Target = FixedVec<T, W, E, B>>,
{
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

impl<T, W, E, B> FusedIterator for Chunks<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
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

impl<T, W, E, B> FusedIterator for Windows<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
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
        unsafe { self.iter.next().unwrap_unchecked() }
    }

    /// Returns the next element from the back without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method when the iterator is exhausted is undefined behavior.
    #[inline]
    pub unsafe fn next_back_unchecked(&mut self) -> T {
        unsafe { self.iter.next_back().unwrap_unchecked() }
    }
}
