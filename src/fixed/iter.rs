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
use std::{marker::PhantomData, ops::Deref};

use std::cmp::min;

/// An iterator over the decompressed values of a borrowed [`FixedVec`].
///
/// This struct is created by the [`iter`](FixedVec::iter) method. It acts as a
/// safe wrapper around the high-performance unchecked iterators, providing
/// bounds checking before delegating the core logic.
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
    fwd_reader: FixedVecUncheckedIter<'a, T, W, E, B>,
    bwd_reader: FixedVecReverseUncheckedIter<'a, T, W, E, B>,
    front_index: usize,
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
            // SAFETY: The readers are initialized with the bounds of the vector.
            fwd_reader: unsafe { FixedVecUncheckedIter::new(vec) },
            bwd_reader: unsafe { FixedVecReverseUncheckedIter::new(vec) },
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
        // SAFETY: The iterator's logic guarantees we do not call `next_unchecked`
        // more than `len` times for the forward reader.
        let value = unsafe { self.fwd_reader.next_unchecked() };
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
        // SAFETY: The iterator's logic guarantees we do not call `next_unchecked`
        // more than `len` times for the backward reader.
        let value = unsafe { self.bwd_reader.next_unchecked() };
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
/// owned `FixedVec`. It implements a stateful bitstream reader for maximum performance.
pub struct FixedVecIntoIter<T, W, E, B = Vec<W>>
where
    T: Storable<W> + 'static,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + 'static,
{
    // This field holds the owned data, ensuring it lives as long as the iterator.
    _vec_owner: B,
    // The high-performance stateful reader which borrows the owned data.
    reader: FixedVecUncheckedIter<'static, T, W, E, B>,
    _phantom: PhantomData<T>,
}

impl<T, W, E> FixedVecIntoIter<T, W, E, Vec<W>>
where
    T: Storable<W> + 'static,
    W: Word,
    E: Endianness + 'static,
{
    /// Creates a new consuming iterator from an owned `FixedVec`.
    pub(super) fn new(vec: FixedVec<T, W, E, Vec<W>>) -> Self {
        let reader = unsafe {
            let vec_ref: &'static FixedVec<T, W, E, Vec<W>> =
                std::mem::transmute(&vec as &FixedVec<T, W, E, Vec<W>>);
            FixedVecUncheckedIter::new(vec_ref)
        };
        Self {
            _vec_owner: vec.bits,
            reader,
            _phantom: PhantomData,
        }
    }
}

impl<T, W, E, B> Iterator for FixedVecIntoIter<T, W, E, B>
where
    T: Storable<W> + 'static,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + 'static,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.reader.items_remaining == 0 {
            return None;
        }
        // SAFETY: The items_remaining check ensures we don't iterate past the end.
        Some(unsafe { self.reader.next_unchecked() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.reader.items_remaining,
            Some(self.reader.items_remaining),
        )
    }
}

impl<T, W, E, B> ExactSizeIterator for FixedVecIntoIter<T, W, E, B>
where
    T: Storable<W> + 'static,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + 'static,
{
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        self.reader.items_remaining
    }
}

/// An iterator over the decompressed values of a [`FixedVec`] that does not
/// perform bounds checking.
///
/// This struct is created by the `iter_unchecked` method on `FixedVec`.
/// It implements a stateful bitstream reader for high-performance sequential access.
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
    // The current buffer of bits from the underlying storage.
    window: W,
    // The number of valid bits remaining in the `window`, starting from the LSB.
    bits_in_window: usize,
    // The index of the next word to read from the `bits` slice.
    word_index: usize,
    // The number of elements remaining to be iterated.
    items_remaining: usize,
    _phantom: PhantomData<(T, E)>,
}

impl<'a, T, W, E, B> FixedVecUncheckedIter<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Creates a new unchecked iterator.
    ///
    /// # Safety
    /// The vector must not be modified while the iterator is alive.
    pub(super) unsafe fn new(vec: &'a FixedVec<T, W, E, B>) -> Self {
        let (window, bits_in_window, word_index) = if E::IS_LITTLE && !vec.is_empty() {
            (*vec.as_limbs().get_unchecked(0), <W as Word>::BITS, 1)
        } else {
            (W::ZERO, 0, 0)
        };

        Self {
            vec,
            window,
            bits_in_window,
            word_index,
            items_remaining: vec.len(),
            _phantom: PhantomData,
        }
    }

    /// Returns the next element in the iterator without performing bounds checks.
    ///
    /// # Safety
    /// This method must not be called if the iterator is exhausted.
    #[inline]
    pub unsafe fn next_unchecked(&mut self) -> T {
        debug_assert!(self.items_remaining > 0);

        let bit_width = self.vec.bit_width();
        // Fast path for bit_width == word size. This avoids the `>> 64` panic
        // and handles endianness correctly.
        if bit_width == <W as Word>::BITS {
            let index = self.vec.len() - self.items_remaining;
            self.items_remaining -= 1;
            let val = *self.vec.as_limbs().get_unchecked(index);
            let final_val = if E::IS_BIG { W::from_be(val) } else { val };
            return <T as Storable<W>>::from_word(final_val);
        }

        self.items_remaining -= 1;

        if E::IS_LITTLE {
            let mask = self.vec.mask;
            if self.bits_in_window >= bit_width {
                let value = self.window & mask;
                self.window >>= bit_width;
                self.bits_in_window -= bit_width;
                return <T as Storable<W>>::from_word(value);
            }

            let limbs = self.vec.as_limbs();
            let bits_from_old_window = self.bits_in_window;
            let mut result = self.window;

            self.window = *limbs.get_unchecked(self.word_index);
            self.word_index += 1;
            result |= self.window << bits_from_old_window;
            let value = result & mask;

            let bits_from_new_window = bit_width - bits_from_old_window;
            self.window >>= bits_from_new_window;
            self.bits_in_window = <W as Word>::BITS - bits_from_new_window;

            <T as Storable<W>>::from_word(value)
        } else {
            // Fallback for BE: use the original `get_unchecked` logic.
            let index = self.vec.len() - self.items_remaining - 1;
            self.vec.get_unchecked(index)
        }
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
    window: W,
    bits_in_window: usize,
    word_index: usize,
    items_remaining: usize,
    _phantom: PhantomData<T>,
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
        if vec.is_empty() {
            return Self {
                vec,
                items_remaining: 0,
                window: W::ZERO,
                bits_in_window: 0,
                word_index: 0,
                _phantom: PhantomData,
            };
        }

        let limbs = vec.as_limbs();
        let bits_per_word = <W as Word>::BITS;
        let total_bits = vec.len() * vec.bit_width();

        let word_index = (total_bits.saturating_sub(1)) / bits_per_word;
        let window = *limbs.get_unchecked(word_index);

        let bits_in_window = total_bits % bits_per_word;
        let bits_in_window = if bits_in_window == 0 {
            bits_per_word
        } else {
            bits_in_window
        };

        Self {
            vec,
            window,
            bits_in_window,
            word_index,
            items_remaining: vec.len(),
            _phantom: PhantomData,
        }
    }

    /// Returns the next element from the back of the iterator without bounds checks.
    ///
    /// # Safety
    /// This method must not be called if the iterator is exhausted.
    #[inline]
    pub unsafe fn next_unchecked(&mut self) -> T {
        debug_assert!(self.items_remaining > 0);

        let bit_width = self.vec.bit_width();
        if bit_width == <W as Word>::BITS {
            self.items_remaining -= 1;
            let index = self.items_remaining;
            let val = *self.vec.as_limbs().get_unchecked(index);
            let final_val = if E::IS_BIG { W::from_be(val) } else { val };
            return <T as Storable<W>>::from_word(final_val);
        }

        self.items_remaining -= 1;

        if E::IS_BIG {
            return self.vec.get_unchecked(self.items_remaining);
        }

        let bits_per_word = <W as Word>::BITS;

        if self.bits_in_window >= bit_width {
            self.bits_in_window -= bit_width;
            let val = (self.window >> self.bits_in_window) & self.vec.mask;
            return <T as Storable<W>>::from_word(val);
        }

        let limbs = self.vec.as_limbs();
        let bits_from_old = self.bits_in_window;
        let mut result = self.window;

        self.word_index -= 1;
        self.window = *limbs.get_unchecked(self.word_index);

        result &= (W::ONE << bits_from_old).wrapping_sub(W::ONE);
        let bits_from_new = bit_width - bits_from_old;
        result <<= bits_from_new;
        result |= self.window >> (bits_per_word - bits_from_new);

        self.bits_in_window = bits_per_word - bits_from_new;
        <T as Storable<W>>::from_word(result)
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