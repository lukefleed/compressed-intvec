//! Iterators over the elements of a [`FixedVec`](super::FixedVec).

use super::{Backend, layout};
use crate::Storable;
use std::{fmt, iter::FusedIterator, marker::PhantomData};

/// The words of an iteration, its remaining elements `[front, back)`, and their layout.
///
/// Borrowing and owning iterators hold a cursor over borrowed or owned words. Each element is
/// read independently, as by `FixedVec::get`, and skipping takes constant time.
///
/// Invariant: `words` holds at least `back` elements of `bit_width` bits, plus the padding word
/// if `back > 0`; `bit_width` is at most 64 and `mask` is its mask. Only the unsafe iterator
/// constructors create a cursor, and the methods never raise `back`, so every read is from the
/// words that the cursor was built for. `W` is a sealed [`Backend`], so `as_ref` returns the
/// same words on every call.
#[derive(Clone)]
struct Cursor<W> {
    words: W,
    front: usize,
    back: usize,
    bit_width: usize,
    mask: u64,
}

impl<W: Backend> Cursor<W> {
    #[inline]
    fn len(&self) -> usize {
        self.back - self.front
    }

    /// Returns element `index`, which must be below `back`.
    #[inline(always)]
    unsafe fn read(&self, index: usize) -> u64 {
        // SAFETY: `index < back` (caller), and the words hold `back` elements (invariant).
        unsafe { layout::read(self.words.as_ref(), self.bit_width, self.mask, index) }
    }

    #[inline]
    fn next(&mut self) -> Option<u64> {
        if self.front >= self.back {
            return None;
        }
        let index = self.front;
        self.front += 1;
        // SAFETY: `index < back`.
        Some(unsafe { self.read(index) })
    }

    #[inline]
    fn next_back(&mut self) -> Option<u64> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        // SAFETY: `back` is below its previous value.
        Some(unsafe { self.read(self.back) })
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<u64> {
        if n >= self.len() {
            self.front = self.back;
            return None;
        }
        self.front += n;
        self.next()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<u64> {
        if n >= self.len() {
            self.back = self.front;
            return None;
        }
        self.back -= n;
        self.next_back()
    }

    /// Calls `f` on the remaining elements from the front, as [`Iterator::fold`].
    #[inline]
    fn fold<A>(self, init: A, mut f: impl FnMut(A, u64) -> A) -> A {
        (self.front..self.back).fold(init, |acc, index| {
            // SAFETY: `index < back`.
            f(acc, unsafe { self.read(index) })
        })
    }
}

/// A borrowed iterator yielding decoded elements of a [`FixedVec`](crate::FixedVec) or
/// [`FixedVecSlice`](crate::FixedVecSlice) by value.
///
/// [`FixedVec::iter`](crate::FixedVec::iter) and
/// [`FixedVecSlice::iter`](crate::FixedVecSlice::iter) create an iterator that borrows the packed
/// words and decodes each element on demand. It implements [`DoubleEndedIterator`],
/// [`ExactSizeIterator`] and [`FusedIterator`], with constant-time [`nth`](Iterator::nth) and
/// [`nth_back`](DoubleEndedIterator::nth_back).
#[derive(Clone)]
pub struct FixedVecIter<'a, T> {
    cursor: Cursor<&'a [u64]>,
    element: PhantomData<fn() -> T>,
}

impl<'a, T> FixedVecIter<'a, T> {
    /// Iterates over the elements `[start, end)` of `words`.
    ///
    /// # Safety
    ///
    /// `start <= end`, `bit_width <= 64`, `mask` is the mask of `bit_width`, and `words` hold
    /// `end` elements of `bit_width` bits, plus the padding word if `end > 0`.
    pub(crate) unsafe fn new(
        words: &'a [u64],
        start: usize,
        end: usize,
        bit_width: usize,
        mask: u64,
    ) -> Self {
        Self {
            cursor: Cursor {
                words,
                front: start,
                back: end,
                bit_width,
                mask,
            },
            element: PhantomData,
        }
    }
}

impl<T: Storable> Iterator for FixedVecIter<'_, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.cursor.next().map(T::from_word)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<T> {
        self.cursor.nth(n).map(T::from_word)
    }

    #[inline]
    fn fold<A, F: FnMut(A, T) -> A>(self, init: A, mut f: F) -> A {
        self.cursor
            .fold(init, |acc, word| f(acc, T::from_word(word)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.cursor.len(), Some(self.cursor.len()))
    }
}

impl<T: Storable> DoubleEndedIterator for FixedVecIter<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.cursor.next_back().map(T::from_word)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<T> {
        self.cursor.nth_back(n).map(T::from_word)
    }
}

impl<T: Storable> ExactSizeIterator for FixedVecIter<'_, T> {}

impl<T: Storable> FusedIterator for FixedVecIter<'_, T> {}

impl<T: Storable + fmt::Debug> fmt::Debug for FixedVecIter<'_, T> {
    /// Shows the iterator's remaining elements.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// An owning iterator yielding decoded elements of a [`FixedVec`](crate::FixedVec) by value.
///
/// [`into_iter`](IntoIterator::into_iter) consumes a `FixedVec<T>` with `Vec<u64>` storage and
/// transfers its packed word buffer to this iterator. The iterator decodes each element on
/// demand. It implements [`DoubleEndedIterator`], [`ExactSizeIterator`] and
/// [`FusedIterator`], with constant-time [`nth`](Iterator::nth) and
/// [`nth_back`](DoubleEndedIterator::nth_back).
#[derive(Clone)]
pub struct FixedVecIntoIter<T> {
    cursor: Cursor<Vec<u64>>,
    element: PhantomData<fn() -> T>,
}

impl<T> FixedVecIntoIter<T> {
    /// Iterates over the `len` elements of `words`.
    ///
    /// # Safety
    ///
    /// As for [`FixedVecIter::new`], with `start = 0` and `end = len`.
    pub(crate) unsafe fn new(words: Vec<u64>, len: usize, bit_width: usize, mask: u64) -> Self {
        Self {
            cursor: Cursor {
                words,
                front: 0,
                back: len,
                bit_width,
                mask,
            },
            element: PhantomData,
        }
    }
}

impl<T: Storable> Iterator for FixedVecIntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.cursor.next().map(T::from_word)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<T> {
        self.cursor.nth(n).map(T::from_word)
    }

    #[inline]
    fn fold<A, F: FnMut(A, T) -> A>(self, init: A, mut f: F) -> A {
        self.cursor
            .fold(init, |acc, word| f(acc, T::from_word(word)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.cursor.len(), Some(self.cursor.len()))
    }
}

impl<T: Storable> DoubleEndedIterator for FixedVecIntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.cursor.next_back().map(T::from_word)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<T> {
        self.cursor.nth_back(n).map(T::from_word)
    }
}

impl<T: Storable> ExactSizeIterator for FixedVecIntoIter<T> {}

impl<T: Storable> FusedIterator for FixedVecIntoIter<T> {}

impl<T: Storable + fmt::Debug> fmt::Debug for FixedVecIntoIter<T> {
    /// Shows the iterator's remaining elements.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}
