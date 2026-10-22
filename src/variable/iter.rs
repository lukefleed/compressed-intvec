//! Iteration and batch access over a [`VarVec`](super::VarVec).

use super::View;
use crate::Storable;
use crate::common::Decoder;
use std::{fmt, iter::FusedIterator, marker::PhantomData};

/// A decoder that knows which element it reads next, so that it seeks only when the target is
/// behind it or in a later block.
pub(crate) struct Cursor<'a> {
    view: View<'a>,
    decoder: Decoder<'a>,
    /// The index of the element at the decoder's position.
    next: usize,
}

impl<'a> Cursor<'a> {
    /// Creates a cursor at element 0, which starts at bit 0 (invariant of `VarVec`).
    #[inline]
    pub(crate) fn new(view: View<'a>) -> Self {
        Self {
            decoder: Decoder::new(view.data, view.codec),
            view,
            next: 0,
        }
    }

    /// Returns the word of element `index`, which is below the length of the vector.
    #[inline]
    pub(crate) fn read_at(&mut self, index: usize) -> u64 {
        if index != self.next {
            self.move_to(index);
        }
        self.next = index + 1;
        self.decoder.read()
    }

    /// Moves the decoder to element `index`: forward when it is later in the current block,
    /// otherwise from the sample of its block.
    fn move_to(&mut self, index: usize) {
        let k = self.view.sampling_rate;
        let block = index / k;
        if index < self.next || block != self.next / k {
            let sample = self
                .view
                .samples
                .get(block)
                .expect("every element below the length has a sample");
            self.decoder.seek(sample);
            self.next = block * k;
        }
        for _ in self.next..index {
            self.decoder.read();
        }
        self.next = index;
    }
}

/// An iterator yielding the elements of a [`VarVec`](super::VarVec) by value.
///
/// Created by [`VarVec::iter`](super::VarVec::iter), or by iterating over `&VarVec`. Forward
/// iteration decodes each element once and uses constant auxiliary space. [`nth`](Iterator::nth)
/// and [`nth_back`](DoubleEndedIterator::nth_back) use the sampled index to skip, requiring at
/// most one seek and `sampling_rate` codeword decodes to reach their target.
///
/// Reverse iteration decodes one sample block into a cache, then yields its elements from the
/// end. A complete reverse traversal decodes each element once and caches at most
/// `sampling_rate` words, allocating when the first block is decoded. Mixing forward and
/// backward iteration still yields each element at most once, but can decode elements more
/// than once because the forward cursor and the reverse cache share a decoder.
///
/// # Examples
///
/// ```
/// use compressed_intvec::VarVec;
///
/// let vec = VarVec::from_slice(&[2u32, 3, 5, 7, 11]);
/// assert_eq!(vec.iter().skip(1).take(3).collect::<Vec<_>>(), [3, 5, 7]);
/// assert_eq!(vec.iter().rev().collect::<Vec<_>>(), [11, 7, 5, 3, 2]);
/// ```
pub struct VarVecIter<'a, T> {
    cursor: Cursor<'a>,
    /// The next element from the front.
    front: usize,
    /// One past the next element from the back.
    back: usize,
    /// The words of the elements `back - back_words.len()..back`, which `next_back` yields from
    /// the end. Those below `front` are never yielded, since `next_back` stops at `front`.
    back_words: Vec<u64>,
    element: PhantomData<fn() -> T>,
}

impl<'a, T: Storable> VarVecIter<'a, T> {
    /// Creates an iterator over the elements `front..back` of `view`.
    #[inline]
    pub(crate) fn new(view: View<'a>, front: usize, back: usize) -> Self {
        Self {
            cursor: Cursor::new(view),
            front,
            back,
            back_words: Vec::new(),
            element: PhantomData,
        }
    }

    /// Decodes the elements before `back`, from the sample of the element `back - 1` or from
    /// `front` if it is later, into the empty `back_words`.
    fn decode_back(&mut self) {
        let k = self.cursor.view.sampling_rate;
        let start = ((self.back - 1) / k * k).max(self.front);
        let cursor = &mut self.cursor;
        // Exact, so that the capacity never exceeds `sampling_rate` words.
        self.back_words.reserve_exact(self.back - start);
        self.back_words
            .extend((start..self.back).map(|index| cursor.read_at(index)));
    }
}

impl<T: Storable> Iterator for VarVecIter<'_, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.front == self.back {
            return None;
        }
        let word = self.cursor.read_at(self.front);
        self.front += 1;
        Some(T::from_word(word))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }

    fn nth(&mut self, n: usize) -> Option<T> {
        if n >= self.back - self.front {
            self.front = self.back;
            return None;
        }
        self.front += n;
        self.next()
    }

    fn count(self) -> usize {
        self.back - self.front
    }

    fn last(mut self) -> Option<T> {
        // A single element, so it is read without buffering the elements before it.
        (self.front < self.back).then(|| T::from_word(self.cursor.read_at(self.back - 1)))
    }
}

impl<T: Storable> DoubleEndedIterator for VarVecIter<'_, T> {
    fn next_back(&mut self) -> Option<T> {
        if self.front == self.back {
            return None;
        }
        if self.back_words.is_empty() {
            self.decode_back();
        }
        self.back -= 1;
        let word = self
            .back_words
            .pop()
            .expect("decode_back decodes the element before back");
        Some(T::from_word(word))
    }

    fn nth_back(&mut self, n: usize) -> Option<T> {
        if n >= self.back - self.front {
            self.back = self.front;
            self.back_words.clear();
            return None;
        }
        self.back -= n;
        // Keeps the words of the elements that are still before `back`.
        self.back_words
            .truncate(self.back_words.len().saturating_sub(n));
        self.next_back()
    }
}

impl<T: Storable> ExactSizeIterator for VarVecIter<'_, T> {}

impl<T: Storable> FusedIterator for VarVecIter<'_, T> {}

impl<T> fmt::Debug for VarVecIter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VarVecIter")
            .field("remaining", &(self.back - self.front))
            .finish_non_exhaustive()
    }
}

/// A lazy iterator yielding an [`Option<T>`] for each requested [`VarVec`](super::VarVec) index.
///
/// Created by [`VarVec::get_many`](super::VarVec::get_many). Preserves input order and
/// duplicates, yielding `None` for out-of-bounds indices. Reads share a decoding cursor, so
/// strictly increasing indices decode each element at most once. The cursor uses constant
/// auxiliary space.
///
/// The iterator has the same length as its input iterator, including invalid indices. It
/// implements [`ExactSizeIterator`] and [`FusedIterator`] when the input iterator does.
pub struct VarVecGetMany<'a, T, I> {
    cursor: Cursor<'a>,
    indices: I,
    element: PhantomData<fn() -> T>,
}

impl<'a, T: Storable, I> VarVecGetMany<'a, T, I> {
    #[inline]
    pub(crate) fn new(view: View<'a>, indices: I) -> Self {
        Self {
            cursor: Cursor::new(view),
            indices,
            element: PhantomData,
        }
    }
}

impl<T: Storable, I: Iterator<Item = usize>> Iterator for VarVecGetMany<'_, T, I> {
    type Item = Option<T>;

    #[inline]
    fn next(&mut self) -> Option<Option<T>> {
        let index = self.indices.next()?;
        Some((index < self.cursor.view.len).then(|| T::from_word(self.cursor.read_at(index))))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl<T: Storable, I: ExactSizeIterator<Item = usize>> ExactSizeIterator
    for VarVecGetMany<'_, T, I>
{
}

impl<T: Storable, I: FusedIterator<Item = usize>> FusedIterator for VarVecGetMany<'_, T, I> {}

impl<T, I: fmt::Debug> fmt::Debug for VarVecGetMany<'_, T, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VarVecGetMany")
            .field("indices", &self.indices)
            .finish_non_exhaustive()
    }
}
