//! Iteration over a [`SeqVec`](super::SeqVec) and over its sequences.

use super::View;
use crate::Storable;
use crate::common::Decoder;
use std::{fmt, iter::FusedIterator, marker::PhantomData};

/// Where the codewords of a sequence end.
enum End {
    /// After this many more codewords (the lengths are stored).
    Count(u64),
    /// At bit position `end`, which is `bits` bits after the start of the sequence.
    Bit { end: u64, bits: u64 },
}

/// A lazy iterator over the values of one sequence in a [`SeqVec`](super::SeqVec).
///
/// Returned by [`SeqVec::get`](super::SeqVec::get) and yielded by [`SeqVecIter`]. The iterator
/// borrows the encoded bits and yields decoded values of `T` from front to back. Advancing
/// or skipping elements decodes the corresponding codewords.
///
/// With [`SeqVecBuilder::store_lengths`](crate::SeqVecBuilder::store_lengths),
/// [`count`](Iterator::count) returns the remaining stored length in constant time, and
/// [`size_hint`](Iterator::size_hint) is exact when that length fits `usize`. For vectors
/// that use sequence boundaries, counting decodes the remaining values and the size hint
/// is an upper bound based on the sequence's encoded bit length.
pub struct SeqIter<'a, T> {
    decoder: Decoder<'a>,
    end: End,
    element: PhantomData<fn() -> T>,
}

impl<'a, T: Storable> SeqIter<'a, T> {
    /// Creates an iterator over sequence `index` of `view`, which is below its length.
    pub(crate) fn new(view: View<'a>, index: usize) -> Self {
        let (start, end) = view.bounds(index);
        let mut decoder = Decoder::new(view.data, view.codec);
        decoder.seek(start);
        let end = match view.stored_len(index) {
            Some(count) => End::Count(count),
            None => End::Bit {
                end,
                bits: end - start,
            },
        };
        Self {
            decoder,
            end,
            element: PhantomData,
        }
    }
}

impl<T: Storable> Iterator for SeqIter<'_, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match &mut self.end {
            End::Count(0) => return None,
            End::Count(count) => *count -= 1,
            End::Bit { end, .. } => {
                if self.decoder.bit_pos() >= *end {
                    return None;
                }
            }
        }
        Some(T::from_word(self.decoder.read()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.end {
            End::Count(count) => {
                let count = usize::try_from(count).ok();
                (count.unwrap_or(usize::MAX), count)
            }
            // Every codeword takes at least one bit, so the bits of the sequence bound the
            // number of elements left.
            End::Bit { bits, .. } => (0, usize::try_from(bits).ok()),
        }
    }

    fn count(self) -> usize {
        match self.end {
            End::Count(count) => {
                usize::try_from(count).expect("a stored length counts elements held in memory")
            }
            End::Bit { .. } => self.fold(0, |count, _| count + 1),
        }
    }
}

impl<T: Storable> FusedIterator for SeqIter<'_, T> {}

impl<T> fmt::Debug for SeqIter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SeqIter").finish_non_exhaustive()
    }
}

/// A double-ended, exact-size iterator over the sequences in a [`SeqVec`](super::SeqVec).
///
/// Returned by [`SeqVec::iter`](super::SeqVec::iter) and iteration over `&SeqVec`. Each item
/// is a [`SeqIter`] that decodes one sequence lazily. The outer iterator reads sequence
/// offsets to advance in either direction or skip to another sequence in constant time.
pub struct SeqVecIter<'a, T> {
    view: View<'a>,
    /// The next sequence from the front.
    front: usize,
    /// One past the next sequence from the back.
    back: usize,
    element: PhantomData<fn() -> T>,
}

impl<'a, T: Storable> SeqVecIter<'a, T> {
    pub(crate) fn new(view: View<'a>) -> Self {
        Self {
            view,
            front: 0,
            back: view.len(),
            element: PhantomData,
        }
    }
}

impl<'a, T: Storable> Iterator for SeqVecIter<'a, T> {
    type Item = SeqIter<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<SeqIter<'a, T>> {
        if self.front == self.back {
            return None;
        }
        self.front += 1;
        Some(SeqIter::new(self.view, self.front - 1))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }

    fn nth(&mut self, n: usize) -> Option<SeqIter<'a, T>> {
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

    fn last(mut self) -> Option<SeqIter<'a, T>> {
        self.next_back()
    }
}

impl<T: Storable> DoubleEndedIterator for SeqVecIter<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        Some(SeqIter::new(self.view, self.back))
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.back - self.front {
            self.back = self.front;
            return None;
        }
        self.back -= n;
        self.next_back()
    }
}

impl<T: Storable> ExactSizeIterator for SeqVecIter<'_, T> {}

impl<T: Storable> FusedIterator for SeqVecIter<'_, T> {}

impl<T> fmt::Debug for SeqVecIter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SeqVecIter")
            .field("remaining", &(self.back - self.front))
            .finish_non_exhaustive()
    }
}
