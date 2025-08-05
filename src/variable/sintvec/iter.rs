//! # `SIntVec` Sequential Iterator
//!
//! This module provides [`SIntVecIter`], an iterator for performing efficient,
//! sequential decompression of a signed integer vector.

use super::SIntVec;
use crate::variable::intvec::{IntVecBitReader, IntVecIter};
use dsi_bitstream::{
    codes::ToInt,
    prelude::{BitRead, BitSeek, CodesRead, Endianness},
};

/// An iterator over the decompressed `i64` values of an [`SIntVec`].
///
/// This struct is created by the [`iter`](SIntVec::iter) method on [`SIntVec`].
/// It wraps the underlying [`IntVecIter`] and applies the inverse ZigZag
/// transformation to each decompressed `u64` value on the fly, yielding the
/// original `i64` values.
pub struct SIntVecIter<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// The inner iterator over the ZigZag-encoded `u64` values.
    inner_iter: IntVecIter<'a, E, B>,
}

impl<'a, E, B> SIntVecIter<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new `SIntVecIter` that wraps the inner `IntVec`'s iterator.
    /// This is `pub(super)` and is called by [`SIntVec::iter`].
    pub(super) fn new(sintvec: &'a SIntVec<E, B>) -> Self {
        Self {
            inner_iter: sintvec.inner.iter(),
        }
    }
}

impl<'a, E, B> Iterator for SIntVecIter<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = i64;

    /// Advances the iterator and returns the next decompressed `i64` value.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner_iter
            .next()
            .map(|unsigned_val| unsigned_val.to_int())
    }

    /// Returns a hint about the number of remaining items in the iterator.
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner_iter.size_hint()
    }
}

impl<'a, E, B> std::iter::ExactSizeIterator for SIntVecIter<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        self.inner_iter.len()
    }
}