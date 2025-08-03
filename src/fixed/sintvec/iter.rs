//! # `SFixedVec` Iterators
//!
//! This module provides iterators for performing efficient, sequential
//! decompression of a [`SFixedVec`]. It includes:
//! - [`SFixedVecIter`]: An iterator over a borrowed `SFixedVec`.
//! - [`SFixedVecIntoIter`]: An iterator that consumes an owned `SFixedVec`.

use super::SFixedVec;
use crate::fixed::intvec::iter::{FixedVecIntoIter, FixedVecIter};
use dsi_bitstream::prelude::{Endianness, ToInt};

/// An iterator over the decompressed `i64` values of a borrowed [`SFixedVec`].
///
/// This struct is created by the [`iter`](SFixedVec::iter) method. It wraps
/// the underlying [`FixedVecIter`] and applies the inverse ZigZag transformation
/// to each decompressed `u64` value on the fly, yielding the original `i64` values.
pub struct SFixedVecIter<'a, E: Endianness, B: AsRef<[u64]>> {
    /// The inner iterator over the ZigZag-encoded `u64` values.
    inner_iter: FixedVecIter<'a, E, B>,
}

impl<'a, E: Endianness, B: AsRef<[u64]>> SFixedVecIter<'a, E, B> {
    /// Creates a new `SFixedVecIter` that wraps the inner `FixedVec`'s iterator.
    ///
    /// This is `pub(super)` and is called by [`SFixedVec::iter`].
    pub(super) fn new(s_fixed_vec: &'a SFixedVec<E, B>) -> Self {
        Self {
            inner_iter: s_fixed_vec.inner.iter(),
        }
    }
}

impl<'a, E: Endianness, B: AsRef<[u64]>> Iterator for SFixedVecIter<'a, E, B> {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Get the next unsigned value from the inner iterator and apply
        // the inverse ZigZag transformation (to_int).
        self.inner_iter.next().map(ToInt::to_int)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner_iter.size_hint()
    }
}

impl<'a, E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for SFixedVecIter<'a, E, B> {
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        self.inner_iter.len()
    }
}

/// An iterator that consumes an owned [`SFixedVec`] and yields its decompressed `i64` values.
///
/// This struct is created by the [`into_iter`](SFixedVec::into_iter) method on an
/// owned `SFixedVec`. It wraps the underlying [`FixedVecIntoIter`] and applies
/// the inverse ZigZag transformation on the fly.
pub struct SFixedVecIntoIter<E: Endianness> {
    /// The inner consuming iterator over the ZigZag-encoded `u64` values.
    inner_iter: FixedVecIntoIter<E>,
}

impl<E: Endianness> SFixedVecIntoIter<E> {
    /// Creates a new consuming iterator from an owned `SFixedVec`.
    ///
    /// This is `pub(super)` and is called by [`SFixedVec::into_iter`].
    pub(super) fn new(s_fixed_vec: SFixedVec<E, Vec<u64>>) -> Self {
        Self {
            inner_iter: s_fixed_vec.inner.into_iter(),
        }
    }
}

impl<E: Endianness> Iterator for SFixedVecIntoIter<E> {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner_iter.next().map(ToInt::to_int)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner_iter.size_hint()
    }
}

impl<E: Endianness> ExactSizeIterator for SFixedVecIntoIter<E> {
    fn len(&self) -> usize {
        self.inner_iter.len()
    }
}
