//! # `SFixedVec` Sequential Iterator
//!
//! This module provides [`SFixedVecIter`], an iterator for performing efficient,
//! sequential decompression of a [`SFixedVec`].

use super::SFixedVec;
use crate::fixed::intvec::iter::FixedVecIter;
use dsi_bitstream::prelude::{Endianness, ToInt};

/// An iterator over the decompressed `i64` values of an [`SFixedVec`].
///
/// This struct is created by the [`iter`](SFixedVec::iter) method. It wraps
/// the underlying [`FixedVecIter`] and applies the inverse ZigZag transformation
/// to each decompressed `u64` value on the fly, yielding the original `i64` values.
pub struct SFixedVecIter<'a, E: Endianness> {
    /// The inner iterator over the ZigZag-encoded `u64` values.
    inner_iter: FixedVecIter<'a, E>,
}

impl<'a, E: Endianness> SFixedVecIter<'a, E> {
    /// Creates a new `SFixedVecIter` that wraps the inner `FixedVec`'s iterator.
    ///
    /// This is `pub(super)` and is called by [`SFixedVec::iter`].
    pub(super) fn new(s_fixed_vec: &'a SFixedVec<E>) -> Self {
        Self {
            inner_iter: s_fixed_vec.inner.iter(),
        }
    }
}

impl<E: Endianness> Iterator for SFixedVecIter<'_, E> {
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

impl<E: Endianness> ExactSizeIterator for SFixedVecIter<'_, E> {
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        self.inner_iter.len()
    }
}
