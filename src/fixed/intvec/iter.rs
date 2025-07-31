//! # `FixedVec` Sequential Iterator
//!
//! This module provides [`FixedVecIter`], an iterator for performing efficient,
//! sequential decompression of a [`FixedVec`].

use super::{FixedVec, FixedVecBitReader};
use dsi_bitstream::prelude::{BitRead, BitSeek, Endianness};

/// An iterator over the decompressed `u64` values of a [`FixedVec`].
///
/// This struct is created by the [`iter`](FixedVec::iter) method. It provides
/// a sequential, forward-only scan over the compressed data, decompressing
/// values on the fly.
///
/// # Example
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[u64] = &;
/// let fixed_vec = LEFixedVec::builder(data).build().unwrap();
///
/// // The iterator decompresses values as it is consumed.
/// for (index, value) in fixed_vec.iter().enumerate() {
///     assert_eq!(value, data[index]);
/// }
/// ```
pub struct FixedVecIter<'a, E: Endianness> {
    vec: &'a FixedVec<E>,
    current_index: usize,
}

impl<'a, E: Endianness> FixedVecIter<'a, E> {
    /// Creates a new iterator for a given `FixedVec`.
    ///
    /// This is `pub(super)` and is called by [`FixedVec::iter`].
    pub(super) fn new(vec: &'a FixedVec<E>) -> Self {
        Self {
            vec,
            current_index: 0,
        }
    }
}

impl<E: Endianness> Iterator for FixedVecIter<'_, E>
where
    for<'b> FixedVecBitReader<'b, E>:
        BitRead<E, Error = core::convert::Infallible> + BitSeek<Error = core::convert::Infallible>,
{
    type Item = u64;

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

impl<E: Endianness> ExactSizeIterator for FixedVecIter<'_, E>
where
    for<'b> FixedVecBitReader<'b, E>:
        BitRead<E, Error = core::convert::Infallible> + BitSeek<Error = core::convert::Infallible>,
{
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        self.vec.len().saturating_sub(self.current_index)
    }
}
