//! # `FixedVec` Sequential Iterator
//!
//! This module provides [`FixedVecIter`], an iterator for performing efficient,
//! sequential decompression of a [`FixedVec`].

use super::FixedVec;
use dsi_bitstream::prelude::Endianness;

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
pub struct FixedVecIter<'a, E: Endianness, B: AsRef<[u64]>> {
    vec: &'a FixedVec<E, B>,
    current_index: usize,
}

impl<'a, E: Endianness, B: AsRef<[u64]>> FixedVecIter<'a, E, B> {
    /// Creates a new iterator for a given `FixedVec`.
    ///
    /// This is `pub(super)` and is called by [`FixedVec::iter`].
    pub(super) fn new(vec: &'a FixedVec<E, B>) -> Self {
        Self {
            vec,
            current_index: 0,
        }
    }
}

impl<E: Endianness, B: AsRef<[u64]>> Iterator for FixedVecIter<'_, E, B> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.vec.len() {
            return None;
        }
        // SAFETY: The iterator's logic guarantees the index is in bounds.
        // This now calls the highly optimized, direct-access get_unchecked method.
        let value = unsafe { self.vec.get_unchecked(self.current_index) };
        self.current_index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len().saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

impl<E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for FixedVecIter<'_, E, B> {
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        self.vec.len().saturating_sub(self.current_index)
    }
}
