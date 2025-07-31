//! # `SIntVec` Sequential Iterator
//!
//! This module provides [`SIntVecIter`], an iterator for performing efficient,
//! sequential decompression of a signed integer vector.
//!
//! ## Implementation
//!
//! The iterator is implemented as a lightweight wrapper around the underlying
//! [`IntVecIter`]. It fetches the compressed `u64` values from the inner
//! iterator and applies the inverse ZigZag transformation ([`ToInt`]) on the
//! fly to restore the original `i64` values. This approach ensures that the
//! performance characteristics are nearly identical to those of the `IntVec`
//! iterator, as the `to_int` operation has negligible overhead.
//!
//! [`ToInt`]: dsi_bitstream::prelude::ToInt

use super::SIntVec;
use crate::variable::intvec::IntVecIter;
use dsi_bitstream::{codes::ToInt, prelude::Endianness};

/// An iterator over the decompressed `i64` values of an [`SIntVec`].
///
/// This struct is created by the [`iter`](SIntVec::iter) method on [`SIntVec`].
/// It wraps the underlying [`IntVecIter`] and applies the inverse ZigZag
/// transformation to each decompressed `u64` value on the fly, yielding the
/// original `i64` values.
///
/// # Example
///
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[i64] = &[-10, 200, 30, -40, 50];
/// let sintvec = LESIntVec::builder(data)
///     .codec(CodecSpec::Gamma)
///     .build()
///     .unwrap();
///
/// // The iterator is an ExactSizeIterator.
/// assert_eq!(sintvec.iter().len(), data.len());
///
/// // It decompresses and transforms values on the fly.
/// let collected_values: Vec<i64> = sintvec.iter().collect();
/// assert_eq!(collected_values, data);
/// ```
pub struct SIntVecIter<'a, E: Endianness>
where
    for<'b> crate::variable::intvec::IntVecBitReader<'b, E>: dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
        + dsi_bitstream::dispatch::CodesRead<E>
        + dsi_bitstream::prelude::BitSeek<Error = core::convert::Infallible>,
{
    /// The inner iterator over the ZigZag-encoded `u64` values.
    inner_iter: IntVecIter<'a, E>,
}

impl<'a, E: Endianness> SIntVecIter<'a, E>
where
    for<'b> crate::variable::intvec::IntVecBitReader<'b, E>: dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
        + dsi_bitstream::dispatch::CodesRead<E>
        + dsi_bitstream::prelude::BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new `SIntVecIter` that wraps the inner `IntVec`'s iterator.
    /// This is `pub(super)` and is called by [`SIntVec::iter`].
    pub(super) fn new(sintvec: &'a SIntVec<E>) -> Self {
        Self {
            inner_iter: sintvec.inner.iter(),
        }
    }
}

impl<E: Endianness> Iterator for SIntVecIter<'_, E>
where
    for<'b> crate::variable::intvec::IntVecBitReader<'b, E>: dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
        + dsi_bitstream::dispatch::CodesRead<E>
        + dsi_bitstream::prelude::BitSeek<Error = core::convert::Infallible>,
{
    type Item = i64;

    /// Advances the iterator and returns the next decompressed `i64` value.
    ///
    /// It fetches the next `u64` value from the inner iterator and then applies
    /// the inverse ZigZag transformation (`to_int`) to restore the original signed integer.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Get the next unsigned value from the inner iterator and apply
        // the inverse ZigZag transformation (to_int).
        self.inner_iter
            .next()
            .map(|unsigned_val| unsigned_val.to_int())
    }

    /// Returns a hint about the number of remaining items in the iterator.
    /// This is delegated to the inner `IntVecIter`.
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner_iter.size_hint()
    }
}

impl<E: Endianness> std::iter::ExactSizeIterator for SIntVecIter<'_, E>
where
    for<'b> crate::variable::intvec::IntVecBitReader<'b, E>: dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
        + dsi_bitstream::dispatch::CodesRead<E>
        + dsi_bitstream::prelude::BitSeek<Error = core::convert::Infallible>,
{
    /// Returns the exact number of remaining items in the iterator.
    /// This is delegated to the inner `IntVecIter`.
    fn len(&self) -> usize {
        self.inner_iter.len()
    }
}
