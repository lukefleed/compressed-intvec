//! Iterators for [`IntVec`].
//!
//! This module provides the iterators for [`IntVec`]. Due to the nature of
//! variable-length encoding, an [`IntVec`] is immutable once created, as
//! modifying an element would require re-encoding the rest of the data stream.
//!
//! The provided iterators, [`IntVecIter`] and [`IntVecIntoIter`], offer efficient,
//! read-only access to the compressed data by decompressing values on the fly.
//!
//! [`IntVec`]: crate::variable::IntVec

use super::{traits::Storable, IntVec, IntVecBitReader};
use dsi_bitstream::{
    dispatch::{CodesRead, FuncCodeReader, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};
use std::marker::PhantomData;

/// A borrowing iterator over the values of an [`IntVec`].
///
/// This struct is created by the [`iter`](IntVec::iter) method on [`IntVec`].
/// It provides a sequential, forward-only scan over the compressed data,
/// decompressing values on the fly.
///
/// # Examples
///
/// ```
/// use compressed_intvec::variable::{IntVec, UIntVec};
///
/// let data: &[u32] = &[10, 20, 30, 40];
/// let vec: UIntVec<u32> = IntVec::from_slice(data).unwrap();
///
/// let mut sum = 0;
/// for value in vec.iter() {
///     sum += value;
/// }
///
/// assert_eq!(sum, 100);
/// ```
pub struct IntVecIter<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    len: usize,
    reader: IntVecBitReader<'a, E>,
    code_reader: FuncCodeReader<E, IntVecBitReader<'a, E>>,
    current_index: usize,
    _markers: PhantomData<(&'a B, T)>,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> IntVecIter<'a, T, E, B>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new iterator.
    pub(super) fn new(intvec: &'a IntVec<T, E, B>) -> Self {
        let reader = IntVecBitReader::<E>::new(dsi_bitstream::impls::MemWordReader::new(
            intvec.data.as_ref(),
        ));
        let code_reader = FuncCodeReader::new(intvec.encoding)
            .expect("Failed to create code reader for DSI encoding.");

        Self {
            len: intvec.len,
            reader,
            code_reader,
            current_index: 0,
            _markers: PhantomData,
        }
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> Iterator for IntVecIter<'a, T, E, B>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.len {
            return None;
        }
        let value = self.code_reader.read(&mut self.reader).ok()?;
        self.current_index += 1;
        Some(Storable::from_word(value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len.saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for IntVecIter<'a, T, E, B>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn len(&self) -> usize {
        self.len.saturating_sub(self.current_index)
    }
}

/// An owning iterator over the values of an [`IntVec`].
///
/// This struct is created by the [`into_iter`](IntVec::into_iter) method on
/// [`IntVec`] (or by using a `for` loop on an owned `IntVec`). It takes ownership
/// of the vector and decodes its values on the fly.
///
/// # Examples
///
/// ```
/// use compressed_intvec::variable::{IntVec, SIntVec};
///
/// let data: &[i16] = &[-1, -2, -3, -4];
/// let vec: SIntVec<i16> = IntVec::from_slice(data).unwrap();
///
/// // The `into_iter` call is implicit in the for loop.
/// // This loop consumes `vec`.
/// let collected: Vec<i16> = vec.into_iter().map(|v| v * 2).collect();
///
/// assert_eq!(collected, &[-2, -4, -6, -8]);
/// ```
pub struct IntVecIntoIter<T: Storable, E: Endianness, B: AsRef<[u64]>> {
    vec: IntVec<T, E, B>,
    current_index: usize,
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> IntVecIntoIter<T, E, B> {
    /// Creates a new owning iterator.
    pub(super) fn new(vec: IntVec<T, E, B>) -> Self {
        Self {
            vec,
            current_index: 0,
        }
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> Iterator for IntVecIntoIter<T, E, B>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.vec.len() {
            return None;
        }
        // This get is amortized O(1), but less efficient for full scans than IntVecIter
        let value = self.vec.get(self.current_index);
        self.current_index += 1;
        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len().saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for IntVecIntoIter<T, E, B>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn len(&self) -> usize {
        self.vec.len().saturating_sub(self.current_index)
    }
}