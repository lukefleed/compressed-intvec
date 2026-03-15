//! Iterators for [`VarVec`].
//!
//! This module provides the iterators for [`VarVec`]. Due to the nature of
//! variable-length encoding, an [`VarVec`] is immutable once created, as
//! modifying an element would require re-encoding the rest of the data stream.
//!
//! [`VarVec`]: crate::variable::VarVec

use super::{traits::Storable, VarVec};
use crate::common::codec_reader::{CodecReader, VarVecBitReader};
use dsi_bitstream::{
    dispatch::{CodesRead, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};
use std::fmt;
use std::iter::FusedIterator;
use std::marker::PhantomData;

/// A borrowing iterator over the values of an [`VarVec`].
///
/// This struct is created by the [`iter`](VarVec::iter) method on [`VarVec`].
/// It provides a sequential, forward-only scan over the compressed data,
/// decompressing values on the fly.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use compressed_intvec::variable::{VarVec, UVarVec};
///
/// let data: &[u32] = &[10, 20, 30, 40, 50];
/// let vec: UVarVec<u32> = VarVec::from_slice(data)?;
///
/// let mut sum = 0;
/// for value in vec.iter() {
///     sum += value;
/// }
///
/// assert_eq!(sum, 150);
/// # Ok(())
/// # }
/// ```
pub struct VarVecIter<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    len: usize,
    reader: VarVecBitReader<'a, E>,
    /// The hybrid dispatcher that handles codec reading robustly.
    code_reader: CodecReader<'a, E>,
    current_index: usize,
    _markers: PhantomData<(&'a B, T)>,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> VarVecIter<'a, T, E, B>
where
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new iterator.
    pub(super) fn new(intvec: &'a VarVec<T, E, B>) -> Self {
        let reader = VarVecBitReader::<E>::new(dsi_bitstream::impls::MemWordReader::new_inf(
            intvec.data.as_ref(),
        ));
        // Instantiate the robust hybrid dispatcher. This will not panic.
        let code_reader = CodecReader::new(intvec.encoding);

        Self {
            len: intvec.len,
            reader,
            code_reader,
            current_index: 0,
            _markers: PhantomData,
        }
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> Iterator for VarVecIter<'_, T, E, B>
where
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.len {
            return None;
        }
        // The read operation is infallible due to the robust dispatcher.
        let value = self.code_reader.read(&mut self.reader).unwrap();
        self.current_index += 1;
        Some(Storable::from_word(value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len.saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for VarVecIter<'_, T, E, B>
where
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn len(&self) -> usize {
        self.len.saturating_sub(self.current_index)
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> FusedIterator for VarVecIter<'_, T, E, B> where
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>
{
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> fmt::Debug for VarVecIter<'_, T, E, B>
where
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VarVecIter")
            .field("remaining", &self.len.saturating_sub(self.current_index))
            .finish()
    }
}

/// An owning iterator over the values of an [`VarVec`].
///
/// This struct is created by the [`into_iter`](VarVec::into_iter) method on
/// [`VarVec`] (or by using a `for` loop on an owned [`VarVec`]). It takes ownership
/// of the vector and decodes its values on the fly.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use compressed_intvec::variable::{VarVec, SVarVec};
///
/// let data: &[i16] = &[-1, -2, -3, -4];
/// let vec: SVarVec<i16> = VarVec::from_slice(data)?;
///
/// // The `into_iter` call is implicit in the for loop.
/// // This loop consumes `vec`.
/// let collected: Vec<i16> = vec.into_iter().map(|v| v * 2).collect();
///
/// assert_eq!(collected, &[-2, -4, -6, -8]);
/// # Ok(())
/// # }
/// ```
pub struct VarVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// The number of elements remaining in the iterator.
    len: usize,
    /// The current position in the sequence.
    current_index: usize,
    /// A stateful reader that borrows from `_data_owner`.
    reader: VarVecBitReader<'static, E>,
    /// The hybrid dispatcher for decoding.
    code_reader: CodecReader<'static, E>,
    /// This field owns the data buffer, ensuring it lives as long as the iterator.
    _data_owner: Vec<u64>,
    /// Phantom data to hold the generic type `T`.
    _markers: PhantomData<T>,
}

impl<T, E> VarVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new, efficient owning iterator from a [`VarVec`].
    pub(super) fn new(vec: VarVec<T, E, Vec<u64>>) -> Self {
        // This is a self-referential struct. We move the owned data buffer into `_data_owner`.
        // Then, we create a 'static reference to that data to initialize the reader.
        // This is safe because `_data_owner` is part of the same struct as `reader`,
        // guaranteeing that the data outlives the reference.
        let data_ref: &'static [u64] = unsafe { std::mem::transmute(vec.data.as_slice()) };

        let reader = VarVecBitReader::<E>::new(dsi_bitstream::impls::MemWordReader::new_inf(data_ref));
        let code_reader = CodecReader::new(vec.encoding);

        Self {
            len: vec.len,
            current_index: 0,
            reader,
            code_reader,
            _data_owner: vec.data,
            _markers: PhantomData,
        }
    }
}

impl<T, E> Iterator for VarVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.len {
            return None;
        }
        let value = self.code_reader.read(&mut self.reader).unwrap();
        self.current_index += 1;
        Some(Storable::from_word(value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len.saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

impl<T, E> ExactSizeIterator for VarVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn len(&self) -> usize {
        self.len.saturating_sub(self.current_index)
    }
}

impl<T, E> FusedIterator for VarVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
}

impl<T, E> fmt::Debug for VarVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VarVecIntoIter")
            .field("remaining", &self.len.saturating_sub(self.current_index))
            .finish()
    }
}
