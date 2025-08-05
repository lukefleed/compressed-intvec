//! # `IntVec` Sequential Iterator
//!
//! This module provides [`IntVecIter`], an iterator for performing efficient,
//! sequential decompression of an [`IntVec`]. The iterator is designed for
//! forward-only scans, decompressing values from the underlying bitstream on
//! the fly.
//!
//! [`IntVec`]: super::IntVec

use super::{IntVec, IntVecBitReader};
use dsi_bitstream::{
    dispatch::{CodesRead, FuncCodeReader, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};

/// An iterator over the decompressed `u64` values of an [`IntVec`].
///
/// This struct is created by the [`iter`](IntVec::iter) method on [`IntVec`].
/// It provides a sequential, forward-only scan over the compressed data,
/// decompressing values on the fly. It also implements [`ExactSizeIterator`],
/// allowing the user to know exactly how many items are remaining.
pub struct IntVecIter<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    // This where clause now guarantees that the reader implements all necessary traits.
    IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// The total number of elements in the vector.
    len: usize,
    /// The underlying bitstream reader used for decoding.
    reader: IntVecBitReader<'a, E>,
    /// The pre-configured code reader for this iterator instance.
    code_reader: FuncCodeReader<E, IntVecBitReader<'a, E>>,
    /// The index of the next element to be returned.
    current_index: usize,
    /// A flag to track if the bitstream is still valid.
    valid: bool,
    /// A phantom data field to associate the backend type B.
    _phantom: std::marker::PhantomData<&'a B>,
}

impl<'a, E, B> IntVecIter<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new iterator for a given `IntVec`.
    ///
    /// This is `pub(super)` and is called by [`IntVec::iter`].
    pub(super) fn new(intvec: &'a IntVec<E, B>) -> Self {
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
            valid: true,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, E, B> Iterator for IntVecIter<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = u64;

    /// Advances the iterator and returns the next decompressed value.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if !self.valid || self.current_index >= self.len {
            return None;
        }

        let result = self.code_reader.read(&mut self.reader);

        match result {
            Ok(value) => {
                self.current_index += 1;
                Some(value)
            }
            Err(_) => {
                self.valid = false;
                None
            }
        }
    }

    /// Returns a hint about the number of remaining items in the iterator.
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.valid {
            let remaining = self.len.saturating_sub(self.current_index);
            (remaining, Some(remaining))
        } else {
            (0, Some(0))
        }
    }
}

impl<'a, E, B> std::iter::ExactSizeIterator for IntVecIter<'a, E, B>
where
    E: Endianness,
    B: AsRef<[u64]>,
    IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        if self.valid {
            self.len.saturating_sub(self.current_index)
        } else {
            0
        }
    }
}