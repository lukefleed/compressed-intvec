//! A stateless reader for efficient repeated access to a [`SeqVec`].
//!
//! This module provides [`SeqVecReader`], a reusable reader that maintains an
//! internal bitstream reader to avoid setup overhead when accessing multiple
//! sequences. Each access performs an independent seek operation.
//!
//! For access patterns with sequential locality (indices mostly increasing),
//! consider using [`SeqVecSeqReader`] instead.
//!
//! [`SeqVec`]: crate::seq::SeqVec
//! [`SeqVecSeqReader`]: crate::seq::SeqVecSeqReader

use super::iter::{CodecReader, SeqIter, SeqVecBitReader};
use super::{SeqVec, SeqVecError};
use crate::variable::traits::Storable;
use dsi_bitstream::{
    dispatch::CodesRead,
    impls::MemWordReader,
    prelude::{BitRead, BitSeek, Endianness},
};

/// A stateless reader for efficient repeated access to a [`SeqVec`].
///
/// This reader maintains an internal bitstream reader that is reused across
/// multiple access operations, avoiding the overhead of creating a new reader
/// for each call. However, each [`get`](Self::get) operation performs an
/// independent seek to the target sequence's bit offset.
///
/// Use this reader when:
/// - You need to access multiple sequences.
/// - The access pattern is unpredictable (random order).
/// - You want to avoid the per-call overhead of [`SeqVec::get`].
///
/// For sequential or mostly-increasing access patterns, [`SeqVecSeqReader`]
/// is more efficient as it can decode forward without seeking.
///
/// # Examples
///
/// ```ignore
/// use compressed_intvec::seq::{SeqVec, LESeqVec};
///
/// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4], &[5, 6]];
/// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
///
/// let mut reader = vec.reader();
///
/// // Random access
/// assert_eq!(reader.get_vec(2), Some(vec![5, 6]));
/// assert_eq!(reader.get_vec(0), Some(vec![1, 2]));
/// assert_eq!(reader.get_vec(1), Some(vec![3, 4]));
/// ```
///
/// [`SeqVec`]: crate::seq::SeqVec
/// [`SeqVecSeqReader`]: crate::seq::SeqVecSeqReader
pub struct SeqVecReader<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Reference to the parent SeqVec.
    seqvec: &'a SeqVec<T, E, B>,
    /// The reusable bitstream reader.
    reader: SeqVecBitReader<'a, E>,
    /// The codec dispatcher.
    code_reader: CodecReader<'a, T, E, B>,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVecReader<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new reader for the given [`SeqVec`].
    #[inline]
    pub(crate) fn new(seqvec: &'a SeqVec<T, E, B>) -> Self {
        let reader = SeqVecBitReader::<E>::new(MemWordReader::new(seqvec.data.as_ref()));
        let code_reader = CodecReader::new(seqvec.encoding);

        Self {
            seqvec,
            reader,
            code_reader,
        }
    }

    /// Returns the number of sequences in the underlying [`SeqVec`].
    #[inline]
    pub fn num_sequences(&self) -> usize {
        self.seqvec.num_sequences()
    }

    /// Returns an iterator over the elements of sequence `index`.
    ///
    /// Returns `None` if `index >= num_sequences()`.
    ///
    /// Note: Unlike the iterator returned by [`SeqVec::get`], this method
    /// reuses the internal reader state, making repeated calls more efficient.
    /// However, calling this method invalidates any previously returned iterator
    /// (they share the underlying reader).
    #[inline]
    pub fn get(&mut self, index: usize) -> Option<ReaderSeqIter<'_, 'a, T, E, B>> {
        if index >= self.seqvec.num_sequences() {
            return None;
        }
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Returns an iterator over the elements of sequence `index` without
    /// bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < num_sequences()`.
    #[inline]
    pub unsafe fn get_unchecked(&mut self, index: usize) -> ReaderSeqIter<'_, 'a, T, E, B> {
        debug_assert!(
            index < self.seqvec.num_sequences(),
            "index {} out of bounds for {} sequences",
            index,
            self.seqvec.num_sequences()
        );

        let start_bit = self.seqvec.bit_offsets.get_unchecked(index);
        let end_bit = self.seqvec.bit_offsets.get_unchecked(index + 1);

        // Seek to the start of the sequence.
        self.reader.set_bit_pos(start_bit).unwrap();

        ReaderSeqIter {
            reader: &mut self.reader,
            code_reader: &self.code_reader,
            end_bit,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the elements of sequence `index` as a newly allocated `Vec`.
    ///
    /// Returns `None` if `index >= num_sequences()`.
    #[inline]
    pub fn get_vec(&mut self, index: usize) -> Option<Vec<T>> {
        self.get(index).map(|iter| iter.collect())
    }

    /// Decodes sequence `index` into the provided buffer.
    ///
    /// The buffer is cleared before use. Returns the number of elements
    /// decoded, or `None` if `index >= num_sequences()`.
    #[inline]
    pub fn get_into(&mut self, index: usize, buf: &mut Vec<T>) -> Option<usize> {
        let iter = self.get(index)?;
        buf.clear();
        buf.extend(iter);
        Some(buf.len())
    }
}

/// An iterator over elements of a sequence, using a borrowed reader.
///
/// This iterator is returned by [`SeqVecReader::get`] and borrows the reader's
/// internal bitstream. It cannot outlive the mutable borrow of the reader.
pub struct ReaderSeqIter<'r, 'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Mutable reference to the reader's bitstream.
    reader: &'r mut SeqVecBitReader<'a, E>,
    /// Reference to the codec dispatcher.
    code_reader: &'r CodecReader<'a, T, E, B>,
    /// The bit position at which this sequence ends.
    end_bit: u64,
    /// Marker for the element type.
    _marker: std::marker::PhantomData<T>,
}

impl<'r, 'a, T: Storable, E: Endianness, B: AsRef<[u64]>> Iterator
    for ReaderSeqIter<'r, 'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.reader.bit_pos() >= self.end_bit {
            return None;
        }

        let word = self.code_reader.read(self.reader).unwrap();
        Some(T::from_word(word))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let bits_remaining = self.end_bit.saturating_sub(self.reader.bit_pos());
        if bits_remaining == 0 {
            return (0, Some(0));
        }
        (1, Some(bits_remaining as usize))
    }
}

impl<'r, 'a, T: Storable, E: Endianness, B: AsRef<[u64]>> std::iter::FusedIterator
    for ReaderSeqIter<'r, 'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
}
