//! A stateful reader optimized for sequential access to a [`SeqVec`].
//!
//! This module provides [`SeqVecSeqReader`], a reader that maintains internal
//! state to optimize access patterns where sequences are accessed in increasing
//! order or with high locality.
//!
//! When accessing sequence `i` followed by sequence `i+1`, the reader can
//! simply continue decoding from its current position instead of seeking.
//! This avoids the overhead of seeking for sequential traversals.
//!
//! [`SeqVec`]: crate::seq::SeqVec

use super::iter::{CodecReader, SeqVecBitReader};
use super::{SeqVec, SeqVecError};
use crate::variable::traits::Storable;
use dsi_bitstream::{
    dispatch::CodesRead,
    impls::MemWordReader,
    prelude::{BitRead, BitSeek, Endianness},
};
use std::marker::PhantomData;

/// A stateful reader optimized for sequential access to a [`SeqVec`].
///
/// This reader maintains an internal cursor tracking the current position in
/// the bitstream. When sequences are accessed in increasing order, the reader
/// can avoid seeking by decoding forward from its current position.
///
/// # Optimization Strategy
///
/// When [`get`](Self::get) is called with index `i`:
///
/// 1. **Fast path (no seek)**: If the reader's cursor is already positioned at
///    the start of sequence `i` (because sequence `i-1` was just fully decoded),
///    no seek is needed. The reader simply begins decoding.
///
/// 2. **Slow path (seek required)**: If the cursor is elsewhere (backward access,
///    skip, or first access), the reader seeks to sequence `i`'s bit offset.
///
/// This makes sequential iteration over all sequences nearly as fast as a
/// single linear scan of the bitstream.
///
/// # Examples
///
/// ```ignore
/// use compressed_intvec::seq::{SeqVec, LESeqVec};
///
/// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4], &[5, 6], &[7, 8]];
/// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
///
/// let mut reader = vec.seq_reader();
///
/// // Sequential access is optimized (no seeks after the first)
/// for i in 0..vec.num_sequences() {
///     let seq: Vec<u32> = reader.get(i).unwrap().collect();
///     println!("Sequence {}: {:?}", i, seq);
/// }
///
/// // Backward access triggers a seek
/// let _ = reader.get_vec(0); // Seeks back to start
/// ```
///
/// [`SeqVec`]: crate::seq::SeqVec
pub struct SeqVecSeqReader<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
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
    /// The index of the next sequence to be read (i.e., the sequence whose
    /// start position the cursor is currently at, if no partial read occurred).
    /// This is `usize::MAX` if the cursor position is unknown/invalid.
    next_sequence_index: usize,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVecSeqReader<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new sequential reader for the given [`SeqVec`].
    #[inline]
    pub(crate) fn new(seqvec: &'a SeqVec<T, E, B>) -> Self {
        let reader = SeqVecBitReader::<E>::new(MemWordReader::new(seqvec.data.as_ref()));
        let code_reader = CodecReader::new(seqvec.encoding);

        Self {
            seqvec,
            reader,
            code_reader,
            next_sequence_index: 0, // Cursor starts at the beginning.
        }
    }

    /// Returns the number of sequences in the underlying [`SeqVec`].
    #[inline]
    pub fn num_sequences(&self) -> usize {
        self.seqvec.num_sequences()
    }

    /// Returns the index of the next sequence that would be read without seeking.
    ///
    /// This is useful for understanding the reader's current state. If the
    /// returned value equals the index you want to access, the access will
    /// be fast (no seek required).
    #[inline]
    pub fn current_position(&self) -> usize {
        self.next_sequence_index
    }

    /// Resets the reader to the beginning of the [`SeqVec`].
    ///
    /// After calling this, the next access to sequence 0 will not require
    /// a seek operation.
    #[inline]
    pub fn reset(&mut self) {
        self.reader.set_bit_pos(0).unwrap();
        self.next_sequence_index = 0;
    }

    /// Returns an iterator over the elements of sequence `index`.
    ///
    /// Returns `None` if `index >= num_sequences()`.
    ///
    /// If `index == current_position()`, no seek is performed (fast path).
    /// Otherwise, the reader seeks to the sequence's bit offset.
    #[inline]
    pub fn get(&mut self, index: usize) -> Option<SeqReaderIter<'_, 'a, T, E, B>> {
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
    pub unsafe fn get_unchecked(&mut self, index: usize) -> SeqReaderIter<'_, 'a, T, E, B> {
        debug_assert!(
            index < self.seqvec.num_sequences(),
            "index {} out of bounds for {} sequences",
            index,
            self.seqvec.num_sequences()
        );

        let start_bit = self.seqvec.bit_offsets.get_unchecked(index);
        let end_bit = self.seqvec.bit_offsets.get_unchecked(index + 1);

        // Decide whether to seek or continue from current position.
        if index != self.next_sequence_index {
            // Slow path: need to seek.
            self.reader.set_bit_pos(start_bit).unwrap();
        }
        // else: fast path, cursor is already at the right position.

        // After the iterator is fully consumed, the cursor will be at `end_bit`,
        // which is the start of sequence `index + 1`. We update the index now,
        // assuming the caller will fully consume the iterator. If they don't,
        // the next access will require a seek anyway.
        self.next_sequence_index = index + 1;

        SeqReaderIter {
            reader: &mut self.reader,
            code_reader: &self.code_reader,
            end_bit,
            _marker: PhantomData,
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

    /// Skips the next `n` sequences, advancing the cursor.
    ///
    /// This is useful when you know you want to skip some sequences without
    /// decoding them. It performs a single seek to the start of the sequence
    /// at `current_position() + n`.
    ///
    /// Returns `false` if skipping would go past the end of the [`SeqVec`].
    #[inline]
    pub fn skip(&mut self, n: usize) -> bool {
        let target = self.next_sequence_index.saturating_add(n);
        if target > self.seqvec.num_sequences() {
            return false;
        }

        if target == self.seqvec.num_sequences() {
            // Position at the end (after last sequence).
            let end_bit = unsafe {
                self.seqvec
                    .bit_offsets
                    .get_unchecked(self.seqvec.num_sequences())
            };
            self.reader.set_bit_pos(end_bit).unwrap();
        } else {
            let target_bit = unsafe { self.seqvec.bit_offsets.get_unchecked(target) };
            self.reader.set_bit_pos(target_bit).unwrap();
        }

        self.next_sequence_index = target;
        true
    }
}

/// An iterator over elements of a sequence from a sequential reader.
///
/// This iterator is returned by [`SeqVecSeqReader::get`] and borrows the
/// reader's internal bitstream.
pub struct SeqReaderIter<'r, 'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
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
    _marker: PhantomData<T>,
}

impl<'r, 'a, T: Storable, E: Endianness, B: AsRef<[u64]>> Iterator
    for SeqReaderIter<'r, 'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Branch-predicted comparison: almost always false until sequence ends.
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
    for SeqReaderIter<'r, 'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
}
