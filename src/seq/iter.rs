//! Iterators for [`SeqVec`].
//!
//! This module provides iterator types for accessing sequences and their elements
//! within a [`SeqVec`]. The design prioritizes zero-allocation iteration and
//! efficient bit-level termination detection.
//!
//! The primary types are:
//!
//! - [`SeqIter`]: Iterator over the elements of a single sequence.
//! - [`SeqVecIter`]: Iterator over all sequences in a [`SeqVec`].
//!
//! [`SeqVec`]: crate::seq::SeqVec

use crate::common::codec_reader::CodecReader;
use crate::fixed::FixedVec;
use crate::variable::traits::Storable;
use dsi_bitstream::{
    codes::params::DefaultReadParams,
    dispatch::{Codes, CodesRead, StaticCodeRead},
    impls::{BufBitReader, MemWordReader},
    prelude::{BitRead, BitSeek, Endianness},
};
use std::cell::Cell;
use std::marker::PhantomData;

/// Type alias for the bit reader used internally by [`SeqVec`] accessors.
///
/// This reader is configured for in-memory buffers with infallible reads,
/// matching the configuration used in the `variable` module.
pub(crate) type SeqVecBitReader<'a, E> =
    BufBitReader<E, MemWordReader<u64, &'a [u64]>, DefaultReadParams>;

/// A zero-allocation iterator over the elements of a single sequence.
///
/// This iterator decodes elements lazily from the compressed bitstream. It
/// determines when to stop by comparing the current bit position against the
/// end boundary of the sequence. This approach avoids storing explicit lengths.
///
/// ## Termination Logic
///
/// Each call to [`next`](Iterator::next) performs:
/// 1. A comparison: `current_bit_pos >= end_bit`.
/// 2. If not at end, a variable-length decode operation.
///
/// The comparison involves two `u64` values and is highly predictable (almost
/// always `false` until the sequence ends), making it effectively free due to
/// branch prediction.
///
/// ## Trait Implementations
///
/// - [`Iterator`]: Core iteration functionality.
/// - [`std::iter::FusedIterator`]: Guarantees that after returning `None`, all subsequent
///   calls return `None`.
///
/// Implements [`ExactSizeIterator`] when explicit lengths are available. If
/// lengths are not stored, computing an exact size hint requires decoding the
/// remaining elements to count them.
///
/// ## Examples
///
/// ```
/// use compressed_intvec::seq::{SeqVec, LESeqVec};
///
/// let sequences: &[&[u32]] = &[&[10, 20, 30], &[100, 200]];
/// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
///
/// // Iterate over the first sequence
/// let mut sum = 0u32;
/// for value in vec.get(0).unwrap() {
///     sum += value;
/// }
/// assert_eq!(sum, 60);
/// ```
pub struct SeqIter<'a, T: Storable, E: Endianness>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// The underlying data buffer for optional length computations.
    data: &'a [u64],
    /// The bitstream reader, positioned at the current element.
    reader: SeqVecBitReader<'a, E>,
    /// The hybrid codec reader for decoding elements.
    /// Provides fast-path optimization via function pointers for common codecs,
    /// with fallback to dynamic dispatch for uncommon parameter combinations.
    code_reader: CodecReader<'a, E>,
    /// The bit position at which this sequence ends (exclusive).
    end_bit: u64,
    /// The codec used for this sequence.
    encoding: Codes,
    /// Cached bit position used only for the `size_hint()` method.
    /// This avoids calling the mutable `bit_pos()` method during non-mutable size queries.
    cached_bit_pos: u64,
    /// Optional remaining length of the sequence.
    ///
    /// When available, this enables exact size hints without extra decoding.
    remaining_len: Cell<Option<usize>>,
    /// Marker for the element type.
    _marker: PhantomData<T>,
}

impl<'a, T: Storable, E: Endianness> SeqIter<'a, T, E>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new iterator over a sequence.
    ///
    /// The iterator decodes elements starting from `start_bit` and stops when
    /// the bit position reaches or exceeds `end_bit`.
    ///
    /// ## Arguments
    ///
    /// * `data` - The compressed data buffer containing all sequences.
    /// * `start_bit` - The bit offset where this sequence begins.
    /// * `end_bit` - The bit offset where this sequence ends (exclusive).
    /// * `encoding` - The codec used for compression.
    #[inline]
    pub(crate) fn new(data: &'a [u64], start_bit: u64, end_bit: u64, encoding: Codes) -> Self {
        let mut reader = SeqVecBitReader::<E>::new(MemWordReader::new(data));

        // Seek to the start of this sequence. The operation is infallible for
        // in-memory readers, but we handle the Result for type correctness.
        let _ = reader.set_bit_pos(start_bit);

        // Create the hybrid codec reader for efficient decoding.
        let code_reader = CodecReader::new(encoding);

        Self {
            data,
            reader,
            code_reader,
            end_bit,
            encoding,
            cached_bit_pos: start_bit,
            remaining_len: Cell::new(None),
            _marker: PhantomData,
        }
    }

    /// Creates a new iterator over a sequence with a known length.
    ///
    /// This constructor is used when explicit sequence lengths are stored,
    /// enabling exact size hints without additional decoding.
    #[inline]
    pub(crate) fn new_with_len(
        data: &'a [u64],
        start_bit: u64,
        end_bit: u64,
        encoding: Codes,
        len: Option<usize>,
    ) -> Self {
        let iter = Self::new(data, start_bit, end_bit, encoding);
        if let Some(len) = len {
            iter.remaining_len.set(Some(len));
        }
        iter
    }

    /// Returns the remaining length of the sequence, computing it if necessary.
    #[inline]
    fn remaining_len(&self) -> usize {
        if let Some(len) = self.remaining_len.get() {
            return len;
        }

        let mut reader = SeqVecBitReader::<E>::new(MemWordReader::new(self.data));
        let _ = reader.set_bit_pos(self.cached_bit_pos);
        let code_reader = CodecReader::new(self.encoding);

        let mut count = 0usize;
        while reader.bit_pos().unwrap_or(self.cached_bit_pos) < self.end_bit {
            let _ = code_reader.read(&mut reader).unwrap();
            count += 1;
        }

        self.remaining_len.set(Some(count));
        count
    }

    /// Returns the ending bit position for this sequence (exclusive).
    ///
    /// This is useful for understanding the memory footprint of the sequence
    /// when compressed.
    #[inline]
    pub fn end_bit(&self) -> u64 {
        self.end_bit
    }
}

impl<'a, T: Storable, E: Endianness> Iterator for SeqIter<'a, T, E>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Termination check: Query current bit position and compare against end.
        // This branch is highly predictable (false until sequence ends).
        let current_bit = self.reader.bit_pos().unwrap_or(self.cached_bit_pos);
        if current_bit >= self.end_bit {
            return None;
        }

        // Decode the next element using the optimized codec reader.
        // CodecReader provides fast-path dispatch via function pointers for
        // common codecs, with fallback to dynamic dispatch for uncommon parameters.
        let word = self.code_reader.read(&mut self.reader).unwrap();

        // Update cached position for size_hint() using the reader's state.
        self.cached_bit_pos = self.reader.bit_pos().unwrap_or(current_bit);

        if let Some(len) = self.remaining_len.get() {
            self.remaining_len.set(Some(len.saturating_sub(1)));
        }

        Some(T::from_word(word))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining_len();
        (remaining, Some(remaining))
    }
}

impl<'a, T: Storable, E: Endianness> ExactSizeIterator for SeqIter<'a, T, E>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    #[inline]
    fn len(&self) -> usize {
        self.remaining_len()
    }
}

impl<'a, T: Storable, E: Endianness> std::iter::FusedIterator for SeqIter<'a, T, E> where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>
{
}

/// An iterator over all sequences in a [`SeqVec`].
///
/// This iterator is created by the [`iter`] method on [`SeqVec`]. Each call to
/// [`next`](Iterator::next) returns a [`SeqIter`] for the corresponding sequence.
///
/// ## Trait Implementations
///
/// - [`Iterator`]: Core iteration.
/// - [`ExactSizeIterator`]: The number of sequences is known.
/// - [`std::iter::FusedIterator`]: After `None`, always returns `None`.
/// - [`DoubleEndedIterator`]: Supports iteration from both ends.
///
/// ## Examples
///
/// ```
/// use compressed_intvec::seq::{SeqVec, LESeqVec};
///
/// let sequences: &[&[u32]] = &[&[1, 2], &[3], &[4, 5, 6]];
/// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
///
/// // Collect all sequences into vectors
/// let all: Vec<Vec<u32>> = vec.iter().map(|s| s.collect()).collect();
/// assert_eq!(all, vec![vec![1, 2], vec![3], vec![4, 5, 6]]);
/// ```
///
/// [`SeqVec`]: crate::seq::SeqVec
/// [`iter`]: crate::seq::SeqVec::iter
pub struct SeqVecIter<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    SeqVecBitReader<'a, E>: CodesRead<E>,
{
    /// Reference to the compressed data buffer.
    data: &'a [u64],
    /// Reference to the bit offsets index.
    bit_offsets: &'a FixedVec<u64, u64, E, B>,
    /// Optional reference to stored sequence lengths.
    seq_lengths: Option<&'a FixedVec<usize, u64, E, Vec<u64>>>,
    /// The codec used for compression.
    encoding: Codes,
    /// Current front index (for forward iteration).
    front: usize,
    /// Current back index (exclusive, for backward iteration).
    back: usize,
    /// Markers for type parameters.
    _marker: PhantomData<(T, E)>,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVecIter<'a, T, E, B>
where
    SeqVecBitReader<'a, E>: CodesRead<E>,
{
    /// Creates a new iterator over all sequences.
    ///
    /// ## Arguments
    ///
    /// * `data` - The compressed data buffer.
    /// * `bit_offsets` - The index of bit offsets for each sequence.
    /// * `encoding` - The codec used for compression.
    /// * `num_sequences` - The total number of sequences.
    #[inline]
    pub(crate) fn new(
        data: &'a [u64],
        bit_offsets: &'a FixedVec<u64, u64, E, B>,
        seq_lengths: Option<&'a FixedVec<usize, u64, E, Vec<u64>>>,
        encoding: Codes,
        num_sequences: usize,
    ) -> Self {
        Self {
            data,
            bit_offsets,
            seq_lengths,
            encoding,
            front: 0,
            back: num_sequences,
            _marker: PhantomData,
        }
    }

    /// Returns the number of sequences remaining in this iterator.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.back.saturating_sub(self.front)
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> Iterator for SeqVecIter<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = SeqIter<'a, T, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }

        // SAFETY: front < back <= num_sequences, and bit_offsets has
        // num_sequences + 1 elements, so indices front and front + 1 are valid.
        let start_bit = unsafe { self.bit_offsets.get_unchecked(self.front) };
        let end_bit = unsafe { self.bit_offsets.get_unchecked(self.front + 1) };
        let len = self
            .seq_lengths
            .map(|lengths| unsafe { lengths.get_unchecked(self.front) });

        self.front += 1;

        Some(SeqIter::new_with_len(
            self.data,
            start_bit,
            end_bit,
            self.encoding,
            len,
        ))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }

    #[inline]
    fn count(self) -> usize {
        self.remaining()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.remaining() {
            self.front = self.back;
            return None;
        }
        self.front += n;
        self.next()
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> DoubleEndedIterator
    for SeqVecIter<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }

        self.back -= 1;

        // SAFETY: back was decremented from a value > front, so back is a valid
        // index, and back + 1 <= original num_sequences.
        let start_bit = unsafe { self.bit_offsets.get_unchecked(self.back) };
        let end_bit = unsafe { self.bit_offsets.get_unchecked(self.back + 1) };
        let len = self
            .seq_lengths
            .map(|lengths| unsafe { lengths.get_unchecked(self.back) });

        Some(SeqIter::new_with_len(
            self.data,
            start_bit,
            end_bit,
            self.encoding,
            len,
        ))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.remaining() {
            self.back = self.front;
            return None;
        }
        self.back -= n;
        self.next_back()
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for SeqVecIter<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    #[inline]
    fn len(&self) -> usize {
        self.remaining()
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> std::iter::FusedIterator
    for SeqVecIter<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
}

/// An owning iterator over all sequences in a [`crate::seq::SeqVec`].
///
/// This iterator consumes a [`crate::seq::SeqVec<T, E, Vec<u64>>`] and yields [`SeqIter`]
/// instances for each sequence. The iterator owns the underlying data buffer,
/// ensuring it lives as long as the iterator.
///
/// ## Design
///
/// This is a self-referential struct similar to [`IntVecIntoIter`](crate::variable::iter::IntVecIntoIter).
/// The data buffer is stored in `_data_owner`, and a transmuted `'static` reference
/// is used to create the reader. This is safe because `_data_owner` is part of
/// the same struct, guaranteeing the data outlives the reader.
///
/// ## Trait Implementations
///
/// - [`Iterator`]: Core iteration.
/// - [`ExactSizeIterator`]: The number of sequences is known.
/// - [`DoubleEndedIterator`]: Supports iteration from both ends.
/// - [`std::iter::FusedIterator`]: After `None`, always returns `None`.
///
/// ## Examples
///
/// ```
/// use compressed_intvec::seq::{SeqVec, LESeqVec};
///
/// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5]];
/// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
///
/// // Consume the vec and iterate over sequences
/// let mut count = 0;
/// for seq in vec {
///     count += 1;
///     let _ = seq.collect::<Vec<_>>();
/// }
/// assert_eq!(count, 2);
/// ```
pub struct SeqVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// The number of sequences remaining in the iterator.
    num_sequences: usize,
    /// The current sequence index being iterated.
    current_index: usize,
    /// Reference to the bit offsets index. This is a transmuted reference
    /// that is guaranteed to be valid for the lifetime of `_data_owner`.
    bit_offsets: &'static [u64],
    /// Reference to the compressed data buffer. This is a transmuted reference
    /// that is guaranteed to be valid for the lifetime of `_data_owner`.
    data: &'static [u64],
    /// The codec used for decoding.
    encoding: Codes,
    /// Optional stored sequence lengths.
    seq_lengths: Option<FixedVec<usize, u64, E, Vec<u64>>>,
    /// This field owns the data buffer, ensuring it lives as long as the iterator.
    _data_owner: Vec<u64>,
    /// This field owns the bit offsets buffer.
    _bit_offsets_owner: Vec<u64>,
    /// Phantom data to hold the generic types.
    _markers: PhantomData<(T, E)>,
}

impl<T, E> SeqVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new owning iterator from a [`SeqVec`] with owned data.
    pub(crate) fn new(vec: super::SeqVec<T, E, Vec<u64>>) -> Self {
        let encoding = vec.encoding;
        let num_sequences = vec.num_sequences();
        let seq_lengths = vec.seq_lengths;

        // Extract the owned buffers.
        //
        // IMPORTANT: `FixedVec::as_limbs()` returns the *packed* backing storage,
        // not the logical element values. For `bit_offsets` we need the decoded
        // offsets (N+1 values, including the sentinel).
        let _data_owner = vec.data;
        let mut _bit_offsets_owner = Vec::with_capacity(vec.bit_offsets.len());
        for i in 0..vec.bit_offsets.len() {
            _bit_offsets_owner.push(unsafe { vec.bit_offsets.get_unchecked(i) });
        }

        // Create transmuted 'static references. This is safe because _data_owner
        // and _bit_offsets_owner are part of this struct.
        let data_ref: &'static [u64] = unsafe { std::mem::transmute(_data_owner.as_slice()) };
        let bit_offsets_ref: &'static [u64] =
            unsafe { std::mem::transmute(_bit_offsets_owner.as_slice()) };

        Self {
            num_sequences,
            current_index: 0,
            bit_offsets: bit_offsets_ref,
            data: data_ref,
            encoding,
            seq_lengths,
            _data_owner,
            _bit_offsets_owner,
            _markers: PhantomData,
        }
    }
}

impl<T, E> Iterator for SeqVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = SeqIter<'static, T, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.num_sequences {
            return None;
        }

        let start_bit = self.bit_offsets[self.current_index];
        let end_bit = self.bit_offsets[self.current_index + 1];
        let len = self
            .seq_lengths
            .as_ref()
            .map(|lengths| unsafe { lengths.get_unchecked(self.current_index) });

        self.current_index += 1;

        Some(SeqIter::new_with_len(
            self.data,
            start_bit,
            end_bit,
            self.encoding,
            len,
        ))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.num_sequences.saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

impl<T, E> ExactSizeIterator for SeqVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    #[inline]
    fn len(&self) -> usize {
        self.num_sequences.saturating_sub(self.current_index)
    }
}

impl<T, E> DoubleEndedIterator for SeqVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.num_sequences {
            return None;
        }

        self.num_sequences -= 1;

        let start_bit = self.bit_offsets[self.num_sequences];
        let end_bit = self.bit_offsets[self.num_sequences + 1];

        let len = self
            .seq_lengths
            .as_ref()
            .map(|lengths| unsafe { lengths.get_unchecked(self.num_sequences) });

        Some(SeqIter::new_with_len(
            self.data,
            start_bit,
            end_bit,
            self.encoding,
            len,
        ))
    }
}

impl<T, E> std::iter::FusedIterator for SeqVecIntoIter<T, E>
where
    T: Storable + 'static,
    E: Endianness + 'static,
    for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
}
