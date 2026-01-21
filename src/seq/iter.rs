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
/// ## Exact Size Hint
///
/// This iterator only implements [`ExactSizeIterator`] when the sequence length
/// is known upfront (e.g., when created from `SeqVecIter` with pre-stored lengths).
/// If lengths are not pre-stored, computing the size requires decoding the entire
/// remaining sequence—an O(n) operation that would be deceptive if hidden behind
/// the `ExactSizeIterator` contract (which users expect to be O(1)).
///
/// Call [`size_hint()`](Iterator::size_hint) for a lower bound without full decoding.
///
/// ## Trait Implementations
///
/// - [`Iterator`]: Core iteration functionality.
/// - [`std::iter::FusedIterator`]: Guarantees that after returning `None`, all subsequent
///   calls return `None`.
/// - [`ExactSizeIterator`]: Only when sequence length is known upfront.
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
    /// The bitstream reader, positioned at the current element.
    reader: SeqVecBitReader<'a, E>,
    /// The hybrid codec reader for decoding elements.
    /// Provides fast-path optimization via function pointers for common codecs,
    /// with fallback to dynamic dispatch for uncommon parameter combinations.
    code_reader: CodecReader<'a, E>,
    /// The bit position at which this sequence ends (exclusive).
    end_bit: u64,
    /// The known length of the sequence, if available.
    /// When `None`, length is not pre-computed; call `count()` on the iterator
    /// if an exact count is needed.
    known_len: Option<usize>,
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
            reader,
            code_reader,
            end_bit,
            known_len: None,
            _marker: PhantomData,
        }
    }

    /// Creates a new iterator over a sequence with a known length.
    ///
    /// This constructor is used when explicit sequence lengths are stored,
    /// enabling [`ExactSizeIterator`] without additional decoding.
    #[inline]
    pub(crate) fn new_with_len(
        data: &'a [u64],
        start_bit: u64,
        end_bit: u64,
        encoding: Codes,
        len: Option<usize>,
    ) -> Self {
        let mut iter = Self::new(data, start_bit, end_bit, encoding);
        iter.known_len = len;
        iter
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
        // When length is known, use counted termination to avoid bit_pos() overhead.
        // The branch discriminant is stable across all iterations, so the branch
        // predictor learns the pattern with ~100% accuracy after 1-2 iterations,
        // making misprediction cost amortized to essentially zero.
        if let Some(ref mut remaining) = self.known_len {
            if *remaining == 0 {
                return None;
            }
            *remaining -= 1;

            // Decode element using optimized codec reader.
            let word = self.code_reader.read(&mut self.reader).unwrap();
            return Some(T::from_word(word));
        }

        // Fallback to bit-bounded termination when length is unknown.
        // This path performs a multi-instruction bit_pos() computation per element,
        // so only use it when length is not pre-stored.
        if self.reader.bit_pos().unwrap() >= self.end_bit {
            return None;
        }

        // Decode element using optimized codec reader.
        let word = self.code_reader.read(&mut self.reader).unwrap();
        Some(T::from_word(word))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // If length is known, provide exact size hint.
        if let Some(len) = self.known_len {
            return (len, Some(len));
        }
        // Otherwise, provide no hint (length unknown).
        (0, None)
    }
}

impl<'a, T: Storable, E: Endianness> std::iter::FusedIterator for SeqIter<'a, T, E> where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>
{
}

/// Conditional [`ExactSizeIterator`] implementation for [`SeqIter`].
///
/// This is only implemented when the sequence length is known upfront (i.e., when
/// created with `new_with_len` and `known_len` is `Some`). This ensures the
/// `len()` method is O(1) and doesn't require expensive decoding.
///
/// If the length is unknown, users can call `count()` on the iterator to consume
/// it and get the exact count, but this is explicit and O(n).
impl<'a, T: Storable, E: Endianness> ExactSizeIterator for SeqIter<'a, T, E>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Returns the exact number of remaining elements.
    ///
    /// # Panics
    ///
    /// Panics if the length was not known upfront (i.e., this iterator was created
    /// with `new` instead of `new_with_len` with a `Some` value). Use `count()`
    /// instead if the length is unknown; it will consume the iterator but provide
    /// the exact count.
    #[inline]
    fn len(&self) -> usize {
        self.known_len.expect(
            "SeqIter::len() called on iterator with unknown length. \
             This can only happen if the iterator was not created with a pre-stored \
             length. Use count() instead.",
        )
    }
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
/// This is a self-referential struct similar to [`VarVecIntoIter`](crate::variable::iter::VarVecIntoIter).
/// The data buffer is stored in `_data_owner`, and a transmuted `'static` reference
/// is used to create the reader. This is safe because `_data_owner` is part of
/// the same struct, guaranteeing the data outlives the reader.
///
/// ## Bit Offsets Storage
///
/// Rather than decoding all offsets from the `FixedVec` upfront (O(n) operation),
/// we store the raw bit width and offsets backing storage. Offsets are decoded
/// lazily during iteration via `get_offset()`.
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
    // Immutable fields (configuration and backing storage)
    /// Reference to the compressed data buffer. This is a transmuted reference
    /// that is guaranteed to be valid for the lifetime of `_data_owner`.
    data: &'static [u64],
    /// The codec used for decoding.
    encoding: Codes,
    /// The bit offsets backing storage (raw limbs).
    bit_offsets_data: &'static [u64],
    /// Number of bits per offset value.
    bit_offsets_num_bits: u64,
    /// Total number of bit offsets (including sentinel at N+1).
    bit_offsets_len: usize,
    /// Optional stored sequence lengths.
    seq_lengths: Option<FixedVec<usize, u64, E, Vec<u64>>>,
    /// This field owns the data buffer, ensuring it lives as long as the iterator.
    _data_owner: Vec<u64>,
    /// This field owns the bit offsets buffer.
    _bit_offsets_owner: Vec<u64>,
    /// Phantom data to hold the generic types.
    _markers: PhantomData<(T, E)>,

    // Mutable fields (iteration state)
    /// The number of sequences remaining in the iterator.
    num_sequences: usize,
    /// The current sequence index being iterated.
    current_index: usize,
    /// Cached end_bit from previous iteration to avoid redundant offset decoding.
    /// For sequential access, the end_bit of iteration i becomes start_bit of iteration i+1.
    last_end_bit: u64,
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
    ///
    /// This constructor avoids decoding all offsets upfront. Instead, it stores
    /// the raw bit width and backing data of the `FixedVec`, allowing offsets to
    /// be decoded lazily during iteration.
    pub(crate) fn new(vec: super::SeqVec<T, E, Vec<u64>>) -> Self {
        let encoding = vec.encoding;
        let num_sequences = vec.num_sequences();
        let seq_lengths = vec.seq_lengths;
        let _data_owner = vec.data;
        let _bit_offsets_owner = vec.bit_offsets.as_limbs().to_vec();

        // Store bit offset metadata for lazy decoding during iteration.
        let bit_offsets_num_bits = vec.bit_offsets.bit_width() as u64;
        let bit_offsets_len = vec.bit_offsets.len();

        // Create transmuted 'static references. This is safe because _data_owner
        // and _bit_offsets_owner are part of this struct.
        let data_ref: &'static [u64] = unsafe { std::mem::transmute(_data_owner.as_slice()) };
        let bit_offsets_ref: &'static [u64] =
            unsafe { std::mem::transmute(_bit_offsets_owner.as_slice()) };

        Self {
            num_sequences,
            current_index: 0,
            data: data_ref,
            encoding,
            bit_offsets_data: bit_offsets_ref,
            bit_offsets_num_bits,
            bit_offsets_len,
            seq_lengths,
            _data_owner,
            _bit_offsets_owner,
            _markers: PhantomData,
            last_end_bit: 0,
        }
    }

    /// Lazily decodes the bit offset for the given index.
    ///
    /// This avoids decoding all offsets upfront, achieving O(1) amortized cost
    /// during forward iteration.
    #[inline]
    fn get_offset(&self, index: usize) -> u64 {
        if index >= self.bit_offsets_len {
            return 0; // Out of bounds: return sentinel
        }

        let offset_bits = self.bit_offsets_num_bits;
        let start_bit = (index as u64) * offset_bits;

        // Create a reader over the bit offsets data and extract the value.
        let mut reader = SeqVecBitReader::<E>::new(MemWordReader::new(self.bit_offsets_data));
        let _ = reader.set_bit_pos(start_bit);

        // Read the offset value. For typical cases with explicit bit widths,
        // this is a simple bit extraction.
        reader.read_bits(offset_bits as usize).unwrap_or(0)
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

        // Optimize sequential access: reuse cached end_bit from previous iteration.
        // For forward iteration, the end_bit of iteration i becomes the start_bit
        // of iteration i+1, eliminating a redundant offset decode operation.
        let start_bit = if self.current_index == 0 {
            self.get_offset(0)
        } else {
            self.last_end_bit
        };

        let end_bit = self.get_offset(self.current_index + 1);
        self.last_end_bit = end_bit;

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

        let start_bit = self.get_offset(self.num_sequences);
        let end_bit = self.get_offset(self.num_sequences + 1);

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
