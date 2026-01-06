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

use crate::fixed::FixedVec;
use crate::variable::traits::Storable;
use dsi_bitstream::{
    dispatch::{Codes, CodesRead, FuncCodeReader, StaticCodeRead},
    impls::{BufBitReader, MemWordReader},
    prelude::{BitRead, BitSeek, Endianness, LE},
    traits::DefaultReadParams,
};
use std::marker::PhantomData;

/// Type alias for the bit reader used internally by [`SeqVec`] accessors.
///
/// This reader is configured for in-memory buffers with infallible reads,
/// matching the configuration used in the `variable` module.
pub(crate) type SeqVecBitReader<'a, E> =
    BufBitReader<E, MemWordReader<u64, &'a [u64]>, DefaultReadParams>;

/// A hybrid codec dispatcher for reading variable-length codes.
///
/// This enum implements a two-tier dispatch strategy to balance performance
/// and compatibility across all supported codecs.
///
/// ## Dispatch Strategy
///
/// 1. **Fast Path** ([`CodecReader::Fast`]): For codecs with common parameters
///    (e.g., Gamma, Delta, Zeta with k ≤ 10), the `dsi-bitstream` library provides
///    pre-compiled function pointers via [`FuncCodeReader`]. This avoids runtime
///    dispatch overhead on every read operation.
///
/// 2. **Slow Path** ([`CodecReader::Slow`]): For codecs with parameters outside
///    the pre-compiled set (e.g., `Golomb { b: 15 }`), the dispatcher stores the
///    [`Codes`] enum and performs a `match` on each read. This is slower but
///    guarantees that any validly constructed [`SeqVec`] can be read.
///
/// The constructor automatically selects the appropriate path based on the codec.
pub(crate) enum CodecReader<E: Endianness, R: CodesRead<E>> {
    /// Fast path using a pre-resolved function pointer.
    Fast(FuncCodeReader<E, R>),
    /// Slow path using runtime dispatch on the [`Codes`] enum.
    Slow(Codes),
}

impl<E: Endianness, R: CodesRead<E>> CodecReader<E, R> {
    /// Creates a new codec reader, selecting the optimal dispatch strategy.
    ///
    /// Attempts to use the fast function-pointer path. Falls back to the slower
    /// match-based dispatch if the codec's parameters are not in the pre-compiled set.
    #[inline]
    pub(crate) fn new(code: Codes) -> Self {
        match FuncCodeReader::new(code) {
            Ok(func_reader) => Self::Fast(func_reader),
            Err(_) => Self::Slow(code),
        }
    }

    /// Reads and decodes a single value from the bitstream.
    #[inline(always)]
    pub(crate) fn read(&self, reader: &mut R) -> Result<u64, R::Error> {
        match self {
            Self::Fast(func_reader) => func_reader.read(reader),
            Self::Slow(code) => code.read(reader),
        }
    }
}

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
/// - [`FusedIterator`]: Guarantees that after returning `None`, all subsequent
///   calls return `None`.
///
/// Does **not** implement [`ExactSizeIterator`] because the element count is
/// unknown without fully decoding the sequence. The [`size_hint`](Iterator::size_hint)
/// method provides bounds based on the remaining bits.
///
/// ## Examples
///
/// ```ignore
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
    SeqVecBitReader<'a, E>: CodesRead<E>,
{
    /// The bitstream reader, positioned at the current element.
    reader: SeqVecBitReader<'a, E>,
    /// The codec dispatcher for decoding elements.
    code_reader: CodecReader<E, SeqVecBitReader<'a, E>>,
    /// The bit position at which this sequence ends (exclusive).
    end_bit: u64,
    /// Marker for the element type.
    _marker: PhantomData<T>,
}

impl<'a, T: Storable, E: Endianness> SeqIter<'a, T, E>
where
    SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E> + BitSeek,
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

        Self {
            reader,
            code_reader: CodecReader::new(encoding),
            end_bit,
            _marker: PhantomData,
        }
    }

    /// Returns the current bit position within the bitstream.
    #[inline]
    pub fn bit_pos(&self) -> u64 {
        self.reader.bit_pos()
    }

    /// Returns the ending bit position for this sequence (exclusive).
    #[inline]
    pub fn end_bit(&self) -> u64 {
        self.end_bit
    }

    /// Returns the number of bits remaining in this sequence.
    ///
    /// This is the number of compressed bits, not the number of elements.
    /// The element count depends on the values and codec used.
    #[inline]
    pub fn bits_remaining(&self) -> u64 {
        self.end_bit.saturating_sub(self.reader.bit_pos())
    }
}

impl<'a, T: Storable, E: Endianness> Iterator for SeqIter<'a, T, E>
where
    SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Termination check: compare current position against end boundary.
        // This branch is highly predictable (false until sequence ends).
        if self.reader.bit_pos() >= self.end_bit {
            return None;
        }

        // Decode the next element. Infallible for in-memory buffers.
        let word = self.code_reader.read(&mut self.reader).unwrap();
        Some(T::from_word(word))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let bits_remaining = self.bits_remaining();

        if bits_remaining == 0 {
            return (0, Some(0));
        }

        // Lower bound: at least one element exists if any bits remain.
        // Upper bound: maximum elements assuming each uses 1 bit (minimum for
        // unary/gamma codes encoding the value 0).
        (1, Some(bits_remaining as usize))
    }
}

impl<'a, T: Storable, E: Endianness> std::iter::FusedIterator for SeqIter<'a, T, E> where
    SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E>
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
/// - [`FusedIterator`]: After `None`, always returns `None`.
/// - [`DoubleEndedIterator`]: Supports iteration from both ends.
///
/// ## Examples
///
/// ```ignore
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
    bit_offsets: &'a FixedVec<u64, u64, LE, B>,
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
        bit_offsets: &'a FixedVec<u64, u64, LE, B>,
        encoding: Codes,
        num_sequences: usize,
    ) -> Self {
        Self {
            data,
            bit_offsets,
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
    SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E> + BitSeek,
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

        self.front += 1;

        Some(SeqIter::new(self.data, start_bit, end_bit, self.encoding))
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
    SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E> + BitSeek,
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

        Some(SeqIter::new(self.data, start_bit, end_bit, self.encoding))
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
    SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E> + BitSeek,
{
    #[inline]
    fn len(&self) -> usize {
        self.remaining()
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> std::iter::FusedIterator
    for SeqVecIter<'a, T, E, B>
where
    SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E> + BitSeek,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsi_bitstream::prelude::LE;

    // Helper to create a simple test scenario with known bit patterns.
    // For unit tests, we rely on integration with SeqVec in mod.rs tests.

    #[test]
    fn test_seq_vec_iter_size_hint() {
        // This test verifies the size_hint logic without needing actual data.
        // We test the iterator state management.

        // Create a minimal FixedVec with 4 sequences (5 offsets).
        let offsets_data: Vec<u64> = vec![0, 10, 20, 30, 40];
        let bit_offsets = FixedVec::<u64, u64, LE>::try_from(offsets_data.as_slice()).unwrap();
        let data: Vec<u64> = vec![0; 10]; // Dummy data

        let iter: SeqVecIter<'_, u32, LE, Vec<u64>> =
            SeqVecIter::new(&data, &bit_offsets, Codes::Gamma, 4);

        assert_eq!(iter.size_hint(), (4, Some(4)));
        assert_eq!(iter.remaining(), 4);
    }

    #[test]
    fn test_seq_vec_iter_double_ended() {
        let offsets_data: Vec<u64> = vec![0, 5, 10, 15];
        let bit_offsets = FixedVec::<u64, u64, LE>::try_from(offsets_data.as_slice()).unwrap();
        let data: Vec<u64> = vec![0; 10];

        let mut iter: SeqVecIter<'_, u32, LE, Vec<u64>> =
            SeqVecIter::new(&data, &bit_offsets, Codes::Gamma, 3);

        // Verify front/back state without actually decoding.
        assert_eq!(iter.remaining(), 3);

        // Advance from front.
        let _ = iter.next();
        assert_eq!(iter.remaining(), 2);

        // Advance from back.
        let _ = iter.next_back();
        assert_eq!(iter.remaining(), 1);

        // One element left.
        let _ = iter.next();
        assert_eq!(iter.remaining(), 0);

        // Exhausted.
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }
}
