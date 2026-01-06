//! Iterators for the [`SeqVec`] data structure.
//!
//! This module provides the iterator types used to access elements within
//! a [`SeqVec`]. The primary type is [`SeqIter`], a zero-allocation iterator
//! over the elements of a single sequence.
//!
//! [`SeqVec`]: crate::seq::SeqVec

use crate::variable::traits::Storable;
use dsi_bitstream::{
    dispatch::{Codes, CodesRead, FuncCodeReader, StaticCodeRead},
    impls::{BufBitReader, DefaultReadParams, MemWordReader},
    prelude::{BitRead, BitSeek, Endianness},
};
use std::marker::PhantomData;

/// Type alias for the bit reader used internally by `SeqVec` accessors.
pub(crate) type SeqVecBitReader<'a, E> =
    BufBitReader<E, MemWordReader<u64, &'a [u64]>, DefaultReadParams>;

/// A hybrid codec dispatcher for reading variable-length codes.
///
/// This enum provides two dispatch strategies to maximize performance while
/// guaranteeing correctness for all supported codecs.
///
/// The fast path uses pre-compiled function pointers for common codec configurations.
/// The slow path uses runtime dispatch via a `match` statement for less common
/// configurations. This ensures that any validly constructed `SeqVec` can be read.
pub(crate) enum CodecReader<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> {
    /// Fast path: uses a function pointer for common codecs.
    Fast(FuncCodeReader),
    /// Slow path: stores the codec enum and dispatches at runtime.
    Slow(Codes),
    /// Zero-sized variant to carry the type parameters.
    _Phantom(PhantomData<(&'a T, E, B)>),
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> CodecReader<'a, T, E, B> {
    /// Creates a new codec reader, selecting the optimal dispatch strategy.
    ///
    /// This constructor attempts to use the fast function-pointer path. If the
    /// codec's parameters are not supported by the pre-compiled set, it falls
    /// back to the slower but universally compatible match-based dispatch.
    #[inline]
    pub(crate) fn new(code: Codes) -> Self {
        match FuncCodeReader::try_new(code) {
            Ok(func_reader) => Self::Fast(func_reader),
            Err(_) => Self::Slow(code),
        }
    }

    /// Reads and decodes a single value from the bitstream.
    #[inline(always)]
    pub(crate) fn read<R>(&self, reader: &mut R) -> Result<u64, R::Error>
    where
        R: BitRead<E> + CodesRead<E>,
    {
        match self {
            Self::Fast(func_reader) => func_reader.read(reader),
            Self::Slow(code) => code.read(reader),
            Self::_Phantom(_) => unreachable!(),
        }
    }
}

/// A zero-allocation iterator over the elements of a single sequence in a [`SeqVec`].
///
/// This iterator is created by the [`get`] method on [`SeqVec`]. It decodes
/// elements lazily from the compressed bitstream, stopping when the bit position
/// reaches the end of the sequence's allocated bit range.
///
/// # Performance
///
/// Each call to [`next`] performs:
/// 1. A comparison of the current bit position against the end boundary (two `u64`).
/// 2. A variable-length decode operation.
///
/// The comparison is highly predictable (almost always `false` until the sequence
/// ends), making it effectively free due to branch prediction.
///
/// # Implementing Traits
///
/// `SeqIter` implements [`Iterator`] and [`FusedIterator`]. It does **not**
/// implement [`ExactSizeIterator`] because the number of elements is unknown
/// without decoding the entire sequence. The [`size_hint`] method provides a
/// lower bound of 0 and an upper bound estimated from the available bits.
///
/// [`SeqVec`]: crate::seq::SeqVec
/// [`get`]: crate::seq::SeqVec::get
/// [`next`]: Iterator::next
/// [`size_hint`]: Iterator::size_hint
pub struct SeqIter<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E>,
{
    /// The bitstream reader positioned at the current element.
    reader: SeqVecBitReader<'a, E>,
    /// The codec dispatcher for decoding elements.
    code_reader: CodecReader<'a, T, E, B>,
    /// The bit position at which this sequence ends (exclusive).
    /// When `reader.bit_pos() >= end_bit`, the sequence is exhausted.
    end_bit: u64,
    /// Marker for the element type.
    _marker: PhantomData<T>,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> SeqIter<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E>,
{
    /// Creates a new iterator over a sequence.
    ///
    /// The iterator will decode elements starting from `start_bit` until
    /// the bit position reaches `end_bit`.
    ///
    /// # Arguments
    ///
    /// * `data` - The compressed data buffer.
    /// * `start_bit` - The bit offset where the sequence begins.
    /// * `end_bit` - The bit offset where the sequence ends (exclusive).
    /// * `encoding` - The codec used for compression.
    #[inline]
    pub(crate) fn new(data: &'a [u64], start_bit: u64, end_bit: u64, encoding: Codes) -> Self
    where
        SeqVecBitReader<'a, E>: BitSeek<Error = core::convert::Infallible>,
    {
        let mut reader = SeqVecBitReader::<E>::new(MemWordReader::new(data));
        // Seek is infallible for in-memory readers.
        reader.set_bit_pos(start_bit).unwrap();

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
    /// Note that this is the number of *bits*, not elements. The number of
    /// elements depends on the values and the codec used.
    #[inline]
    pub fn bits_remaining(&self) -> u64 {
        self.end_bit.saturating_sub(self.reader.bit_pos())
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> Iterator for SeqIter<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Check if we have reached or passed the end of this sequence.
        // This comparison is highly predictable (almost always false).
        if self.reader.bit_pos() >= self.end_bit {
            return None;
        }

        // Decode the next element. The read is infallible for in-memory buffers.
        let word = self.code_reader.read(&mut self.reader).unwrap();
        Some(T::from_word(word))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let bits_remaining = self.bits_remaining();
        if bits_remaining == 0 {
            return (0, Some(0));
        }

        // Lower bound: at least one element if there are any bits left.
        // Upper bound: maximum elements if each used only 1 bit (minimum for
        // unary/gamma codes encoding the value 0).
        let upper = bits_remaining as usize;
        (1, Some(upper))
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> std::iter::FusedIterator
    for SeqIter<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E>,
{
}

/// An iterator over all sequences in a [`SeqVec`].
///
/// This iterator is created by the [`iter`] method on [`SeqVec`]. Each call
/// to [`next`] returns a [`SeqIter`] for the next sequence.
///
/// [`SeqVec`]: crate::seq::SeqVec
/// [`iter`]: crate::seq::SeqVec::iter
/// [`next`]: Iterator::next
pub struct SeqVecIter<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E>,
{
    /// Reference to the compressed data.
    data: &'a [u64],
    /// Reference to the bit offsets for each sequence.
    bit_offsets: &'a crate::fixed::FixedVec<u64, u64, dsi_bitstream::prelude::LE, B>,
    /// The codec used for compression.
    encoding: Codes,
    /// The current sequence index.
    current: usize,
    /// The total number of sequences.
    num_sequences: usize,
    /// Markers for type parameters.
    _marker: PhantomData<(T, E)>,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVecIter<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E>,
{
    /// Creates a new iterator over all sequences.
    #[inline]
    pub(crate) fn new(
        data: &'a [u64],
        bit_offsets: &'a crate::fixed::FixedVec<u64, u64, dsi_bitstream::prelude::LE, B>,
        encoding: Codes,
        num_sequences: usize,
    ) -> Self {
        Self {
            data,
            bit_offsets,
            encoding,
            current: 0,
            num_sequences,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> Iterator for SeqVecIter<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = SeqIter<'a, T, E, B>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.num_sequences {
            return None;
        }

        // SAFETY: We have verified that current < num_sequences, and
        // bit_offsets has num_sequences + 1 elements.
        let start_bit = unsafe { self.bit_offsets.get_unchecked(self.current) };
        let end_bit = unsafe { self.bit_offsets.get_unchecked(self.current + 1) };

        self.current += 1;

        Some(SeqIter::new(self.data, start_bit, end_bit, self.encoding))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.num_sequences - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> ExactSizeIterator for SeqVecIter<'a, T, E, B> where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>
{
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> std::iter::FusedIterator
    for SeqVecIter<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
}
