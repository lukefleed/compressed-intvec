//! Vectors of sequences of integers, stored with a variable-length code.

#![forbid(unsafe_code)]

mod builder;
mod iter;
#[cfg(feature = "parallel")]
mod parallel;
#[cfg(feature = "serde")]
mod serde;

pub use builder::SeqVecBuilder;
pub use iter::{SeqIter, SeqVecIter};

use crate::common::{CheckedDecoder, Codec, CodewordError, ValidCodec};
use crate::{Backend, FixedVec, FixedVecSlice, Storable};
use std::{
    error::Error,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

/// A compact, immutable vector of integer sequences.
///
/// Use it for lists that you build once and read repeatedly, such as graph adjacency lists.
/// Sequences can have different lengths or be empty. [`len`](Self::len) counts sequences,
/// including empty ones.
///
/// Each integer is encoded with the selected [`Codec`], using variable-length codes
/// implemented by [dsi-bitstream](https://docs.rs/dsi-bitstream/). The encoded values occupy
/// one bitstream. A bit-packed [`FixedVec`] indexes sequence boundaries with one start
/// offset per sequence followed by the final end offset. Compression depends on the values
/// and the chosen code.
///
/// [`from_slices`](Self::from_slices) and [`collect`](Iterator::collect) choose a code
/// automatically. The [`builder`](Self::builder) supports an explicit code and optional
/// stored sequence lengths.
///
/// [`get`](Self::get) returns a [`SeqIter`] in constant time. The iterator borrows the
/// encoded data and yields decoded values of `T` as it advances. [`iter`](Self::iter)
/// returns a [`SeqVecIter`] that yields these sequence iterators in order. Equality compares
/// the sequences' decoded values.
///
/// [`Stored lengths`](SeqVecBuilder::store_lengths) make counting a sequence constant-time
/// at the cost of an additional bit-packed buffer.
///
/// `T` implements [`Storable`]. The [`Backend`] parameter `B` holds the bitstream, offsets and
/// optional lengths, with owned `Vec<u64>` storage by default. [`as_borrowed`](Self::as_borrowed)
/// creates a view of an existing vector, and [`from_parts`](Self::from_parts) validates
/// externally supplied buffers.
///
/// With the `serde` feature, the serialized format records its version, the element signedness,
/// the code, the stream from the first to the last offset, the offsets and whether lengths are
/// stored. Deserialization checks the version and the element signedness, then validates every
/// sequence against `T::BITS` as [`from_parts`](Self::from_parts) does, and counts the lengths
/// in the same pass. Every 1.x release loads the vectors that it and earlier 1.x releases
/// serialized.
///
/// # Examples
///
/// ```
/// use compressed_intvec::SeqVec;
///
/// let lists: [&[u32]; 3] = [&[1, 2, 3], &[], &[10, 20]];
/// let vec = SeqVec::from_slices(&lists);
/// assert_eq!(vec.len(), 3);
/// assert_eq!(vec.get(2).unwrap().collect::<Vec<_>>(), [10, 20]);
/// assert_eq!(vec.get(1).unwrap().count(), 0);
/// assert!(vec.get(3).is_none());
/// assert_eq!(vec, lists[..]);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize, mem_dbg::MemDbg))]
pub struct SeqVec<T, B = Vec<u64>> {
    // Invariants, established by `from_parts` (and by the builder, which encodes the stream
    // itself) and never changed afterwards:
    // - `offsets` holds `n + 1` positions, for `n` sequences, and `lengths`, when stored, holds
    //   `n` lengths;
    // - for every `i < n`, decoding from bit `offsets[i]` of `data` (`lengths[i]` codewords when
    //   the lengths are stored, otherwise codewords until the position reaches `offsets[i + 1]`)
    //   succeeds, ends exactly at `offsets[i + 1]`, and yields words below `2^T::BITS` (for the
    //   last property, the builder relies on the contract of `Storable::to_word`); every
    //   codeword encodes a word, so dsi-bitstream's decoders read them without overflow.
    // No unsafe code relies on them: a broken invariant makes reads panic or return wrong
    // values.
    data: B,
    offsets: FixedVec<u64, B>,
    lengths: Option<FixedVec<u64, B>>,
    codec: ValidCodec,
    element: PhantomData<fn() -> T>,
}

/// The encoded buffers and metadata used to import or export a [`SeqVec`].
///
/// [`SeqVec::into_parts`] transfers these parts to the caller. Public fields allow callers
/// to assemble parts from external storage. [`SeqVec::from_parts`] validates their contents
/// and relationships before constructing a vector.
///
/// Sequence `i` contains the codewords of `codec` in the half-open bit range
/// `offsets[i]..offsets[i + 1]` of `data`. Equal consecutive offsets describe an empty sequence.
/// The offsets contain one entry per sequence plus a final end position. If `lengths` is
/// present, each entry must match the number of codewords in that sequence.
///
/// Bit `k` of the stream is bit `k % 8` of byte `k / 8` of `data` in
/// memory, which on little-endian targets is bit `k % 64` of word `k / 64`.
pub struct SeqVecParts<B> {
    /// The words containing the encoded values.
    pub data: B,
    /// The bit position where each sequence starts, followed by the position where the last
    /// one ends: one more entry than there are sequences.
    pub offsets: FixedVec<u64, B>,
    /// The number of elements of each sequence, if stored.
    pub lengths: Option<FixedVec<u64, B>>,
    /// The code used for every element.
    pub codec: Codec,
}

impl<B: Clone> Clone for SeqVecParts<B> {
    fn clone(&self) -> Self {
        let Self {
            data,
            offsets,
            lengths,
            codec,
        } = self;
        Self {
            data: data.clone(),
            offsets: offsets.clone(),
            lengths: lengths.clone(),
            codec: *codec,
        }
    }
}

impl<B: Backend> fmt::Debug for SeqVecParts<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            data,
            offsets,
            lengths,
            codec,
        } = self;
        f.debug_struct("SeqVecParts")
            .field("data", &data.as_ref())
            .field("offsets", offsets)
            .field("lengths", lengths)
            .field("codec", codec)
            .finish()
    }
}

/// Invalid encoded data or inconsistent metadata supplied to [`SeqVec::from_parts`].
///
/// Returned by [`SeqVec::from_parts`]. Deserialization, with the `serde` feature, reports it in
/// the message of its error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SeqVecPartsError {
    /// The parameter of the code is outside the range that its [`Codec`] variant documents.
    InvalidParameter(Codec),
    /// The offsets are empty: they hold one more entry than there are sequences.
    NoOffsets,
    /// The number of lengths is not the number of sequences.
    LengthCount {
        /// The number of sequences.
        expected: usize,
        /// The number of lengths in the parts.
        found: usize,
    },
    /// The stream ends before a sequence does.
    Truncated {
        /// The index of the sequence.
        sequence: usize,
    },
    /// A codeword is malformed or its decoded value does not fit in a `u64`.
    /// See [`Codec`] for codes that cannot encode `u64::MAX`.
    InvalidCodeword {
        /// The index of the sequence.
        sequence: usize,
    },
    /// The codewords of a sequence do not end where the next sequence starts.
    Misaligned {
        /// The index of the sequence.
        sequence: usize,
        /// The offset of the next sequence.
        expected: u64,
        /// The bit position where the codewords of the sequence end.
        found: u64,
    },
    /// A decoded word does not fit the element type: it is not below `2^T::BITS`.
    WordOutOfRange {
        /// The index of the sequence.
        sequence: usize,
        /// The decoded word.
        word: u64,
    },
}

impl fmt::Display for SeqVecPartsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidParameter(codec) => {
                write!(f, "the parameter of {codec:?} is out of range")
            }
            Self::NoOffsets => write!(f, "the offsets are empty"),
            Self::LengthCount { expected, found } => {
                write!(f, "{found} lengths for {expected} sequences")
            }
            Self::Truncated { sequence } => {
                write!(f, "the stream ends before the end of sequence {sequence}")
            }
            Self::InvalidCodeword { sequence } => {
                write!(
                    f,
                    "a codeword of sequence {sequence} encodes no 64-bit word"
                )
            }
            Self::Misaligned {
                sequence,
                expected,
                found,
            } => write!(
                f,
                "sequence {sequence} ends at bit {found} instead of the next offset {expected}"
            ),
            Self::WordOutOfRange { sequence, word } => {
                write!(
                    f,
                    "word {word} of sequence {sequence} does not fit the element type"
                )
            }
        }
    }
}

impl Error for SeqVecPartsError {}

/// The borrowed representation used by the read paths of every storage backend.
#[derive(Clone, Copy)]
pub(crate) struct View<'a> {
    data: &'a [u64],
    offsets: FixedVecSlice<'a, u64>,
    lengths: Option<FixedVecSlice<'a, u64>>,
    codec: ValidCodec,
}

impl<'a> View<'a> {
    /// Returns the number of sequences.
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Returns the start and the end of sequence `index`, which is below the length.
    fn bounds(&self, index: usize) -> (u64, u64) {
        let offset = |i| {
            self.offsets
                .get(i)
                .expect("every sequence below the length has two offsets")
        };
        (offset(index), offset(index + 1))
    }

    /// Returns the stored length of sequence `index`, which is below the length.
    fn stored_len(&self, index: usize) -> Option<u64> {
        self.lengths.map(|lengths| {
            lengths
                .get(index)
                .expect("the stored lengths cover every sequence")
        })
    }
}

impl<T: Storable> SeqVec<T> {
    /// Compresses `sequences`, choosing a [`Codec`] automatically from all values.
    ///
    /// Equivalent to [`Self::builder`] followed by [`SeqVecBuilder::build`]. The sequence
    /// order, element order and empty sequences are preserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::SeqVec;
    ///
    /// let vec = SeqVec::from_slices(&[vec![-3i64, 0, 3], vec![7]]);
    /// assert_eq!(vec.get(0).unwrap().collect::<Vec<_>>(), [-3, 0, 3]);
    /// ```
    pub fn from_slices<S: AsRef<[T]>>(sequences: &[S]) -> Self {
        Self::builder().build(sequences)
    }

    /// Returns a builder with automatic code selection.
    ///
    /// By default, sequence iterators stop at each sequence's end offset. Choose a code with
    /// [`SeqVecBuilder::codec`] or add lengths with [`SeqVecBuilder::store_lengths`].
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::{Codec, SeqVec};
    ///
    /// let vec = SeqVec::builder()
    ///     .codec(Codec::Delta)
    ///     .store_lengths()
    ///     .build(&[[1u32, 2], [3, 4]])?;
    /// assert_eq!(vec.get(1).unwrap().count(), 2);
    /// # Ok::<(), compressed_intvec::CodecError<u32>>(())
    /// ```
    pub fn builder() -> SeqVecBuilder<T> {
        SeqVecBuilder::new()
    }
}

impl<T, B> SeqVec<T, B> {
    /// Returns the number of sequences, including empty sequences.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Returns `true` if [`Self::len`] is zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the code used for every element.
    #[inline]
    pub fn codec(&self) -> Codec {
        self.codec.get()
    }

    /// Consumes the vector and transfers its encoded buffers and metadata to the caller.
    ///
    /// Pass the returned parts to [`Self::from_parts`] to rebuild the vector with validation.
    pub fn into_parts(self) -> SeqVecParts<B> {
        let Self {
            data,
            offsets,
            lengths,
            codec,
            element: _,
        } = self;
        SeqVecParts {
            data,
            offsets,
            lengths,
            codec: codec.get(),
        }
    }
}

impl<T, B: Backend> SeqVec<T, B> {
    /// Returns a view that borrows the bitstream, offsets and optional lengths.
    ///
    /// The returned vector uses slices of the existing packed buffers and shares their
    /// validated representation. Construction is allocation-free and takes constant time.
    /// The view's lifetime is tied to the borrow of `self`.
    ///
    /// ```
    /// use compressed_intvec::SeqVec;
    ///
    /// let sequences: [&[u64]; 2] = [&[1, 2], &[3]];
    /// let owned = SeqVec::from_slices(&sequences);
    /// let view = owned.as_borrowed();
    /// assert_eq!(view.get(0).unwrap().collect::<Vec<_>>(), [1, 2]);
    /// let boxed: SeqVec<u64, Box<[u64]>> = owned.into();
    /// assert_eq!(boxed.get(1).unwrap().collect::<Vec<_>>(), [3]);
    /// ```
    #[inline]
    pub fn as_borrowed(&self) -> SeqVec<T, &[u64]> {
        SeqVec {
            data: self.data.as_ref(),
            offsets: self.offsets.as_borrowed(),
            lengths: self.lengths.as_ref().map(FixedVec::as_borrowed),
            codec: self.codec,
            element: PhantomData,
        }
    }
}

/// Converts the owned stream, offsets and optional lengths directly to boxed storage.
///
/// The conversion preserves the packed words and metadata of the validated representation.
/// Boxing may reallocate.
impl<T> From<SeqVec<T>> for SeqVec<T, Box<[u64]>> {
    fn from(vec: SeqVec<T>) -> Self {
        let SeqVec {
            data,
            offsets,
            lengths,
            codec,
            element,
        } = vec;
        Self {
            data: data.into_boxed_slice(),
            offsets: offsets.into(),
            lengths: lengths.map(Into::into),
            codec,
            element,
        }
    }
}

impl<T: Storable> SeqVec<T> {
    /// Validates `data` and `offsets` as [`from_parts`](SeqVec::from_parts) does without
    /// lengths, and, if `store_lengths`, stores the number of codewords of every sequence,
    /// counted in the same decode.
    #[cfg(feature = "serde")]
    fn from_stream(
        data: Vec<u64>,
        offsets: FixedVec<u64>,
        store_lengths: bool,
        codec: Codec,
    ) -> Result<Self, SeqVecPartsError> {
        let codec = codec
            .validate()
            .ok_or(SeqVecPartsError::InvalidParameter(codec))?;
        if offsets.is_empty() {
            return Err(SeqVecPartsError::NoOffsets);
        }
        let lengths = if offsets.bit_width() == 0 {
            // Every sequence is empty, as in `from_parts`, and their number is not bounded by the
            // size of the input, so the lengths are the zero-width offsets without the last one.
            let mut lengths = offsets.clone();
            lengths.pop();
            lengths
        } else {
            // One length per sequence, and the offsets hold at least one bit per sequence.
            let mut lengths = Vec::new();
            check_stream(
                View {
                    data: &data,
                    offsets: offsets.as_slice(),
                    lengths: None,
                    codec,
                },
                T::BITS,
                |count| {
                    if store_lengths {
                        lengths.push(count);
                    }
                },
            )?;
            FixedVec::from_slice(&lengths)
        };
        Ok(Self {
            data,
            offsets,
            lengths: store_lengths.then_some(lengths),
            codec,
            element: PhantomData,
        })
    }
}

impl<T: Storable, B: Backend> SeqVec<T, B> {
    /// Validates encoded buffers and metadata, then constructs a vector from those buffers.
    ///
    /// Checks the code's parameters, the number of offsets and optional lengths, and every
    /// declared sequence. Each sequence must decode into words that fit `T::BITS` and end at
    /// its next offset. Stored lengths, if present, must match the decoded counts.
    /// Validation takes time proportional to the declared sequences and their encoded bits.
    /// Unused bits before the first offset or after the final offset are retained;
    /// serialization, with the `serde` feature, leaves them out.
    ///
    /// # Errors
    ///
    /// A [`SeqVecPartsError`] if the parts do not form a vector of `T`, as described in
    /// [`SeqVecParts`].
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::{SeqVec, SeqVecPartsError};
    ///
    /// let vec = SeqVec::from_slices(&[[1u32, 2], [3, 4]]);
    /// let mut parts = vec.clone().into_parts();
    /// assert_eq!(SeqVec::<u32>::from_parts(parts.clone())?, vec);
    ///
    /// parts.lengths = None;
    /// parts.offsets = compressed_intvec::FixedVec::from_slice(&[0, 1, 2]);
    /// assert!(matches!(
    ///     SeqVec::<u32>::from_parts(parts),
    ///     Err(SeqVecPartsError::Misaligned { sequence: 0, .. })
    /// ));
    /// # Ok::<(), SeqVecPartsError>(())
    /// ```
    pub fn from_parts(parts: SeqVecParts<B>) -> Result<Self, SeqVecPartsError> {
        let SeqVecParts {
            data,
            offsets,
            lengths,
            codec,
        } = parts;
        let codec = codec
            .validate()
            .ok_or(SeqVecPartsError::InvalidParameter(codec))?;
        let Some(len) = offsets.len().checked_sub(1) else {
            return Err(SeqVecPartsError::NoOffsets);
        };
        if let Some(lengths) = &lengths
            && lengths.len() != len
        {
            return Err(SeqVecPartsError::LengthCount {
                expected: len,
                found: lengths.len(),
            });
        }
        // A bit width of 0 means that every offset, and every stored length, is 0, so every
        // sequence is empty and there is nothing to decode. The check is skipped there because
        // the number of such sequences is not bounded by the size of the parts: a few serialized
        // bytes can claim 2^60 of them.
        let all_empty = offsets.bit_width() == 0
            && lengths
                .as_ref()
                .is_none_or(|lengths| lengths.bit_width() == 0);
        if !all_empty {
            check_stream(
                View {
                    data: data.as_ref(),
                    offsets: offsets.as_slice(),
                    lengths: lengths.as_ref().map(FixedVec::as_slice),
                    codec,
                },
                T::BITS,
                |_| {},
            )?;
        }
        Ok(Self {
            data,
            offsets,
            lengths,
            codec,
            element: PhantomData,
        })
    }

    #[inline]
    fn view(&self) -> View<'_> {
        View {
            data: self.data.as_ref(),
            offsets: self.offsets.as_slice(),
            lengths: self.lengths.as_ref().map(FixedVec::as_slice),
            codec: self.codec,
        }
    }

    /// Returns an iterator over the elements of sequence `index`, or `None` if `index >= len`.
    ///
    /// Creating the iterator takes constant time. The iterator borrows the sequence's
    /// encoded data and yields decoded values of `T` as it advances. An empty sequence
    /// returns `Some` with an empty iterator.
    #[inline]
    pub fn get(&self, index: usize) -> Option<SeqIter<'_, T>> {
        (index < self.len()).then(|| SeqIter::new(self.view(), index))
    }

    /// Returns an iterator over the sequences in order, each as a lazy [`SeqIter`].
    ///
    /// The outer iterator is double-ended and knows its exact remaining number of sequences.
    /// Each step creates an inner iterator positioned at a sequence's start offset.
    #[inline]
    pub fn iter(&self) -> SeqVecIter<'_, T> {
        SeqVecIter::new(self.view())
    }
}

/// Checks that each sequence of `view` decodes from its offset to the next one, in the stored
/// number of codewords if any, into words of at most `element_bits` bits. Passes the number of
/// codewords of every sequence, in order, to `count`.
fn check_stream(
    view: View<'_>,
    element_bits: u32,
    mut count: impl FnMut(u64),
) -> Result<(), SeqVecPartsError> {
    let check_word = |sequence, word: u64| {
        if element_bits < u64::BITS && word >> element_bits != 0 {
            Err(SeqVecPartsError::WordOutOfRange { sequence, word })
        } else {
            Ok(())
        }
    };
    let undecodable = |sequence, error| match error {
        CodewordError::Truncated => SeqVecPartsError::Truncated { sequence },
        CodewordError::Invalid => SeqVecPartsError::InvalidCodeword { sequence },
    };
    let mut decoder = CheckedDecoder::new(view.data, view.codec);
    for sequence in 0..view.len() {
        let (start, end) = view.bounds(sequence);
        // The reader fails only at the end of the words, which `Truncated` reports.
        decoder
            .try_seek(start)
            .map_err(|_| SeqVecPartsError::Truncated { sequence })?;
        // Decodes the stored number of codewords, or codewords until the position reaches the
        // end. Every codeword takes at least one bit, and decoding stops past the end, so the
        // loop stops within the stream whatever the stored count.
        let mut remaining = view.stored_len(sequence);
        let mut decoded = 0;
        loop {
            match &mut remaining {
                Some(0) => break,
                Some(remaining) => *remaining -= 1,
                None if decoder.bit_pos() >= end => break,
                None => {}
            }
            let word = decoder
                .try_read()
                .map_err(|error| undecodable(sequence, error))?;
            check_word(sequence, word)?;
            decoded += 1;
            if decoder.bit_pos() > end {
                break;
            }
        }
        let found = decoder.bit_pos();
        if found != end {
            return Err(SeqVecPartsError::Misaligned {
                sequence,
                expected: end,
                found,
            });
        }
        count(decoded);
    }
    Ok(())
}

impl<T: Storable + fmt::Debug, B: Backend> fmt::Debug for SeqVec<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.iter().map(Iterator::collect::<Vec<T>>))
            .finish()
    }
}

/// Compares sequence lengths and decoded elements.
impl<T: Storable + PartialEq, B: Backend, B2: Backend> PartialEq<SeqVec<T, B2>> for SeqVec<T, B> {
    fn eq(&self, other: &SeqVec<T, B2>) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a.eq(b))
    }
}

impl<T: Storable + Eq, B: Backend> Eq for SeqVec<T, B> {}

impl<T: Storable + PartialEq, S: AsRef<[T]>, B: Backend> PartialEq<[S]> for SeqVec<T, B> {
    fn eq(&self, other: &[S]) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other)
                .all(|(a, b)| a.eq(b.as_ref().iter().copied()))
    }
}

/// Hashes the sequences, consistently with [`PartialEq`].
impl<T: Storable + Hash, B: Backend> Hash for SeqVec<T, B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len());
        for sequence in self.iter() {
            let mut len = 0;
            for element in sequence {
                element.hash(state);
                len += 1;
            }
            state.write_usize(len);
        }
    }
}

/// An empty vector.
impl<T: Storable> Default for SeqVec<T> {
    fn default() -> Self {
        Self::from_slices::<&[T]>(&[])
    }
}

impl<T: Storable, S: AsRef<[T]>> From<&[S]> for SeqVec<T> {
    /// Equivalent to [`SeqVec::from_slices`].
    fn from(sequences: &[S]) -> Self {
        Self::from_slices(sequences)
    }
}

impl<T: Storable, S: AsRef<[T]>> FromIterator<S> for SeqVec<T> {
    /// Collects sequences with automatic code selection.
    ///
    /// The sequence objects are buffered so all values can be examined before encoding.
    fn from_iter<I: IntoIterator<Item = S>>(sequences: I) -> Self {
        Self::builder().build_from_iter(sequences)
    }
}

impl<'a, T: Storable, B: Backend> IntoIterator for &'a SeqVec<T, B> {
    type Item = SeqIter<'a, T>;
    type IntoIter = SeqVecIter<'a, T>;

    fn into_iter(self) -> SeqVecIter<'a, T> {
        self.iter()
    }
}
