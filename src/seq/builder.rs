//! The builder of [`SeqVec`].

use super::SeqVec;
use crate::common::{Automatic, Codec, CodecError, Encoder, ValidCodec, select};
use crate::{FixedVec, Storable};
use std::{fmt, marker::PhantomData};

/// Configures the compression and optional sequence lengths of a [`SeqVec`].
///
/// Start with [`SeqVec::builder`]. By default, `C =` [`Automatic`] and construction returns a
/// vector directly: automatic selection compares estimated sizes of eligible [`Codec`]
/// candidates that can encode every value. Calling [`codec`](Self::codec) sets `C = Codec`,
/// and construction returns `Result<SeqVec<T>, CodecError<T>>` because an explicit code can
/// have an invalid parameter or be unable to encode a value.
///
/// By default, sequence iterators stop at each sequence's end offset. Use
/// [`store_lengths`](Self::store_lengths) for constant-time counts of sequence elements.
///
/// # Examples
///
/// ```
/// use compressed_intvec::{Codec, SeqVec};
///
/// let lists = vec![vec![1u32, 2, 3], vec![4, 5]];
/// let vec = SeqVec::builder().store_lengths().build(&lists);
/// assert_eq!(vec.get(0).unwrap().size_hint(), (3, Some(3)));
///
/// let vec = SeqVec::builder()
///     .codec(Codec::Rice { log2_b: 2 })
///     .build_from_iter(lists.iter())?;
/// assert_eq!(vec, lists[..]);
/// # Ok::<(), compressed_intvec::CodecError<u32>>(())
/// ```
#[derive(Clone, Copy)]
pub struct SeqVecBuilder<T, C = Automatic> {
    codec: C,
    store_lengths: bool,
    element: PhantomData<fn() -> T>,
}

/// Prints the codec and whether the lengths are stored, whatever the element type.
impl<T, C: fmt::Debug> fmt::Debug for SeqVecBuilder<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SeqVecBuilder")
            .field("codec", &self.codec)
            .field("store_lengths", &self.store_lengths)
            .finish()
    }
}

impl<T> SeqVecBuilder<T> {
    pub(crate) fn new() -> Self {
        Self {
            codec: Automatic,
            store_lengths: false,
            element: PhantomData,
        }
    }
}

impl<T, C> SeqVecBuilder<T, C> {
    /// Stores the number of elements of every sequence.
    ///
    /// Each [`SeqIter`](crate::SeqIter) then stops after the stored number of elements,
    /// reports an exact [`size_hint`](Iterator::size_hint), and implements
    /// [`count`](Iterator::count) as a constant-time read of the remaining stored length.
    ///
    /// Lengths occupy an additional [`FixedVec`] whose bit width is determined by the
    /// longest sequence.
    pub fn store_lengths(self) -> Self {
        Self {
            store_lengths: true,
            ..self
        }
    }

    /// Sets the code used for every element.
    ///
    /// The resulting builder's construction methods return a [`Result`] and report invalid
    /// parameters or values that the code cannot encode.
    pub fn codec(self, codec: Codec) -> SeqVecBuilder<T, Codec> {
        SeqVecBuilder {
            codec,
            store_lengths: self.store_lengths,
            element: PhantomData,
        }
    }
}

impl<T: Storable> SeqVecBuilder<T> {
    /// Compresses `sequences` with an automatically selected code.
    ///
    /// Returns the vector directly because selection considers only codes that can encode
    /// every value. All values are examined before encoding. Each source is borrowed once
    /// through [`AsRef`], and the same slice is used for selection and encoding.
    pub fn build<S: AsRef<[T]>>(self, sequences: &[S]) -> SeqVec<T> {
        let sequences: Vec<&[T]> = sequences.iter().map(AsRef::as_ref).collect();
        let words = sequences
            .iter()
            .flat_map(|sequence| sequence.iter().map(|value| value.to_word()));
        encode(&sequences, select(words), self.store_lengths).unwrap_or_else(|_| {
            unreachable!("the automatic choice picks a valid code that encodes every value")
        })
    }

    /// Collects and compresses sequences with an automatically selected code.
    ///
    /// Buffers the sequence objects so all values can be examined before encoding. For
    /// construction that consumes sequences as they arrive, select an explicit code with
    /// [`Self::codec`].
    ///
    /// Returns the vector directly, as [`build`](SeqVecBuilder::build) does.
    pub fn build_from_iter<S: AsRef<[T]>>(
        self,
        sequences: impl IntoIterator<Item = S>,
    ) -> SeqVec<T> {
        self.build(&sequences.into_iter().collect::<Vec<S>>())
    }
}

impl<T: Storable> SeqVecBuilder<T, Codec> {
    /// Compresses `sequences` with the chosen code.
    ///
    /// # Errors
    ///
    /// - [`CodecError::InvalidParameter`] if the parameter of the code is out of range;
    /// - [`CodecError::Unencodable`] if the code cannot encode one of the values.
    pub fn build<S: AsRef<[T]>>(self, sequences: &[S]) -> Result<SeqVec<T>, CodecError<T>> {
        self.build_from_iter(sequences)
    }

    /// Compresses sequences with the chosen code as the input iterator yields them.
    ///
    /// Each sequence is encoded before the next is requested, consuming the input in
    /// a single pass.
    ///
    /// # Errors
    ///
    /// As [`build`](SeqVecBuilder::build).
    pub fn build_from_iter<S: AsRef<[T]>>(
        self,
        sequences: impl IntoIterator<Item = S>,
    ) -> Result<SeqVec<T>, CodecError<T>> {
        let codec = self
            .codec
            .validate()
            .ok_or(CodecError::InvalidParameter(self.codec))?;
        encode(sequences, codec, self.store_lengths)
    }
}

/// Encodes `sequences` with `codec`.
fn encode<T: Storable, S: AsRef<[T]>>(
    sequences: impl IntoIterator<Item = S>,
    codec: ValidCodec,
    store_lengths: bool,
) -> Result<SeqVec<T>, CodecError<T>> {
    let mut encoder = Encoder::new(codec);
    let mut offsets = vec![0];
    let mut lengths = Vec::new();
    for sequence in sequences {
        let sequence = sequence.as_ref();
        for &value in sequence {
            encoder
                .write(value.to_word())
                .map_err(|_| CodecError::Unencodable {
                    codec: codec.get(),
                    value,
                })?;
        }
        offsets.push(encoder.bit_len());
        if store_lengths {
            // Lossless: usize is at most 64 bits wide on every supported target.
            lengths.push(sequence.len() as u64);
        }
    }
    // The stream, the offsets and the lengths are consistent by construction, which is what
    // `from_parts` would check.
    Ok(SeqVec {
        data: encoder.finish(),
        offsets: FixedVec::from_slice(&offsets),
        lengths: store_lengths.then(|| FixedVec::from_slice(&lengths)),
        codec,
        element: PhantomData,
    })
}
