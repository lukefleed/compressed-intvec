//! Serialization of [`SeqVec`] with the `serde` feature.
//!
//! The serialized form is tagged with the format version, and its body records the element
//! signedness ([`Storable::SIGNED`]), the code, the words of the stream up to the last offset,
//! the offsets, and whether the vector stores the lengths. Deserialization requires the same
//! signedness, decodes every sequence with the checked decoder, which also checks that it fits
//! the element type, and counts the lengths in that pass.

use super::{SeqVec, SeqVecPartsError};
use crate::common::serde::{
    CodecWire, StreamError, StreamWords, check_stream_words, stream_words_from_wire,
};
use crate::{Backend, FixedVec, Storable};
use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// The serialized form of a [`SeqVec`], with the stream words as `D` and the offsets as `P`.
///
/// Format version 1 carries the element signedness, the code, the stream words up to the one
/// that holds the bit before the last offset, with the bits before the first offset and from the
/// last offset on zero, the offsets as a serialized [`FixedVec`], and whether the vector stores
/// the lengths, which the reader counts. Loading requires the signedness of `T`, a code with its
/// parameter in range, at least one offset, sequences whose codewords each encode a word below
/// `2^T::BITS` and end at the next offset, no word after the last offset, and zero bits outside
/// the offsets.
///
/// The variant is the format version, which every serde format decodes before the body, so an
/// unknown version is refused before the body is read. Binary formats write the variant's
/// position in this declaration, so versions are only appended.
///
/// Once released, a version is never changed: a format change appends a variant, and readers
/// keep loading every earlier version.
#[derive(Serialize, Deserialize)]
#[serde(rename = "SeqVec")]
enum Wire<D, P> {
    V1(V1<D, P>),
}

/// The body of format version 1 of a [`SeqVec`].
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct V1<D, P> {
    element_signed: bool,
    codec: CodecWire,
    data: D,
    offsets: P,
    store_lengths: bool,
}

impl<T: Storable, B: Backend> Serialize for SeqVec<T, B> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let offset = |i| {
            self.offsets
                .get(i)
                .expect("the offsets hold one more entry than there are sequences")
        };
        Wire::V1(V1 {
            element_signed: T::SIGNED,
            codec: self.codec.get().into(),
            data: StreamWords {
                words: self.data.as_ref(),
                start: offset(0),
                end: offset(self.len()),
            },
            offsets: &self.offsets,
            store_lengths: self.lengths.is_some(),
        })
        .serialize(serializer)
    }
}

/// A failure to deserialize a [`SeqVec`] of the requested element type.
enum WireError {
    Signedness { found: bool },
    Parts(SeqVecPartsError),
    Stream(StreamError),
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let signedness = |signed: bool| if signed { "signed" } else { "unsigned" };
        match self {
            Self::Signedness { found } => write!(
                f,
                "serialized elements are {}, expected {}",
                signedness(*found),
                signedness(!found)
            ),
            Self::Parts(error) => write!(f, "invalid serialized SeqVec: {error}"),
            Self::Stream(error) => write!(f, "invalid serialized SeqVec: {error}"),
        }
    }
}

/// Builds the vector that `wire` describes, checking every property of the format.
fn parse<T: Storable>(wire: Wire<Vec<u64>, FixedVec<u64>>) -> Result<SeqVec<T>, WireError> {
    let Wire::V1(V1 {
        element_signed,
        codec,
        data,
        offsets,
        store_lengths,
    }) = wire;
    if element_signed != T::SIGNED {
        return Err(WireError::Signedness {
            found: element_signed,
        });
    }
    let vec = SeqVec::from_stream(
        stream_words_from_wire(data),
        offsets,
        store_lengths,
        codec.into(),
    )
    .map_err(WireError::Parts)?;
    let offset = |i| vec.offsets.get(i).expect("the offsets were checked");
    check_stream_words(&vec.data, offset(0), offset(vec.len())).map_err(WireError::Stream)?;
    Ok(vec)
}

impl<'de, T: Storable> Deserialize<'de> for SeqVec<T> {
    /// Deserializes a vector, checking the format version before the body, then the signedness
    /// of the elements, every sequence against the offsets and the element type, and that the
    /// stream holds no bits outside the offsets.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        parse(Wire::<Vec<u64>, FixedVec<u64>>::deserialize(deserializer)?)
            .map_err(de::Error::custom)
    }
}
