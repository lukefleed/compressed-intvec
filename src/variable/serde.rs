//! Serialization of [`VarVec`] with the `serde` feature.
//!
//! The serialized form is tagged with the format version, and its body records the element
//! signedness ([`Storable::SIGNED`]), the code, the sampling rate, the length and the words of
//! the stream up to its last codeword. Deserialization requires the same signedness, decodes
//! every element with the checked decoder, which also checks that it fits the element type,
//! and rebuilds the samples in that pass.

use super::{VarVec, VarVecPartsError};
use crate::common::Decoder;
use crate::common::serde::{
    CodecWire, StreamError, StreamWords, check_stream_words, stream_words_from_wire,
};
use crate::{Backend, Storable};
use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{fmt, num::NonZeroUsize};

/// The serialized form of a [`VarVec`], with the stream words as `D`.
///
/// Format version 1 carries the element signedness, the code, the sampling rate, the length and
/// the stream words up to the one that holds the last bit of the last codeword, with the bits
/// after that codeword zero. The samples are not serialized: the reader rebuilds them. Loading
/// requires the signedness of `T`, a code with its parameter in range, a nonzero sampling rate,
/// `len` codewords that each encode a word below `2^T::BITS`, no word after the last codeword,
/// and zero bits after it.
///
/// The variant is the format version, which every serde format decodes before the body, so an
/// unknown version is refused before the body is read. Binary formats write the variant's
/// position in this declaration, so versions are only appended.
///
/// Once released, a version is never changed: a format change appends a variant, and readers
/// keep loading every earlier version.
#[derive(Serialize, Deserialize)]
#[serde(rename = "VarVec")]
enum Wire<D> {
    V1(V1<D>),
}

/// The body of format version 1 of a [`VarVec`].
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct V1<D> {
    element_signed: bool,
    codec: CodecWire,
    sampling_rate: u64,
    len: u64,
    data: D,
}

impl<T: Storable, B: Backend> VarVec<T, B> {
    /// Returns the bit position after the last codeword, decoding from the last sample.
    fn stream_end(&self) -> u64 {
        let Some(last) = self.len.checked_sub(1) else {
            return 0;
        };
        let block = last / self.sampling_rate.get();
        let mut decoder = Decoder::new(self.data.as_ref(), self.codec);
        decoder.seek(
            self.samples
                .get(block)
                .expect("every element below the length has a sample"),
        );
        for _ in block * self.sampling_rate.get()..self.len {
            decoder.read();
        }
        decoder.bit_pos()
    }
}

impl<T: Storable, B: Backend> Serialize for VarVec<T, B> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Lossless: usize is at most 64 bits wide on every supported target.
        Wire::V1(V1 {
            element_signed: T::SIGNED,
            codec: self.codec.get().into(),
            sampling_rate: self.sampling_rate.get() as u64,
            len: self.len as u64,
            data: StreamWords {
                words: self.data.as_ref(),
                start: 0,
                end: self.stream_end(),
            },
        })
        .serialize(serializer)
    }
}

/// A failure to deserialize a [`VarVec`] with the requested element type.
enum WireError {
    Signedness { found: bool },
    SamplingRate(u64),
    Length(u64),
    Parts(VarVecPartsError),
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
            Self::SamplingRate(rate) => {
                write!(
                    f,
                    "serialized sampling rate {rate} is zero or does not fit usize"
                )
            }
            Self::Length(len) => write!(f, "serialized length {len} does not fit usize"),
            Self::Parts(error) => write!(f, "invalid serialized VarVec: {error}"),
            Self::Stream(error) => write!(f, "invalid serialized VarVec: {error}"),
        }
    }
}

/// Builds the vector that `wire` describes, checking every property of the format.
fn parse<T: Storable>(wire: Wire<Vec<u64>>) -> Result<VarVec<T>, WireError> {
    let Wire::V1(V1 {
        element_signed,
        codec,
        sampling_rate,
        len,
        data,
    }) = wire;
    if element_signed != T::SIGNED {
        return Err(WireError::Signedness {
            found: element_signed,
        });
    }
    let sampling_rate = usize::try_from(sampling_rate)
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or(WireError::SamplingRate(sampling_rate))?;
    let len = usize::try_from(len).map_err(|_| WireError::Length(len))?;
    let (vec, end) = VarVec::from_stream(
        stream_words_from_wire(data),
        len,
        sampling_rate,
        codec.into(),
    )
    .map_err(WireError::Parts)?;
    check_stream_words(&vec.data, 0, end).map_err(WireError::Stream)?;
    Ok(vec)
}

impl<'de, T: Storable> Deserialize<'de> for VarVec<T> {
    /// Deserializes a vector, checking the format version before the body, then the signedness
    /// of the elements, every codeword against the length and the element type, and that the
    /// stream ends with the last codeword.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        parse(Wire::<Vec<u64>>::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}
