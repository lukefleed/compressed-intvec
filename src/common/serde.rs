//! What the serialized forms of [`VarVec`](crate::VarVec) and [`SeqVec`](crate::SeqVec)
//! share: the serialized form of a [`Codec`] and the stream words.
//!
//! The form of a [`Codec`] is a type of its own, so that renaming a variant of [`Codec`] does not
//! change the format. It accepts any parameter: the vector that it is part of checks the code
//! when it is rebuilt.

use super::Codec;
use serde::{Deserialize, Serialize, Serializer};
use std::fmt;

/// The serialized form of a [`Codec`].
///
/// Binary formats write the position of the variant in this declaration as its wire number, so
/// the variants are never reordered or removed, and new ones are appended.
#[derive(Serialize, Deserialize)]
pub(crate) enum CodecWire {
    Unary,
    Gamma,
    Delta,
    Omega,
    VByteLe,
    VByteBe,
    Rice(u32),
    Golomb(u64),
    ExpGolomb(u32),
    Zeta(u32),
    Pi(u32),
}

impl From<Codec> for CodecWire {
    fn from(codec: Codec) -> Self {
        match codec {
            Codec::Unary => Self::Unary,
            Codec::Gamma => Self::Gamma,
            Codec::Delta => Self::Delta,
            Codec::Omega => Self::Omega,
            Codec::VByteLe => Self::VByteLe,
            Codec::VByteBe => Self::VByteBe,
            Codec::Rice { log2_b } => Self::Rice(log2_b),
            Codec::Golomb { b } => Self::Golomb(b),
            Codec::ExpGolomb { k } => Self::ExpGolomb(k),
            Codec::Zeta { k } => Self::Zeta(k),
            Codec::Pi { k } => Self::Pi(k),
        }
    }
}

impl From<CodecWire> for Codec {
    fn from(wire: CodecWire) -> Self {
        match wire {
            CodecWire::Unary => Self::Unary,
            CodecWire::Gamma => Self::Gamma,
            CodecWire::Delta => Self::Delta,
            CodecWire::Omega => Self::Omega,
            CodecWire::VByteLe => Self::VByteLe,
            CodecWire::VByteBe => Self::VByteBe,
            CodecWire::Rice(log2_b) => Self::Rice { log2_b },
            CodecWire::Golomb(b) => Self::Golomb { b },
            CodecWire::ExpGolomb(k) => Self::ExpGolomb { k },
            CodecWire::Zeta(k) => Self::Zeta { k },
            CodecWire::Pi(k) => Self::Pi { k },
        }
    }
}

/// The stream words of a vector as they are serialized: the words up to the one that holds bit
/// `end - 1`, with every bit outside `start..end` cleared.
///
/// In the serialized sequence, bit `k` of the stream is bit `k % 64` of word `k / 64` on every
/// host. The in-memory words hold the bytes of the stream in memory order, so on big-endian
/// hosts each word is byte-swapped on the way out. On little-endian hosts the conversion is the
/// identity.
pub(crate) struct StreamWords<'a> {
    pub(crate) words: &'a [u64],
    pub(crate) start: u64,
    pub(crate) end: u64,
}

impl Serialize for StreamWords<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Fewer words than `end` needs only for a `SeqVec` without sequences, whose single
        // offset no codeword reaches.
        let count = usize::try_from(self.end.div_ceil(64))
            .map_or(self.words.len(), |count| count.min(self.words.len()));
        serializer.collect_seq(
            self.words[..count]
                .iter()
                .enumerate()
                .map(|(index, &word)| {
                    let (before, after) = outside(index, self.start, self.end);
                    u64::from_le(word) & !(before | after)
                }),
        )
    }
}

/// A serialized stream that is not in the one form that the writer produces.
#[derive(Debug)]
pub(crate) enum StreamError {
    /// The stream has words after the one that holds its last bit.
    TrailingWords { found: usize, expected: u64 },
    /// A bit before the first codeword is set.
    LeadingBits,
    /// A bit after the last codeword is set.
    TrailingBits,
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::TrailingWords { found, expected } => write!(
                f,
                "the stream has {found} words, but its codewords end within {expected}"
            ),
            Self::LeadingBits => write!(f, "a stream bit before the first codeword is set"),
            Self::TrailingBits => write!(f, "a stream bit after the last codeword is set"),
        }
    }
}

/// Checks that the in-memory `words` of a stream are what [`StreamWords`] writes for the
/// codewords in `start..end`: no words after the one that holds bit `end - 1`, and no bit set
/// outside `start..end`.
pub(crate) fn check_stream_words(words: &[u64], start: u64, end: u64) -> Result<(), StreamError> {
    let expected = end.div_ceil(64);
    // Lossless: usize is at most 64 bits wide on every supported target.
    if words.len() as u64 > expected {
        return Err(StreamError::TrailingWords {
            found: words.len(),
            expected,
        });
    }
    for (index, &word) in words.iter().enumerate() {
        let (before, after) = outside(index, start, end);
        let word = u64::from_le(word);
        if word & before != 0 {
            return Err(StreamError::LeadingBits);
        }
        if word & after != 0 {
            return Err(StreamError::TrailingBits);
        }
    }
    Ok(())
}

/// Returns the masks of the bits of stream word `index` that lie before bit `start` and at or
/// after bit `end`.
fn outside(index: usize, start: u64, end: u64) -> (u64, u64) {
    let low_bits = |n: u64| if n >= 64 { u64::MAX } else { (1 << n) - 1 };
    // Saturates only for a word past bit 2^64, which no stream in memory reaches.
    let base = (index as u64).saturating_mul(64);
    (
        low_bits(start.saturating_sub(base)),
        !low_bits(end.saturating_sub(base)),
    )
}

/// Turns the words of the serialized form into the in-memory words of a stream.
///
/// Inverse of [`StreamWords`]: it byte-swaps each word on big-endian hosts, and is the identity
/// on little-endian hosts.
pub(crate) fn stream_words_from_wire(mut words: Vec<u64>) -> Vec<u64> {
    for word in &mut words {
        *word = word.to_le();
    }
    words
}
