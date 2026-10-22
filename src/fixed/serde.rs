//! Serialization of [`FixedVec`] with the `serde` feature.
//!
//! The serialized form is tagged with the format version, and its body records the element
//! signedness ([`Storable::SIGNED`]), the bit width, the length and the data words. A payload
//! loads as any element type with the same signedness whose `BITS` is at least the bit width. A
//! `u32` payload loads as `u64`, and a `usize` payload written on a 64-bit target loads on a
//! 32-bit one when its bit width is at most 32.
//!
//! Padding words are omitted and unused trailing bits are cleared, and the reader rejects a
//! payload whose unused bits are set, while stored element words are preserved. Custom
//! [`Storable`] aliases or element equality that identifies distinct stored words can therefore
//! produce equal vectors with different serialized words.

use super::{Backend, FixedVec, FixedVecError, checked_bit_width, layout};
use crate::Storable;
use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// The serialized form of a [`FixedVec`], with the data words as `W`.
///
/// Format version 1 carries the element signedness, the bit width, the length and the
/// `ceil(len * bit_width / 64)` data words, in which the bits past the last element are zero.
/// Loading requires the signedness of `T`, a bit width of at most `T::BITS`, that many words,
/// and zero unused bits.
///
/// The variant is the format version, which every serde format decodes before the body, so an
/// unknown version is refused before the body is read. Binary formats write the variant's
/// position in this declaration, so versions are only appended.
///
/// Once released, a version is never changed: a format change appends a variant, and readers
/// keep loading every earlier version.
#[derive(Serialize, Deserialize)]
#[serde(rename = "FixedVec")]
enum Wire<W> {
    V1(V1<W>),
}

/// The body of format version 1 of a [`FixedVec`].
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct V1<W> {
    element_signed: bool,
    bit_width: u32,
    len: u64,
    words: W,
}

/// The data words of a vector, with the bits past the last element cleared.
struct DataWords<'a> {
    words: &'a [u64],
    last_mask: u64,
}

impl Serialize for DataWords<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let last = self.words.len().saturating_sub(1);
        serializer.collect_seq(self.words.iter().enumerate().map(|(index, &word)| {
            if index == last {
                word & self.last_mask
            } else {
                word
            }
        }))
    }
}

/// Returns the mask of the bits of the last data word that hold elements, when the elements
/// take `bits` bits.
fn last_mask(bits: usize) -> u64 {
    match bits % layout::WORD_BITS {
        0 => u64::MAX,
        used => layout::mask(used),
    }
}

impl<T: Storable, B: Backend> Serialize for FixedVec<T, B> {
    /// Serializes the metadata and packed data words, omitting padding words and clearing unused
    /// trailing bits.
    ///
    /// Stored element representations are preserved. Custom [`Storable`] aliases or equality
    /// that identifies distinct stored words can produce equal vectors with different serialized
    /// words.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Does not overflow: the words hold `len` elements (invariant).
        let bits = self.len * self.bit_width;
        Wire::V1(V1 {
            element_signed: T::SIGNED,
            bit_width: self.bit_width as u32,
            // Lossless: usize is at most 64 bits wide on every supported target.
            len: self.len as u64,
            words: DataWords {
                words: &self.words.as_ref()[..bits.div_ceil(layout::WORD_BITS)],
                last_mask: last_mask(bits),
            },
        })
        .serialize(serializer)
    }
}

/// A validation failure in a serialized [`FixedVec`] of the requested element type.
enum WireError {
    Length(u64),
    Signedness { found: bool, expected: bool },
    WordCount { found: usize, expected: usize },
    UnusedBits,
    Parts(FixedVecError),
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let signedness = |signed: bool| if signed { "signed" } else { "unsigned" };
        match *self {
            Self::Length(len) => write!(f, "serialized length {len} does not fit usize"),
            Self::Signedness { found, expected } => write!(
                f,
                "serialized elements are {}, expected {}",
                signedness(found),
                signedness(expected)
            ),
            Self::WordCount { found, expected } => write!(
                f,
                "serialized vector has {found} data words, expected {expected}"
            ),
            Self::UnusedBits => write!(f, "a bit past the last element is set"),
            Self::Parts(error) => write!(f, "invalid serialized vector: {error}"),
        }
    }
}

/// Builds the vector that `wire` describes, or returns the first check that it fails.
fn parse<T: Storable>(wire: Wire<Vec<u64>>) -> Result<FixedVec<T>, WireError> {
    let Wire::V1(V1 {
        element_signed,
        bit_width,
        len,
        mut words,
    }) = wire;
    if element_signed != T::SIGNED {
        return Err(WireError::Signedness {
            found: element_signed,
            expected: T::SIGNED,
        });
    }
    let len = usize::try_from(len).map_err(|_| WireError::Length(len))?;
    let width =
        checked_bit_width::<T>(bit_width).map_err(|error| WireError::Parts(error.into()))?;
    let required = layout::required_words(len, width).ok_or(WireError::Parts(
        FixedVecError::LengthOverflow { len, bit_width },
    ))?;
    let data_words = required.saturating_sub(1);
    if words.len() != data_words {
        return Err(WireError::WordCount {
            found: words.len(),
            expected: data_words,
        });
    }
    // Does not overflow: `required_words` computed it.
    if let Some(&last) = words.last()
        && last & !last_mask(len * width) != 0
    {
        return Err(WireError::UnusedBits);
    }
    if len > 0 {
        words.push(0);
    }
    FixedVec::from_parts(words, len, bit_width).map_err(WireError::Parts)
}

impl<'de, T: Storable> Deserialize<'de> for FixedVec<T> {
    /// Deserializes a vector, checking the format version before the body, then the element
    /// signedness, that the bit width fits `T`, that the number of words matches the length and
    /// the bit width, and that the bits past the last element are zero.
    ///
    /// The stored bit width determines compatibility with `T`: for built-in integers, a payload
    /// can load into another type with the same signedness when its stored bit width fits that
    /// type. Padding is restored before [`FixedVec::from_parts`] validates the reconstructed
    /// buffer.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        parse(Wire::<Vec<u64>>::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}
