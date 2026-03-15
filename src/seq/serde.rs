//! [`serde`] support for [`SeqVec`].
//!
//! This module provides `Serialize` and `Deserialize` implementations for
//! [`SeqVec`], allowing it to be easily serialized and deserialized with formats
//! like JSON, Bincode, etc. This is enabled by the `serde` feature flag.
//!
//! # Implementation
//!
//! A manual implementation is necessary because `SeqVec` uses generic type
//! parameters that don't directly map to serde's derive capabilities. The
//! `Codes` type from `dsi-bitstream` natively supports serde in version 0.9+.
//!
//! # Examples
//!
//! Serializing and deserializing a `SeqVec` using `serde_json`:
//!
//! ```
//! # #[cfg(feature = "serde")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::seq::{SeqVec, LESeqVec};
//!
//! let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
//! let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
//!
//! // Serialize the vector to a JSON string
//! let serialized = serde_json::to_string(&vec)?;
//!
//! // Deserialize the JSON string back into a SeqVec of the same type.
//! let deserialized: LESeqVec<u32> = serde_json::from_str(&serialized)?;
//!
//! // Verify equality
//! assert_eq!(vec, deserialized);
//! #     Ok(())
//! # }
//! # #[cfg(not(feature = "serde"))]
//! # fn main() {}
//! ```
//!
//! [`serde`]: https://serde.rs/
//! [`SeqVec`]: super::SeqVec

use super::SeqVec;
use crate::fixed::FixedVec;
use crate::variable::traits::Storable;
use dsi_bitstream::prelude::{Codes, Endianness};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

/// A proxy struct for serializing `SeqVec`.
///
/// This struct extracts only the data fields without generic type parameters,
/// allowing serde to serialize the contents without requiring the generic types
/// to implement `Serialize`.
#[derive(Serialize)]
struct SeqVecSerializeProxy<'a> {
    data: &'a [u64],
    bit_offsets_data: &'a [u64],
    bit_offsets_len: usize,
    bit_offsets_bit_width: usize,
    seq_lengths: Option<SeqLengthsSerializeProxy<'a>>,
    encoding: Codes,
}

/// A proxy struct for deserializing `SeqVec`.
///
/// This struct holds the raw data fields needed to reconstruct a `SeqVec`.
/// It decouples deserialization from the generic type parameters.
#[derive(Deserialize)]
struct SeqVecDeserializeProxy {
    data: Vec<u64>,
    bit_offsets_data: Vec<u64>,
    bit_offsets_len: usize,
    bit_offsets_bit_width: usize,
    seq_lengths: Option<SeqLengthsDeserializeProxy>,
    encoding: Codes,
}

/// A proxy struct for serializing sequence lengths.
#[derive(Serialize)]
struct SeqLengthsSerializeProxy<'a> {
    data: &'a [u64],
    len: usize,
    bit_width: usize,
}

/// A proxy struct for deserializing sequence lengths.
#[derive(Deserialize)]
struct SeqLengthsDeserializeProxy {
    data: Vec<u64>,
    len: usize,
    bit_width: usize,
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> Serialize for SeqVec<T, E, B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let proxy = SeqVecSerializeProxy {
            data: self.data.as_ref(),
            bit_offsets_data: self.bit_offsets.as_limbs(),
            bit_offsets_len: self.bit_offsets.len(),
            bit_offsets_bit_width: self.bit_offsets.bit_width(),
            seq_lengths: self
                .seq_lengths
                .as_ref()
                .map(|lengths| SeqLengthsSerializeProxy {
                    data: lengths.as_limbs(),
                    len: lengths.len(),
                    bit_width: lengths.bit_width(),
                }),
            encoding: self.encoding,
        };
        proxy.serialize(serializer)
    }
}

impl<'de, T: Storable + 'static, E: Endianness> Deserialize<'de> for SeqVec<T, E, Vec<u64>> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let proxy = SeqVecDeserializeProxy::deserialize(deserializer)?;

        // Validate bit_width
        let word_size_bits = std::mem::size_of::<u64>() * 8;
        if proxy.bit_offsets_bit_width > word_size_bits {
            return Err(de::Error::custom(format!(
                "Deserialized bit_offsets bit_width ({}) cannot be greater than word size ({})",
                proxy.bit_offsets_bit_width, word_size_bits
            )));
        }

        // Validate buffer size
        let required_bits = proxy
            .bit_offsets_len
            .saturating_mul(proxy.bit_offsets_bit_width);
        let required_data_words = required_bits.div_ceil(word_size_bits);

        if proxy.bit_offsets_data.len() < required_data_words {
            return Err(de::Error::custom(format!(
                "Deserialized bit_offsets buffer is too small. It has {} words, but at least {} are required.",
                proxy.bit_offsets_data.len(),
                required_data_words
            )));
        }

        // Reconstruct the FixedVec<u64, u64, E, Vec<u64>>
        let bit_offsets = unsafe {
            FixedVec::new_unchecked(
                proxy.bit_offsets_data,
                proxy.bit_offsets_len,
                proxy.bit_offsets_bit_width,
            )
        };

        let seq_lengths = if let Some(lengths_proxy) = proxy.seq_lengths {
            if lengths_proxy.bit_width > word_size_bits {
                return Err(de::Error::custom(format!(
                    "Deserialized seq_lengths bit_width ({}) cannot be greater than word size ({})",
                    lengths_proxy.bit_width, word_size_bits
                )));
            }

            let required_bits = lengths_proxy.len.saturating_mul(lengths_proxy.bit_width);
            let required_data_words = required_bits.div_ceil(word_size_bits);

            if lengths_proxy.data.len() < required_data_words {
                return Err(de::Error::custom(format!(
                    "Deserialized seq_lengths buffer is too small. It has {} words, but at least {} are required.",
                    lengths_proxy.data.len(),
                    required_data_words
                )));
            }

            if lengths_proxy.len + 1 != proxy.bit_offsets_len {
                return Err(de::Error::custom(
                    "Deserialized seq_lengths length must match number of sequences",
                ));
            }

            Some(unsafe {
                FixedVec::new_unchecked(
                    lengths_proxy.data,
                    lengths_proxy.len,
                    lengths_proxy.bit_width,
                )
            })
        } else {
            None
        };

        // Reconstruct the SeqVec
        Ok(unsafe {
            SeqVec::from_raw_parts_with_lengths(
                proxy.data,
                bit_offsets,
                seq_lengths,
                proxy.encoding,
            )
        })
    }
}
