//! [`serde`] support for [`SeqVec`].
//!
//! This module provides `Serialize` and `Deserialize` implementations for
//! [`SeqVec`], allowing it to be easily serialized and deserialized with formats
//! like JSON, Bincode, etc. This is enabled by the `serde` feature flag.
//!
//! # Implementation
//!
//! A manual implementation is necessary because the underlying `dsi-bitstream::codes::Codes`
//! enum does not implement the `serde` traits. This module uses a serializable
//! "proxy" enum to handle this conversion.
//!
//! # Examples
//!
//! Serializing and deserializing a `SeqVec` using `serde_json`:
//!
//! ```
//! # #[cfg(feature = "serde")] {
//! use compressed_intvec::seq::{SeqVec, LESeqVec};
//!
//! let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
//! let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
//!
//! // Serialize the vector to a JSON string
//! let serialized = serde_json::to_string(&vec).unwrap();
//!
//! // Deserialize the JSON string back into a SeqVec of the same type.
//! let deserialized: LESeqVec<u32> = serde_json::from_str(&serialized).unwrap();
//!
//! // Verify equality
//! assert_eq!(vec, deserialized);
//! # }
//! ```
//!
//! [`serde`]: https://serde.rs/
//! [`SeqVec`]: super::SeqVec

use super::SeqVec;
use crate::fixed::FixedVec;
use crate::variable::traits::Storable;
use dsi_bitstream::prelude::{Codes, Endianness, LE};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A serializable proxy for `dsi-bitstream::prelude::Codes`.
/// This is an internal detail to bridge `Codes` with `serde`.
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
enum CodesSerde {
    Gamma,
    Delta,
    Zeta { k: usize },
    Rice { log2_b: usize },
    Unary,
    Golomb { b: usize },
    Omega,
    Pi { k: usize },
    ExpGolomb { k: usize },
    VByteLe,
    VByteBe,
}

impl From<Codes> for CodesSerde {
    fn from(code: Codes) -> Self {
        match code {
            Codes::Gamma => CodesSerde::Gamma,
            Codes::Delta => CodesSerde::Delta,
            Codes::Zeta { k } => CodesSerde::Zeta { k },
            Codes::Rice { log2_b } => CodesSerde::Rice { log2_b },
            Codes::Unary => CodesSerde::Unary,
            Codes::Golomb { b } => CodesSerde::Golomb { b },
            Codes::Omega => CodesSerde::Omega,
            Codes::Pi { k } => CodesSerde::Pi { k },
            Codes::ExpGolomb { k } => CodesSerde::ExpGolomb { k },
            Codes::VByteLe => CodesSerde::VByteLe,
            Codes::VByteBe => CodesSerde::VByteBe,
            _ => unimplemented!("Serialization for this code is not implemented"),
        }
    }
}

impl From<CodesSerde> for Codes {
    fn from(proxy: CodesSerde) -> Self {
        match proxy {
            CodesSerde::Gamma => Codes::Gamma,
            CodesSerde::Delta => Codes::Delta,
            CodesSerde::Zeta { k } => Codes::Zeta { k },
            CodesSerde::Rice { log2_b } => Codes::Rice { log2_b },
            CodesSerde::Unary => Codes::Unary,
            CodesSerde::Golomb { b } => Codes::Golomb { b },
            CodesSerde::Omega => Codes::Omega,
            CodesSerde::Pi { k } => Codes::Pi { k },
            CodesSerde::ExpGolomb { k } => Codes::ExpGolomb { k },
            CodesSerde::VByteLe => Codes::VByteLe,
            CodesSerde::VByteBe => Codes::VByteBe,
        }
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]> + Serialize> Serialize for SeqVec<T, E, B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct SerializeProxy<'a, B: AsRef<[u64]> + Serialize> {
            data: &'a [u64],
            bit_offsets: &'a FixedVec<u64, u64, LE, B>,
            encoding: CodesSerde,
        }

        let proxy = SerializeProxy {
            data: self.data.as_ref(),
            bit_offsets: &self.bit_offsets,
            encoding: self.encoding.into(),
        };
        proxy.serialize(serializer)
    }
}

/// A helper struct for deserializing an owned `SeqVec`.
#[derive(Deserialize)]
#[serde(rename = "SeqVec")]
struct SeqVecProxy {
    data: Vec<u64>,
    bit_offsets: FixedVec<u64, u64, LE, Vec<u64>>,
    encoding: CodesSerde,
}

impl<'de, T: Storable, E: Endianness> Deserialize<'de> for SeqVec<T, E, Vec<u64>> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = SeqVecProxy::deserialize(deserializer)?;
        // SAFETY: The deserialized proxy struct contains all necessary components,
        // which are assumed to be consistent as they were serialized together.
        Ok(unsafe { SeqVec::from_raw_parts(helper.data, helper.bit_offsets, helper.encoding.into()) })
    }
}
