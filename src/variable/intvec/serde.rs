//! Manual [`serde`] implementation for [`IntVec`].
//!
//! This module provides the serialization and deserialization logic for [`IntVec`].
//! A manual implementation is necessary because [`IntVec`] contains fields from
//! external crates (like `dsi_bitstream::codes::Codes`) that do not implement
//! [`serde`] traits.
//!
//! The strategy is to use "shadow" or "proxy" structs/enums that are
//! easily derivable with [`serde`]. The `serialize` and `deserialize` functions
//! then perform conversions between the main types and these serializable proxies.
//! This encapsulates the serialization logic cleanly and decouples it from the
//! public API.
//!
//! [`serde`]: https://crates.io/crates/serde
//! [`IntVec`]: crate::intvec::IntVec

#![cfg_attr(docsrs, doc(cfg(feature = "serde")))]

use super::{Encoding, Endianness, IntVec, PhantomData, Samples};
use dsi_bitstream::prelude::Codes;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A serializable proxy for `dsi_bitstream::prelude::Codes`.
///
/// This enum replicates the structure of `Codes` for the variants supported
/// by this library, allowing it to derive [`serde`] traits.
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
    /// Converts a `Codes` instance into its serializable proxy.
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
            // Explicitly panic for unsupported codes to make it clear during
            // development if a new code needs to be added to the proxy.
            _ => unimplemented!("Serialization for this code is not implemented"),
        }
    }
}

impl From<CodesSerde> for Codes {
    /// Converts a serializable proxy back into a `Codes` instance.
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

/// A serializable proxy for the `Encoding` enum.
#[derive(Serialize, Deserialize)]
enum EncodingSerde {
    Dsi(CodesSerde),
    Fixed { num_bits: usize },
}

impl From<&Encoding> for EncodingSerde {
    /// Converts an `Encoding` reference into its serializable proxy.
    fn from(encoding: &Encoding) -> Self {
        match encoding {
            Encoding::Dsi(code) => EncodingSerde::Dsi((*code).into()),
            Encoding::Fixed { num_bits } => EncodingSerde::Fixed {
                num_bits: *num_bits,
            },
        }
    }
}

impl From<EncodingSerde> for Encoding {
    /// Converts a serializable proxy back into an `Encoding` instance.
    fn from(proxy: EncodingSerde) -> Self {
        match proxy {
            EncodingSerde::Dsi(code_proxy) => Encoding::Dsi(code_proxy.into()),
            EncodingSerde::Fixed { num_bits } => Encoding::Fixed { num_bits },
        }
    }
}

/// A private helper struct for serializing and deserializing an [`IntVec`].
///
/// This struct mirrors the layout of [`IntVec`] but uses the serializable
/// proxy enums. It is the core of the manual [`serde`] implementation.
#[derive(Serialize, Deserialize)]
struct IntVecSerde {
    data: Vec<u64>,
    samples: Option<Samples>,
    k: Option<usize>,
    len: usize,
    encoding: EncodingSerde,
}

impl<E: Endianness> Serialize for IntVec<E> {
    /// Serializes the [`IntVec`] using the proxy-based approach.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert the non-serializable fields to their proxy counterparts.
        let helper = IntVecSerde {
            data: self.data.clone(),
            samples: self.samples.clone(),
            k: self.k,
            len: self.len,
            encoding: (&self.encoding).into(),
        };
        // Delegate the actual serialization to the fully serializable helper struct.
        helper.serialize(serializer)
    }
}

impl<'de, E: Endianness> Deserialize<'de> for IntVec<E> {
    /// Deserializes an [`IntVec`] using the proxy-based approach.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // First, deserialize the data into the helper struct.
        let helper = IntVecSerde::deserialize(deserializer)?;
        // Then, construct the IntVec, converting proxy types back to the main types.
        Ok(IntVec {
            data: helper.data,
            samples: helper.samples,
            k: helper.k,
            len: helper.len,
            encoding: helper.encoding.into(),
            endian: PhantomData, // The endianness is a zero-sized marker.
        })
    }
}
