// src/variable/serde.rs

//! Manual [`serde`] implementation for [`IntVec`].
//!
//! A manual implementation is necessary because `dsi-bitstream::codes::Codes`
//! does not implement [`serde`] traits. This module uses a serializable
//! "proxy" enum to handle this cleanly.

#![cfg_attr(docsrs, doc(cfg(feature = "serde")))]

use super::{traits::Storable, Endianness, IntVec};
use crate::fixed::{FixedVec, LEFixedVec};
use dsi_bitstream::prelude::{Codes, LE};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A serializable proxy for `dsi_bitstream::prelude::Codes`.
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
        // FIX: Corrected the match arms to return the `Codes` enum variants.
        // The previous version had a typo in the last arm.
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

// FIX: Added the `B: Serialize` trait bound to the main impl signature.
impl<T: Storable, E: Endianness, B: AsRef<[u64]> + Serialize> Serialize for IntVec<T, E, B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // FIX: Renamed generic parameter to UpperCamelCase and ensured its bounds are correct.
        #[derive(Serialize)]
        struct SerializeProxy<'a, BSamples: AsRef<[u64]> + Serialize> {
            data: &'a [u64],
            samples: &'a FixedVec<u64, u64, LE, BSamples>,
            k: usize,
            len: usize,
            encoding: CodesSerde,
        }

        let proxy = SerializeProxy {
            data: self.data.as_ref(),
            samples: &self.samples,
            k: self.k,
            len: self.len,
            encoding: self.encoding.into(),
        };
        proxy.serialize(serializer)
    }
}

#[derive(Deserialize)]
#[serde(rename = "IntVec")]
struct IntVecProxy {
    data: Vec<u64>,
    samples: LEFixedVec,
    k: usize,
    len: usize,
    encoding: CodesSerde,
}

impl<'de, T: Storable, E: Endianness> Deserialize<'de> for IntVec<T, E, Vec<u64>> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = IntVecProxy::deserialize(deserializer)?;
        // SAFETY: The deserialized proxy struct contains all necessary components,
        // which are assumed to be consistent as they were serialized together.
        Ok(unsafe {
            IntVec::new_unchecked(
                helper.data,
                helper.samples,
                helper.k,
                helper.len,
                helper.encoding.into(),
            )
        })
    }
}