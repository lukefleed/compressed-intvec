//! Manual [`serde`] implementation for [`IntVec`].
//!
//! A manual implementation is necessary because `dsi_bitstream::codes::Codes`
//! does not implement [`serde`] traits. This module uses a serializable
//! "proxy" enum to handle this cleanly.

#![cfg_attr(docsrs, doc(cfg(feature = "serde")))]

use super::{Endianness, IntVec, PhantomData, Samples};
use dsi_bitstream::prelude::Codes;
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

/// A private helper struct for serializing and deserializing an [`IntVec`].
#[derive(Serialize, Deserialize)]
struct IntVecSerde {
    data: Vec<u64>,
    samples: Option<Samples>,
    k: Option<usize>,
    len: usize,
    encoding: CodesSerde,
}

impl<E: Endianness> Serialize for IntVec<E> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let helper = IntVecSerde {
            data: self.data.clone(),
            samples: self.samples.clone(),
            k: self.k,
            len: self.len,
            encoding: self.encoding.into(),
        };
        helper.serialize(serializer)
    }
}

impl<'de, E: Endianness> Deserialize<'de> for IntVec<E> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = IntVecSerde::deserialize(deserializer)?;
        Ok(IntVec {
            data: helper.data,
            samples: helper.samples,
            k: helper.k,
            len: helper.len,
            encoding: helper.encoding.into(),
            endian: PhantomData,
        })
    }
}
