//! Shared serde support utilities for integer vector types.
//!
//! This module provides common serialization infrastructure used by variable-width
//! and sequential integer vectors, particularly the serializable proxy for the
//! `dsi-bitstream::codes::Codes` enum which does not implement serde traits directly.

use dsi_bitstream::prelude::Codes;
use serde::{Deserialize, Serialize};

/// A serializable proxy for `dsi-bitstream::prelude::Codes`.
///
/// This enum bridges the `Codes` type from the dsi-bitstream crate with serde,
/// allowing complete serialization and deserialization of integer vector types
/// that use variable-width encoding.
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub(crate) enum CodesSerde {
    /// Gamma code encoding.
    Gamma,
    /// Delta code encoding.
    Delta,
    /// Zeta code encoding with parameter `k`.
    Zeta { k: usize },
    /// Rice code encoding with log2 of the base parameter.
    Rice { log2_b: usize },
    /// Unary code encoding.
    Unary,
    /// Golomb code encoding with parameter `b`.
    Golomb { b: usize },
    /// Omega code encoding.
    Omega,
    /// Pi code encoding with parameter `k`.
    Pi { k: usize },
    /// Exponential Golomb code encoding with parameter `k`.
    ExpGolomb { k: usize },
    /// Variable-byte encoding with little-endian byte order.
    VByteLe,
    /// Variable-byte encoding with big-endian byte order.
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
