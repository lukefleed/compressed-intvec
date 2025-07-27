//! # Codec Specification and Strategy Selection
//!
//! This module defines the mechanisms for selecting and configuring the
//! compression strategy for an [`IntVec`]. The choice of codec is critical, as
//! its effectiveness is highly dependent on the statistical properties of the
//! data being compressed.
//!
//! ## Encoding Strategies
//!
//! The library supports two fundamental encoding families:
//!
//! 1.  **Variable-Length Instantaneous Codes**: Sourced from the [`dsi-bitstream`]
//!     crate, these codes (e.g., Gamma, Delta, Zeta) are designed to compress
//!     integers by using shorter bit sequences for more frequent values, making
//!     them ideal for skewed data distributions.
//!
//! 2.  **Fixed-Width Integer Encoding**: This strategy uses the same number of
//!     bits for every integer. It is optimal for data that is uniformly
//!     distributed within a known range, providing the fastest possible random
//!     access.
//!
//! ## The [`CodecSpec`] Enum
//!
//! The primary user-facing API for this module is the [`CodecSpec`] enum. It
//! provides a high-level interface for specifying the desired compression
//! strategy, allowing for:
//! - Direct selection of a parameter-free codec (e.g., `Gamma`).
//! - Explicit parameterization of tunable codecs (e.g., `Zeta { k: Some(3) }`).
//! - Automatic parameter selection, where the library analyzes the data to find
//!   the optimal configuration (e.g., `Auto` or `FixedLength { num_bits: None }`).
//!
//! The [`resolve_codec`] function translates a user's [`CodecSpec`] into a concrete
//! [`Encoding`] variant that the [`IntVec`] can use for its internal operations.
//!
//! [`IntVec`]: crate::intvec::IntVec
//! [`dsi-bitstream`]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/

use crate::intvec::IntVecError;
use dsi_bitstream::dispatch::FuncCodeWriter;
use dsi_bitstream::impls::{BufBitWriter, MemWordWriterVec};
use dsi_bitstream::prelude::{Codes, CodesStats, BE};
use mem_dbg::{CopyType, MemDbgImpl, MemSize, SizeFlags, True};

/// Specifies the compression codec and its parameters for an [`IntVec`].
///
/// This enum allows for either explicitly setting the parameters for codes
/// like Rice and Zeta, or requesting that [`IntVec`] automatically selects
/// suitable parameters based on the data distribution during construction.
///
/// [`IntVec`]: crate::intvec::IntVec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodecSpec {
    /// Use Elias γ-coding.
    ///
    /// The implied probability distribution is approximately `1 / (2*x^2)`.
    /// This code is parameter-free and is generally effective for data distributions
    /// skewed towards small values.
    Gamma,
    /// Use Elias δ-coding. This is the default codec spec.
    ///
    /// The implied probability distribution is approximately `1 / (2*x*log(x)^2)`.
    /// Delta coding is also parameter-free and tends to be more efficient than
    /// Gamma for larger integer values.
    #[default]
    Delta,
    /// Use Unary coding.
    ///
    /// Represents the number `n` with `n` zeros followed by a one. It is only
    /// efficient for very small integers, particularly `0` and `1`.
    Unary,
    /// Use Rice-coding.
    ///
    /// A special case of Golomb coding suitable for geometrically distributed data.
    /// - If `log2_b` is `Some(val)`, uses the specified parameter.
    /// - If `log2_b` is `None` (on slice-based builder), an optimal parameter
    ///   is estimated from the data's average value.
    Rice { log2_b: Option<u8> },
    /// Use Boldi-Vigna ζ-coding.
    ///
    /// The implied probability distribution is approximately `1 / x^(1 + 1/k)`.
    /// This code is effective for power-law distributions.
    /// - If `k` is `Some(val)`, uses the specified parameter (`k > 0`).
    /// - If `k` is `None` (on slice-based builder), a default of `k=3` is used.
    Zeta { k: Option<u64> },
    /// Use Golomb-coding.
    ///
    /// Suitable for geometrically distributed data.
    /// - If `b` is `Some(val)`, uses the specified parameter (`b > 0`).
    /// - If `b` is `None` (on slice-based builder), an optimal parameter
    ///   is estimated from the data's average value.
    Golomb { b: Option<u64> },
    /// Use Elias-Fano ω-coding, a universal code for positive integers.
    Omega,
    /// Use an alternative universal code for positive integers.
    /// - If `k` is `Some(val)`, uses the specified parameter (`k > 0`).
    /// - If `k` is `None` (on slice-based builder), a default of `k=3` is used.
    Pi { k: Option<u64> },
    /// Use Elias-Fano Exponential-Golomb coding.
    /// - If `k` is `Some(val)`, uses the specified parameter.
    /// - If `k` is `None` (on slice-based builder), a default of `k=2` is used.
    ExpGolomb { k: Option<u64> },
    /// Use VByte encoding with Little-Endian byte order. Efficient for integers
    /// that fit within a few bytes.
    VByteLe,
    /// Use VByte encoding with Big-Endian byte order.
    VByteBe,
    /// Use fixed-width integer encoding.
    ///
    /// Optimal for uniformly distributed data within a known range.
    /// - If `num_bits` is `Some(val)`, uses the specified number of bits.
    /// - If `num_bits` is `None` (on slice-based builder), the minimum bits
    ///   required for the largest value in the data is used.
    FixedLength { num_bits: Option<u8> },
    /// Automatically select the best variable-length code based on the data.
    ///
    /// This is the recommended default for the slice-based builder.
    /// This option is **not** supported for the iterator-based builder.
    Auto,
    /// Use an explicitly provided code from the [dsi-bitstream](https://docs.rs/dsi-bitstream/latest/dsi_bitstream/) library enum.
    ///
    /// This is an escape hatch for advanced use cases or for codes not yet
    /// directly enumerated in [`CodecSpec`].
    Explicit(Codes),
}

/// Represents the chosen encoding strategy for an `IntVec`.
///
/// This is the concrete, resolved encoding that an `IntVec` instance stores and
/// uses for its internal compression and decompression logic. It is the result
/// of resolving a [`CodecSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// A variable-length, bit-level instantaneous code from the [dsi-bitstream](https://docs.rs/dsi-bitstream/latest/dsi_bitstream/) library.
    Dsi(Codes),
    /// Fixed-width encoding, where each integer is stored with `num_bits`.
    Fixed { num_bits: usize },
}

// Manual implementation because the derive macro fails on the `Codes` type
// from the external `dsi-bitstream` crate.
impl CopyType for Encoding {
    type Copy = True;
}

// Manual implementation for `MemSize`. `Encoding` is a `Copy` enum and does not
// own any heap memory, so its size is just its stack size.
impl MemSize for Encoding {
    fn mem_size(&self, _flags: SizeFlags) -> usize {
        core::mem::size_of::<Self>()
    }
}

// Manual implementation for `MemDbgImpl`. We use the default implementation for
// `_mem_dbg_rec_on` which does not recurse, correctly treating this enum as a
// leaf in the memory layout tree.
impl MemDbgImpl for Encoding {}

/// Resolves a user-provided [`CodecSpec`] into a concrete [`Encoding`] variant.
///
/// This function is the core of the codec selection mechanism. It translates the
/// user's high-level request (the `spec`) into a fully-parameterized, concrete
/// [`Encoding`] that can be used for compression.
///
/// If the `spec` includes requests for automatic parameter selection (e.g.,
/// `CodecSpec::Auto` or variants with `None` parameters), this function analyzes
/// the provided `input` data slice to determine the optimal settings.
///
/// # Arguments
/// * `input`: The data slice used to determine optimal parameters for automatic
///   selection. This is ignored for specs with fully-fixed parameters.
/// * `spec`: The [`CodecSpec`] indicating the desired codec and parameter settings.
///
/// # Returns
/// A `Result` containing the concrete [`Encoding`] variant or an
/// [`IntVecError::InvalidParameters`] if the configuration is invalid.
///
/// # Heuristics and Justification for Automatic Selection
///
/// When a [`CodecSpec`] variant with `None` parameters or [`CodecSpec::Auto`] is
/// provided, this function uses data-driven heuristics.
///
/// - **`CodecSpec::FixedLength { num_bits: None }`**: The function scans the
///   entire `input` slice to find the maximum value. It then calculates the
///   minimum number of bits required to represent this value. A full scan is
///   necessary here to guarantee correctness.
///
/// - **`CodecSpec::Rice { log2_b: None }` / `CodecSpec::Golomb { b: None }`**:
///   These codes are optimal for geometrically distributed data. This function
///   computes the average of the `input` data to estimate the optimal parameter.
///
/// - **`CodecSpec::Zeta { k: None }`, `Pi { k: None }`, `ExpGolomb { k: None }`**:
///   These fall back to reasonable default parameters (`k=3`, `k=3`, `k=2` respectively).
///
/// - **`CodecSpec::Auto`**: This triggers the most sophisticated heuristic. It uses
///   a dynamic sampling strategy to balance analysis speed and accuracy:
///   1.  **For small inputs (<= 10,000 elements)**, it analyzes the *entire* dataset.
///       This gives a perfect statistical profile, ensuring the best possible codec
///       choice without a noticeable performance penalty.
///   2.  **For larger inputs**, it takes a **uniform sample** of ~10,000 elements
///       by selecting values at regular intervals across the entire input slice.
///       This provides a high-quality, representative sample while ensuring the
///       analysis step remains extremely fast, regardless of input size.
///   3.  Based on this analysis, it uses the [`CodesStats`] utility
///       to select the variable-length code predicted to be the most space-efficient.
pub fn resolve_codec(input: &[u64], spec: CodecSpec) -> Result<Encoding, IntVecError> {
    match spec {
        // Parameter-free codecs
        CodecSpec::Gamma => Ok(Encoding::Dsi(Codes::Gamma)),
        CodecSpec::Delta => Ok(Encoding::Dsi(Codes::Delta)),
        CodecSpec::Unary => Ok(Encoding::Dsi(Codes::Unary)),
        CodecSpec::Omega => Ok(Encoding::Dsi(Codes::Omega)),
        CodecSpec::VByteLe => Ok(Encoding::Dsi(Codes::VByteLe)),
        CodecSpec::VByteBe => Ok(Encoding::Dsi(Codes::VByteBe)),

        // Passthrough for advanced usage
        CodecSpec::Explicit(codes) => Ok(Encoding::Dsi(codes)),

        // Codecs with optional parameters
        CodecSpec::Rice { log2_b } => {
            let final_log2_b = log2_b.unwrap_or_else(|| {
                if input.is_empty() {
                    return 0;
                }
                let sum: u128 = input.iter().map(|&x| x as u128).sum();
                let avg = sum as f64 / input.len() as f64;
                if avg < 1.0 {
                    return 0;
                }
                let ideal_log2_b = (avg / std::f64::consts::LN_2).log2().round();
                ideal_log2_b.clamp(0.0, 10.0) as u8
            });
            Ok(Encoding::Dsi(Codes::Rice {
                log2_b: final_log2_b as usize,
            }))
        }

        CodecSpec::Zeta { k } => {
            let final_k = k.unwrap_or(3);
            if final_k == 0 {
                return Err(IntVecError::InvalidParameters(
                    "Zeta parameter k cannot be zero".to_string(),
                ));
            }
            Ok(Encoding::Dsi(Codes::Zeta {
                k: final_k as usize,
            }))
        }

        CodecSpec::Golomb { b } => {
            let final_b = b.unwrap_or_else(|| {
                if input.is_empty() {
                    return 1;
                }
                let sum: u128 = input.iter().map(|&x| x as u128).sum();
                let avg = sum as f64 / input.len() as f64;
                // Heuristic for optimal Golomb parameter 'b'
                (avg * 0.69).round().max(1.0) as u64
            });
            if final_b == 0 {
                return Err(IntVecError::InvalidParameters(
                    "Golomb parameter b cannot be zero".to_string(),
                ));
            }
            Ok(Encoding::Dsi(Codes::Golomb {
                b: final_b as usize,
            }))
        }

        CodecSpec::Pi { k } => {
            let final_k = k.unwrap_or(3);
            if final_k == 0 {
                return Err(IntVecError::InvalidParameters(
                    "Pi parameter k cannot be zero".to_string(),
                ));
            }
            Ok(Encoding::Dsi(Codes::Pi {
                k: final_k as usize,
            }))
        }

        CodecSpec::ExpGolomb { k } => {
            let final_k = k.unwrap_or(2);
            Ok(Encoding::Dsi(Codes::ExpGolomb {
                k: final_k as usize,
            }))
        }

        CodecSpec::FixedLength { num_bits } => {
            let final_num_bits = num_bits.map(|n| n as usize).unwrap_or_else(|| {
                let max_val = input.iter().max().copied().unwrap_or(0);
                if max_val == 0 {
                    1
                } else {
                    (u64::BITS - max_val.leading_zeros()) as usize
                }
            });

            if final_num_bits > 64 {
                return Err(IntVecError::InvalidParameters(
                    "FixedLength num_bits cannot be greater than 64".to_string(),
                ));
            }
            Ok(Encoding::Fixed {
                num_bits: final_num_bits,
            })
        }

        CodecSpec::Auto => {
            if input.is_empty() {
                return Ok(Encoding::Dsi(Codes::Gamma));
            }

            const TARGET_SAMPLE_SIZE: usize = 10_000;
            let mut stats = CodesStats::<10, 20, 10, 10, 10>::default();

            if input.len() <= TARGET_SAMPLE_SIZE {
                // For small inputs, analyze the entire dataset for perfect accuracy.
                for &value in input {
                    stats.update(value);
                }
            } else {
                // For large inputs, take a uniform sample to keep analysis fast.
                let step = input.len() as f64 / TARGET_SAMPLE_SIZE as f64;
                let sample_iter =
                    (0..TARGET_SAMPLE_SIZE).map(|i| input[((i as f64) * step) as usize]);
                for value in sample_iter {
                    stats.update(value);
                }
            }

            let (best_code, _) = stats.best_code();

            // This check ensures that the selected code is supported by the writer.
            // It acts as a safeguard against future changes in dsi-bitstream.
            if FuncCodeWriter::<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>>::new(
                best_code,
            )
            .is_ok()
            {
                Ok(Encoding::Dsi(best_code))
            } else {
                // Fallback to a safe, universally supported code if the best
                // one is not available in the writer dispatch.
                Ok(Encoding::Dsi(Codes::Delta))
            }
        }
    }
}
