//! # Codec Specification and Strategy Selection
//!
//! This module defines the mechanisms for selecting and configuring the
//! compression strategy for an [`IntVec`]. The choice of codec is critical, as
//! its effectiveness is highly dependent on the statistical properties of the
//! data being compressed.
//!
//! ## Encoding Strategies
//!
//! The library supports variable-length instantaneous codes sourced from the
//! [`dsi-bitstream`] crate. These codes (e.g., Gamma, Delta, Zeta) are designed
//! to compress integers by using shorter bit sequences for more frequent values,
//! making them ideal for skewed data distributions.
//!
//! ## The [`VariableCodecSpec`] Enum
//!
//! The primary user-facing API for this module is the [`VariableCodecSpec`] enum.
//! It provides a high-level interface for specifying the desired compression
//! strategy, allowing for:
//! - Direct selection of a parameter-free codec (e.g., `Gamma`).
//! - Explicit parameterization of tunable codecs (e.g., `Zeta { k: Some(3) }`).
//! - Automatic parameter selection (`Auto`), where the library analyzes the data
//!   to find the optimal codec configuration.
//!
//! The [`resolve_codec`] function translates a user's [`VariableCodecSpec`] into a
//! concrete [`Codes`] variant that the [`IntVec`] can use for its internal operations.
//!
//! [`IntVec`]: crate::variable::intvec::IntVec
//! [`dsi-bitstream`]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/

use crate::variable::intvec::IntVecError;
use dsi_bitstream::dispatch::FuncCodeWriter;
use dsi_bitstream::impls::{BufBitWriter, MemWordWriterVec};
use dsi_bitstream::prelude::{Codes, CodesStats, BE};

/// Specifies the compression codec and its parameters for an [`IntVec`].
///
/// This enum allows for either explicitly setting the parameters for codes
/// like Rice and Zeta, or requesting that [`IntVec`] automatically selects
/// suitable parameters based on the data distribution during construction.
///
/// [`IntVec`]: crate::variable::intvec::IntVec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VariableCodecSpec {
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
    /// Automatically select the best variable-length code based on the data.
    ///
    /// This is the recommended default for the slice-based builder.
    /// This option is **not** supported for the iterator-based builder.
    Auto,
    /// Use an explicitly provided code from the [dsi-bitstream](https://docs.rs/dsi-bitstream/latest/dsi_bitstream/) library enum.
    ///
    /// This is an escape hatch for advanced use cases or for codes not yet
    /// directly enumerated in [`VariableCodecSpec`].
    Explicit(Codes),
}

/// Resolves a user-provided [`VariableCodecSpec`] into a concrete [`Codes`] variant.
///
/// This function is the core of the codec selection mechanism. It translates the
/// user's high-level request (the `spec`) into a fully-parameterized, concrete
/// [`Codes`] variant that can be used for compression.
///
/// If the `spec` includes requests for automatic parameter selection (e.g.,
/// `VariableCodecSpec::Auto` or variants with `None` parameters), this function
/// analyzes the provided `input` data slice to determine the optimal settings.
///
/// # Arguments
/// * `input`: The data slice used to determine optimal parameters for automatic
///   selection. This is ignored for specs with fully-fixed parameters.
/// * `spec`: The [`VariableCodecSpec`] indicating the desired codec and parameter settings.
///
/// # Returns
/// A `Result` containing the concrete [`Codes`] variant or an
/// [`IntVecError::InvalidParameters`] if the configuration is invalid.
///
/// # Heuristics and Justification for Automatic Selection
///
/// When a [`VariableCodecSpec`] variant with `None` parameters or `Auto` is
/// provided, this function uses data-driven heuristics.
///
/// - **`Rice { log2_b: None }` / `Golomb { b: None }`**: These codes are optimal
///   for geometrically distributed data. This function computes the average of
///   the `input` data to estimate the optimal parameter.
///
/// - **`Zeta { k: None }`, `Pi { k: None }`, `ExpGolomb { k: None }`**: These
///   fall back to reasonable default parameters (`k=3`, `k=3`, `k=2` respectively).
///
/// - **`Auto`**: This triggers the most sophisticated heuristic. It uses
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
pub(crate) fn resolve_codec(input: &[u64], spec: VariableCodecSpec) -> Result<Codes, IntVecError> {
    match spec {
        // Parameter-free codecs
        VariableCodecSpec::Gamma => Ok(Codes::Gamma),
        VariableCodecSpec::Delta => Ok(Codes::Delta),
        VariableCodecSpec::Unary => Ok(Codes::Unary),
        VariableCodecSpec::Omega => Ok(Codes::Omega),
        VariableCodecSpec::VByteLe => Ok(Codes::VByteLe),
        VariableCodecSpec::VByteBe => Ok(Codes::VByteBe),

        // Passthrough for advanced usage
        VariableCodecSpec::Explicit(codes) => Ok(codes),

        // Codecs with optional parameters
        VariableCodecSpec::Rice { log2_b } => {
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
            Ok(Codes::Rice {
                log2_b: final_log2_b as usize,
            })
        }

        VariableCodecSpec::Zeta { k } => {
            let final_k = k.unwrap_or(3);
            if final_k == 0 {
                return Err(IntVecError::InvalidParameters(
                    "Zeta parameter k cannot be zero".to_string(),
                ));
            }
            Ok(Codes::Zeta {
                k: final_k as usize,
            })
        }

        VariableCodecSpec::Golomb { b } => {
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
            Ok(Codes::Golomb {
                b: final_b as usize,
            })
        }

        VariableCodecSpec::Pi { k } => {
            let final_k = k.unwrap_or(3);
            if final_k == 0 {
                return Err(IntVecError::InvalidParameters(
                    "Pi parameter k cannot be zero".to_string(),
                ));
            }
            Ok(Codes::Pi {
                k: final_k as usize,
            })
        }

        VariableCodecSpec::ExpGolomb { k } => {
            let final_k = k.unwrap_or(2);
            Ok(Codes::ExpGolomb {
                k: final_k as usize,
            })
        }

        VariableCodecSpec::Auto => {
            if input.is_empty() {
                return Ok(Codes::Gamma);
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
                Ok(best_code)
            } else {
                // Fallback to a safe, universally supported code if the best
                // one is not available in the writer dispatch.
                Ok(Codes::Delta)
            }
        }
    }
}
