//! # Codec Specification and Strategy Selection
//!
//! This module defines the mechanisms for selecting and configuring the
//! compression strategy for an [`IntVec`]. The choice of codec is critical, as
//! its effectiveness is highly dependent on the statistical properties of the
//! data being compressed.
//!
//! [`IntVec`]: crate::variable::IntVec

use super::IntVecError;
use dsi_bitstream::dispatch::FuncCodeWriter;
use dsi_bitstream::impls::{BufBitWriter, MemWordWriterVec};
use dsi_bitstream::prelude::{Codes, CodesStats, BE};

/// Specifies the compression codec and its parameters for an [`IntVec`].
///
/// This enum allows for either explicitly setting the parameters for codes
/// like Rice and Zeta, or requesting that [`IntVec`] automatically selects
/// suitable parameters based on the data distribution during construction.
///
/// [`IntVec`]: crate::variable::IntVec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VariableCodecSpec {
    /// Use Elias γ-coding. This is the default codec spec.
    #[default]
    Gamma,
    /// Use Elias δ-coding.
    Delta,
    /// Use Unary coding.
    Unary,
    /// Use Rice-coding.
    /// - If `log2_b` is `Some(val)`, uses the specified parameter.
    /// - If `log2_b` is `None`, an optimal parameter is estimated from the data.
    Rice { log2_b: Option<u8> },
    /// Use Boldi-Vigna ζ-coding.
    /// - If `k` is `Some(val)`, uses the specified parameter (`k > 0`).
    /// - If `k` is `None`, a default of `k=3` is used.
    Zeta { k: Option<u64> },
    /// Use Golomb-coding.
    /// - If `b` is `Some(val)`, uses the specified parameter (`b > 0`).
    /// - If `b` is `None`, an optimal parameter is estimated from the data.
    Golomb { b: Option<u64> },
    /// Use Elias-Fano ω-coding.
    Omega,
    /// Use an alternative universal code.
    /// - If `k` is `Some(val)`, uses the specified parameter (`k > 0`).
    /// - If `k` is `None`, a default of `k=3` is used.
    Pi { k: Option<u64> },
    /// Use Elias-Fano Exponential-Golomb coding.
    /// - If `k` is `Some(val)`, uses the specified parameter.
    /// - If `k` is `None`, a default of `k=2` is used.
    ExpGolomb { k: Option<u64> },
    /// Use VByte encoding with Little-Endian byte order.
    VByteLe,
    /// Use VByte encoding with Big-Endian byte order.
    VByteBe,
    /// Automatically select the best variable-length code based on the data.
    /// This option is **not** supported for the iterator-based builder.
    Auto,
    /// Use an explicitly provided code from the `dsi-bitstream` library.
    Explicit(Codes),
}

/// Resolves a user-provided [`VariableCodecSpec`] into a concrete [`Codes`] variant.
///
/// This function translates the user's high-level request into a fully-parameterized,
/// concrete [`Codes`] variant that can be used for compression.
///
/// If the `spec` includes requests for automatic parameter selection, this function
/// analyzes the provided `input` data slice to determine the optimal settings.
pub(crate) fn resolve_codec<U>(input: &[U], spec: VariableCodecSpec) -> Result<Codes, IntVecError>
where
    U: Into<u64> + Copy,
{
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
                let sum: u128 = input.iter().map(|&x| x.into() as u128).sum();
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
                let sum: u128 = input.iter().map(|&x| x.into() as u128).sum();
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
                    stats.update(value.into());
                }
            } else {
                // For large inputs, take a uniform sample to keep analysis fast.
                let step = input.len() as f64 / TARGET_SAMPLE_SIZE as f64;
                let sample_iter =
                    (0..TARGET_SAMPLE_SIZE).map(|i| input[((i as f64) * step) as usize]);
                for value in sample_iter {
                    stats.update(value.into());
                }
            }

            let (best_code, _) = stats.best_code();

            // This check ensures that the selected code is supported by the writer.
            if FuncCodeWriter::<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>>::new(
                best_code,
            )
            .is_ok()
            {
                Ok(best_code)
            } else {
                // Fallback to a safe, universally supported code.
                Ok(Codes::Delta)
            }
        }
    }
}