//! Codec selection for variable-length integer compression.
//!
//! This module defines [`VariableCodecSpec`], an enum that allows users to
//! control the compression strategy for an [`IntVec`]. The choice of codec is a
//! critical performance parameter, as its effectiveness depends on the statistical
//! properties of the data being compressed.
//!
//! For most use cases, [`VariableCodecSpec::Auto`] is recommended, as it analyzes
//! the data to select a well-suited codec automatically. However, for users with
//! specific knowledge about their data's distribution, manually selecting a
//! codec can provide more control.
//!
//! [`IntVec`]: crate::variable::IntVec

use super::IntVecError;
use dsi_bitstream::dispatch::FuncCodeWriter;
use dsi_bitstream::impls::{BufBitWriter, MemWordWriterVec};
use dsi_bitstream::prelude::{Codes, CodesStats, BE};

/// Specifies the compression codec and its parameters for an [`IntVec`].
///
/// This enum allows for either explicitly setting the parameters for codes
/// like Rice and Zeta, or requesting that the [`IntVecBuilder`](super::IntVecBuilder)
/// automatically select suitable parameters based on the data distribution.
///
/// # Examples
///
/// Selecting a codec using the builder:
///
/// ```
/// use compressed_intvec::variable::{IntVec, UIntVec, VariableCodecSpec};
///
/// let data: &[u32] = &[1, 2, 3, 4, 5];
///
/// // Build a vector using Delta coding
/// let vec: UIntVec<u32> = IntVec::builder(data)
///     .codec(VariableCodecSpec::Delta)
///     .build()
///     .unwrap();
/// ```
///
/// [`IntVec`]: crate::variable::IntVec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VariableCodecSpec {
    /// Elias γ-coding. A simple, universal code that is effective for integers
    /// with a distribution skewed towards small values. It is the default codec
    /// for the iterator-based builder.
    #[default]
    Gamma,
    /// Elias δ-coding. A universal code that is generally more efficient than
    /// Gamma for larger integer values.
    Delta,
    /// Unary coding. Encodes an integer `n` as `n` zeros followed by a one. It is
    /// only efficient for extremely small values (e.g., 0, 1, 2).
    Unary,
    /// Rice-coding with a parameter `log2_b`. This code is optimal for data with
    /// a geometric distribution.
    ///
    /// - If `log2_b` is `Some(val)`, the specified parameter is used.
    /// - If `log2_b` is `None`, an optimal parameter is estimated from the data
    ///   during the build process.
    Rice { log2_b: Option<u8> },
    /// Boldi-Vigna ζ-coding with a parameter `k`. This code is effective for
    /// data with a power-law distribution, which is common in web graphs and
    /// social networks.
    ///
    /// - If `k` is `Some(val)`, the specified parameter is used (`k > 0`).
    /// - If `k` is `None`, a default of `k=3` is used.
    Zeta { k: Option<u64> },
    /// Golomb-coding with a parameter `b`. This is a generalization of Rice coding
    /// and is also suitable for geometric distributions.
    ///
    /// - If `b` is `Some(val)`, the specified parameter is used (`b > 0`).
    /// - If `b` is `None`, an optimal parameter is estimated from the data.
    Golomb { b: Option<u64> },
    /// Elias-Fano ω-coding. A universal code.
    Omega,
    /// An alternative universal code with a parameter `k`.
    ///
    /// - If `k` is `Some(val)`, the specified parameter is used (`k > 0`).
    /// - If `k` is `None`, a default of `k=3` is used.
    Pi { k: Option<u64> },
    /// Elias-Fano Exponential-Golomb coding with a parameter `k`.
    ///
    /// - If `k` is `Some(val)`, the specified parameter is used.
    /// - If `k` is `None`, a default of `k=2` is used.
    ExpGolomb { k: Option<u64> },
    /// VByte encoding with Little-Endian byte order. This is often one of the
    /// fastest codecs for decoding, though it may not offer the best compression.
    VByteLe,
    /// VByte encoding with Big-Endian byte order.
    VByteBe,
    /// Automatically select the best variable-length code based on the data.
    ///
    /// When this option is used, the builder will analyze a sample of the input
    /// data to estimate which codec will provide the best compression ratio.
    ///
    /// **Note:** This option is **not** supported for the iterator-based builder,
    /// as it requires pre-analyzing the data.
    Auto,
    /// Use an explicitly provided `Codes` variant from `dsi-bitstream`.
    ///
    /// This is for advanced use cases where the user has already constructed
    /// a `Codes` enum instance.
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
        // Parameter-free codecs are a direct mapping.
        VariableCodecSpec::Gamma => Ok(Codes::Gamma),
        VariableCodecSpec::Delta => Ok(Codes::Delta),
        VariableCodecSpec::Unary => Ok(Codes::Unary),
        VariableCodecSpec::Omega => Ok(Codes::Omega),
        VariableCodecSpec::VByteLe => Ok(Codes::VByteLe),
        VariableCodecSpec::VByteBe => Ok(Codes::VByteBe),

        // Passthrough for advanced usage.
        VariableCodecSpec::Explicit(codes) => Ok(codes),

        // Codecs where parameters can be estimated from the data.
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
                // Heuristic for optimal Rice parameter.
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
                // Heuristic for optimal Golomb parameter 'b'.
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

            // To keep analysis fast, we analyze a sample of the data, not the whole set.
            const TARGET_SAMPLE_SIZE: usize = 10_000;
            let mut stats = CodesStats::<10, 20, 10, 10, 10>::default();

            if input.len() <= TARGET_SAMPLE_SIZE {
                // For small inputs, analyze the entire dataset for perfect accuracy.
                for &value in input {
                    stats.update(value.into());
                }
            } else {
                // For large inputs, take a uniform sample.
                let step = input.len() as f64 / TARGET_SAMPLE_SIZE as f64;
                let sample_iter =
                    (0..TARGET_SAMPLE_SIZE).map(|i| input[((i as f64) * step) as usize]);
                for value in sample_iter {
                    stats.update(value.into());
                }
            }

            let (best_code, _) = stats.best_code();

            // This check ensures that the selected code is supported by the writer implementation.
            if FuncCodeWriter::<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>>::new(
                best_code,
            )
            .is_ok()
            {
                Ok(best_code)
            } else {
                // Fallback to a safe, universally supported code if the best is not available.
                Ok(Codes::Delta)
            }
        }
    }
}