//! Codec selection for variable-length integer compression.
//!
//! This module defines [`VariableCodecSpec`], an enum for controlling the
//! compression strategy of an [`IntVec`]. The choice of codec is a critical
//! performance parameter, as its effectiveness depends on the statistical
//! properties of the data being compressed.
//!
//! # Codec Selection Strategy
//!
//! Codec selection is performed by a statistical analysis of the entire
//! input dataset at construction time. 
//!
//! The [`VariableCodecSpec`] enum provides several ways to specify the compression method:
//!
//! 1.  **Explicit Specification**: A specific codec and all its parameters are
//!     provided. This is suitable when the data characteristics are known in
//!     advance.
//!     - Non-parametric examples: [`Gamma`](VariableCodecSpec::Gamma), [`Delta`](VariableCodecSpec::Delta).
//!     - Parametric example: `Zeta { k: Some(3) }`.
//! 
//! ```
//! use compressed_intvec::prelude::*;
//! 
//! let data: &[u32] = &(0..1000).collect::<Vec<_>>(); 
//! 
//! // Explicitly specify a non-parametric codec
//! let delta_vec: UIntVec<u32> = IntVec::builder()
//!     .codec(VariableCodecSpec::Delta)
//!     .k(16)
//!     .build(&data)
//!     .unwrap();
//!  
//! // Explicitly specify a parametric codec with a fixed parameter
//! let zeta_vec: UIntVec<u32> = IntVec::builder()
//!     .codec(VariableCodecSpec::Zeta { k: Some(3) })
//!     .build(&data)
//!     .unwrap();
//! ```
//!
//! 2.  **Automatic Parameter Estimation**: A specific codec family is chosen, but
//!     the optimal parameter is determined by the builder based on a full data
//!     analysis. This is achieved by providing `None` as the parameter value.
//!     - Example: `Rice { log2_b: None }` will find the best `log2_b` for the
//!       given data.
//! 
//! ```
//! use compressed_intvec::prelude::*;
//!
//! let data: &[u32] = &(0..1000).collect::<Vec<_>>();
//!
//! // Automatically select the best Rice parameter
//! let rice_vec: UIntVec<u32> = IntVec::builder()
//!     .codec(VariableCodecSpec::Rice { log2_b: None })
//!     .build(&data)
//!     .unwrap();
//! ```
//!
//! 3.  **Fully Automatic Selection**: The builder analyzes the data against all
//!     available codecs and their standard parameter ranges to find the single
//!     best configuration. This is activated by using [`VariableCodecSpec::Auto`].
//! 
//! ```
//! use compressed_intvec::prelude::*;
//! 
//! let data: &[u32] = &(0..1000).collect::<Vec<_>>();
//! // Automatically select the best codec and parameters for the data
//! let auto_vec: UIntVec<u32> = IntVec::builder()
//!    .codec(VariableCodecSpec::Auto)
//!    .build(&data)
//!    .unwrap();
//! ```
//! 
//!
//! ## Analysis Mechanism
//!
//! The selection logic uses the [`CodesStats`] utility from the [`dsi-bitstream`]
//! crate. For a given sequence of integers, [`CodesStats`] calculates the exact
//! total bit cost for encoding the sequence with a wide range of instantaneous
//! codes and their common parameterizations. 
//!
//! ## Construction Overhead
//!
//! The full-dataset analysis has a one-time computational cost at construction.
//! The complexity is `O(N * C)`, where `N` is the number of elements in the
//! input and `C` is the number of codec configurations tested by [`CodesStats`]
//! (approximately 70).
//!
//! This trade-off is suitable for read-heavy workloads where a higher initial
//! cost is acceptable for better compression and subsequent read performance.
//!
//! # Implementation Notes
//!
//! - The parameter ranges for codecs like Zeta and Rice are defined by the `const
//!   generics` of the [`CodesStats`] struct in [`dsi-bitstream`]. The default
//!   values cover common and effective parameter ranges.
//! - If a data distribution benefits from a parameter outside of the tested
//!   range (e.g., Zeta with `k=20`), it must be specified explicitly in the
//!   builder via `.codec(VariableCodecSpec::Zeta { k: Some(20) })`.
//!
//! [`IntVec`]: crate::variable::IntVec
//! [`IntVecBuilder`]: crate::variable::builder::IntVecBuilder
//! [`dsi-bitstream`]: https://crates.io/crates/dsi-bitstream

use super::IntVecError;
use dsi_bitstream::prelude::{Codes, CodesStats};

/// Specifies the compression codec and its parameters for an [`IntVec`](super::IntVec).
///
/// This enum allows for either explicitly setting the parameters for codes
/// like Rice and Zeta, or requesting that the [`IntVecBuilder`](super::builder::IntVecBuilder)
/// automatically select suitable parameters by performing a full analysis of
/// the data distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VariableCodecSpec {
    /// Elias γ-coding. A simple, universal code that is effective for integers
    /// with a distribution skewed towards small values. It is the default codec
    /// for the iterator-based builder, which cannot perform data analysis.
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
    /// - If `log2_b` is `None`, an optimal parameter is estimated by analyzing the
    ///   entire dataset.
    Rice { log2_b: Option<u8> },

    /// Boldi-Vigna ζ-coding with a parameter `k`. This code is effective for
    /// data with a power-law distribution, common in web graphs and social networks.
    ///
    /// - If `k` is `Some(val)`, the specified parameter is used (`k > 0`).
    /// - If `k` is `None`, an optimal parameter is estimated by analyzing the
    ///   entire dataset.
    Zeta { k: Option<u64> },

    /// Golomb-coding with a parameter `b`. This is a generalization of Rice coding
    /// and is also suitable for geometric distributions.
    ///
    /// - If `b` is `Some(val)`, the specified parameter is used (`b > 0`).
    /// - If `b` is `None`, an optimal parameter is estimated by analyzing the
    ///   entire dataset.
    Golomb { b: Option<u64> },

    /// Elias-Fano ω-coding. A universal code.
    Omega,

    /// Streamlined Apostolico–Drovandi π code with a parameter `k`.
    ///
    /// - If `k` is `Some(val)`, the specified parameter is used (`k > 0`).
    /// - If `k` is `None`, an optimal parameter is estimated by analyzing the
    ///   entire dataset.
    Pi { k: Option<u64> },

    /// Elias-Fano Exponential-Golomb coding with a parameter `k`.
    ///
    /// - If `k` is `Some(val)`, the specified parameter is used.
    /// - If `k` is `None`, an optimal parameter is estimated by analyzing the
    ///   entire dataset.
    ExpGolomb { k: Option<u64> },

    /// VByte encoding with Little-Endian byte order. This is often one of the
    /// fastest codecs for decoding, though it may not offer the best compression.
    VByteLe,

    /// VByte encoding with Big-Endian byte order.
    VByteBe,

    /// Automatically select the best variable-length code based on the data.
    ///
    /// When this option is used, the builder performs a statistical
    /// analysis on the entire input dataset to determine which codec and
    /// parameterization provides the best compression ratio.
    ///
    /// # Note
    /// 
    /// This option is **not** supported for the iterator-based builder,
    /// as it requires pre-analyzing the data.
    Auto,

    /// Use an explicitly provided [`Codes`] variant from [`dsi-bitstream`](https://crates.io/crates/dsi-bitstream).
    ///
    /// This is for advanced use cases where the user has already constructed
    /// a [`Codes`] enum instance.
    Explicit(Codes),
}

impl VariableCodecSpec {
    /// Returns `true` if this codec specification requires data analysis to resolve.
    ///
    /// Analysis is needed when parameters are not explicitly specified and must
    /// be determined by analyzing the data distribution.
    #[inline]
    pub(crate) fn requires_analysis(&self) -> bool {
        matches!(
            self,
            VariableCodecSpec::Auto
                | VariableCodecSpec::Rice { log2_b: None }
                | VariableCodecSpec::Zeta { k: None }
                | VariableCodecSpec::Golomb { b: None }
                | VariableCodecSpec::Pi { k: None }
                | VariableCodecSpec::ExpGolomb { k: None }
        )
    }
}

/// Resolves a user-provided [`VariableCodecSpec`] into a concrete [`Codes`] variant.
///
/// This function translates the user's high-level request into a fully-parameterized,
/// concrete [`Codes`] variant that can be used for compression.
///
/// If the `spec` includes requests for automatic parameter selection (e.g., `Auto`
/// or `Zeta { k: None }`), this function analyzes the **entire** provided `input`
/// data slice to determine the optimal settings.
pub(crate) fn resolve_codec<U>(input: &[U], spec: VariableCodecSpec) -> Result<Codes, IntVecError>
where
    U: Into<u64> + Copy,
{
    match spec {
        // Parameter-free codecs: direct mapping, no data needed.
        VariableCodecSpec::Gamma => Ok(Codes::Gamma),
        VariableCodecSpec::Delta => Ok(Codes::Delta),
        VariableCodecSpec::Unary => Ok(Codes::Unary),
        VariableCodecSpec::Omega => Ok(Codes::Omega),
        VariableCodecSpec::VByteLe => Ok(Codes::VByteLe),
        VariableCodecSpec::VByteBe => Ok(Codes::VByteBe),
        VariableCodecSpec::Explicit(codes) => Ok(codes),

        // Codecs with explicit parameters: direct mapping.
        VariableCodecSpec::Rice { log2_b: Some(p) } => Ok(Codes::Rice { log2_b: p as usize }),
        VariableCodecSpec::Zeta { k: Some(p) } => Ok(Codes::Zeta { k: p as usize }),
        VariableCodecSpec::Golomb { b: Some(p) } => Ok(Codes::Golomb { b: p as usize }),
        VariableCodecSpec::Pi { k: Some(p) } => Ok(Codes::Pi { k: p as usize }),
        VariableCodecSpec::ExpGolomb { k: Some(p) } => Ok(Codes::ExpGolomb { k: p as usize }),

        // Codecs requiring analysis: return error if no data provided.
        VariableCodecSpec::Auto
        | VariableCodecSpec::Rice { log2_b: None }
        | VariableCodecSpec::Zeta { k: None }
        | VariableCodecSpec::Golomb { b: None }
        | VariableCodecSpec::Pi { k: None }
        | VariableCodecSpec::ExpGolomb { k: None } => {
            if input.is_empty() {
                return Ok(Codes::Gamma);  // Safe default only for analysis codecs
            }

            // Define a type alias for the default [`CodesStats`] configuration for clarity.
            // These const generics define the range of parameters to test.
            type DefaultCodesStats = CodesStats<10, 20, 10, 10, 10>;

            // Create a stats object and populate it by iterating through the entire dataset.
            let mut stats = DefaultCodesStats::default();
            for &value in input {
                stats.update(value.into());
            }

            // Use the populated stats to resolve the codec specification.
            match spec {
                VariableCodecSpec::Auto => {
                    let (best_code, _) = stats.best_code();
                    Ok(best_code)
                }
                VariableCodecSpec::Rice { log2_b: None } => {
                    let (best_param, _) = stats
                        .rice
                        .iter()
                        .enumerate()
                        .min_by_key(|&(_, cost)| cost)
                        .unwrap_or((0, &0)); // Fallback to 0 if array is empty.
                    Ok(Codes::Rice { log2_b: best_param })
                }
                VariableCodecSpec::Zeta { k: None } => {
                    let (best_param, _) = stats
                        .zeta
                        .iter()
                        .enumerate()
                        .min_by_key(|&(_, cost)| cost)
                        .unwrap_or((0, &0));
                    Ok(Codes::Zeta { k: best_param + 1 }) // Zeta params are 1-based.
                }
                VariableCodecSpec::Golomb { b: None } => {
                    let (best_param, _) = stats
                        .golomb
                        .iter()
                        .enumerate()
                        .min_by_key(|&(_, cost)| cost)
                        .unwrap_or((0, &0));
                    Ok(Codes::Golomb { b: best_param + 1 }) // Golomb params are 1-based.
                }
                VariableCodecSpec::Pi { k: None } => {
                    let (best_param, _) = stats
                        .pi
                        .iter()
                        .enumerate()
                        .min_by_key(|&(_, cost)| cost)
                        .unwrap_or((0, &0));
                    Ok(Codes::Pi { k: best_param + 2 }) // Pi params are offset by 2.
                }
                VariableCodecSpec::ExpGolomb { k: None } => {
                    let (best_param, _) = stats
                        .exp_golomb
                        .iter()
                        .enumerate()
                        .min_by_key(|&(_, cost)| cost)
                        .unwrap_or((0, &0));
                    Ok(Codes::ExpGolomb { k: best_param })
                }
                // This arm is guaranteed to be unreachable because the outer match
                // ensures `spec` is one of the variants handled above.
                _ => unreachable!(),
            }
        }
    }
}

/// Resolves a codec specification by analyzing data from an iterator.
///
/// This function avoids intermediate allocations by consuming the iterator
/// directly. It should only be called when `spec.requires_analysis()` returns
/// `true`.
///
/// # Arguments
///
/// * `iter` - An iterator of u64 values to analyze for codec parameter selection.
///
/// * `spec` - The codec specification requesting analysis (Auto or a codec with
///   None parameters).
pub(crate) fn resolve_codec_from_iter<I>(
    iter: I,
    spec: VariableCodecSpec,
) -> Result<Codes, IntVecError>
where
    I: Iterator<Item = u64>,
{
    // Define the default CodesStats configuration for parameter analysis.
    type DefaultCodesStats = CodesStats<10, 20, 10, 10, 10>;

    let mut stats = DefaultCodesStats::default();
    let mut count = 0usize;

    // Analyze the data stream without materializing it to a vector.
    for value in iter {
        stats.update(value);
        count += 1;
    }

    // If the stream is empty, return a safe default.
    if count == 0 {
        return Ok(Codes::Gamma);
    }

    // Use the accumulated statistics to determine the best codec.
    match spec {
        VariableCodecSpec::Auto => {
            let (best_code, _) = stats.best_code();
            Ok(best_code)
        }
        VariableCodecSpec::Rice { log2_b: None } => {
            let (best_param, _) = stats
                .rice
                .iter()
                .enumerate()
                .min_by_key(|&(_, cost)| cost)
                .unwrap_or((0, &0));
            Ok(Codes::Rice { log2_b: best_param })
        }
        VariableCodecSpec::Zeta { k: None } => {
            let (best_param, _) = stats
                .zeta
                .iter()
                .enumerate()
                .min_by_key(|&(_, cost)| cost)
                .unwrap_or((0, &0));
            Ok(Codes::Zeta { k: best_param + 1 })
        }
        VariableCodecSpec::Golomb { b: None } => {
            let (best_param, _) = stats
                .golomb
                .iter()
                .enumerate()
                .min_by_key(|&(_, cost)| cost)
                .unwrap_or((0, &0));
            Ok(Codes::Golomb { b: best_param + 1 })
        }
        VariableCodecSpec::Pi { k: None } => {
            let (best_param, _) = stats
                .pi
                .iter()
                .enumerate()
                .min_by_key(|&(_, cost)| cost)
                .unwrap_or((0, &0));
            Ok(Codes::Pi { k: best_param + 2 })
        }
        VariableCodecSpec::ExpGolomb { k: None } => {
            let (best_param, _) = stats
                .exp_golomb
                .iter()
                .enumerate()
                .min_by_key(|&(_, cost)| cost)
                .unwrap_or((0, &0));
            Ok(Codes::ExpGolomb { k: best_param })
        }
        _ => unreachable!("resolve_codec_from_iter called with non-analysis codec"),
    }
}