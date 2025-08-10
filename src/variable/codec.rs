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
use dsi_bitstream::prelude::{Codes, CodesStats};

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
/// let data: &[u32] = &;
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
    /// - If `log2_b` is `None`, an optimal parameter is estimated from the data.
    Rice { log2_b: Option<u8> },
    /// Boldi-Vigna ζ-coding with a parameter `k`. This code is effective for
    /// data with a power-law distribution, which is common in web graphs and
    /// social networks.
    ///
    /// - If `k` is `Some(val)`, the specified parameter is used (`k > 0`).
    /// - If `k` is `None`, an optimal parameter is estimated from the data.
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
    /// - If `k` is `None`, an optimal parameter is estimated from the data.
    Pi { k: Option<u64> },
    /// Elias-Fano Exponential-Golomb coding with a parameter `k`.
    ///
    /// - If `k` is `Some(val)`, the specified parameter is used.
    /// - If `k` is `None`, an optimal parameter is estimated from the data.
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
    /// Use an explicitly provided [`Codes`] variant from [`dsi-bitstream`](https://crates.io/crates/dsi-bitstream).
    ///
    /// This is for advanced use cases where the user has already constructed
    /// a [`Codes`] enum instance.
    Explicit(Codes),
}

/// A helper function to perform statistical analysis on the data.
fn get_stats<U: Into<u64> + Copy>(input: &[U]) -> CodesStats<10, 20, 10, 10, 10> {
    const TARGET_SAMPLE_SIZE: usize = 10_000;
    let mut stats = CodesStats::<10, 20, 10, 10, 10>::default();

    if input.len() <= TARGET_SAMPLE_SIZE {
        for &value in input {
            stats.update(value.into());
        }
    } else {
        let step = input.len() as f64 / TARGET_SAMPLE_SIZE as f64;
        let sample_iter = (0..TARGET_SAMPLE_SIZE).map(|i| input[((i as f64) * step) as usize]);
        for value in sample_iter {
            stats.update(value.into());
        }
    }
    stats
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
    if input.is_empty() {
        return Ok(Codes::Gamma); // A safe default for empty data
    }

    match spec {
        // Parameter-free codecs are a direct mapping.
        VariableCodecSpec::Gamma => Ok(Codes::Gamma),
        VariableCodecSpec::Delta => Ok(Codes::Delta),
        VariableCodecSpec::Unary => Ok(Codes::Unary),
        VariableCodecSpec::Omega => Ok(Codes::Omega),
        VariableCodecSpec::VByteLe => Ok(Codes::VByteLe),
        VariableCodecSpec::VByteBe => Ok(Codes::VByteBe),
        VariableCodecSpec::Explicit(codes) => Ok(codes),

        // Codecs with optional, user-provided parameters.
        VariableCodecSpec::Rice { log2_b: Some(p) } => Ok(Codes::Rice { log2_b: p as usize }),
        VariableCodecSpec::Zeta { k: Some(p) } => Ok(Codes::Zeta { k: p as usize }),
        VariableCodecSpec::Golomb { b: Some(p) } => Ok(Codes::Golomb { b: p as usize }),
        VariableCodecSpec::Pi { k: Some(p) } => Ok(Codes::Pi { k: p as usize }),
        VariableCodecSpec::ExpGolomb { k: Some(p) } => Ok(Codes::ExpGolomb { k: p as usize }),

        // Codecs where we must estimate the best parameters.
        VariableCodecSpec::Rice { log2_b: None } => {
            let stats = get_stats(input);
            let (best_param, _) = stats
                .rice
                .iter()
                .enumerate()
                .min_by_key(|&(_, cost)| cost)
                .unwrap_or((0, &0));
            Ok(Codes::Rice { log2_b: best_param })
        }
        VariableCodecSpec::Zeta { k: None } => {
            let stats = get_stats(input);
            let (best_param, _) = stats
                .zeta
                .iter()
                .enumerate()
                .min_by_key(|&(_, cost)| cost)
                .unwrap_or((0, &0));
            Ok(Codes::Zeta { k: best_param + 1 }) // Zeta params are 1-based
        }
        VariableCodecSpec::Golomb { b: None } => {
            let stats = get_stats(input);
            let (best_param, _) = stats
                .golomb
                .iter()
                .enumerate()
                .min_by_key(|&(_, cost)| cost)
                .unwrap_or((0, &0));
            Ok(Codes::Golomb { b: best_param + 1 }) // Golomb params are 1-based
        }
        VariableCodecSpec::Pi { k: None } => {
            let stats = get_stats(input);
            let (best_param, _) = stats
                .pi
                .iter()
                .enumerate()
                .min_by_key(|&(_, cost)| cost)
                .unwrap_or((0, &0));
            Ok(Codes::Pi { k: best_param + 2 }) // Pi params are offset by 2 in stats
        }
        VariableCodecSpec::ExpGolomb { k: None } => {
            let stats = get_stats(input);
            let (best_param, _) = stats
                .exp_golomb
                .iter()
                .enumerate()
                .min_by_key(|&(_, cost)| cost)
                .unwrap_or((0, &0));
            Ok(Codes::ExpGolomb { k: best_param })
        }

        // The fully automatic case: find the best of all codecs.
        VariableCodecSpec::Auto => {
            let stats = get_stats(input);
            let (best_code, _) = stats.best_code();
            Ok(best_code)
        }
    }
}
