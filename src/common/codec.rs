//! The codes of this crate, their parameter ranges, and the automatic choice of a code.

#![forbid(unsafe_code)]

use dsi_bitstream::prelude::{Codes, CodesStats};
use std::{error::Error, fmt};

/// An integer encoding used by [`VarVec`](crate::VarVec) and
/// [`SeqVec`](crate::SeqVec).
///
/// Encoding and hot-path decoding use the implementations from
/// [dsi-bitstream](https://docs.rs/dsi-bitstream/). This crate provides the codec enum,
/// automatic selection and checked decoding of external data.
///
/// Each code encodes the word returned by [`Storable::to_word`](crate::Storable::to_word), so
/// signed integers are zig-zag encoded before the code is applied. Compression depends on how
/// often the values with short codewords occur. The builders select a codec automatically;
/// an explicit choice configures the encoding for a known distribution or existing data.
///
/// # Automatic selection
///
/// The builders choose the smallest estimated encoded payload among unary, gamma, delta, omega,
/// [`VByteLe`](Codec::VByteLe), zeta with `k` from 1 to 10, Golomb with `b` from 1 to 10,
/// exponential Golomb and Rice with parameters from 0 to 9, and pi with `k` from 2 to 10.
/// Only codes that can encode every input word are considered. These candidates select their
/// decoding implementation once when the decoder is created.
///
/// Selection measures codeword lengths, excluding index storage and word padding. For this
/// estimate, `u64::MAX` is treated as `u64::MAX - 1`, underestimating an eligible code's payload
/// by at most one bit per occurrence. The resulting choice minimizes this payload estimate.
///
/// The chosen code can encode every value, and the vector records it, so
/// [`VarVec::codec`](crate::VarVec::codec) and [`SeqVec::codec`](crate::SeqVec::codec) return
/// it and serialized data carries it. The candidates and the estimate are the current policy.
/// A minor release may change them, and with them the code that a builder chooses for the same
/// input and the size of the result.
///
/// # Parameters
///
/// A code with a parameter outside the range that its variant documents makes the builders
/// return [`CodecError::InvalidParameter`].
///
/// # Codes that cannot encode `u64::MAX`
///
/// Gamma, delta, omega, zeta, pi and exponential Golomb with `k == 0` encode `n + 1`, so they
/// cannot encode the word `u64::MAX`, which stores the `u64` value `u64::MAX` and the `i64`
/// value `i64::MIN`. Building a vector that holds such a value with one of these codes returns
/// [`CodecError::Unencodable`]; the automatic choice never picks them for such data.
///
/// # Examples
///
/// ```
/// use compressed_intvec::{Codec, VarVec};
///
/// let values: Vec<u32> = (0..1000).collect();
/// let vec = VarVec::builder().codec(Codec::Zeta { k: 3 }).build(&values)?;
/// assert_eq!(vec.codec(), Codec::Zeta { k: 3 });
/// assert_eq!(vec.get(500), Some(500));
/// # Ok::<(), compressed_intvec::CodecError<u32>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize), mem_size(flat))]
#[non_exhaustive]
pub enum Codec {
    /// The unary code: `n` zeros and a one, so `n + 1` bits. It suits only words that are
    /// almost always 0 or 1, since large words take as many bits as their value.
    Unary,
    /// The Elias gamma code: `floor(log2(n + 1))` in unary, then the bits of `n + 1` below its
    /// leading one, so `2 * floor(log2(n + 1)) + 1` bits.
    Gamma,
    /// The Elias delta code: `floor(log2(n + 1))` in gamma, then the bits of `n + 1` below its
    /// leading one. It generally takes fewer bits than gamma for large words.
    Delta,
    /// The Elias omega code, which writes the length of the binary representation of `n + 1`
    /// recursively.
    Omega,
    /// The variable-length byte code with 7 bits of the word in each byte, least significant
    /// group first. It takes a multiple of 8 bits.
    VByteLe,
    /// The variable-length byte code with 7 bits of the word in each byte, most significant
    /// group first. It takes as many bits as [`Codec::VByteLe`].
    VByteBe,
    /// The Rice code: `n >> log2_b` in unary, then the low `log2_b` bits of `n`. It suits
    /// geometric distributions. `log2_b < 64`; `Rice { log2_b: 0 }` is the unary code.
    Rice {
        /// The base-2 logarithm of the divisor.
        log2_b: u32,
    },
    /// The Golomb code: `n / b` in unary, then `n % b` in minimal binary. It suits geometric
    /// distributions. `b >= 1`; `Golomb { b: 1 }` is the unary code.
    Golomb {
        /// The divisor.
        b: u64,
    },
    /// The exponential Golomb code: `n >> k` in gamma, then the low `k` bits of `n`. `k < 64`;
    /// `ExpGolomb { k: 0 }` is the gamma code.
    ExpGolomb {
        /// The number of low bits written in binary.
        k: u32,
    },
    /// The zeta code of Boldi and Vigna, which suits power-law distributions. `1 <= k < 64`;
    /// `Zeta { k: 1 }` is the gamma code.
    Zeta {
        /// The shrinking factor.
        k: u32,
    },
    /// The streamlined pi code of Apostolico and Drovandi: `floor(log2(n + 1))` in the Rice code
    /// with `log2_b = k`, then the bits of `n + 1` below its leading one. `k < 64`;
    /// `Pi { k: 0 }` is the gamma code.
    Pi {
        /// The parameter of the Rice code of the length.
        k: u32,
    },
}

// The derived `MemDbg` of an enum with fields panics under `DbgFlags::RUST_LAYOUT` on stable
// Rust, so a vector's report shows its code as one line, without the variant's fields.
#[cfg(feature = "mem_dbg")]
impl mem_dbg::MemDbgImpl for Codec {}

impl Codec {
    /// Returns the code as a [`ValidCodec`], or `None` if its parameter is outside the range of
    /// its variant.
    pub(crate) fn validate(self) -> Option<ValidCodec> {
        let valid = match self {
            Self::Unary
            | Self::Gamma
            | Self::Delta
            | Self::Omega
            | Self::VByteLe
            | Self::VByteBe => true,
            Self::Rice { log2_b: k } | Self::ExpGolomb { k } | Self::Pi { k } => k < 64,
            Self::Zeta { k } => (1..64).contains(&k),
            Self::Golomb { b } => b >= 1,
        };
        valid.then_some(ValidCodec(self))
    }

    /// Returns `true` if the code can encode the word `u64::MAX`.
    pub(crate) fn encodes_max(self) -> bool {
        match self {
            Self::Gamma | Self::Delta | Self::Omega | Self::Zeta { .. } | Self::Pi { .. } => false,
            Self::ExpGolomb { k } => k > 0,
            Self::Unary
            | Self::VByteLe
            | Self::VByteBe
            | Self::Rice { .. }
            | Self::Golomb { .. } => true,
        }
    }

    /// Returns the simplest variant with the same codewords.
    fn canonical(self) -> Self {
        match self {
            Self::Rice { log2_b: 0 } | Self::Golomb { b: 1 } => Self::Unary,
            Self::ExpGolomb { k: 0 } | Self::Zeta { k: 1 } | Self::Pi { k: 0 } => Self::Gamma,
            Self::Golomb { b } if b.is_power_of_two() => Self::Rice {
                log2_b: b.trailing_zeros(),
            },
            other => other,
        }
    }
}

/// A [`Codec`] whose parameter is in the range of its variant.
///
/// [`Codec::validate`] is its only constructor, so the encoder and the decoders, which take it,
/// never see a parameter on which their arithmetic would divide by zero or overflow a shift.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize), mem_size(flat))]
pub(crate) struct ValidCodec(Codec);

// A vector's report shows its code as one line, as for `Codec`.
#[cfg(feature = "mem_dbg")]
impl mem_dbg::MemDbgImpl for ValidCodec {}

impl ValidCodec {
    /// Returns the code.
    pub(crate) fn get(self) -> Codec {
        self.0
    }

    /// Returns the dsi-bitstream code with the same codewords.
    pub(crate) fn to_codes(self) -> Codes {
        // Lossless: the parameters are `u32`, and usize is at least 32 bits wide on every
        // supported target.
        match self.0 {
            Codec::Unary => Codes::Unary,
            Codec::Gamma => Codes::Gamma,
            Codec::Delta => Codes::Delta,
            Codec::Omega => Codes::Omega,
            Codec::VByteLe => Codes::VByteLe,
            Codec::VByteBe => Codes::VByteBe,
            Codec::Rice { log2_b } => Codes::Rice(log2_b as usize),
            Codec::Golomb { b } => Codes::Golomb(b),
            Codec::ExpGolomb { k } => Codes::ExpGolomb(k as usize),
            Codec::Zeta { k } => Codes::Zeta(k as usize),
            Codec::Pi { k } => Codes::Pi(k as usize),
        }
    }
}

/// The automatic candidates, whose dsi-bitstream decoders select their implementation once.
type Stats = CodesStats<10, 10, 10, 10, 9>;

/// Returns the code with the smallest estimated payload among candidates that encode every
/// word. The estimate treats `u64::MAX` as `u64::MAX - 1`.
pub(crate) fn select(words: impl IntoIterator<Item = u64>) -> ValidCodec {
    let mut stats = Stats::default();
    let mut has_max = false;
    for word in words {
        has_max |= word == u64::MAX;
        // `CodesStats` computes `n + 1` for some codes, so it cannot take `u64::MAX`. The codes
        // that can encode `u64::MAX` take at most one bit more for it than for `u64::MAX - 1`, so
        // the totals of the codes that can be chosen are off by at most one bit per such word.
        stats.update(word.min(u64::MAX - 1));
    }
    let candidates = [
        (Codec::Unary, stats.unary),
        (Codec::Gamma, stats.gamma),
        (Codec::Delta, stats.delta),
        (Codec::Omega, stats.omega),
    ]
    .into_iter()
    .chain(
        (1..)
            .zip(stats.zeta)
            .map(|(k, bits)| (Codec::Zeta { k }, bits)),
    )
    .chain(
        (1..)
            .zip(stats.golomb)
            .map(|(b, bits)| (Codec::Golomb { b }, bits)),
    )
    .chain(
        (0..)
            .zip(stats.exp_golomb)
            .map(|(k, bits)| (Codec::ExpGolomb { k }, bits)),
    )
    .chain(
        (0..)
            .zip(stats.rice)
            .map(|(log2_b, bits)| (Codec::Rice { log2_b }, bits)),
    )
    .chain((2..).zip(stats.pi).map(|(k, bits)| (Codec::Pi { k }, bits)));
    // VByte encodes every word, and its total cannot saturate for any vector that fits in
    // memory, so the choice never falls on a code whose total saturated.
    let mut best = (Codec::VByteLe, stats.vbyte);
    for (codec, bits) in candidates {
        if bits < best.1 && (!has_max || codec.encodes_max()) {
            best = (codec, bits);
        }
    }
    best.0
        .canonical()
        .validate()
        .expect("every automatic candidate has a parameter in range")
}

/// A builder marker for automatic codec selection.
///
/// This is the default codec type parameter of [`VarVecBuilder`](crate::VarVecBuilder) and
/// [`SeqVecBuilder`](crate::SeqVecBuilder). Their `build` methods choose a code that can encode
/// every value and return the vector directly. See [`Codec`] for the selection policy.
///
/// [`codec`](crate::VarVecBuilder::codec) changes the type parameter of the builder from
/// `Automatic` to [`Codec`]. The `build` methods of that builder return a [`CodecError`] when
/// the chosen code cannot build the vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Automatic;

/// An invalid codec parameter or a value that the chosen codec cannot encode.
///
/// Returned by the builders of [`VarVec`](crate::VarVec) and [`SeqVec`](crate::SeqVec) after
/// [`codec`](crate::VarVecBuilder::codec) selects an explicit code. Automatic builders return
/// the vector directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodecError<T> {
    /// The parameter of the code is outside the range that its [`Codec`] variant documents.
    InvalidParameter(Codec),
    /// The code cannot encode `value`, whose word is `u64::MAX` (see [`Codec`]).
    Unencodable {
        /// The chosen code.
        codec: Codec,
        /// The first value that the code cannot encode.
        value: T,
    },
}

impl<T: fmt::Debug> fmt::Display for CodecError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(codec) => {
                write!(f, "the parameter of {codec:?} is out of range")
            }
            Self::Unencodable { codec, value } => {
                write!(f, "{codec:?} cannot encode the value {value:?}")
            }
        }
    }
}

impl<T: fmt::Debug> Error for CodecError<T> {}
