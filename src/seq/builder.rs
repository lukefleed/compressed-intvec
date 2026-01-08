//! Builders for constructing a [`SeqVec`].
//!
//! This module provides two builders for creating a [`SeqVec`]:
//!
//! - [`SeqVecBuilder`]: For building from a collection of sequences already in
//!   memory. This builder can analyze the data to automatically select an
//!   optimal compression codec.
//!
//! - [`SeqVecFromIterBuilder`]: For building from an iterator of sequences.
//!   This is suitable for large datasets generated on the fly, but requires
//!   the codec to be specified manually since single-pass construction cannot
//!   perform data analysis.
//!
//! [`SeqVec`]: crate::seq::SeqVec

use super::{SeqVec, SeqVecBitWriter, SeqVecError, VariableCodecSpec};
use crate::fixed::{BitWidth, FixedVec};
use crate::variable::codec;
use crate::variable::traits::Storable;
use dsi_bitstream::{
    codes::{
        DeltaWrite, ExpGolombWrite, GammaWrite, GolombWrite, OmegaWrite, PiWrite, RiceWrite,
        VByteBeWrite, VByteLeWrite, ZetaWrite,
    },
    impls::MemWordWriterVec,
    prelude::{BitWrite, Codes, CodesWrite, Endianness},
};
use std::marker::PhantomData;

/// A builder for creating a [`SeqVec`] from a collection of sequences.
///
/// This builder is the primary entry point for constructing a compressed
/// sequence vector when the sequences are already available in memory. It
/// allows configuration of the compression codec.
///
/// The builder always produces an owned `SeqVec<T, E, Vec<u64>>`.
///
/// ## Construction Strategy
///
/// When the codec is [`VariableCodecSpec::Auto`] or requires parameter
/// estimation (e.g., `Rice { log2_b: None }`), the builder performs a two-pass
/// construction:
///
/// 1. **Analysis pass**: Collects all elements to determine the optimal codec.
/// 2. **Encoding pass**: Compresses the data using the selected codec.
///
/// When a fully-specified codec is provided (e.g., `Gamma`, `Delta`,
/// `Zeta { k: Some(3) }`), the builder performs **single-pass construction**,
/// avoiding the temporary allocation of all elements.
///
/// ## Examples
///
/// ```
/// use compressed_intvec::seq::{SeqVec, LESeqVec, VariableCodecSpec};
///
/// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
///
/// // Automatic codec selection (two-pass)
/// let vec_auto: LESeqVec<u32> = SeqVec::builder()
///     .codec(VariableCodecSpec::Auto)
///     .build(sequences)
///     .unwrap();
///
/// // Explicit codec (single-pass, more efficient)
/// let vec_gamma: LESeqVec<u32> = SeqVec::builder()
///     .codec(VariableCodecSpec::Gamma)
///     .build(sequences)
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SeqVecBuilder<T: Storable, E: Endianness> {
    codec_spec: VariableCodecSpec,
    _markers: PhantomData<(T, E)>,
}

impl<T: Storable, E: Endianness> Default for SeqVecBuilder<T, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Storable, E: Endianness> SeqVecBuilder<T, E> {
    /// Creates a new builder with default settings.
    ///
    /// The default codec is [`VariableCodecSpec::Auto`], which analyzes the
    /// data to select the best codec.
    #[inline]
    pub fn new() -> Self {
        Self {
            codec_spec: VariableCodecSpec::Auto,
            _markers: PhantomData,
        }
    }

    /// Sets the compression codec to use.
    ///
    /// ## Codec Selection Guidelines
    ///
    /// - [`Auto`](VariableCodecSpec::Auto): Best compression ratio, but requires
    ///   two-pass construction with O(N) temporary allocation.
    /// - [`Gamma`](VariableCodecSpec::Gamma): Good general-purpose choice for
    ///   data skewed towards small values. Single-pass.
    /// - [`Delta`](VariableCodecSpec::Delta): Better than Gamma for larger values.
    ///   Single-pass.
    /// - [`Zeta { k: Some(k) }`](VariableCodecSpec::Zeta): Optimal for power-law
    ///   distributions. Single-pass when `k` is specified.
    #[inline]
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the [`SeqVec`] from a slice of sequences.
    ///
    /// Each element of `sequences` is a sequence that will be compressed and
    /// stored. Empty sequences are supported.
    ///
    /// ## Type Requirements
    ///
    /// The sequences can be any type that implements `AsRef<[T]>`, such as
    /// `&[T]`, `Vec<T>`, or `Box<[T]>`.
    ///
    /// ## Errors
    ///
    /// Returns a [`SeqVecError`] if:
    /// - Codec resolution fails.
    /// - An I/O error occurs during encoding.
    ///
    /// ## Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// // From slice of slices
    /// let data: &[&[u32]] = &[&[1, 2], &[3, 4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::builder().build(data).unwrap();
    ///
    /// // From Vec of Vecs
    /// let data: Vec<Vec<u32>> = vec![vec![1, 2], vec![3, 4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::builder().build(&data).unwrap();
    /// ```
    pub fn build<S: AsRef<[T]>>(
        self,
        sequences: &[S],
    ) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        T: 'static,
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Determine if we can use single-pass construction.
        // Single-pass is possible only when the codec is fully specified.
        if self.codec_spec.requires_analysis() {
            self.build_two_pass(sequences)
        } else {
            self.build_single_pass(sequences)
        }
    }

    /// Two-pass construction: analyze data first, then encode.
    ///
    /// Used when the codec requires data analysis (Auto, or parameter estimation).
    fn build_two_pass<S: AsRef<[T]>>(
        self,
        sequences: &[S],
    ) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        T: 'static,
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Pass 1: Collect all elements for codec analysis.
        let all_words: Vec<u64> = sequences
            .iter()
            .flat_map(|seq| seq.as_ref().iter().map(|x| x.to_word()))
            .collect();

        // Resolve the codec based on data distribution.
        let resolved_codec = codec::resolve_codec(&all_words, self.codec_spec)
            .map_err(|e| SeqVecError::CodecDispatch(e.to_string()))?;

        // Pass 2: Encode with the selected codec.
        self.encode_sequences(sequences, resolved_codec)
    }

    /// Single-pass construction: encode directly without data analysis.
    ///
    /// Used when the codec is fully specified (no Auto, no parameter estimation).
    fn build_single_pass<S: AsRef<[T]>>(
        self,
        sequences: &[S],
    ) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        T: 'static,
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Resolve the codec without data analysis. Pass an empty u64 slice
        // since fully-specified codecs do not require data.
        let resolved_codec = codec::resolve_codec::<u64>(&[], self.codec_spec)
            .map_err(|e| SeqVecError::CodecDispatch(e.to_string()))?;

        self.encode_sequences(sequences, resolved_codec)
    }

    /// Core encoding logic shared by both construction paths.
    fn encode_sequences<S: AsRef<[T]>>(
        self,
        sequences: &[S],
        resolved_codec: Codes,
    ) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        let num_sequences = sequences.len();

        // Handle empty input.
        if num_sequences == 0 {
            let empty_offsets = FixedVec::<u64, u64, E>::builder()
                .bit_width(BitWidth::Minimal)
                .build(&[0u64])?;
            return Ok(SeqVec {
                data: Vec::new(),
                bit_offsets: empty_offsets,
                encoding: resolved_codec,
                _markers: PhantomData,
            });
        }

        let (data, offsets) = encode_sequences_impl(
            sequences.iter(),
            resolved_codec,
            Vec::with_capacity(num_sequences + 1),
        )?;

        // Build the bit offsets index with minimal bit width.
        let bit_offsets = FixedVec::<u64, u64, E>::builder()
            .bit_width(BitWidth::Minimal)
            .build(&offsets)?;

        Ok(SeqVec {
            data,
            bit_offsets,
            encoding: resolved_codec,
            _markers: PhantomData,
        })
    }
}

/// A builder for creating a [`SeqVec`] from an iterator of sequences.
///
/// This builder is designed for constructing a [`SeqVec`] from a data source
/// that produces sequences on the fly. It consumes the iterator in a single
/// pass, compressing sequences as they arrive.
///
/// ## Limitations
///
/// This builder does **not** support:
/// - [`VariableCodecSpec::Auto`]: Automatic codec selection requires analyzing
///   all data, which is impossible in a single pass.
/// - Parameter estimation for codecs like `Rice { log2_b: None }` or
///   `Zeta { k: None }`.
///
/// The codec must be fully specified. If an unsupported codec is provided,
/// the [`build`](Self::build) method will return an error.
///
/// ## Examples
///
/// ```
/// use compressed_intvec::seq::{SeqVec, LESeqVec, VariableCodecSpec};
///
/// // Generate sequences on the fly
/// let sequences_iter = (0..100).map(|i| vec![i as u32, i as u32 + 1]);
///
/// let vec: LESeqVec<u32> = SeqVec::from_iter_builder(sequences_iter)
///     .codec(VariableCodecSpec::Gamma) // Must be specified
///     .build()
///     .unwrap();
///
/// assert_eq!(vec.num_sequences(), 100);
/// ```
#[derive(Debug)]
pub struct SeqVecFromIterBuilder<T: Storable, E: Endianness, I> {
    iter: I,
    codec_spec: VariableCodecSpec,
    _markers: PhantomData<(T, E)>,
}

impl<T, E, I, S> SeqVecFromIterBuilder<T, E, I>
where
    T: Storable,
    E: Endianness,
    I: IntoIterator<Item = S>,
    S: AsRef<[T]>,
{
    /// Creates a new builder from an iterator with default settings.
    ///
    /// The default codec is [`VariableCodecSpec::Gamma`], as automatic
    /// selection is not possible in single-pass construction.
    #[inline]
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            codec_spec: VariableCodecSpec::Gamma,
            _markers: PhantomData,
        }
    }

    /// Sets the compression codec to use.
    ///
    /// The codec must be fully specified (no `Auto`, no `None` parameters).
    ///
    /// ## Errors
    ///
    /// The [`build`](Self::build) method will return an error if a codec
    /// requiring data analysis is provided.
    #[inline]
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the [`SeqVec`] by consuming the iterator.
    ///
    /// ## Errors
    ///
    /// Returns a [`SeqVecError`] if:
    /// - An automatic or parameter-estimating codec spec is used.
    /// - An I/O error occurs during encoding.
    ///
    /// ## Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec, VariableCodecSpec};
    ///
    /// let sequences: Vec<Vec<u32>> = vec![vec![1, 2], vec![3, 4, 5]];
    ///
    /// let vec: LESeqVec<u32> = SeqVec::from_iter_builder(sequences.into_iter())
    ///     .codec(VariableCodecSpec::Delta)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn build(self) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        T: 'static,
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Reject codecs that require data analysis.
        if self.codec_spec.requires_analysis() {
            return Err(SeqVecError::InvalidParameters(
                "Automatic codec selection is not supported for iterator-based construction. \
                 Please provide a fully-specified codec (e.g., Gamma, Delta, Zeta { k: Some(3) })."
                    .to_string(),
            ));
        }

        // Resolve the codec without data analysis. Pass an empty u64 slice
        // since fully-specified codecs do not require data.
        let resolved_codec = codec::resolve_codec::<u64>(&[], self.codec_spec)
            .map_err(|e| SeqVecError::CodecDispatch(e.to_string()))?;

        let iter = self.iter.into_iter();
        // Use size_hint to pre-allocate offsets for efficiency.
        let (lower, _) = iter.size_hint();
        let offsets = Vec::with_capacity(lower.saturating_add(1));

        let (data, offsets) = encode_sequences_impl(iter, resolved_codec, offsets)?;

        // Handle empty iterator case.
        if offsets.is_empty() {
            let empty_offsets = FixedVec::<u64, u64, E>::builder()
                .bit_width(BitWidth::Minimal)
                .build(&[0u64])?;
            return Ok(SeqVec {
                data: Vec::new(),
                bit_offsets: empty_offsets,
                encoding: resolved_codec,
                _markers: PhantomData,
            });
        }

        // Build the bit offsets index.
        let bit_offsets = FixedVec::<u64, u64, E>::builder()
            .bit_width(BitWidth::Minimal)
            .build(&offsets)?;

        Ok(SeqVec {
            data,
            bit_offsets,
            encoding: resolved_codec,
            _markers: PhantomData,
        })
    }
}

/// Extension trait for `VariableCodecSpec` to check if analysis is required.
trait CodecSpecExt {
    /// Returns `true` if this codec spec requires data analysis to resolve.
    fn requires_analysis(&self) -> bool;
}

impl CodecSpecExt for VariableCodecSpec {
    #[inline]
    fn requires_analysis(&self) -> bool {
        matches!(
            self,
            VariableCodecSpec::Auto
                | VariableCodecSpec::Rice { log2_b: None }
                | VariableCodecSpec::Zeta { k: None }
                | VariableCodecSpec::Golomb { b: None }
        )
    }
}

/// Shared implementation for encoding sequences from an iterator.
///
/// This function encodes all sequences using a single resolved codec and
/// pre-allocated offsets vector. It resolves the codec dispatch once at the
/// beginning (via `CodecWriter`) rather than per-element, improving throughput.
///
/// Returns the encoded data (word vector) and bit offset boundaries.
fn encode_sequences_impl<T: Storable, E: Endianness, I, S>(
    sequences: I,
    resolved_codec: Codes,
    mut offsets: Vec<u64>,
) -> Result<(Vec<u64>, Vec<u64>), SeqVecError>
where
    E: Endianness,
    I: IntoIterator<Item = S>,
    S: AsRef<[T]>,
    SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
{
    // Initialize the bit writer.
    let word_writer = MemWordWriterVec::new(Vec::new());
    let mut writer = SeqVecBitWriter::<E>::new(word_writer);
    let mut current_bit_offset: u64 = 0;

    // Process each sequence, recording bit offsets at boundaries.
    for seq in sequences {
        offsets.push(current_bit_offset);

        for elem in seq.as_ref() {
            let bits_written = write_code_value(&mut writer, elem.to_word(), resolved_codec)?;
            current_bit_offset += bits_written as u64;
        }
    }

    // Sentinel: total bit length.
    offsets.push(current_bit_offset);

    // Finalize the writer.
    writer.flush()?;
    let mut data = writer.into_inner()?.into_inner();
    data.shrink_to_fit();

    Ok((data, offsets))
}

/// Writes a single value using the specified codec.
///
/// This function is kept for potential future use. For bulk encoding,
/// `resolve_write_fn` is more efficient as it avoids repeated dispatch.
/// Returns the number of bits written.
#[inline]
fn write_code_value<E: Endianness>(
    writer: &mut SeqVecBitWriter<E>,
    value: u64,
    code: Codes,
) -> Result<usize, SeqVecError>
where
    SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
{
    let bits = match code {
        Codes::Unary => writer.write_unary(value)?,
        Codes::Gamma => writer.write_gamma(value)?,
        Codes::Delta => writer.write_delta(value)?,
        Codes::Omega => writer.write_omega(value)?,
        Codes::VByteLe => writer.write_vbyte_le(value)?,
        Codes::VByteBe => writer.write_vbyte_be(value)?,
        Codes::Zeta { k } => writer.write_zeta(value, k)?,
        Codes::Rice { log2_b } => writer.write_rice(value, log2_b)?,
        Codes::Golomb { b } => writer.write_golomb(value, b as u64)?,
        Codes::ExpGolomb { k } => writer.write_exp_golomb(value, k)?,
        Codes::Pi { k } => writer.write_pi(value, k)?,
        _ => {
            return Err(SeqVecError::CodecDispatch(format!(
                "Unsupported codec for writing: {:?}",
                code
            )));
        }
    };
    Ok(bits)
}

// --- Integration with SeqVec ---

impl<T: Storable + 'static, E: Endianness> SeqVec<T, E, Vec<u64>> {
    /// Creates a builder for constructing a [`SeqVec`] with custom settings.
    ///
    /// This is the most flexible way to create a [`SeqVec`], allowing
    /// customization of the compression codec.
    ///
    /// ## Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec, VariableCodecSpec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20]];
    ///
    /// let vec: LESeqVec<u32> = SeqVec::builder()
    ///     .codec(VariableCodecSpec::Zeta { k: Some(3) })
    ///     .build(sequences)
    ///     .unwrap();
    /// ```
    #[inline]
    pub fn builder() -> SeqVecBuilder<T, E> {
        SeqVecBuilder::new()
    }

    /// Creates a builder for constructing a [`SeqVec`] from an iterator.
    ///
    /// This is useful for large datasets that are generated on the fly.
    /// The codec must be specified explicitly since single-pass construction
    /// cannot perform data analysis.
    ///
    /// ## Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec, VariableCodecSpec};
    ///
    /// // Generate sequences programmatically
    /// let sequences = (0..50).map(|i| vec![i as u32; i % 5 + 1]);
    ///
    /// let vec: LESeqVec<u32> = SeqVec::from_iter_builder(sequences)
    ///     .codec(VariableCodecSpec::Gamma)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(vec.num_sequences(), 50);
    /// ```
    #[inline]
    pub fn from_iter_builder<I, S>(iter: I) -> SeqVecFromIterBuilder<T, E, I>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[T]>,
    {
        SeqVecFromIterBuilder::new(iter)
    }

    /// Creates a `SeqVec` from raw parts without validation.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it does not validate that the `data` and
    /// `bit_offsets` are consistent with each other. The caller must ensure:
    /// - The `bit_offsets` array has at least 2 elements (start and end sentinel).
    /// - All offsets are valid bit positions within the `data` buffer.
    /// - The last offset equals the total number of bits in the compressed data.
    ///
    /// # Arguments
    ///
    /// * `data` - The compressed data buffer.
    /// * `bit_offsets` - The bit offset index for each sequence.
    /// * `encoding` - The codec used to encode the data.
    #[inline]
    pub unsafe fn from_raw_parts(
        data: Vec<u64>,
        bit_offsets: crate::fixed::FixedVec<u64, u64, E, Vec<u64>>,
        encoding: dsi_bitstream::prelude::Codes,
    ) -> Self {
        SeqVec {
            data,
            bit_offsets,
            encoding,
            _markers: PhantomData,
        }
    }
}
