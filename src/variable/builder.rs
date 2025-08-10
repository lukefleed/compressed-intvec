//! Builders for constructing an [`IntVec`].
//!
//! This module provides the two primary builders for creating an [`IntVec`]:
//!
//! - [`IntVecBuilder`]: For building from an existing slice of data. This is the
//!   most flexible builder, as it can analyze the data to automatically select
//!   an optimal compression codec.
//! - [`IntVecFromIterBuilder`]: For building from an iterator. This is suitable
//!   for large datasets that are generated on the fly, but it requires the
//!   compression codec to be specified manually.
//!
//! [`IntVec`]: crate::variable::IntVec

use super::{codec, traits::Storable, IntVec, IntVecBitWriter, IntVecError, VariableCodecSpec};
use crate::fixed::{BitWidth, FixedVec};
use dsi_bitstream::{
    codes::{
        DeltaWrite, ExpGolombWrite, GammaWrite, GolombWrite, OmegaWrite, PiWrite, RiceWrite,
        VByteBeWrite, VByteLeWrite, ZetaWrite,
    },
    impls::MemWordWriterVec,
    prelude::{BitWrite, Codes, CodesWrite, Endianness, LE},
};
use std::marker::PhantomData;

/// A builder for creating an [`IntVec`] from a slice of integers.
///
/// This builder is the primary entry point for constructing a compressed vector
/// when the data is already available in memory. It allows for detailed
/// configuration of the sampling rate (`k`) and the compression codec.
///
/// This builder always produces an owned `IntVec<T, E, Vec<u64>>`. It is obtained
/// by calling [`IntVec::builder`].
#[derive(Debug)]
pub struct IntVecBuilder<'a, T: Storable, E: Endianness> {
    input: &'a [T],
    k: usize,
    codec_spec: VariableCodecSpec,
    _markers: PhantomData<(T, E)>,
}

impl<'a, T: Storable, E: Endianness> IntVecBuilder<'a, T, E> {
    /// Creates a new builder for an `IntVec` with default settings.
    ///
    /// By default, the sampling rate is `k=32` and the codec is chosen
    /// automatically via [`VariableCodecSpec::Auto`].
    pub(super) fn new(input: &'a [T]) -> Self {
        Self {
            input,
            k: 32,
            codec_spec: VariableCodecSpec::Auto,
            _markers: PhantomData,
        }
    }

    /// Sets the sampling rate `k` for random access.
    ///
    /// The sampling rate determines how many elements are stored between each
    /// sample point. A smaller `k` results in faster random access but uses
    /// more memory for the sampling table. See the [module-level documentation](super)
    /// for a detailed explanation.
    ///
    /// # Panics
    ///
    /// The [`build`](IntVecBuilder::build) method will return an error if `k` is 0.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the compression codec to use.
    ///
    /// The choice of codec can significantly impact the compression ratio.
    /// By default, this is [`VariableCodecSpec::Auto`], which analyzes the data
    /// to select the best codec.
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the [`IntVec`], consuming the builder.
    ///
    /// This method performs the compression and builds the sampling table.
    ///
    /// # Errors
    ///
    /// Returns an [`IntVecError`] if the parameters are invalid (e.g., `k=0`) or
    /// if an error occurs during compression.
    ///
    /// # Examples
    ///
    /// ```    /// use compressed_intvec::variable::{IntVec, SIntVec, VariableCodecSpec};
    ///
    /// let data: &[i16] = &[-100, 0, 50, -2, 1000];
    ///
    /// let vec: SIntVec<i16> = IntVec::builder(data)
    ///     .k(2) // Smaller `k` for faster access
    ///     .codec(VariableCodecSpec::Delta) // Explicitly choose Delta coding
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(vec.len(), 5);
    /// assert_eq!(vec.get(0), Some(-100));
    /// ```
    pub fn build(self) -> Result<IntVec<T, E, Vec<u64>>, IntVecError>
    where
        IntVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        if self.k == 0 {
            return Err(IntVecError::InvalidParameters(
                "Sampling rate k cannot be zero".to_string(),
            ));
        }

        // Convert the input data to a vector of u64 words for analysis and compression.
        let words: Vec<u64> = self.input.iter().map(|&x| x.to_word()).collect();
        let resolved_code = codec::resolve_codec(&words, self.codec_spec)?;

        if self.input.is_empty() {
            let empty_samples = FixedVec::<u64, u64, LE>::builder()
                .build(&[0u64; 0])
                .unwrap();
            return Ok(unsafe {
                IntVec::new_unchecked(Vec::new(), empty_samples, self.k, 0, resolved_code)
            });
        }

        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = IntVecBitWriter::<E>::new(word_writer);

        let sample_capacity = self.input.len().div_ceil(self.k);
        let mut temp_samples = Vec::with_capacity(sample_capacity);
        let mut current_bit_offset = 0;

        // Iterate through the data, writing compressed values and recording samples.
        for (i, &value) in self.input.iter().enumerate() {
            if i % self.k == 0 {
                temp_samples.push(current_bit_offset as u64);
            }

            // Use our own dispatcher to call the appropriate write method.
            // This avoids the limitations of dsi-bitstream's FuncCodeWriter.
            let bits_written = match resolved_code {
                Codes::Gamma => writer.write_gamma(value.to_word()).unwrap(),
                Codes::Delta => writer.write_delta(value.to_word()).unwrap(),
                Codes::Zeta { k } => writer.write_zeta(value.to_word(), k).unwrap(),
                Codes::Golomb { b } => writer.write_golomb(value.to_word(), b as u64).unwrap(),
                Codes::Rice { log2_b } => writer.write_rice(value.to_word(), log2_b).unwrap(),
                Codes::Unary => writer.write_unary(value.to_word()).unwrap(),
                Codes::Omega => writer.write_omega(value.to_word()).unwrap(),
                Codes::Pi { k } => writer.write_pi(value.to_word(), k).unwrap(),
                Codes::ExpGolomb { k } => writer.write_exp_golomb(value.to_word(), k).unwrap(),
                Codes::VByteLe => writer.write_vbyte_le(value.to_word()).unwrap(),
                Codes::VByteBe => writer.write_vbyte_be(value.to_word()).unwrap(),
                _ => {
                    return Err(IntVecError::InvalidParameters(
                        "The specified codec is not supported for slice-based construction."
                            .to_string(),
                    ));
                }
            };
            current_bit_offset += bits_written;
        }
        // Write a final stopper to ensure the last value can always be read safely.
        writer.write_bits(u64::MAX, 64).unwrap();

        // Compress the recorded samples into a FixedVec.
        let samples = FixedVec::<u64, u64, LE>::builder()
            .bit_width(BitWidth::Minimal)
            .build(&temp_samples)
            .unwrap();

        writer.flush().unwrap();
        let mut data = writer.into_inner().unwrap().into_inner();
        data.shrink_to_fit();

        Ok(
            unsafe {
                IntVec::new_unchecked(data, samples, self.k, self.input.len(), resolved_code)
            },
        )
    }
}

/// A builder for creating an [`IntVec`] from an iterator.
///
/// This builder is designed for constructing an `IntVec` from a data source that
/// is an iterator. It consumes the iterator and compresses its elements on the fly.
/// It is obtained by calling [`IntVec::from_iter_builder`].
///
/// # Limitations
///
/// This builder does **not** support automatic codec selection (i.e., [`VariableCodecSpec::Auto`])
/// or automatic parameter estimation for codecs like `Rice` or `Golomb`. Since the
/// iterator is consumed in a single pass, the data cannot be pre-analyzed to
/// determine its statistical properties. The user must specify a concrete codec.
#[derive(Debug)]
pub struct IntVecFromIterBuilder<T: Storable, E: Endianness, I: IntoIterator<Item = T>> {
    iter: I,
    k: usize,
    codec_spec: VariableCodecSpec,
    _markers: PhantomData<(T, E)>,
}

impl<T: Storable, E: Endianness, I: IntoIterator<Item = T>> IntVecFromIterBuilder<T, E, I> {
    /// Creates a new builder from an iterator with default settings.
    ///
    /// By default, the sampling rate is `k=32` and the codec is [`VariableCodecSpec::Gamma`],
    /// as automatic selection is not possible.
    pub(super) fn new(iter: I) -> Self {
        Self {
            iter,
            k: 32,
            codec_spec: VariableCodecSpec::Gamma,
            _markers: PhantomData,
        }
    }

    /// Sets the sampling rate `k` for random access.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the compression codec to use.
    ///
    /// # Errors
    ///
    /// The `build` method will return an error if a codec specification that
    /// requires data analysis is provided (e.g., [`VariableCodecSpec::Auto`]).
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the [`IntVec`] by consuming the iterator.
    ///
    /// This method iterates through the provided data source, compresses it,
    /// and builds the sampling table in a single pass.
    ///
    /// # Errors
    ///
    /// Returns an [`IntVecError`] if an automatic codec spec is used or if `k` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::variable::{IntVec, UIntVec, VariableCodecSpec};
    ///
    /// // Create a vector from a range iterator
    /// let data_iter = 0..1000u32;
    ///
    /// let vec: UIntVec<u32> = IntVec::from_iter_builder(data_iter)
    ///     .k(64)
    ///     .codec(VariableCodecSpec::Gamma) // Must be specified
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(vec.len(), 1000);
    /// assert_eq!(vec.get(999), Some(999));
    /// ```
    pub fn build(self) -> Result<IntVec<T, E, Vec<u64>>, IntVecError>
    where
        IntVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Resolve the codec, but return an error if it requires data analysis.
        let resolved_code = match self.codec_spec {
            VariableCodecSpec::Auto
            | VariableCodecSpec::Rice { log2_b: None }
            | VariableCodecSpec::Zeta { k: None }
            | VariableCodecSpec::Golomb { b: None } => {
                return Err(IntVecError::InvalidParameters("Automatic parameter selection is not supported for iterator-based construction. Please provide fixed parameters.".to_string()));
            }
            // Pass an empty slice for validation; the parameters are explicit.
            spec => codec::resolve_codec(&[0u64; 0], spec)?,
        };

        if self.k == 0 {
            return Err(IntVecError::InvalidParameters(
                "Sampling rate k cannot be zero".to_string(),
            ));
        }

        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = IntVecBitWriter::<E>::new(word_writer);
        let mut len = 0;

        let mut temp_samples = Vec::new();
        let mut current_bit_offset = 0;

        for (i, value) in self.iter.into_iter().enumerate() {
            if i % self.k == 0 {
                temp_samples.push(current_bit_offset as u64);
            }

            // Use our own dispatcher to call the appropriate write method.
            let bits_written = match resolved_code {
                Codes::Gamma => writer.write_gamma(value.to_word()).unwrap(),
                Codes::Delta => writer.write_delta(value.to_word()).unwrap(),
                Codes::Zeta { k } => writer.write_zeta(value.to_word(), k).unwrap(),
                Codes::Golomb { b } => writer.write_golomb(value.to_word(), b as u64).unwrap(),
                Codes::Rice { log2_b } => writer.write_rice(value.to_word(), log2_b).unwrap(),
                Codes::Unary => writer.write_unary(value.to_word()).unwrap(),
                Codes::Omega => writer.write_omega(value.to_word()).unwrap(),
                Codes::Pi { k } => writer.write_pi(value.to_word(), k).unwrap(),
                Codes::ExpGolomb { k } => writer.write_exp_golomb(value.to_word(), k).unwrap(),
                Codes::VByteLe => writer.write_vbyte_le(value.to_word()).unwrap(),
                Codes::VByteBe => writer.write_vbyte_be(value.to_word()).unwrap(),
                _ => {
                    return Err(IntVecError::InvalidParameters(
                        "The specified codec is not supported for iterator-based construction."
                            .to_string(),
                    ));
                }
            };
            current_bit_offset += bits_written;
            len += 1;
        }
        writer.write_bits(u64::MAX, 64).unwrap(); // Stopper

        let samples = FixedVec::<u64, u64, LE>::builder()
            .bit_width(BitWidth::Minimal)
            .build(&temp_samples)
            .unwrap();

        writer.flush().unwrap();
        let mut data = writer.into_inner().unwrap().into_inner();
        data.shrink_to_fit();

        Ok(unsafe { IntVec::new_unchecked(data, samples, self.k, len, resolved_code) })
    }
}
