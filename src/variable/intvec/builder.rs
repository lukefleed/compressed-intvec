//! # `IntVec` Builders
//!
//! This module provides the builder implementations for constructing an [`IntVec`].
//! It offers two distinct builder types to accommodate different use cases:
//!
//! 1.  **[`IntVecBuilder`] (from slice `&[u64]`):**
//!     This is the recommended and most flexible builder. Since it has access
//!     to the entire dataset upfront, it can perform a pre-analysis pass to
//!     automatically determine the most space-efficient codec and parameters.
//!     This is ideal when the data fits comfortably in memory.
//!
//! 2.  **[`IntVecFromIterBuilder`] (from `Iterator<Item = u64>`):**
//!     This builder is designed for scenarios where data is generated on-the-fly
//!     or is too large to be materialized into a `Vec<u64>`. It processes the
//!     data in a streaming fashion, which is highly memory-efficient. However,
//!     this comes with a limitation: **it cannot perform automatic parameter
//!     selection**. The user must provide a fully specified [`VariableCodecSpec`].
//!
//! Both builders use a fluent API to configure parameters like the sampling
//! rate `k` and the desired [`VariableCodecSpec`].

use super::{IntVec, IntVecBitWriter, IntVecError};
use crate::fixed::intvec::{BitWidth, LEFixedVec};
use crate::variable::codec::{resolve_codec, VariableCodecSpec};
use dsi_bitstream::{
    dispatch::{FuncCodeWriter, StaticCodeWrite},
    impls::MemWordWriterVec,
    prelude::{BitWrite, CodesWrite, Endianness},
};
use std::marker::PhantomData;

/// A builder for creating an [`IntVec`] from a slice (`&[u64]`).
///
/// This builder is obtained by calling [`IntVec::builder`]. It provides a fluent
/// interface for setting parameters like the sampling rate `k` and the
/// compression `VariableCodecSpec`.
///
/// Because it operates on a slice, it can automatically select the best codec
/// parameters by analyzing the data first. This is the recommended way to construct
/// an [`IntVec`] when all data is available in memory.
/// This builder always produces a `IntVec<E, Vec<u64>>`.
#[derive(Debug)]
pub struct IntVecBuilder<'a, E: Endianness> {
    pub(super) input: &'a [u64],
    pub(super) k: usize,
    pub(super) codec_spec: VariableCodecSpec,
    pub(super) _endian: PhantomData<E>,
}

impl<'a, E: Endianness> IntVecBuilder<'a, E> {
    /// Creates a new builder for an `IntVec`.
    ///
    /// This constructor is `pub(super)`, meaning it is only visible to the parent
    /// module (`intvec`). The public entry point for creating this builder is
    /// [`IntVec::builder`].
    pub(super) fn new(input: &'a [u64]) -> Self {
        Self {
            input,
            k: 32,                               // Default sampling rate
            codec_spec: VariableCodecSpec::Auto, // Default to auto-selection
            _endian: PhantomData,
        }
    }

    /// Sets the sampling rate `k`.
    ///
    /// The sampling rate determines how frequently a sample of the bitstream's position
    /// is stored. A smaller `k` leads to faster random access but increases memory
    /// overhead. A larger `k` reduces memory usage but slows down access.
    ///
    /// The value must be greater than 0.
    ///
    /// # Arguments
    /// * `k`: The sampling rate.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the codec specification for compression.
    ///
    /// This determines which compression algorithm will be used. See [`VariableCodecSpec`]
    /// for available options. It can be a specific codec (e.g., `Gamma`) or a request
    /// for automatic selection (`Auto`).
    ///
    /// # Arguments
    /// * `codec_spec`: The desired codec specification.
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the `IntVec`, consuming the builder.
    ///
    /// This method performs the main compression logic. It resolves the `VariableCodecSpec`,
    /// automatically selecting parameters if requested (e.g., for `VariableCodecSpec::Auto` or
    /// `VariableCodecSpec::Rice { log2_b: None }`). It then encodes the input data and
    /// builds the final `IntVec` structure.
    ///
    /// # Returns
    ///
    /// A `Result` containing the constructed `IntVec<E, Vec<u64>>` on success, or an
    /// `IntVecError` if there's a problem, such as `k=0`.
    pub fn build(self) -> Result<IntVec<E, Vec<u64>>, IntVecError>
    where
        IntVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        if self.k == 0 {
            return Err(IntVecError::InvalidParameters(
                "Sampling rate k cannot be zero".to_string(),
            ));
        }

        let resolved_code = resolve_codec(self.input, self.codec_spec)?;

        if self.input.is_empty() {
            // SAFETY: An empty IntVec with empty samples is valid.
            return Ok(unsafe {
                IntVec::new_unchecked(
                    Vec::new(),
                    LEFixedVec::builder(&[0u64; 0]).build().unwrap(),
                    self.k,
                    0,
                    resolved_code,
                )
            });
        }

        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = IntVecBitWriter::<E>::new(word_writer);

        let code_writer = FuncCodeWriter::new(resolved_code)
            .map_err(|e| IntVecError::CodecDispatch(e.to_string()))?;

        let sample_capacity = (self.input.len() + self.k - 1) / self.k;
        let mut temp_samples = Vec::with_capacity(sample_capacity);
        let mut current_bit_offset = 0;

        for (i, &value) in self.input.iter().enumerate() {
            if i % self.k == 0 {
                temp_samples.push(current_bit_offset as u64);
            }
            let bits_written = code_writer.write(&mut writer, value).unwrap();
            current_bit_offset += bits_written;
        }
        writer.write_bits(u64::MAX, 64).unwrap(); // Stopper

        // Build the compressed sample vector
        let samples = LEFixedVec::builder(&temp_samples)
            .bit_width(BitWidth::Minimal)
            .build()
            .unwrap();

        writer.flush().unwrap();
        let mut data = writer.into_inner().unwrap().into_inner();
        data.shrink_to_fit();

        // SAFETY: The builder ensures all parameters are consistent.
        Ok(unsafe {
            IntVec::new_unchecked(data, samples, self.k, self.input.len(), resolved_code)
        })
    }
}

/// A builder for creating an [`IntVec`] from an iterator.
///
/// This builder is obtained by calling [`IntVec::from_iter_builder`]. It is
/// designed for scenarios where data is too large to fit in memory as a
/// `Vec<u64>` or is generated on-the-fly.
///
/// # Limitations
///
/// **This builder does not support automatic parameter selection for codecs.**
/// Because the data is processed in a stream, the builder cannot analyze it beforehand
/// to determine optimal parameters. You must provide a `VariableCodecSpec` with fixed,
/// pre-determined parameters.
#[derive(Debug)]
pub struct IntVecFromIterBuilder<E: Endianness, I: IntoIterator<Item = u64>> {
    iter: I,
    k: usize,
    codec_spec: VariableCodecSpec,
    _endian: PhantomData<E>,
}

impl<E: Endianness, I: IntoIterator<Item = u64>> IntVecFromIterBuilder<E, I> {
    /// Creates a new builder from an iterator.
    ///
    /// This constructor is `pub(super)`, making it accessible only to the parent
    /// module. The public entry point is [`IntVec::from_iter_builder`].
    pub(super) fn new(iter: I) -> Self {
        Self {
            iter,
            k: 32,
            codec_spec: VariableCodecSpec::Gamma, // Default to a safe, parameter-free codec
            _endian: PhantomData,
        }
    }

    /// Sets the sampling rate `k`.
    ///
    /// Refer to [`IntVecBuilder::k`] for more details.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the codec specification.
    ///
    /// The provided [`VariableCodecSpec`] must have all parameters explicitly defined,
    /// as automatic parameter selection is not supported for iterator-based building.
    ///
    /// # Arguments
    /// * `codec_spec`: The desired codec specification with fixed parameters.
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the `IntVec` by consuming the iterator.
    ///
    /// This method iterates through the provided data, encodes it according to the
    /// specified codec, and constructs the final `IntVec<E, Vec<u64>>`.
    ///
    /// # Returns
    ///
    /// A `Result` containing the constructed `IntVec` on success, or an
    /// `IntVecError` on failure. Errors can occur if:
    /// - An automatic or parameter-less codec spec is provided (e.g., `VariableCodecSpec::Auto`).
    /// - `k=0` is used.
    pub fn build(self) -> Result<IntVec<E, Vec<u64>>, IntVecError>
    where
        IntVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Validate that the codec spec does not require data pre-analysis.
        let resolved_code = match self.codec_spec {
            VariableCodecSpec::Auto
            | VariableCodecSpec::Rice { log2_b: None }
            | VariableCodecSpec::Zeta { k: None }
            | VariableCodecSpec::Golomb { b: None }
            | VariableCodecSpec::Pi { k: None }
            | VariableCodecSpec::ExpGolomb { k: None } => {
                return Err(IntVecError::InvalidParameters("Automatic parameter selection is not supported for iterator-based construction. Please provide fixed parameters.".to_string()));
            }
            // For other codecs, we can resolve them with an empty slice as a dummy.
            spec => resolve_codec(&[], spec)?,
        };

        if self.k == 0 {
            return Err(IntVecError::InvalidParameters(
                "Sampling rate k cannot be zero".to_string(),
            ));
        }

        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = IntVecBitWriter::<E>::new(word_writer);
        let mut len = 0;

        let code_writer = FuncCodeWriter::new(resolved_code)
            .map_err(|e| IntVecError::CodecDispatch(e.to_string()))?;
        let mut temp_samples = Vec::new();
        let mut current_bit_offset = 0;

        for (i, value) in self.iter.into_iter().enumerate() {
            if i % self.k == 0 {
                temp_samples.push(current_bit_offset as u64);
            }
            let bits_written = code_writer.write(&mut writer, value).unwrap();
            current_bit_offset += bits_written;
            len += 1;
        }
        writer.write_bits(u64::MAX, 64).unwrap(); // Stopper

        let samples = LEFixedVec::builder(&temp_samples)
            .bit_width(BitWidth::Minimal)
            .build()
            .unwrap();

        writer.flush().unwrap();
        let mut data = writer.into_inner().unwrap().into_inner();
        data.shrink_to_fit();

        // SAFETY: The builder ensures all parameters are consistent.
        Ok(unsafe { IntVec::new_unchecked(data, samples, self.k, len, resolved_code) })
    }
}