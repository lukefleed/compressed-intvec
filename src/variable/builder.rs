//! Builders for constructing an [`VarVec`].
//!
//! This module provides the two primary builders for creating an [`VarVec`]:
//!
//! - [`VarVecBuilder`]: For building from an existing slice of data. This is the
//!   most flexible builder, as it can analyze the data to automatically select
//!   an optimal compression codec.
//! - [`VarVecFromIterBuilder`]: For building from an iterator. This is suitable
//!   for large datasets that are generated on the fly, but it requires the
//!   compression codec to be specified manually.
//!
//! [`VarVec`]: crate::variable::VarVec

use super::{codec, traits::Storable, VarVec, VarVecBitWriter, VarVecError, VariableCodecSpec};
use crate::common::codec_writer::CodecWriter;
use crate::fixed::{BitWidth, FixedVec};
use dsi_bitstream::{
    dispatch::StaticCodeWrite,
    impls::MemWordWriterVec,
    prelude::{BitWrite, CodesWrite, Endianness, LE},
};
use std::marker::PhantomData;

/// A builder for creating an [`VarVec`] from a slice of integers.
///
/// This builder is the primary entry point for constructing a compressed vector
/// when the data is already available in memory. It allows for detailed
/// configuration of the sampling rate (`k`) and the compression codec.
///
/// This builder always produces an owned `VarVec<T, E, Vec<u64>>`. It is obtained
/// by calling [`VarVec::builder`].
#[derive(Debug)]
pub struct VarVecBuilder<T: Storable, E: Endianness> {
    k: usize,
    codec_spec: VariableCodecSpec,
    _markers: PhantomData<(T, E)>,
}

impl<T: Storable, E: Endianness> VarVecBuilder<T, E> {
    /// Creates a new builder for an `VarVec` with default settings.
    ///
    /// By default, the sampling rate is `k=32` and the codec is chosen
    /// automatically via [`VariableCodecSpec::Auto`].
    pub(super) fn new() -> Self {
        Self {
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
    /// The [`build`](VarVecBuilder::build) method will return an error if `k` is 0.
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

    /// Builds the [`VarVec`] from a slice of data, consuming the builder.
    ///
    /// This method performs the compression and builds the sampling table.
    ///
    /// # Errors
    ///
    /// Returns an [`VarVecError`] if the parameters are invalid (e.g., `k=0`) or
    /// if an error occurs during compression.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::variable::{VarVec, SVarVec, VariableCodecSpec};
    ///
    /// let data: &[i16] = &[-100, 0, 50, -2, 1000];
    ///
    /// let vec: SVarVec<i16> = VarVec::builder()
    ///     .k(2) // Smaller `k` for faster access
    ///     .codec(VariableCodecSpec::Delta) // Explicitly choose Delta coding
    ///     .build(data)
    ///     .unwrap();
    ///
    /// assert_eq!(vec.len(), 5);
    /// assert_eq!(vec.get(0), Some(-100));
    /// ```
    pub fn build(self, input: &[T]) -> Result<VarVec<T, E, Vec<u64>>, VarVecError>
    where
        VarVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        if self.k == 0 {
            return Err(VarVecError::InvalidParameters(
                "Sampling rate k cannot be zero".to_string(),
            ));
        }

        // Resolve codec: only iterate for analysis when necessary.
        let resolved_code = if self.codec_spec.requires_analysis() {
            // Analysis needed: iterate input once, convert on-the-fly.
            codec::resolve_codec_from_iter(input.iter().map(|&x| x.to_word()), self.codec_spec)?
        } else {
            // No analysis needed: resolve directly without data access.
            codec::resolve_codec(&[] as &[u64], self.codec_spec)?
        };

        if input.is_empty() {
            let empty_samples = FixedVec::<u64, u64, LE>::builder()
                .build(&[0u64; 0])
                .unwrap();
            return Ok(unsafe {
                VarVec::new_unchecked(Vec::new(), empty_samples, self.k, 0, resolved_code)
            });
        }

        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = VarVecBitWriter::<E>::new(word_writer);

        let sample_capacity = input.len().div_ceil(self.k);
        let mut temp_samples = Vec::with_capacity(sample_capacity);
        let mut current_bit_offset = 0;

        // Resolve the codec dispatch ONCE at the beginning.
        // This eliminates per-element match overhead for common codecs.
        let code_writer = CodecWriter::new(resolved_code);

        // Iterate through the data, writing compressed values and recording samples.
        for (i, &value) in input.iter().enumerate() {
            if i % self.k == 0 {
                temp_samples.push(current_bit_offset as u64);
            }

            let bits_written = code_writer.write(&mut writer, value.to_word())?;
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

        Ok(unsafe { VarVec::new_unchecked(data, samples, self.k, input.len(), resolved_code) })
    }
}

/// A builder for creating an [`VarVec`] from an iterator.
///
/// This builder is designed for constructing an [`VarVec`] from a data source that
/// is an iterator. It consumes the iterator and compresses its elements on the fly.
/// It is obtained by calling [`VarVec::from_iter_builder`].
///
/// # Limitations
///
/// This builder does **not** support automatic codec selection (i.e., [`VariableCodecSpec::Auto`])
/// or automatic parameter estimation for codecs like [`Rice`](VariableCodecSpec::Rice) or [`Golomb`](VariableCodecSpec::Golomb). Since the
/// iterator is consumed in a single pass, the data cannot be pre-analyzed to
/// determine its statistical properties. The user must specify a concrete codec.
#[derive(Debug)]
pub struct VarVecFromIterBuilder<T: Storable, E: Endianness, I: IntoIterator<Item = T>> {
    iter: I,
    k: usize,
    codec_spec: VariableCodecSpec,
    _markers: PhantomData<(T, E)>,
}

impl<T: Storable, E: Endianness, I: IntoIterator<Item = T>> VarVecFromIterBuilder<T, E, I> {
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
    /// The [`build`](Self::build) method will return an error if a codec specification that
    /// requires data analysis is provided (e.g., [`VariableCodecSpec::Auto`]).
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the [`VarVec`] by consuming the iterator.
    ///
    /// This method iterates through the provided data source, compresses it,
    /// and builds the sampling table in a single pass.
    ///
    /// # Errors
    ///
    /// Returns an [`VarVecError`] if an automatic codec spec is used or if `k` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::variable::{VarVec, UVarVec, VariableCodecSpec};
    ///
    /// // Create a vector from a range iterator
    /// let data_iter = 0..1000u32;
    ///
    /// let vec: UVarVec<u32> = VarVec::from_iter_builder(data_iter)
    ///     .k(64)
    ///     .codec(VariableCodecSpec::Gamma) // Must be specified
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(vec.len(), 1000);
    /// assert_eq!(vec.get(999), Some(999));
    /// ```
    pub fn build(self) -> Result<VarVec<T, E, Vec<u64>>, VarVecError>
    where
        VarVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Resolve the codec, but return an error if it requires data analysis.
        let resolved_code = match self.codec_spec {
            VariableCodecSpec::Auto
            | VariableCodecSpec::Rice { log2_b: None }
            | VariableCodecSpec::Zeta { k: None }
            | VariableCodecSpec::Golomb { b: None } => {
                return Err(VarVecError::InvalidParameters("Automatic parameter selection is not supported for iterator-based construction. Please provide fixed parameters.".to_string()));
            }
            // Pass an empty slice for validation; the parameters are explicit.
            spec => codec::resolve_codec(&[0u64; 0], spec)?,
        };

        if self.k == 0 {
            return Err(VarVecError::InvalidParameters(
                "Sampling rate k cannot be zero".to_string(),
            ));
        }

        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = VarVecBitWriter::<E>::new(word_writer);
        let mut len = 0;

        let mut temp_samples = Vec::new();
        let mut current_bit_offset = 0;

        // Resolve the codec dispatch ONCE at the beginning.
        let code_writer = CodecWriter::new(resolved_code);

        for (i, value) in self.iter.into_iter().enumerate() {
            if i % self.k == 0 {
                temp_samples.push(current_bit_offset as u64);
            }

            let bits_written = code_writer.write(&mut writer, value.to_word())?;
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

        Ok(unsafe { VarVec::new_unchecked(data, samples, self.k, len, resolved_code) })
    }
}
