//! # `IntVec` Builders
//!
//! This module provides the builder implementations for constructing an [`IntVec`].

use super::{IntVec, IntVecBitWriter, IntVecError};
use crate::fixed::intvec::{BitWidth, FixedVec};
use crate::variable::codec::{resolve_codec, VariableCodecSpec};
use common_traits::UnsignedInt;
use dsi_bitstream::{
    dispatch::{FuncCodeWriter, StaticCodeWrite},
    impls::MemWordWriterVec,
    prelude::{BitWrite, CodesWrite, Endianness, LE},
};
use std::marker::PhantomData;

/// A builder for creating an [`IntVec`] from a slice of integers.
///
/// This builder is obtained by calling [`IntVec::builder`]. It is generic over the
/// input integer type `U` (e.g., `u8`, `u16`, `u32`).
///
/// This builder always produces a `IntVec<E, Vec<u64>>`.
#[derive(Debug)]
pub struct IntVecBuilder<'a, E: Endianness, U> {
    pub(super) input: &'a [U],
    pub(super) k: usize,
    pub(super) codec_spec: VariableCodecSpec,
    pub(super) _endian: PhantomData<E>,
    pub(super) _phantom_u: PhantomData<U>,
}

impl<'a, E: Endianness, U> IntVecBuilder<'a, E, U>
where
    U: UnsignedInt + Into<u64> + Copy,
{
    /// Creates a new builder for an `IntVec`.
    pub(super) fn new(input: &'a [U]) -> Self {
        Self {
            input,
            k: 32,
            codec_spec: VariableCodecSpec::Auto,
            _endian: PhantomData,
            _phantom_u: PhantomData,
        }
    }

    /// Sets the sampling rate `k`.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the codec specification for compression.
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the `IntVec`, consuming the builder.
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
            let empty_samples = FixedVec::<LE>::builder(&[0u64; 0]).build().unwrap();
            // SAFETY: An empty IntVec with empty samples is valid.
            return Ok(unsafe {
                IntVec::new_unchecked(Vec::new(), empty_samples, self.k, 0, resolved_code)
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
            let bits_written = code_writer.write(&mut writer, value.into()).unwrap();
            current_bit_offset += bits_written;
        }
        writer.write_bits(u64::MAX, 64).unwrap(); // Stopper

        let samples = FixedVec::<LE>::builder(&temp_samples)
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
/// # Limitations
/// This builder does not support automatic parameter selection for codecs.
#[derive(Debug)]
pub struct IntVecFromIterBuilder<E: Endianness, I: IntoIterator<Item = u64>> {
    iter: I,
    k: usize,
    codec_spec: VariableCodecSpec,
    _endian: PhantomData<E>,
}

impl<E: Endianness, I: IntoIterator<Item = u64>> IntVecFromIterBuilder<E, I> {
    /// Creates a new builder from an iterator.
    pub(super) fn new(iter: I) -> Self {
        Self {
            iter,
            k: 32,
            codec_spec: VariableCodecSpec::Gamma,
            _endian: PhantomData,
        }
    }

    /// Sets the sampling rate `k`.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the codec specification.
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the `IntVec` by consuming the iterator.
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
            spec => resolve_codec(&[0u64; 0], spec)?,
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

        let samples = FixedVec::<LE>::builder(&temp_samples)
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