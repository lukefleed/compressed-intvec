//! # `IntVec` Builders
//!
//! This module provides the builder implementations for constructing a generic [`IntVec`].

use super::{traits::Storable, IntVec, IntVecBitWriter, IntVecError, VariableCodecSpec, codec};
use crate::fixed::{BitWidth, FixedVec};
use dsi_bitstream::{
    dispatch::{FuncCodeWriter, StaticCodeWrite},
    impls::MemWordWriterVec,
    prelude::{BitWrite, CodesWrite, Endianness, LE},
};
use std::marker::PhantomData;

/// A builder for creating an [`IntVec<T, E, ...>`] from a slice of integers.
///
/// This builder is obtained by calling [`IntVec::builder`]. It is generic over the
/// element type `T` (e.g., `u8`, `i32`) and the endianness `E`.
///
/// This builder always produces an owned `IntVec<T, E, Vec<u64>>`.
#[derive(Debug)]
pub struct IntVecBuilder<'a, T: Storable, E: Endianness> {
    pub(super) input: &'a [T],
    pub(super) k: usize,
    pub(super) codec_spec: VariableCodecSpec,
    pub(super) _markers: PhantomData<(T, E)>,
}

impl<'a, T: Storable, E: Endianness> IntVecBuilder<'a, T, E> {
    /// Creates a new builder for an `IntVec`.
    pub(super) fn new(input: &'a [T]) -> Self {
        Self {
            input,
            k: 32,
            codec_spec: VariableCodecSpec::Auto,
            _markers: PhantomData,
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
    pub fn build(self) -> Result<IntVec<T, E, Vec<u64>>, IntVecError>
    where
        IntVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        if self.k == 0 {
            return Err(IntVecError::InvalidParameters(
                "Sampling rate k cannot be zero".to_string(),
            ));
        }

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

        let code_writer = FuncCodeWriter::new(resolved_code)
            .map_err(|e| IntVecError::CodecDispatch(e.to_string()))?;

        let sample_capacity = self.input.len().div_ceil(self.k);
        let mut temp_samples = Vec::with_capacity(sample_capacity);
        let mut current_bit_offset = 0;

        for (i, &value) in self.input.iter().enumerate() {
            if i % self.k == 0 {
                temp_samples.push(current_bit_offset as u64);
            }
            let bits_written = code_writer.write(&mut writer, value.to_word()).unwrap();
            current_bit_offset += bits_written;
        }
        writer.write_bits(u64::MAX, 64).unwrap(); // Stopper

        let samples = FixedVec::<u64, u64, LE>::builder()
            .bit_width(BitWidth::Minimal)
            .build(&temp_samples)
            .unwrap();

        writer.flush().unwrap();
        let mut data = writer.into_inner().unwrap().into_inner();
        data.shrink_to_fit();

        Ok(unsafe {
            IntVec::new_unchecked(data, samples, self.k, self.input.len(), resolved_code)
        })
    }
}

/// A builder for creating an [`IntVec`] from an iterator.
///
/// # Limitations
/// This builder does not support automatic parameter selection for codecs
/// (e.g., `VariableCodecSpec::Auto`). The iterator is consumed once to build the
/// vector, so the data cannot be pre-analyzed. If you need automatic codec
/// selection, collect your data into a `Vec` and use [`IntVec::builder`] instead.
#[derive(Debug)]
pub struct IntVecFromIterBuilder<T: Storable, E: Endianness, I: IntoIterator<Item = T>> {
    iter: I,
    k: usize,
    codec_spec: VariableCodecSpec,
    _markers: PhantomData<(T, E)>,
}

impl<T: Storable, E: Endianness, I: IntoIterator<Item = T>> IntVecFromIterBuilder<T, E, I> {
    /// Creates a new builder from an iterator.
    pub(super) fn new(iter: I) -> Self {
        Self {
            iter,
            k: 32,
            codec_spec: VariableCodecSpec::Gamma,
            _markers: PhantomData,
        }
    }

    /// Sets the sampling rate `k`.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the codec specification.
    ///
    /// # Panics
    /// This method will panic if an automatic codec specification is provided,
    /// as iterators cannot be pre-analyzed.
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the `IntVec` by consuming the iterator.
    pub fn build(self) -> Result<IntVec<T, E, Vec<u64>>, IntVecError>
    where
        IntVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        let resolved_code = match self.codec_spec {
            VariableCodecSpec::Auto
            | VariableCodecSpec::Rice { log2_b: None }
            | VariableCodecSpec::Zeta { k: None }
            | VariableCodecSpec::Golomb { b: None }
            | VariableCodecSpec::Pi { k: None }
            | VariableCodecSpec::ExpGolomb { k: None } => {
                return Err(IntVecError::InvalidParameters("Automatic parameter selection is not supported for iterator-based construction. Please provide fixed parameters.".to_string()));
            }
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

        let code_writer = FuncCodeWriter::new(resolved_code)
            .map_err(|e| IntVecError::CodecDispatch(e.to_string()))?;
        let mut temp_samples = Vec::new();
        let mut current_bit_offset = 0;

        for (i, value) in self.iter.into_iter().enumerate() {
            if i % self.k == 0 {
                temp_samples.push(current_bit_offset as u64);
            }
            let bits_written = code_writer.write(&mut writer, value.to_word()).unwrap();
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