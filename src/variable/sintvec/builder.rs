//! # `SIntVec` Builders
//!
//! This module provides builders for creating a compressed signed integer vector,
//! [`SIntVec`], from slices or iterators.

use crate::prelude::SIntVec;
use crate::variable::codec::VariableCodecSpec;
use crate::variable::intvec::{IntVec, IntVecError};
use common_traits::SignedInt;
use dsi_bitstream::prelude::{BitWrite, CodesWrite, Endianness, ToNat};
use std::marker::PhantomData;

/// A builder for creating an [`SIntVec`] from a slice of signed integers.
///
/// This builder is generic over the input integer type `I` (e.g., `i8`, `i16`).
/// It handles the ZigZag transformation before passing the data to the
/// underlying [`IntVec`] builder for compression.
#[derive(Debug)]
pub struct SIntVecBuilder<'a, E: Endianness, I> {
    input: &'a [I],
    k: usize,
    codec_spec: VariableCodecSpec,
    _endian: PhantomData<E>,
    _phantom_i: PhantomData<I>,
}

impl<'a, E: Endianness, I> SIntVecBuilder<'a, E, I>
where
    I: SignedInt + ToNat + Copy,
    <I as SignedInt>::UnsignedInt: Into<u64>,
{
    /// Creates a new builder from a slice of signed integers.
    pub(super) fn new(input: &'a [I]) -> Self {
        Self {
            input,
            k: 32,
            codec_spec: VariableCodecSpec::Gamma, // A safe default
            _endian: PhantomData,
            _phantom_i: PhantomData,
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

    /// Builds the `SIntVec<E, Vec<u64>>` by transforming and compressing the input data.
    pub fn build(self) -> Result<SIntVec<E, Vec<u64>>, IntVecError>
    where
        for<'b> crate::variable::intvec::IntVecBitWriter<E>:
            BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // 1. Transform signed integers to their unsigned counterparts (e.g., i32 -> u32).
        let unsigned_iter = self.input.iter().map(|&x| x.to_nat());
        // 2. Convert the unsigned counterparts to u64, as required by the inner builder.
        let unsigned_u64_iter = unsigned_iter.map(|x| x.into());

        // Use the iterator-based builder from IntVec to perform the actual encoding.
        let inner_intvec = IntVec::<E>::from_iter_builder(unsigned_u64_iter)
            .k(self.k)
            .codec(self.codec_spec)
            .build()?;

        Ok(SIntVec {
            inner: inner_intvec,
        })
    }
}

/// A builder for creating an [`SIntVec`] from an iterator of `i64`.
///
/// # Limitations
///
/// This builder **requires** that codec parameters be specified manually.
#[derive(Debug)]
pub struct SIntVecFromIterBuilder<E: Endianness, I: IntoIterator<Item = i64>> {
    iter: I,
    k: usize,
    codec_spec: VariableCodecSpec,
    _endian: PhantomData<E>,
}

impl<E: Endianness, I: IntoIterator<Item = i64>> SIntVecFromIterBuilder<E, I> {
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

    /// Sets the codec specification for compression.
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the `SIntVec<E, Vec<u64>>` by consuming the iterator.
    pub fn build(self) -> Result<SIntVec<E, Vec<u64>>, IntVecError>
    where
        for<'b> crate::variable::intvec::IntVecBitWriter<E>:
            BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Create an iterator that applies ZigZag encoding on the fly.
        let unsigned_iter = self.iter.into_iter().map(|x| x.to_nat());

        // Delegate the actual construction to the IntVec iterator builder.
        let inner_intvec = IntVec::<E>::from_iter_builder(unsigned_iter)
            .k(self.k)
            .codec(self.codec_spec)
            .build()?;

        Ok(SIntVec {
            inner: inner_intvec,
        })
    }
}