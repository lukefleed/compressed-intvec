//! # `SFixedVec` Builders
//!
//! This module provides builders for creating a compressed fixed-width signed
//! integer vector, [`SFixedVec`].

use dsi_bitstream::prelude::{BitWrite, Endianness, ToNat};
use std::marker::PhantomData;

use crate::fixed::{
    intvec::{builder::FixedVecBitWriter, FixedVec, FixedVecError},
    sintvec::SFixedVec,
};
use common_traits::SignedInt;

/// A builder for creating a [`SFixedVec`] from a slice of `i64`.
///
/// This builder transparently handles the ZigZag encoding of the input data.
/// It first transforms the `&[i64]` slice into a temporary `Vec<u64>` and then
/// delegates the construction to the underlying [`FixedVecBuilder`].
///
/// Like `FixedVecBuilder`, it can automatically determine the optimal number of
/// bits if not specified.
#[derive(Debug)]
pub struct SFixedVecBuilder<'a, E: Endianness, I> {
    input: &'a [I],
    num_bits: Option<usize>,
    _endian: PhantomData<E>,
}

impl<'a, E: Endianness, I> SFixedVecBuilder<'a, E, I>
where
    I: ToNat + Copy + SignedInt,
    <I as SignedInt>::UnsignedInt: Into<u64> + Ord + Copy + Default,
{
    /// Creates a new builder from a slice of signed integers.
    pub(super) fn new(input: &'a [I]) -> Self {
        Self {
            input,
            num_bits: None,
            _endian: PhantomData,
        }
    }

    /// Sets the number of bits to use for encoding each integer.
    ///
    /// If `Some(bits)`, the specified number of bits will be used.
    /// If `None`, the builder will automatically determine the minimum number of
    /// bits required to store the largest ZigZag-encoded value.
    pub fn num_bits(mut self, num_bits: Option<usize>) -> Self {
        self.num_bits = num_bits;
        self
    }

    /// Builds the `SFixedVec`, consuming the builder.
    pub fn build(self) -> Result<SFixedVec<E>, FixedVecError>
    where
        FixedVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible>,
    {
        // Transform the signed integers to unsigned integers using ZigZag encoding,
        // and convert them to u64 for the underlying builder.
        let unsigned_data: Vec<u64> = self.input.iter().map(|&x| x.to_nat().into()).collect();

        // Delegate the actual construction to the FixedVec builder.
        let inner_fixed_vec = FixedVec::<E>::builder(&unsigned_data)
            .num_bits(self.num_bits)
            .build()?;

        Ok(SFixedVec {
            inner: inner_fixed_vec,
        })
    }
}

/// A builder for creating a [`SFixedVec`] from an iterator of `i64`.
///
/// # Limitations
///
/// This builder **requires** that the number of bits be specified manually, as
/// it cannot pre-analyze the data from a stream.
#[derive(Debug)]
pub struct SFixedVecFromIterBuilder<E: Endianness, I: IntoIterator<Item = i64>> {
    iter: I,
    num_bits: usize,
    _endian: PhantomData<E>,
}

impl<E: Endianness, I: IntoIterator<Item = i64>> SFixedVecFromIterBuilder<E, I> {
    /// Creates a new builder from an iterator and a specified number of bits.
    pub(super) fn new(iter: I, num_bits: usize) -> Self {
        Self {
            iter,
            num_bits,
            _endian: PhantomData,
        }
    }

    /// Builds the `SFixedVec` by consuming the iterator.
    pub fn build(self) -> Result<SFixedVec<E>, FixedVecError>
    where
        FixedVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible>,
    {
        // Create an iterator that applies ZigZag encoding on the fly.
        let unsigned_iter = self.iter.into_iter().map(|x| x.to_nat());

        // Delegate the actual construction to the FixedVec iterator builder.
        let inner_fixed_vec =
            FixedVec::<E>::from_iter_builder(unsigned_iter, self.num_bits).build()?;

        Ok(SFixedVec {
            inner: inner_fixed_vec,
        })
    }
}