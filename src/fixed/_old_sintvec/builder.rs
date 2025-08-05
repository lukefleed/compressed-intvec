//! # `SFixedVec` Builders
//!
//! This module provides builders for creating a compressed fixed-width signed
//! integer vector, [`SFixedVec`].

use crate::fixed::intvec::{BitWidth, FixedVec, FixedVecError};
use crate::fixed::sintvec::SFixedVec;
use common_traits::SignedInt;
use dsi_bitstream::prelude::{BitWrite, Endianness, ToNat};
use std::marker::PhantomData;

use crate::fixed::intvec::builder::FixedVecBitWriter;
use dsi_bitstream::impls::MemWordWriterVec;

/// A builder for creating an owned [`SFixedVec`] from a slice of signed integers.
///
/// This builder transparently handles the ZigZag encoding of the input data.
/// It can automatically determine the optimal number of bits if not specified.
///
/// This builder always produces a `SFixedVec<E, Vec<u64>>`.
#[derive(Debug)]
pub struct SFixedVecBuilder<'a, E: Endianness, I> {
    input: &'a [I],
    bit_width: BitWidth,
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
            bit_width: BitWidth::default(),
            _endian: PhantomData,
        }
    }

    /// Sets the strategy for determining the number of bits for encoding each integer.
    ///
    /// See [`BitWidth`] for available strategies.
    pub fn bit_width(mut self, bit_width: BitWidth) -> Self {
        self.bit_width = bit_width;
        self
    }

    /// Builds the `SFixedVec<E, Vec<u64>>`, consuming the builder.
    ///
    /// This implementation is optimized to avoid intermediate allocations. It makes
    /// two passes over the input data: one to determine the maximum ZigZag-encoded
    /// value (for strategies other than `Explicit`), and a second pass to write the bits.
    pub fn build(self) -> Result<SFixedVec<E, Vec<u64>>, FixedVecError>
    where
        FixedVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible>,
    {
        // Determine the final number of bits based on the selected strategy.
        let final_num_bits = match self.bit_width {
            BitWidth::Explicit(n) => n,
            // For other strategies, we first need to calculate the minimal bits required
            // after ZigZag encoding.
            _ => {
                // First pass: find the max ZigZag value without allocating a new Vec.
                let max_val: u64 = self
                    .input
                    .iter()
                    .map(|&x| x.to_nat().into())
                    .max()
                    .unwrap_or(0);

                let min_bits = if max_val == 0 {
                    1
                } else {
                    (u64::BITS - max_val.leading_zeros()) as usize
                };

                // Apply the selected rounding strategy.
                match self.bit_width {
                    BitWidth::Minimal => min_bits,
                    BitWidth::PowerOfTwo => min_bits.next_power_of_two().min(64),
                    BitWidth::Explicit(_) => unreachable!(),
                }
            }
        };

        if final_num_bits > 64 {
            return Err(FixedVecError::InvalidParameters(
                "num_bits cannot be greater than 64".to_string(),
            ));
        }

        if self.input.is_empty() {
            // SAFETY: An empty vector with 0 length is always valid.
            let inner = unsafe { FixedVec::new_unchecked(Vec::new(), 0, final_num_bits) };
            return Ok(SFixedVec { inner });
        }

        let total_bits = self.input.len() * final_num_bits;
        let num_words = (total_bits + 63) / 64;
        let buffer = vec![0u64; num_words + 1];
        let mut writer = FixedVecBitWriter::<E>::new(MemWordWriterVec::new(buffer));

        let limit = if final_num_bits < 64 {
            1u64 << final_num_bits
        } else {
            u64::MAX
        };

        // Second pass: write the ZigZag-encoded values to the bitstream.
        for (i, &value_i) in self.input.iter().enumerate() {
            let value: u64 = value_i.to_nat().into();
            if final_num_bits < 64 && value >= limit {
                return Err(FixedVecError::ValueTooLarge {
                    value,
                    index: i,
                    num_bits: final_num_bits,
                });
            }
            writer.write_bits(value, final_num_bits).unwrap();
        }

        writer.flush().unwrap();
        let data = writer.into_inner().unwrap().into_inner();

        // SAFETY: The builder correctly constructs the inner vector with necessary padding.
        let inner_fixed_vec =
            unsafe { FixedVec::new_unchecked(data, self.input.len(), final_num_bits) };

        Ok(SFixedVec {
            inner: inner_fixed_vec,
        })
    }
}

/// A builder for creating an owned [`SFixedVec`] from an iterator of `i64`.
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

    /// Builds the `SFixedVec<E, Vec<u64>>` by consuming the iterator.
    pub fn build(self) -> Result<SFixedVec<E, Vec<u64>>, FixedVecError>
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
