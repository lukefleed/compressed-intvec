//! # `FixedVec` Builders
//!
//! This module provides builders for creating a compressed fixed-width integer
//! vector, [`FixedVec`].

use super::{FixedVec, FixedVecError};
use dsi_bitstream::{
    impls::MemWordWriterVec,
    prelude::{BitWrite, BufBitWriter, Endianness},
};
use std::marker::PhantomData;

/// Type alias for the writer used internally by `FixedVec`.
pub type FixedVecBitWriter<E> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;

/// A builder for creating a [`FixedVec`] from a slice of `u64`.
///
/// The builder is highly efficient, as it pre-allocates the exact amount of
/// memory required for the final compressed vector, avoiding reallocations.
#[derive(Debug)]
pub struct FixedVecBuilder<'a, E: Endianness> {
    input: &'a [u64],
    num_bits: Option<usize>,
    _endian: PhantomData<E>,
}

impl<'a, E: Endianness> FixedVecBuilder<'a, E> {
    /// Creates a new builder from a slice of `u64`.
    ///
    /// This is `pub(super)` and is called by [`FixedVec::builder`].
    pub(super) fn new(input: &'a [u64]) -> Self {
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
    /// bits required to store the largest value in the input data.
    pub fn num_bits(mut self, num_bits: Option<usize>) -> Self {
        self.num_bits = num_bits;
        self
    }

    /// Builds the `FixedVec`, consuming the builder.
    pub fn build(self) -> Result<FixedVec<E>, FixedVecError>
    where
        FixedVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible>,
    {
        let final_num_bits = self.num_bits.unwrap_or_else(|| {
            let max_val = self.input.iter().max().copied().unwrap_or(0);
            if max_val == 0 {
                1
            } else {
                (u64::BITS - max_val.leading_zeros()) as usize
            }
        });

        if final_num_bits > 64 {
            return Err(FixedVecError::InvalidParameters(
                "num_bits cannot be greater than 64".to_string(),
            ));
        }

        if self.input.is_empty() {
            return Ok(FixedVec {
                data: Vec::new(),
                len: 0,
                num_bits: final_num_bits,
                _endian: PhantomData,
            });
        }

        // Pre-allocate the exact number of u64 words needed.
        let total_bits = self.input.len() * final_num_bits;
        let num_words = (total_bits + 63) / 64;
        let buffer = vec![0u64; num_words];
        let mut writer = FixedVecBitWriter::<E>::new(MemWordWriterVec::new(buffer));

        let limit = if final_num_bits < 64 {
            1u64 << final_num_bits
        } else {
            u64::MAX
        };

        for (i, &value) in self.input.iter().enumerate() {
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

        Ok(FixedVec {
            data,
            len: self.input.len(),
            num_bits: final_num_bits,
            _endian: PhantomData,
        })
    }
}

/// A builder for creating a [`FixedVec`] from an iterator.
///
/// # Limitations
///
/// This builder **requires** that the number of bits be specified manually, as
/// it cannot pre-analyze the data from a stream.
#[derive(Debug)]
pub struct FixedVecFromIterBuilder<E: Endianness, I: IntoIterator<Item = u64>> {
    iter: I,
    num_bits: usize,
    _endian: PhantomData<E>,
}

impl<E: Endianness, I: IntoIterator<Item = u64>> FixedVecFromIterBuilder<E, I> {
    /// Creates a new builder from an iterator and a specified number of bits.
    ///
    /// This is `pub(super)` and is called by [`FixedVec::from_iter_builder`].
    pub(super) fn new(iter: I, num_bits: usize) -> Self {
        Self {
            iter,
            num_bits,
            _endian: PhantomData,
        }
    }

    /// Builds the `FixedVec` by consuming the iterator.
    pub fn build(self) -> Result<FixedVec<E>, FixedVecError>
    where
        FixedVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible>,
    {
        if self.num_bits > 64 {
            return Err(FixedVecError::InvalidParameters(
                "num_bits cannot be greater than 64".to_string(),
            ));
        }

        let mut writer = FixedVecBitWriter::<E>::new(MemWordWriterVec::new(Vec::new()));
        let mut len = 0;
        let limit = if self.num_bits < 64 {
            1u64 << self.num_bits
        } else {
            u64::MAX
        };

        for (i, value) in self.iter.into_iter().enumerate() {
            if self.num_bits < 64 && value >= limit {
                return Err(FixedVecError::ValueTooLarge {
                    value,
                    index: i,
                    num_bits: self.num_bits,
                });
            }
            writer.write_bits(value, self.num_bits).unwrap();
            len += 1;
        }

        writer.flush().unwrap();
        let mut data = writer.into_inner().unwrap().into_inner();
        data.shrink_to_fit();

        Ok(FixedVec {
            data,
            len,
            num_bits: self.num_bits,
            _endian: PhantomData,
        })
    }
}
