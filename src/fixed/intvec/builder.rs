//! # `FixedVec` Builders
//!
//! This module provides builders for creating a compressed fixed-width integer
//! vector, [`FixedVec`].

use super::{BitWidth, FixedVec, FixedVecError};
use dsi_bitstream::{
    impls::MemWordWriterVec,
    prelude::{BitWrite, BufBitWriter, Endianness},
};
use std::marker::PhantomData;

/// Type alias for the writer used internally by `FixedVec`.
pub type FixedVecBitWriter<E> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;

/// A builder for creating an owned [`FixedVec`] from a slice of integers.
///
/// The builder is generic over the input integer type `U` (e.g., `u8`, `u16`,
/// `u32`, `u64`), allowing for direct construction without intermediate allocations.
/// It is highly efficient, as it pre-allocates the exact amount of
/// memory required for the final compressed vector, avoiding reallocations.
///
/// This builder always produces a `FixedVec<E, Vec<u64>>`.
#[derive(Debug)]
pub struct FixedVecBuilder<'a, E: Endianness, U> {
    input: &'a [U],
    bit_width: BitWidth,
    _endian: PhantomData<E>,
}

impl<'a, E: Endianness, U> FixedVecBuilder<'a, E, U>
where
    U: Into<u64> + Ord + Copy + Default,
{
    /// Creates a new builder from a slice of integers.
    ///
    /// This is `pub(super)` and is called by [`FixedVec::builder`].
    pub(super) fn new(input: &'a [U]) -> Self {
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

    /// Builds the `FixedVec<E, Vec<u64>>`, consuming the builder.
    pub fn build(self) -> Result<FixedVec<E, Vec<u64>>, FixedVecError>
    where
        FixedVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible>,
    {
        let final_num_bits = match self.bit_width {
            BitWidth::Explicit(n) => n,
            BitWidth::Minimal | BitWidth::ByteAligned => {
                let max_val: u64 = self.input.iter().max().copied().unwrap_or_default().into();
                let min_bits = if max_val == 0 {
                    1
                } else {
                    (u64::BITS - max_val.leading_zeros()) as usize
                };

                if let BitWidth::ByteAligned = self.bit_width {
                    // Round up to the nearest multiple of 8, capped at 64.
                    ((min_bits + 7) / 8 * 8).min(64)
                } else {
                    min_bits
                }
            }
        };

        if !self.input.is_empty() && final_num_bits == 0 {
            return Err(FixedVecError::InvalidParameters(
                "num_bits cannot be zero for a non-empty vector".to_string(),
            ));
        }

        if final_num_bits > 64 {
            return Err(FixedVecError::InvalidParameters(
                "num_bits cannot be greater than 64".to_string(),
            ));
        }

        if self.input.is_empty() {
            // SAFETY: An empty vector with 0 length is always valid.
            return Ok(unsafe { FixedVec::new_unchecked(Vec::new(), 0, final_num_bits) });
        }

        // Pre-allocate the exact number of u64 words needed, plus one for padding.
        // The padding word prevents out-of-bounds reads when an element spans
        // the last word boundary.
        let total_bits = self.input.len() * final_num_bits;
        let num_words = (total_bits + 63) / 64;
        let buffer = vec![0u64; num_words + 1];
        let mut writer = FixedVecBitWriter::<E>::new(MemWordWriterVec::new(buffer));

        let limit = if final_num_bits < 64 {
            1u64 << final_num_bits
        } else {
            u64::MAX
        };

        for (i, &value_u) in self.input.iter().enumerate() {
            let value: u64 = value_u.into();
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

        // SAFETY: The builder correctly constructs the vector with the necessary padding.
        Ok(unsafe { FixedVec::new_unchecked(data, self.input.len(), final_num_bits) })
    }
}

/// A builder for creating an owned [`FixedVec`] from an iterator.
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

    /// Builds the `FixedVec<E, Vec<u64>>` by consuming the iterator.
    pub fn build(self) -> Result<FixedVec<E, Vec<u64>>, FixedVecError>
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
        // Add a padding word to prevent out-of-bounds reads when an element
        // spans the last word boundary.
        data.push(0);
        data.shrink_to_fit();

        // SAFETY: The builder correctly constructs the vector with the necessary padding.
        Ok(unsafe { FixedVec::new_unchecked(data, len, self.num_bits) })
    }
}
