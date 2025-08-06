//! # `FixedVec` Builder
//!
//! This module provides the generic builder for creating an owned, compressed
//! fixed-width integer vector, [`FixedVec`].

use crate::fixed::{BitWidth, Error, FixedVec};
use crate::fixed::traits::{Storable, Word};
use dsi_bitstream::{
    impls::{BufBitWriter, MemWordWriterVec},
    prelude::{BitWrite, Endianness},
};
use std::marker::PhantomData;

/// A builder for creating an owned [`FixedVec<T, W, E, Vec<W>>`] from a slice of integers.
///
/// The builder is designed to be ergonomic and efficient, inferring necessary
/// parameters from the data when possible and pre-allocating memory to avoid
/// reallocations.
#[derive(Debug, Clone)]
pub struct FixedVecBuilder<T: Storable<W>, W: Word, E: Endianness> {
    bit_width_strategy: BitWidth,
    _phantom: PhantomData<(T, W, E)>,
}

impl<T, W, E> Default for FixedVecBuilder<T, W, E>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
{
    fn default() -> Self {
        Self {
            bit_width_strategy: BitWidth::default(),
            _phantom: PhantomData,
        }
    }
}

impl<T, W, E> FixedVecBuilder<T, W, E>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    BufBitWriter<E, MemWordWriterVec<W, Vec<W>>>: BitWrite<E, Error = std::convert::Infallible>,
{
    /// Creates a new builder with the default `BitWidth::Minimal` strategy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the strategy for determining the number of bits for encoding each element.
    pub fn bit_width(mut self, strategy: BitWidth) -> Self {
        self.bit_width_strategy = strategy;
        self
    }

    /// Builds the `FixedVec` from a slice of data, consuming the builder.
    pub fn build(self, input: &[T]) -> Result<FixedVec<T, W, E, Vec<W>>, Error> {
        let bits_per_word = <W as crate::fixed::traits::Word>::BITS;

        let final_bit_width = match self.bit_width_strategy {
            BitWidth::Explicit(n) => n,
            _ => {
                let max_val: W = input
                    .iter()
                    .map(|&val| <T as Storable<W>>::into_word(val))
                    .max()
                    .unwrap_or(W::ZERO);

                let min_bits = if max_val == W::ZERO {
                    1
                } else {
                    bits_per_word - max_val.leading_zeros() as usize
                };

                match self.bit_width_strategy {
                    BitWidth::Minimal => min_bits,
                    BitWidth::PowerOfTwo => min_bits.next_power_of_two().min(bits_per_word),
                    BitWidth::Explicit(_) => unreachable!(),
                }
            }
        };

        if !input.is_empty() && final_bit_width == 0 {
            return Err(Error::InvalidParameters(
                "bit_width cannot be zero for a non-empty vector".to_string(),
            ));
        }

        if final_bit_width > bits_per_word {
            return Err(Error::InvalidParameters(format!(
                "bit_width ({}) cannot be greater than the word size ({})",
                final_bit_width, bits_per_word
            )));
        }

        if input.is_empty() {
            return Ok(unsafe { FixedVec::new_unchecked(Vec::new(), 0, final_bit_width) });
        }

        let total_bits = input.len() * final_bit_width;
        let num_words = total_bits.div_ceil(bits_per_word);
        let buffer = vec![W::ZERO; num_words + 1];

        let mut writer = BufBitWriter::new(MemWordWriterVec::new(buffer));

        let limit = if final_bit_width < bits_per_word {
            W::ONE << final_bit_width
        } else {
            W::max_value()
        };

        for (i, &value_t) in input.iter().enumerate() {
            let value_w = <T as Storable<W>>::into_word(value_t);
            if final_bit_width < bits_per_word && value_w >= limit {
                return Err(Error::ValueTooLarge {
                    value: value_w.to_u128().unwrap(),
                    index: i,
                    bit_width: final_bit_width,
                });
            }
            // Convert W to u64 before writing, as required by the BitWrite trait.
            writer.write_bits(value_w.to_u64().unwrap(), final_bit_width).unwrap();
        }

        writer.flush().unwrap();
        let data = writer.into_inner().unwrap().into_inner();

        Ok(unsafe { FixedVec::new_unchecked(data, input.len(), final_bit_width) })
    }
}

/// A builder for creating an owned `FixedVec` from an iterator.
#[derive(Debug)]
pub struct FixedVecFromIterBuilder<T: Storable<W>, W: Word, E: Endianness, I: IntoIterator<Item = T>> {
    iter: I,
    bit_width: usize,
    _phantom: PhantomData<(T, W, E)>,
}

impl<T, W, E, I> FixedVecFromIterBuilder<T, W, E, I>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    I: IntoIterator<Item = T>,
    BufBitWriter<E, MemWordWriterVec<W, Vec<W>>>: BitWrite<E, Error = std::convert::Infallible>,
{
    /// Creates a new builder from an iterator and a specified `bit_width`.
    pub fn new(iter: I, bit_width: usize) -> Self {
        Self {
            iter,
            bit_width,
            _phantom: PhantomData,
        }
    }

    /// Builds the `FixedVec` by consuming the iterator.
    pub fn build(self) -> Result<FixedVec<T, W, E, Vec<W>>, Error> {
        let bits_per_word = <W as crate::fixed::traits::Word>::BITS;
        if self.bit_width > bits_per_word {
            return Err(Error::InvalidParameters(format!(
                "bit_width ({}) cannot be greater than the word size ({})",
                self.bit_width, bits_per_word
            )));
        }

        let mut writer = BufBitWriter::new(MemWordWriterVec::new(Vec::<W>::new()));
        
        let mut len = 0;
        let limit = if self.bit_width < bits_per_word {
            W::ONE << self.bit_width
        } else {
            W::max_value()
        };

        for (i, value_t) in self.iter.into_iter().enumerate() {
            let value_w = <T as Storable<W>>::into_word(value_t);
            if self.bit_width < bits_per_word && value_w >= limit {
                return Err(Error::ValueTooLarge {
                    value: value_w.to_u128().unwrap(),
                    index: i,
                    bit_width: self.bit_width,
                });
            }
            // Convert W to u64 before writing.
            writer.write_bits(value_w.to_u64().unwrap(), self.bit_width).unwrap();
            len += 1;
        }

        writer.flush().unwrap();
        let mut data = writer.into_inner().unwrap().into_inner();
        if len > 0 {
            data.push(W::ZERO);
        }
        data.shrink_to_fit();

        Ok(unsafe { FixedVec::new_unchecked(data, len, self.bit_width) })
    }
}