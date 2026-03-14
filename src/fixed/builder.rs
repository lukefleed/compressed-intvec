//! # [`FixedVec`] Builders
//!
//! This module provides builders for constructing an owned [`FixedVec`].
//!
//! There are two main builders:
//! - [`FixedVecBuilder`]: For building from a slice (`&[T]`). It can automatically
//!   determine the optimal bit width from the data.
//! - [`FixedVecFromIterBuilder`]: For building from an iterator. This is useful
//!   for large datasets, but requires the bit width to be specified manually.
//!
//! # Examples
//!
//! ## Building from a slice
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::fixed::{FixedVec, BitWidth, UFixedVec};
//! use compressed_intvec::fixed::builder::FixedVecBuilder;
//!
//! let data: &[u32] = &[10, 20, 30, 40, 50];
//!
//! // The builder can infer the minimal bit width.
//! let vec: UFixedVec<u32> = FixedVecBuilder::new()
//!     .build(data)?;
//!
//! assert_eq!(vec.bit_width(), 6); // 50 requires 6 bits
//!
//! // Or a specific strategy can be chosen.
//! let vec_pow2: UFixedVec<u32> = FixedVecBuilder::new()
//!     .bit_width(BitWidth::PowerOfTwo)
//!     .build(data)?;
//!
//! assert_eq!(vec_pow2.bit_width(), 8);
//! # Ok(())
//! # }
//! ```

use crate::fixed::traits::{Storable, Word};
use crate::fixed::{BitWidth, Error, FixedVec};
use dsi_bitstream::{
    impls::{BufBitWriter, MemWordWriterVec},
    prelude::{BitWrite, Endianness},
};
use std::marker::PhantomData;

/// A builder for creating a [`FixedVec`] from a slice of integers.
///
/// This builder analyzes a slice to determine the `bit_width` based on
/// the selected [`BitWidth`] strategy and then constructs the compressed vector.
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
    /// Creates a new builder with the default [`BitWidth::Minimal`] strategy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the strategy for determining the number of bits for each element.
    pub fn bit_width(mut self, strategy: BitWidth) -> Self {
        self.bit_width_strategy = strategy;
        self
    }

    /// Builds the [`FixedVec`] from a slice of data.
    ///
    /// This method consumes the builder and returns a new [`FixedVec`].
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if a value in the input is too large for the
    /// specified `bit_width`, or if the parameters are invalid.
    pub fn build(self, input: &[T]) -> Result<FixedVec<T, W, E, Vec<W>>, Error> {
        let bits_per_word = <W as crate::fixed::traits::Word>::BITS;

        // Determine the final bit width based on the chosen strategy.
        let final_bit_width = match self.bit_width_strategy {
            // If the user specified an exact bit width, use it.
            BitWidth::Explicit(n) => n,
            // Otherwise, analyze the data to find the maximum value.
            _ => {
                let max_val: W = input
                    .iter()
                    .map(|&val| <T as Storable<W>>::into_word(val))
                    .max()
                    .unwrap_or(W::ZERO);

                // Calculate the minimum number of bits required.
                let min_bits = (bits_per_word - max_val.leading_zeros() as usize).max(1);

                // Apply the selected strategy.
                match self.bit_width_strategy {
                    BitWidth::Minimal => min_bits,
                    BitWidth::PowerOfTwo => min_bits.next_power_of_two().min(bits_per_word),
                    BitWidth::Explicit(_) => unreachable!(), // Handled above.
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
                "bit_width ({final_bit_width}) cannot be greater than the word size ({bits_per_word})",
            )));
        }

        // Handle the empty case separately.
        if input.is_empty() {
            return Ok(unsafe { FixedVec::new_unchecked(Vec::new(), 0, final_bit_width) });
        }

        // Allocate the buffer for the bit-packed data.
        let total_bits = input.len() * final_bit_width;
        let num_words = total_bits.div_ceil(bits_per_word);
        let buffer = vec![W::ZERO; num_words + 1]; // +1 for padding.

        let mut writer = BufBitWriter::new(MemWordWriterVec::new(buffer));

        // Define the upper bound for values to check against.
        let limit = if final_bit_width < bits_per_word {
            W::ONE << final_bit_width
        } else {
            W::max_value()
        };

        // Write each value to the bitstream.
        for (i, &value_t) in input.iter().enumerate() {
            let value_w = <T as Storable<W>>::into_word(value_t);
            // Ensure the value fits within the calculated bit width.
            if final_bit_width < bits_per_word && value_w >= limit {
                return Err(Error::ValueTooLarge {
                    value: value_w.to_u128().unwrap(),
                    index: i,
                    bit_width: final_bit_width,
                });
            }
            writer
                .write_bits(value_w.to_u64().unwrap(), final_bit_width)
                .unwrap();
        }

        writer.flush().unwrap();
        let data = writer.into_inner().unwrap().into_inner();

        Ok(unsafe { FixedVec::new_unchecked(data, input.len(), final_bit_width) })
    }
}

/// A builder for creating a [`FixedVec`] from an iterator.
///
/// This builder is designed for streaming data from an iterator without first
/// collecting it into a slice. It requires the `bit_width` to be specified
/// manually, as it cannot analyze the data in advance.
#[derive(Debug)]
pub struct FixedVecFromIterBuilder<
    T: Storable<W>,
    W: Word,
    E: Endianness,
    I: IntoIterator<Item = T>,
> {
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

    /// Builds the [`FixedVec`] by consuming the iterator.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if a value from the iterator is too large for the
    /// specified `bit_width`, or if the parameters are invalid.
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

        // Write each value from the iterator to the bitstream.
        for (i, value_t) in self.iter.into_iter().enumerate() {
            let value_w = <T as Storable<W>>::into_word(value_t);
            if self.bit_width < bits_per_word && value_w >= limit {
                return Err(Error::ValueTooLarge {
                    value: value_w.to_u128().unwrap(),
                    index: i,
                    bit_width: self.bit_width,
                });
            }
            writer
                .write_bits(value_w.to_u64().unwrap(), self.bit_width)
                .unwrap();
            len += 1;
        }

        writer.flush().unwrap();
        let mut data = writer.into_inner().unwrap().into_inner();

        // Add padding word to the end of the buffer.
        if len > 0 {
            data.push(W::ZERO);
        }
        data.shrink_to_fit();

        Ok(unsafe { FixedVec::new_unchecked(data, len, self.bit_width) })
    }
}
