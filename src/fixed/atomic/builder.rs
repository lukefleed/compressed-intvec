//! # [`AtomicFixedVec`] Builder
//!
//! This module provides a builder for constructing an owned [`AtomicFixedVec`]
//! from a slice of data (`&[T]`). It mirrors the functionality of the standard
//! [`FixedVecBuilder`](crate::fixed::builder::FixedVecBuilder), allowing for
//! automatic bit-width detection and a consistent API.
//!
//! # Examples
//!
//! ## Building from a slice
//!
//! ```rust
//! use compressed_intvec::prelude::*;
//! use compressed_intvec::fixed::{AtomicFixedVec, UAtomicFixedVec, BitWidth};
//!
//! let data: &[u32] = &[10, 20, 30, 40, 50];
//!
//! // The builder can infer the minimal bit width (6 bits for 50).
//! let vec: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
//!     .build(data)
//!     .unwrap();
//!
//! assert_eq!(vec.bit_width(), 6);
//!
//! // Or a specific strategy can be chosen.
//! let vec_pow2: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
//!     .bit_width(BitWidth::PowerOfTwo)
//!     .build(data)
//!     .unwrap();
//!
//! assert_eq!(vec_pow2.bit_width(), 8);
//! ```

use crate::fixed::atomic::AtomicFixedVec;
use crate::fixed::traits::Storable;
use crate::fixed::{BitWidth, Error};
use num_traits::ToPrimitive;
use std::marker::PhantomData;
use std::sync::atomic::AtomicU64;

/// A builder for creating an [`AtomicFixedVec`] from a slice of integers.
///
/// This builder analyzes a slice to determine the `bit_width` based on
/// the selected [`BitWidth`] strategy and then constructs the thread-safe vector.
///
/// # Example
///
/// ```
/// use compressed_intvec::prelude::*;
/// use compressed_intvec::fixed::{AtomicFixedVec, UAtomicFixedVec, BitWidth};
///
/// let data: &[u32] = &[10, 20, 30, 40, 50];
///
/// // The builder can infer the minimal bit width (6 bits for 50).
/// let vec: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
///     .build(data)
///     .unwrap();
///
/// assert_eq!(vec.bit_width(), 6);
///
/// // Or a specific strategy can be chosen.
/// let vec_pow2: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
///     .bit_width(BitWidth::PowerOfTwo)
///     .build(data)
///     .unwrap();
///
/// assert_eq!(vec_pow2.bit_width(), 8);
///
/// // You can also specify an explicit bit width.
/// let vec_explicit: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
///    .bit_width(BitWidth::Explicit(10))
///    .build(data)
///    .unwrap();
///
/// assert_eq!(vec_explicit.bit_width(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct AtomicFixedVecBuilder<T: Storable<u64>> {
    bit_width_strategy: BitWidth,
    _phantom: PhantomData<T>,
}

impl<T> Default for AtomicFixedVecBuilder<T>
where
    T: Storable<u64>,
{
    fn default() -> Self {
        Self {
            bit_width_strategy: BitWidth::default(),
            _phantom: PhantomData,
        }
    }
}

impl<T> AtomicFixedVecBuilder<T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    /// Creates a new builder with the default [`BitWidth::Minimal`] strategy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the strategy for determining the number of bits for each element.
    ///
    /// This can be one of:
    /// - [`BitWidth::Minimal`]: Automatically determines the minimal bit width needed.
    /// - [`BitWidth::PowerOfTwo`]: Rounds up to the next power of two.
    /// - [`BitWidth::Explicit`]: Uses the specified number of bits explicitly.
    ///
    /// # Note
    ///
    /// Choosing [`BitWidth::Minimal`] and [`BitWidth::PowerOfTwo`] introduces a one-time overhead
    /// at construction time to find the maximum value in the input slice.
    ///
    /// # Performance
    ///
    /// Choosing [`BitWidth::PowerOfTwo`] allows for true atomic operations on the vector, resulting in
    /// better performance for concurrent access patterns. However, it may use more space than necessary.
    pub fn bit_width(mut self, strategy: BitWidth) -> Self {
        self.bit_width_strategy = strategy;
        self
    }

    /// Builds the [`AtomicFixedVec`] from a slice of data.
    ///
    /// This method consumes the builder and returns a new [`AtomicFixedVec`].
    /// The initial data is populated using non-atomic writes, as the vector
    /// is not yet shared during construction.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if a value in the input is too large for the
    /// specified `bit_width`, or if the parameters are invalid.
    pub fn build(self, input: &[T]) -> Result<AtomicFixedVec<T>, Error> {
        let bits_per_word = u64::BITS as usize;

        // Determine the final bit width based on the chosen strategy.
        let final_bit_width = match self.bit_width_strategy {
            BitWidth::Explicit(n) => n,
            _ => {
                let max_val: u64 = input
                    .iter()
                    .map(|&val| T::into_word(val))
                    .max()
                    .unwrap_or(0);

                let min_bits = if max_val == 0 {
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

        // Create a new, zero-initialized AtomicFixedVec.
        let mut atomic_vec = AtomicFixedVec::new(final_bit_width, input.len())?;

        if input.is_empty() {
            return Ok(atomic_vec);
        }

        // --- Initial data population using non-atomic writes ---
        let limit = if final_bit_width < bits_per_word {
            1u64 << final_bit_width
        } else {
            u64::MAX
        };

        for (i, &value_t) in input.iter().enumerate() {
            let value_w = T::into_word(value_t);
            if final_bit_width < bits_per_word && value_w >= limit {
                return Err(Error::ValueTooLarge {
                    value: value_w as u128,
                    index: i,
                    bit_width: final_bit_width,
                });
            }
            // SAFETY: We are writing within the allocated bounds. The vector is
            // not shared at this point, so non-atomic writes are safe and correct.
            unsafe {
                set_unchecked_non_atomic(
                    &mut atomic_vec.storage,
                    i,
                    value_w,
                    final_bit_width,
                    atomic_vec.mask,
                );
            }
        }

        Ok(atomic_vec)
    }
}

/// A non-atomic, unsafe helper to set a value in a slice of [`AtomicU64`].
/// This is only used during the initial construction of the vector.
///
/// # Safety
/// The caller must ensure that `index` is within bounds and that `value_w`
/// fits within the configured `bit_width`.
unsafe fn set_unchecked_non_atomic(
    limbs: &mut [AtomicU64],
    index: usize,
    value: u64,
    bit_width: usize,
    mask: u64,
) {
    let bits_per_word = u64::BITS as usize;
    let bit_pos = index * bit_width;
    let word_index = bit_pos / bits_per_word;
    let bit_offset = bit_pos % bits_per_word;

    // Use `get_mut()` to get a mutable reference to the underlying `u64`.
    // This is safe because we have exclusive access to the vector during building.
    if bit_offset + bit_width <= bits_per_word {
        let word = unsafe { limbs.get_unchecked_mut(word_index).get_mut() };
        *word &= !(mask << bit_offset);
        *word |= value << bit_offset;
    } else {
        // The value spans two words.
        let low_word_ptr = unsafe { limbs.as_mut_ptr().add(word_index) };
        let high_word_ptr = unsafe { limbs.as_mut_ptr().add(word_index + 1) };

        let low_word = unsafe { (*low_word_ptr).get_mut() };
        let high_word = unsafe { (*high_word_ptr).get_mut() };

        *low_word &= !(u64::MAX << bit_offset);
        *low_word |= value << bit_offset;

        let bits_in_high = (bit_offset + bit_width) - bits_per_word;
        let high_mask = (1u64 << bits_in_high).wrapping_sub(1);
        *high_word &= !high_mask;
        *high_word |= value >> (bits_per_word - bit_offset);
    }
}
