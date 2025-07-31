//! # A compressed, randomly accessible vector of `u64` integers with fixed-width encoding.
//!
//! This module provides [`FixedVec`], a data structure optimized for space-efficient
//! storage and O(1) random access for integer sequences where all values fit
//! within a known, fixed number of bits.
//!
//! ## Core Functionality
//!
//! - **Optimal Compression for Uniform Data**: Uses the same number of bits for
//!   every integer, which is the most space-efficient strategy for data that is
//!   uniformly distributed within a known range.
//! - **O(1) Random Access**: The position of any element can be calculated
//!   arithmetically (`index * num_bits`), providing the fastest possible random access.
//! - **Flexible Construction**: Provides a builder API that can determine the
//!   optimal number of bits automatically from a slice of data, or build from an
//!   iterator with a specified bit width.
//!
//! The main struct, [`FixedVec`], is generic over [`Endianness`], allowing
//! users to choose between Little-Endian ([`LEFixedVec`]) and Big-Endian ([`BEFixedVec`])
//! representations.

pub mod builder;
pub mod iter;
#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use builder::{FixedVecBuilder, FixedVecFromIterBuilder};
pub use iter::FixedVecIter;

use dsi_bitstream::prelude::{Endianness, BE, LE};
use mem_dbg::{MemDbg, MemSize};
use std::{any::TypeId, error::Error, fmt, marker::PhantomData};

/// Defines the set of errors that can occur in `FixedVec` operations.
#[derive(Debug)]
pub enum FixedVecError {
    /// An error indicating that a value in the input data does not fit within
    /// the specified number of bits.
    ValueTooLarge {
        value: u64,
        index: usize,
        num_bits: usize,
    },
    /// An error indicating that the provided parameters are invalid for the
    /// requested operation.
    InvalidParameters(String),
    /// An error indicating that a requested index is outside the valid bounds
    /// of the vector.
    IndexOutOfBounds(usize),
}

impl fmt::Display for FixedVecError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FixedVecError::ValueTooLarge {
                value,
                index,
                num_bits,
            } => write!(
                f,
                "value {} at index {} does not fit in {} bits",
                value, index, num_bits
            ),
            FixedVecError::InvalidParameters(s) => write!(f, "Invalid parameters: {}", s),
            FixedVecError::IndexOutOfBounds(index) => write!(f, "Index out of bounds: {}", index),
        }
    }
}

impl Error for FixedVecError {}

/// A compressed, randomly accessible vector of `u64` integers with fixed-width encoding.
///
/// `FixedVec` is optimized for data that is uniformly distributed. It encodes
/// every integer using the same number of bits, which allows for O(1) random
/// access by arithmetically calculating the position of any element.
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FixedVec<E: Endianness> {
    pub(super) data: Vec<u64>,
    pub(super) len: usize,
    pub(super) num_bits: usize,
    /// A mask with the lowest `num_bits` bits set to one.
    pub(super) mask: u64,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(super) _endian: PhantomData<E>,
}
impl<E: Endianness> FixedVec<E> {
    /// Returns a builder for creating a [`FixedVec`] from a slice of data.
    pub fn builder<T: AsRef<[u64]> + ?Sized>(input: &T) -> FixedVecBuilder<E> {
        FixedVecBuilder::new(input.as_ref())
    }

    /// Returns a builder for creating a [`FixedVec`] from an iterator.
    pub fn from_iter_builder<I: IntoIterator<Item = u64>>(
        iter: I,
        num_bits: usize,
    ) -> FixedVecFromIterBuilder<E, I> {
        FixedVecFromIterBuilder::new(iter, num_bits)
    }

    /// Returns the number of integers in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of bits used to encode each integer.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Returns a clone of the underlying storage (`Vec<u64>`).
    pub fn limbs(&self) -> Vec<u64> {
        self.data.clone()
    }
}

impl<E: Endianness> FixedVec<E> {
    /// Retrieves the element at the specified index. Access is O(1).
    #[inline]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }
        // SAFETY: The bounds check has been performed.
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Retrieves the element at the specified index without bounds checking.
    ///
    /// This is a high-performance, low-level implementation that operates
    /// directly on the underlying `u64` slice. It avoids the overhead of any
    /// bitstream reader abstractions and handles endianness correctly.
    ///
    /// In debug builds, this method will panic if the index is out of bounds.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior in release builds.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(
            index < self.len,
            "Index out of bounds: index was {} but length was {}",
            index,
            self.len
        );

        let bit_pos = index as u64 * self.num_bits as u64;
        let word_index = (bit_pos / 64) as usize;
        let bit_offset = (bit_pos % 64) as usize;

        let bits = &self.data;

        // Dispatch based on endianness at compile time.
        if TypeId::of::<E>() == TypeId::of::<LE>() {
            // Little-Endian Path, inspired by sux-rs for maximum performance.
            if bit_offset + self.num_bits <= 64 {
                // Fast path: element is within a single word.
                (*bits.get_unchecked(word_index) >> bit_offset) & self.mask
            } else {
                // Slow path: element spans two words.
                ((*bits.get_unchecked(word_index) >> bit_offset)
                    | (*bits.get_unchecked(word_index + 1) << (64 - bit_offset)))
                    & self.mask
            }
        } else {
            // Big-Endian Path, using 128-bit arithmetic for robust handling
            // of all cases, including num_bits = 64.
            let high_word = bits.get_unchecked(word_index).to_be();
            let low_word = bits.get_unchecked(word_index + 1).to_be();

            let a = u128::from(high_word);
            let b = u128::from(low_word);

            ((((a << 64) | b) << bit_offset) >> (128 - self.num_bits)) as u64
        }
    }

    /// Retrieves multiple elements at the specified indices.
    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<u64>, FixedVecError> {
        for &index in indices {
            if index >= self.len {
                return Err(FixedVecError::IndexOutOfBounds(index));
            }
        }
        // SAFETY: We have pre-checked the bounds of all indices.
        Ok(unsafe { self.get_many_unchecked(indices) })
    }

    /// Retrieves multiple elements at the specified indices without bounds checking.
    ///
    /// In debug builds, this method will panic if any index is out of bounds.
    ///
    /// # Safety
    /// Calling this method with any out-of-bounds index is undefined behavior in release builds.
    pub unsafe fn get_many_unchecked(&self, indices: &[usize]) -> Vec<u64> {
        let mut results = Vec::with_capacity(indices.len());
        for &index in indices {
            // SAFETY: The caller guarantees that the index is in bounds.
            results.push(self.get_unchecked(index));
        }
        results
    }

    /// Returns an iterator over the decompressed `u64` values.
    pub fn iter(&self) -> FixedVecIter<E> {
        FixedVecIter::new(self)
    }

    /// Consumes the [`FixedVec`] and returns a `Vec<u64>`.
    pub fn into_vec(self) -> Vec<u64> {
        self.iter().collect()
    }
}

/// A type alias for a [`FixedVec`] with Big-Endian ([`BE`]) bitstream encoding.
pub type BEFixedVec = FixedVec<BE>;

/// A type alias for a [`FixedVec`] with Little-Endian ([`LE`]) bitstream encoding.
pub type LEFixedVec = FixedVec<LE>;
