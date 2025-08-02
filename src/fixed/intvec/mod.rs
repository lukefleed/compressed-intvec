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
pub mod slice;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use builder::{FixedVecBuilder, FixedVecFromIterBuilder};
pub use iter::FixedVecIter;
pub use slice::FixedVecSlice;

use dsi_bitstream::prelude::{Endianness, BE, LE};
use mem_dbg::{MemDbg, MemSize};
use std::{any::TypeId, cmp::Ordering, error::Error, fmt, marker::PhantomData};

/// Specifies the strategy for determining the number of bits per integer in a `FixedVec`.
#[derive(Debug, Clone, Copy, Default)]
pub enum BitWidth {
    /// Use the exact number of bits specified by the user.
    ///
    /// The user is responsible for ensuring all values fit within this bit width.
    /// An error will be returned during build if a value is too large.
    Explicit(usize),

    /// Automatically determine the minimum number of bits required to store the
    /// largest value in the input data.
    ///
    /// This prioritizes minimal memory usage but may result in bit widths that are
    /// not aligned to byte boundaries (e.g., 9 bits), which can be slightly
    /// less performant for access than byte-aligned widths.
    Minimal,
    
    /// Automatically determine the minimum number of bits and round up to the
    /// nearest multiple of 8 (i.e., a full byte).
    ///
    /// This may use slightly more memory than `Minimal` but can lead to faster
    /// access patterns due to better memory alignment. For example, if the
    /// minimum required bits is 11, this will use 16.
    #[default]
    ByteAligned,
}

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
    ///
    /// The builder is generic over the input integer type `U` (e.g., `u8`, `u16`, `u32`),
    /// allowing for direct construction without an intermediate conversion to `u64`.
    pub fn builder<U, T>(input: &T) -> FixedVecBuilder<E, U>
    where
        U: Into<u64> + Ord + Copy + Default,
        T: AsRef<[U]> + ?Sized,
    {
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

    /// Returns a zero-copy, read-only slice of the underlying storage (`&[u64]`).
    pub fn as_limbs(&self) -> &[u64] {
        &self.data
    }
}

impl<E: Endianness> FixedVec<E> {
    /// Retrieves the element at the specified index. Access is O(1).
    #[inline(always)]
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
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(
            index < self.len,
            "Index out of bounds: index was {} but length was {}",
            index,
            self.len
        );
        // Fast path for 64-bit values. This branch is perfectly predicted
        // by the CPU as `num_bits` is constant for a given FixedVec instance.
        if self.num_bits == 64 {
            let val = *self.data.get_unchecked(index);
            // For Big-Endian, the value was stored with bytes swapped,
            // so we must swap them back to get the correct native representation.
            if TypeId::of::<E>() == TypeId::of::<BE>() {
                return val.to_be();
            }
            return val;
        }

        let bit_pos = index as u64 * self.num_bits as u64;
        let word_index = (bit_pos / 64) as usize;
        let bit_offset = (bit_pos % 64) as usize;

        let bits = &self.data;

        // Dispatch based on endianness at compile time.
        if TypeId::of::<E>() == TypeId::of::<LE>() {
            // Little-Endian Path
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
            // Big-Endian Path, using u64 arithmetic.
            // The .to_be() call is essential to treat the native u64 value
            // as a big-endian sequence of bytes for bitwise operations.
            let word_hi = (*bits.get_unchecked(word_index)).to_be();
            if bit_offset + self.num_bits <= 64 {
                // Fast path: element is within a single word.
                (word_hi << bit_offset) >> (64 - self.num_bits)
            } else {
                // Slow path: element spans two words.
                let word_lo = (*bits.get_unchecked(word_index + 1)).to_be();
                let num_bits_in_first = 64 - bit_offset;
                let num_bits_in_second = self.num_bits - num_bits_in_first;

                // Mask to get the lower bits of the first word.
                let high_part = word_hi & ((1u64 << num_bits_in_first) - 1);
                // Get the most significant bits of the second word.
                let low_part = word_lo >> (64 - num_bits_in_second);

                (high_part << num_bits_in_second) | low_part
            }
        }
    }

    /// Retrieves multiple elements at the specified indices.
    #[inline(always)]
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
    #[inline(always)]
    pub unsafe fn get_many_unchecked(&self, indices: &[usize]) -> Vec<u64> {
        let mut results = Vec::with_capacity(indices.len());
        for &index in indices {
            // SAFETY: The caller guarantees that the index is in bounds.
            results.push(self.get_unchecked(index));
        }
        results
    }

    /// Creates a zero-copy slice of this vector.
    ///
    /// # Arguments
    /// * `start`: The starting index of the slice.
    /// * `len`: The number of elements in the slice.
    ///
    /// # Returns
    /// An `Option` containing the [`FixedVecSlice`] if the specified range is
    /// within the bounds of the vector, or `None` otherwise.
    pub fn slice(&self, start: usize, len: usize) -> Option<FixedVecSlice<E>> {
        if start + len > self.len {
            return None;
        }
        Some(FixedVecSlice::new(self, start..start + len))
    }

    /// Splits the vector into two slices at a given index.
    ///
    /// # Arguments
    /// * `mid`: The index at which to split the vector.
    ///
    /// # Returns
    /// An `Option` containing a tuple of two [`FixedVecSlice`]s if `mid` is
    /// within the bounds of the vector, or `None` otherwise.
    pub fn split_at(&self, mid: usize) -> Option<(FixedVecSlice<E>, FixedVecSlice<E>)> {
        if mid > self.len {
            return None;
        }
        let left = FixedVecSlice::new(self, 0..mid);
        let right = FixedVecSlice::new(self, mid..self.len);
        Some((left, right))
    }

    /// Binary searches this vector for a given element.
    ///
    /// If the vector is not sorted, the returned result is unspecified and
    /// meaningless. For `binary_search` to work correctly, the vector must be
    /// sorted in ascending order.
    ///
    /// If the value is found then `Ok(idx)` is returned, containing the
    /// index of the matching element. If there are multiple matches, then any
    /// one of the matches could be returned.
    /// If the value is not found then `Err(idx)` is returned, containing
    /// the index where a matching element could be inserted while maintaining
    /// sorted order.
    pub fn binary_search(&self, value: u64) -> Result<usize, usize> {
        self.binary_search_by(|probe| probe.cmp(&value))
    }

    /// Binary searches this vector with a comparator function.
    ///
    /// The comparator function should return an `Ordering` that indicates
    /// whether its argument is `Less`, `Equal` or `Greater` than the desired
    /// target.
    /// If the vector is not sorted or if the comparator does not reflect the
    /// vector's ordering, the returned result is unspecified.
    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(u64) -> Ordering,
    {
        let mut low = 0;
        let mut high = self.len();

        while low < high {
            let mid = low + (high - low) / 2;
            // SAFETY: The `low` and `high` bounds are checked in the loop.
            let cmp = f(unsafe { self.get_unchecked(mid) });

            match cmp {
                Ordering::Less => low = mid + 1,
                Ordering::Equal => return Ok(mid),
                Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }

    /// Binary searches this vector with a key extraction function.
    ///
    /// If the vector is not sorted by the key, the returned result is
    /// unspecified.
    #[inline]
    pub fn binary_search_by_key<B, F>(&self, b: &B, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(u64) -> B,
        B: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
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

// Implementations of traits from the standard library
impl<E: Endianness> PartialEq for FixedVec<E> {
    /// Checks for equality between two `FixedVec` instances.
    ///
    /// This method provides a highly efficient comparison. It first checks if the
    /// lengths and bit widths are identical. If they are, it proceeds with an
    /// element-wise comparison using iterators, which decompresses values
    /// on the fly and short-circuits at the first mismatch.
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len || self.num_bits != other.num_bits {
            return false;
        }
        // Use iterators for an efficient, element-by-element comparison
        // that avoids full decompression into a new Vec.
        self.iter().eq(other.iter())
    }
}

impl<E: Endianness> Eq for FixedVec<E> {}

impl<E: Endianness, T: AsRef<[u64]> + ?Sized> PartialEq<T> for FixedVec<E> {
    /// Checks for equality between a `FixedVec` and a slice-like type (e.g., `&[u64]`, `Vec<u64>`).
    ///
    /// The comparison first checks for equal length. If lengths match, it performs
    /// an element-wise comparison between the `FixedVec`'s iterator and the
    /// slice's iterator.
    fn eq(&self, other: &T) -> bool {
        let other_slice = other.as_ref();
        if self.len() != other_slice.len() {
            return false;
        }
        self.iter().eq(other_slice.iter().copied())
    }
}

impl<'a, E: Endianness> IntoIterator for &'a FixedVec<E> {
    type Item = u64;
    type IntoIter = FixedVecIter<'a, E>;

    /// Creates an iterator over the values of the `FixedVec`.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<E: Endianness> IntoIterator for FixedVec<E> {
    type Item = u64;
    type IntoIter = std::vec::IntoIter<u64>;

    /// Consumes the `FixedVec` and creates an iterator over its values.
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

/// A type alias for a [`FixedVec`] with Big-Endian ([`BE`]) bitstream encoding.
pub type BEFixedVec = FixedVec<BE>;

/// A type alias for a [`FixedVec`] with Little-Endian ([`LE`]) bitstream encoding.
pub type LEFixedVec = FixedVec<LE>;