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
//! - **Flexible Backends**: Can be created as an owned vector (`Vec<u64>`) or as a
//!   zero-copy view over an existing slice (`&[u64]`), making it ideal for
//!   memory-mapped files and zero-copy deserialization.
//!
//! The main struct, [`FixedVec`], is generic over [`Endianness`], allowing
//! users to choose between Little-Endian ([`LEFixedVec`]) and Big-Endian ([`BEFixedVec`])
//! representations.

pub mod builder;
pub mod iter;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod slice;

#[cfg(feature = "simd")]
mod simd;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use builder::{FixedVecBuilder, FixedVecFromIterBuilder};
pub use iter::FixedVecIter;
pub use slice::FixedVecSlice;

use dsi_bitstream::prelude::{BitWrite, Endianness, BE, LE};
use mem_dbg::{MemDbg, MemSize};
use std::{any::TypeId, cmp::Ordering, error::Error, fmt, marker::PhantomData};

use crate::fixed::intvec::iter::FixedVecIntoIter;

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
    #[default]
    Minimal,

    /// Automatically determine the minimum number of bits and round up to the
    /// nearest multiple of 8 (i.e., a full byte).
    ///
    /// This may use slightly more memory than `Minimal` but can lead to faster
    /// access patterns due to better memory alignment. For example, if the
    /// minimum required bits is 11, this will use 16.
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
///
/// It is generic over the backend storage `B`, which can be an owned `Vec<u64>`
/// or a borrowed slice `&[u64]`.
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FixedVec<E: Endianness, B: AsRef<[u64]> = Vec<u64>> {
    /// The underlying storage for the bit-packed data.
    bits: B,
    /// The number of elements in the vector.
    len: usize,
    /// The number of bits used to encode each integer.
    num_bits: usize,
    /// A mask with the lowest `num_bits` bits set to one.
    mask: u64,
    /// Zero-sized markers for endianness and backend type parameters.
    #[cfg_attr(feature = "serde", serde(skip))]
    _endian: PhantomData<(E, B)>,
}

impl<E: Endianness> FixedVec<E, Vec<u64>> {
    /// Returns a builder for creating an owned [`FixedVec`] from a slice of data.
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

    /// Returns a builder for creating an owned [`FixedVec`] from an iterator.
    pub fn from_iter_builder<I: IntoIterator<Item = u64>>(
        iter: I,
        num_bits: usize,
    ) -> FixedVecFromIterBuilder<E, I> {
        FixedVecFromIterBuilder::new(iter, num_bits)
    }

    /// Creates an owned `FixedVec` directly from a slice of data.
    ///
    /// This is a convenient alias for `FixedVec::builder(slice).build()`.
    /// The bit width will be automatically determined using the `BitWidth::Minimal` strategy.
    /// To specify a different strategy, use the builder directly.
    pub fn from_slice<U, T>(slice: &T) -> Result<Self, FixedVecError>
    where
        U: Into<u64> + Ord + Copy + Default,
        T: AsRef<[U]> + ?Sized,
        Self: Sized,
        builder::FixedVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible>,
    {
        Self::builder(slice).build()
    }
}

impl<E: Endianness, B: AsRef<[u64]>> FixedVec<E, B> {
    /// Creates a `FixedVec` from its constituent parts.
    ///
    /// This is the primary constructor for creating a `FixedVec` view over an
    /// existing data slice (e.g., `&[u64]` or a memory-mapped buffer).
    ///
    /// # Arguments
    /// * `bits`: The backend storage containing the compressed data.
    /// * `len`: The number of elements in the vector.
    /// * `num_bits`: The number of bits used for each element.
    ///
    /// # Errors
    /// Returns an error if the provided parameters are invalid (e.g., `num_bits > 64`
    /// or the `bits` buffer is too small for the given `len` and `num_bits`).
    ///
    /// # Safety Note
    /// This constructor is safe because it validates that the `bits` buffer is large
    /// enough to contain all encoded data **plus one padding word**. This padding
    /// is essential to prevent `get_unchecked` from reading past the end of the buffer.
    pub fn from_parts(bits: B, len: usize, num_bits: usize) -> Result<Self, FixedVecError> {
        if num_bits > 64 {
            return Err(FixedVecError::InvalidParameters(
                "num_bits cannot be greater than 64".to_string(),
            ));
        }

        let total_bits = len * num_bits;
        let data_words = (total_bits + 63) / 64;

        // Essential safety check: ensure the buffer is large enough for the data
        // AND the padding word required by get_unchecked.
        if bits.as_ref().len() < data_words + 1 {
            return Err(FixedVecError::InvalidParameters(format!(
                "The provided buffer is too small. It has {} words, but {} data words + 1 padding word are required.",
                bits.as_ref().len(),
                data_words
            )));
        }

        // SAFETY: We have performed all necessary checks.
        Ok(unsafe { Self::new_unchecked(bits, len, num_bits) })
    }

    /// Creates a new `FixedVec` from its raw parts without performing safety checks.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// 1. `len * num_bits` is not larger than the number of bits available in `bits`.
    /// 2. The `bits` slice has at least one extra padding word at the end
    ///    to prevent out-of-bounds reads during `get_unchecked`.
    /// 3. `num_bits` is not greater than 64.
    pub(crate) unsafe fn new_unchecked(bits: B, len: usize, num_bits: usize) -> Self {
        let mask = if num_bits == 64 {
            u64::MAX
        } else {
            (1u64 << num_bits) - 1
        };

        Self {
            bits,
            len,
            num_bits,
            mask,
            _endian: PhantomData,
        }
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
        self.bits.as_ref().to_vec()
    }

    /// Returns a zero-copy, read-only slice of the underlying storage (`&[u64]`).
    pub fn as_limbs(&self) -> &[u64] {
        self.bits.as_ref()
    }
}

impl<E: Endianness, B: AsRef<[u64]>> FixedVec<E, B> {
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
            let val = *self.as_limbs().get_unchecked(index);
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

        let bits = self.as_limbs();

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

                // Extract the relevant bits from the first word.
                let high_part = word_hi << bit_offset >> (64 - num_bits_in_first);
                // Get the most significant bits of the second word.
                let low_part = word_lo >> (64 - num_bits_in_second);

                (high_part << num_bits_in_second) | low_part
            }
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
        #[cfg(debug_assertions)]
        {
            for &index in indices {
                debug_assert!(
                    index < self.len,
                    "Index out of bounds: index was {} but length was {}",
                    index,
                    self.len
                );
            }
        }

        // This block is compiled if the "simd" feature is NOT enabled.
        // It provides the basic, correct, and portable scalar implementation.
        #[cfg(not(feature = "simd"))]
        {
            let mut results = Vec::with_capacity(indices.len());
            for &index in indices {
                // SAFETY: The caller guarantees that the index is in bounds.
                results.push(self.get_unchecked(index));
            }
            results
        }

        // This block is compiled ONLY if the "simd" feature is enabled.
        #[cfg(feature = "simd")]
        {
            match self.num_bits {
                // For these byte-aligned bit-widths, we can use a SIMD-accelerated path.
                8 | 16 | 32 | 64 => {
                    if indices.is_empty() {
                        return vec![];
                    }

                    let mut results = vec![0; indices.len()];

                    // Pair each index with its original position to restore order after sorting.
                    let mut indexed_indices: Vec<(usize, usize)> = indices
                        .iter()
                        .enumerate()
                        .map(|(original_pos, &idx)| (idx, original_pos))
                        .collect();

                    // Sort by the access index to create a sequential access pattern.
                    // If the `parallel` feature is enabled, use a parallel sort for large inputs.
                    #[cfg(feature = "parallel")]
                    {
                        use rayon::prelude::{IntoParallelIterator, ParallelSliceMut};
                        indexed_indices.par_sort_unstable_by_key(|&(idx, _)| idx);
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        indexed_indices.sort_unstable_by_key(|&(idx, _)| idx);
                    }

                    let mut i = 0;
                    while i < indexed_indices.len() {
                        // Find a contiguous run of indices.
                        let run_start_index = indexed_indices[i].0;
                        let mut j = i + 1;
                        while j < indexed_indices.len()
                            && indexed_indices[j].0 == indexed_indices[j - 1].0 + 1
                        {
                            j += 1;
                        }

                        let run_len = j - i;
                        if run_len > 4 {
                            // Use SIMD only for reasonably long runs.
                            // This slice contains the (index, original_pos) pairs for the run.
                            let run_slice = &indexed_indices[i..j];
                            // We need a temporary buffer to hold the gathered SIMD results.
                            let mut temp_run_results = vec![0; run_len];

                            // Call the high-performance SIMD gather function.
                            // SAFETY: We have sorted the indices and verified the run is contiguous.
                            // The bounds of each index were checked at the start of the function.
                            simd::gather_simd(self, run_start_index, &mut temp_run_results);

                            // Scatter the results from the temp buffer back to the final
                            // results vector in their original order.
                            for (k, &(_, original_pos)) in run_slice.iter().enumerate() {
                                results[original_pos] = temp_run_results[k];
                            }
                        } else {
                            // For short runs, a scalar loop is often faster.
                            for k in i..j {
                                let (idx, original_pos) = indexed_indices[k];
                                results[original_pos] = self.get_unchecked(idx);
                            }
                        }

                        // Move to the start of the next potential run.
                        i = j;
                    }

                    results
                }
                // For non-byte-aligned bit-widths, fall back to the simple scalar loop.
                // The compiler optimizes this match away, creating a zero-cost abstraction.
                _ => {
                    let mut results = Vec::with_capacity(indices.len());
                    for &index in indices {
                        results.push(self.get_unchecked(index));
                    }
                    results
                }
            }
        }
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
    pub fn slice(&self, start: usize, len: usize) -> Option<FixedVecSlice<E, B>> {
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
    pub fn split_at(&self, mid: usize) -> Option<(FixedVecSlice<E, B>, FixedVecSlice<E, B>)> {
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
    pub fn binary_search_by_key<B1, F>(&self, b: &B1, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(u64) -> B1,
        B1: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Returns an iterator over the decompressed `u64` values.
    pub fn iter(&self) -> FixedVecIter<E, B> {
        FixedVecIter::new(self)
    }
}

impl<E: Endianness> FixedVec<E, Vec<u64>> {
    /// Consumes the owned [`FixedVec`] and returns its underlying `Vec<u64>`.
    pub fn into_limbs(self) -> Vec<u64> {
        self.bits
    }

    /// Consumes the owned [`FixedVec`] and returns a `Vec<u64>` of its decoded values.
    pub fn into_vec(self) -> Vec<u64> {
        self.into_iter().collect()
    }
}

// Implementations of traits from the standard library
impl<E: Endianness, B: AsRef<[u64]>, B2: AsRef<[u64]>> PartialEq<FixedVec<E, B2>>
    for FixedVec<E, B>
{
    /// Checks for equality between two `FixedVec` instances, regardless of backend.
    fn eq(&self, other: &FixedVec<E, B2>) -> bool {
        if self.len != other.len || self.num_bits != other.num_bits {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

impl<E: Endianness, B: AsRef<[u64]>> Eq for FixedVec<E, B> {}

macro_rules! impl_partial_eq_for_uint_slice {
    ($($t:ty),*) => {$(
        impl<E: Endianness, B: AsRef<[u64]>> PartialEq<&[$t]> for FixedVec<E, B> {
            fn eq(&self, other: &&[$t]) -> bool {
                self.eq(*other)
            }
        }

        impl<E: Endianness, B: AsRef<[u64]>> PartialEq<[$t]> for FixedVec<E, B> {
            fn eq(&self, other: &[$t]) -> bool {
                if self.len() != other.len() {
                    return false;
                }
                self.iter().eq(other.iter().map(|&x| x as u64))
            }
        }
    )*};
}

impl_partial_eq_for_uint_slice!(u8, u16, u32, u64);

impl<'a, E: Endianness, B: AsRef<[u64]>> IntoIterator for &'a FixedVec<E, B> {
    type Item = u64;
    type IntoIter = FixedVecIter<'a, E, B>;

    /// Creates an iterator over the values of the `FixedVec`.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<E: Endianness> IntoIterator for FixedVec<E, Vec<u64>> {
    type Item = u64;
    type IntoIter = FixedVecIntoIter<E>;

    /// Consumes the `FixedVec` and creates an iterator over its decompressed values.
    ///
    /// This implementation is "lazy" and decodes values on the fly without
    /// allocating an intermediate `Vec<u64>`.
    fn into_iter(self) -> Self::IntoIter {
        FixedVecIntoIter::new(self)
    }
}

impl<E: Endianness, B: AsRef<[u64]>, B2: AsRef<[u64]>> PartialEq<FixedVecSlice<'_, E, B2>>
    for FixedVec<E, B>
{
    fn eq(&self, other: &FixedVecSlice<'_, E, B2>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

/// A type alias for an owned [`FixedVec`] with Big-Endian ([`BE`]) bitstream encoding.
pub type BEFixedVec = FixedVec<BE, Vec<u64>>;

/// A type alias for an owned [`FixedVec`] with Little-Endian ([`LE`]) bitstream encoding.
pub type LEFixedVec = FixedVec<LE, Vec<u64>>;
