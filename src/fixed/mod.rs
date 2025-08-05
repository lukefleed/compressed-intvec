//! # A generic, compressed, and randomly accessible vector with fixed-width encoding.
//!
//! This module provides [`FixedVec`], a highly generic data structure optimized
//! for space-efficient storage and O(1) random access for integer sequences where
//! all values fit within a known, fixed number of bits.

// Declare and export submodules that will be created in subsequent steps.
#[macro_use]
pub mod macros;
pub mod builder;
pub mod iter;
pub mod traits;
pub mod view;

// Conditionally compile the atomic module.
#[cfg(feature = "atomic")]
pub mod atomic;

// Conditionally compile the serde module.
#[cfg(feature = "serde")]
mod serde;

use dsi_bitstream::prelude::Endianness;
use mem_dbg::{MemDbg, MemSize};
use num_traits::Bounded;
use std::{error::Error as StdError, fmt, marker::PhantomData};
use traits::{Storable, Word};

/// Specifies the strategy for determining the number of bits per integer in a `FixedVec`.
///
/// For maximum random access performance, bit widths that are a power of two
/// (e.g., 8, 16, 32, 64) are optimal as they allow the access logic to use
/// highly efficient bit-shift operations. The `PowerOfTwo` strategy enforces this.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BitWidth {
    /// Automatically determine the minimum number of bits required to store the
    /// largest value in the input data. This prioritizes minimal memory usage.
    #[default]
    Minimal,

    /// Rounds up the minimal bit width to the next power of two (e.g., 8, 16, 32, 64).
    /// This prioritizes maximum random access speed.
    PowerOfTwo,

    /// Use the exact number of bits specified by the user. An error will be
    /// returned during the build process if a value in the input data is too
    /// large to be represented with the given number of bits.
    Explicit(usize),
}

/// Defines the set of errors that can occur in `FixedVec` operations.
#[derive(Debug)]
pub enum Error {
    /// An error indicating that a value in the input data does not fit within
    /// the specified number of bits.
    ValueTooLarge {
        /// The value that caused the error.
        value: u128,
        /// The index of the value in the input data.
        index: usize,
        /// The specified number of bits.
        bit_width: usize,
    },
    /// An error indicating that the provided parameters are invalid for the
    /// requested operation (e.g., `bit_width` is 0 for a non-empty vector).
    InvalidParameters(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::ValueTooLarge {
                value,
                index,
                bit_width,
            } => write!(
                f,
                "value {} at index {} does not fit in {} bits",
                value, index, bit_width
            ),
            Error::InvalidParameters(s) => write!(f, "Invalid parameters: {}", s),
        }
    }
}

impl StdError for Error {}

/// A compressed, randomly accessible vector of integers with fixed-width encoding.
///
/// `FixedVec` is a highly generic data structure for storing sequences of integers
/// where each element is encoded using the same number of bits. This allows for
/// O(1) random access by arithmetically calculating the memory location of any element.
///
/// The structure is generic over several parameters:
/// - `T`: The user-facing element type (e.g., `u32`, `i16`). Must implement [`Storable`].
/// - `W`: The underlying storage word (e.g., `u64`, `usize`). Must implement [`Word`].
/// - `E`: The [`Endianness`] for bitstream operations.
/// - `B`: The backend storage buffer (e.g., `Vec<W>`, `&[W]`).
///
/// For common use cases, a set of convenient type aliases are provided in the prelude.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct FixedVec<
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> = Vec<W>,
> {
    /// The underlying storage for the bit-packed data.
    bits: B,
    /// The number of bits used to encode each element.
    bit_width: usize,
    /// A mask with the lowest `bit_width` bits set to one.
    mask: W,
    /// The number of elements in the vector.
    len: usize,
    /// Zero-sized markers for the generic type parameters.
    _phantom: PhantomData<(T, W, E)>,
}

// This block is for owned `FixedVec`s (`B = Vec<W>`) and exposes the builder APIs.
impl<T, W, E> FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    // The trait bound is required here to satisfy the `build` methods in the builders.
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
{
    /// Returns a builder for creating an owned [`FixedVec`] from a slice of data.
    ///
    /// The builder allows for detailed configuration of the vector's properties,
    /// such as the bit width strategy.
    pub fn builder() -> builder::FixedVecBuilder<T, W, E> {
        builder::FixedVecBuilder::new()
    }

    /// Returns a builder for creating an owned [`FixedVec`] from an iterator.
    ///
    /// # Limitations
    /// This builder requires that the number of bits be specified manually, as it
    /// cannot pre-analyze the data from a stream.
    pub fn from_iter_builder<I: IntoIterator<Item = T>>(
        iter: I,
        bit_width: usize,
    ) -> builder::FixedVecFromIterBuilder<T, W, E, I> {
        builder::FixedVecFromIterBuilder::new(iter, bit_width)
    }
}

// This block contains the core immutable API, available for all `FixedVec` instances,
// including both owned vectors and borrowed views (`&[W]`).
impl<T, W, E, B> FixedVec<T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Returns the number of elements in the vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of bits used to encode each element.
    #[inline]
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// Returns a zero-copy, read-only slice of the underlying storage words.
    #[inline]
    pub fn as_limbs(&self) -> &[W] {
        self.bits.as_ref()
    }

    /// Creates a `FixedVec` from its constituent parts, enabling zero-copy views.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// 1. `len * bit_width` is not larger than the number of bits available in `bits`.
    /// 2. The `bits` slice has at least one extra padding word at the end
    ///    to prevent out-of-bounds reads during `get_unchecked`.
    /// 3. `bit_width` is not greater than `W::BITS`.
    pub(crate) unsafe fn new_unchecked(bits: B, len: usize, bit_width: usize) -> Self {
        let mask = if bit_width == <W as traits::Word>::BITS {
            W::max_value()
        } else {
            (W::ONE << bit_width) - W::ONE
        };

        Self {
            bits,
            len,
            bit_width,
            mask,
            _phantom: PhantomData,
        }
    }

    /// Retrieves the element at the specified index. Access is O(1).
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Retrieves the element at the specified index without bounds checking.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        debug_assert!(index < self.len);

        let bits_per_word = <W as traits::Word>::BITS;
        if self.bit_width == bits_per_word {
            let val = *self.as_limbs().get_unchecked(index);
            let final_val = if E::IS_BIG { val.to_be() } else { val };
            return <T as Storable<W>>::from_word(final_val);
        }
        
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / bits_per_word;
        let bit_offset = bit_pos % bits_per_word;

        let limbs = self.as_limbs();
        let final_word: W;

        if E::IS_LITTLE {
            if bit_offset + self.bit_width <= bits_per_word {
                final_word = (*limbs.get_unchecked(word_index) >> bit_offset) & self.mask;
            } else {
                let low = *limbs.get_unchecked(word_index) >> bit_offset;
                let high = *limbs.get_unchecked(word_index + 1) << (bits_per_word - bit_offset);
                final_word = (low | high) & self.mask;
            }
        } else {
            let word_hi = (*limbs.get_unchecked(word_index)).to_be();
            if bit_offset + self.bit_width <= bits_per_word {
                final_word = (word_hi << bit_offset) >> (bits_per_word - self.bit_width);
            } else {
                let word_lo = (*limbs.get_unchecked(word_index + 1)).to_be();
                let bits_in_first = bits_per_word - bit_offset;
                let high = word_hi << bit_offset >> (bits_per_word - bits_in_first);
                let low = word_lo >> (bits_per_word - (self.bit_width - bits_in_first));
                final_word = (high << (self.bit_width - bits_in_first)) | low;
            }
        }
        <T as Storable<W>>::from_word(final_word)
    }

    /// Returns a safe iterator over the decompressed values.
    pub fn iter(&self) -> iter::FixedVecIter<T, W, E, B> {
        iter::FixedVecIter::new(self)
    }

    /// Returns an iterator that does not perform bounds checking.
    ///
    /// # Safety
    /// The returned iterator is unsafe to use. The caller must ensure that the
    /// iterator's `next_unchecked` method is not called more times than the
    /// length of the vector.
    pub unsafe fn iter_unchecked(&self) -> iter::FixedVecUncheckedIter<T, W, E, B> {
        iter::FixedVecUncheckedIter::new(self)
    }
}

/// Implements `IntoIterator` for a borrowed `FixedVec`.
/// This allows for iterating over the vector using `for val in &my_vec`.
impl<'a, T, W, E, B> IntoIterator for &'a FixedVec<T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    type Item = T;
    type IntoIter = iter::FixedVecIter<'a, T, W, E, B>;

    /// Creates an iterator over the values of the `FixedVec`.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Implements `IntoIterator` for an owned `FixedVec`.
/// This allows for iterating over the vector using `for val in my_vec`, consuming it.
impl<T, W, E> IntoIterator for FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
{
    type Item = T;
    type IntoIter = iter::FixedVecIntoIter<T, W, E>;

    /// Consumes the `FixedVec` and creates an iterator over its decompressed values.
    ///
    /// This implementation is "lazy" and decodes values on the fly without
    /// allocating an intermediate `Vec<T>`.
    fn into_iter(self) -> Self::IntoIter {
        iter::FixedVecIntoIter::new(self)
    }
}