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
pub mod slice;
pub mod parallel;
pub mod proxy;
pub mod iter_mut;

// Conditionally compile the atomic module.
#[cfg(feature = "atomic")]
pub mod atomic;

// Conditionally compile the serde module.
#[cfg(feature = "serde")]
mod serde;

use dsi_bitstream::{prelude::Endianness, traits::{BE, LE}};
use mem_dbg::{MemDbg, MemSize};
use std::{error::Error as StdError, fmt, marker::PhantomData, iter::FromIterator};
use traits::{Storable, Word};
use num_traits::ToPrimitive;
use crate::fixed::iter_mut::ChunksMut;

use crate::fixed::proxy::MutProxy;

// Type aliases for common `FixedVec` configurations.

/// A generic fixed-width vector for unsigned integers, using `usize` as the
/// storage word and Little-Endian byte order.
///
/// This is the recommended alias for general-purpose use with unsigned types.
/// `T` can be `u8`, `u16`, `u32`, `u64`, `u128`, or `usize`.
pub type UFixedVec<T, B = Vec<usize>> = FixedVec<T, usize, LE, B>;

/// A generic fixed-width vector for signed integers, using `usize` as the
/// storage word and Little-Endian byte order.
///
/// This is the recommended alias for general-purpose use with signed types.
/// `T` can be `i8`, `i16`, `i32`, `i64`, `i128`, or `isize`.
pub type SFixedVec<T, B = Vec<usize>> = FixedVec<T, usize, LE, B>;

// --- Concrete Aliases for `u64`/`i64` elements with a `u64` backend ---
// These are provided for backward compatibility and for cases where a `u64`
// storage word is explicitly required.

/// A `FixedVec` for `u64` elements with a `u64` backend and Little-Endian layout.
pub type LEFixedVec<B = Vec<u64>> = FixedVec<u64, u64, LE, B>;
/// A `FixedVec` for `i64` elements with a `u64` backend and Little-Endian layout.
pub type LESFixedVec<B = Vec<u64>> = FixedVec<i64, u64, LE, B>;

/// A `FixedVec` for `u64` elements with a `u64` backend and Big-Endian layout.
pub type BEFixedVec<B = Vec<u64>> = FixedVec<u64, u64, BE, B>;
/// A `FixedVec` for `i64` elements with a `u64` backend and Big-Endian layout.
pub type BESFixedVec<B = Vec<u64>> = FixedVec<i64, u64, BE, B>;

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

    /// Creates a `FixedVec` from its constituent parts, enabling zero-copy views.
    pub fn from_parts(bits: B, len: usize, bit_width: usize) -> Result<Self, Error> {
        if bit_width > <W as traits::Word>::BITS {
            return Err(Error::InvalidParameters(format!(
                "bit_width ({}) cannot be greater than the word size ({})",
                bit_width, <W as traits::Word>::BITS
            )));
        }

        let total_bits = len * bit_width;
        let data_words = (total_bits + <W as traits::Word>::BITS - 1) / <W as traits::Word>::BITS;

        if bits.as_ref().len() < data_words + 1 {
            return Err(Error::InvalidParameters(format!(
                "The provided buffer is too small. It has {} words, but {} data words + 1 padding word are required.",
                bits.as_ref().len(),
                data_words
            )));
        }

        Ok(unsafe { Self::new_unchecked(bits, len, bit_width) })
    }

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

        /// Creates a zero-copy immutable view (slice) of this vector.
    ///
    /// # Arguments
    /// * `start`: The starting index of the slice.
    /// * `len`: The number of elements in the slice.
    ///
    /// # Returns
    /// An `Option` containing the [`FixedVecSlice`] if the specified range is
    /// within the bounds of the vector, or `None` otherwise.
    pub fn slice(&self, start: usize, len: usize) -> Option<slice::FixedVecSlice<&Self>> {
        if start.saturating_add(len) > self.len {
            return None;
        }
        Some(slice::FixedVecSlice::new(self, start..start + len))
    }

    /// Splits the vector into two views at a given index.
    ///
    /// # Arguments
    /// * `mid`: The index at which to split the vector.
    ///
    /// # Returns
    /// An `Option` containing a tuple of two [`FixedVecSlice`]s if `mid` is
    /// within the bounds of the vector, or `None` otherwise.
    pub fn split_at(&self, mid: usize) -> Option<(slice::FixedVecSlice<&Self>, slice::FixedVecSlice<&Self>)> {
        if mid > self.len {
            return None;
        }
        let left = slice::FixedVecSlice::new(self, 0..mid);
        let right = slice::FixedVecSlice::new(self, mid..self.len);
        Some((left, right))
    }

    /// Returns an iterator over non-overlapping chunks of the vector.
    ///
    /// Each chunk is a `FixedVecSlice` of length `chunk_size`, except for the
    /// last chunk which may be shorter.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    pub fn chunks(&self, chunk_size: usize) -> iter::Chunks<T, W, E, B> {
        iter::Chunks::new(self, chunk_size)
    }

    /// Binary searches this vector for a given element.
    pub fn binary_search(&self, value: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        // We can't use self.binary_search_by directly because of Ord vs FnMut,
        // so we reimplement the logic here for clarity.
        let mut low = 0;
        let mut high = self.len();

        while low < high {
            let mid = low + (high - low) / 2;
            let mid_val = unsafe { self.get_unchecked(mid) };
            
            match mid_val.cmp(value) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => return Ok(mid),
                std::cmp::Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }

    /// Binary searches this vector with a comparator function.
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> std::cmp::Ordering, // Accetta T, non &T
    {
        let mut low = 0;
        let mut high = self.len();

        while low < high {
            let mid = low + (high - low) / 2;
            let mid_val = unsafe { self.get_unchecked(mid) };
            // Passa la proprietà di mid_val alla closure
            let cmp = f(mid_val); 

            match cmp {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => return Ok(mid),
                std::cmp::Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }

    /// Binary searches this vector with a key extraction function.
    pub fn binary_search_by_key<K: Ord, F>(&self, key: &K, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> K,
    {
        self.binary_search_by(|probe| f(probe).cmp(key))
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

impl<T, W, E> FromIterator<T> for FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
{
    /// Creates a `FixedVec` from an iterator.
    ///
    /// The bit width is determined automatically using the `BitWidth::Minimal`
    /// strategy for optimal space usage. This involves collecting the iterator
    /// into a temporary `Vec<T>` to analyze its contents.
    ///
    /// # Example
    /// ```
    /// use compressed_intvec::prelude::*;
    ///
    /// let data = 0u32..100;
    /// let vec: UFixedVec<u32> = data.collect();
    ///
    /// assert_eq!(vec.len(), 100);
    /// assert_eq!(vec.get(50), Some(50));
    /// ```
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let data: Vec<T> = iter.into_iter().collect();
        // The builder defaults to BitWidth::Minimal, which is appropriate for
        // FromIterator as it adapts to the collected data. This build call
        // should not fail unless there's an internal logic error.
        Self::builder().build(&data).unwrap()
    }
}


// This block implements resizing and capacity management methods for `FixedVec`
// instances that have an owned `Vec<W>` backend.
impl<T, W, E> FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W> + ToPrimitive,
    W: Word,
    E: Endianness,
{
    /// Creates a new, empty `FixedVec` with a specified bit width.
    pub fn new(bit_width: usize) -> Result<Self, Error> {
        if bit_width > <W as traits::Word>::BITS {
            return Err(Error::InvalidParameters(format!(
                "bit_width ({}) cannot be greater than the word size ({})",
                bit_width, <W as traits::Word>::BITS
            )));
        }
        Ok(unsafe { Self::new_unchecked(Vec::new(), 0, bit_width) })
    }

    /// Appends an element to the back of the vector.
    pub fn push(&mut self, value: T) {
        let value_w = <T as Storable<W>>::into_word(value);
        let bits_per_word = <W as traits::Word>::BITS;

        let limit = if self.bit_width < bits_per_word {
            W::ONE << self.bit_width
        } else {
            W::max_value()
        };

        if self.bit_width < bits_per_word && value_w >= limit {
            panic!("Value {:?} does not fit in the configured bit_width of {}", value_w, self.bit_width);
        }

        let bit_pos = self.len * self.bit_width;
        let required_total_bits = bit_pos + self.bit_width;
        let required_words = (required_total_bits + bits_per_word - 1) / bits_per_word;
        
        // +1 for the padding word.
        let required_vec_len_for_write = required_words + 1;

        if self.bits.capacity() < required_vec_len_for_write {
            // Reallocation is needed. Let Vec decide how much to grow.
            self.bits.reserve(required_vec_len_for_write - self.bits.len());
        }

        if self.bits.len() < required_vec_len_for_write {
            // Ensure the vec's length is sufficient for the write operation
            // without over-allocating if capacity is already sufficient.
            self.bits.resize(required_vec_len_for_write, W::ZERO);
        }

        unsafe {
            // Write to the original length index.
            self.set_unchecked(self.len, value_w);
        }
        
        self.len += 1;
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let value = self.get(self.len - 1).unwrap(); 
        self.len -= 1;
        Some(value)
    }

    /// Removes all elements from the vector.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Creates a new, empty `FixedVec` with a specified bit width and capacity.
    ///
    /// The vector will be able to hold at least `capacity` elements without
    /// reallocating.
    pub fn with_capacity(bit_width: usize, capacity: usize) -> Result<Self, Error> {
        if bit_width > <W as traits::Word>::BITS {
            return Err(Error::InvalidParameters(format!(
                "bit_width ({}) cannot be greater than the word size ({})",
                bit_width, <W as traits::Word>::BITS
            )));
        }
        let bits_per_word = <W as traits::Word>::BITS;
        let total_bits = capacity.saturating_mul(bit_width);
        let num_words = (total_bits + bits_per_word - 1) / bits_per_word;
        
        // +1 for the padding word, unless capacity is 0.
        let buffer = if capacity == 0 {
            Vec::new()
        } else {
            Vec::with_capacity(num_words + 1)
        };
        
        Ok(unsafe { Self::new_unchecked(buffer, 0, bit_width) })
    }

    /// Returns the number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        if self.bit_width == 0 {
            // For zero-sized elements, capacity is conceptually infinite.
            return usize::MAX;
        }
        let word_capacity = self.bits.capacity();
        if word_capacity == 0 {
            return 0;
        }
        // Subtract 1 for the padding word before calculating element capacity.
        ((word_capacity - 1) * <W as traits::Word>::BITS) / self.bit_width
    }

    /// Returns the capacity of the underlying storage in words.
    pub fn word_capacity(&self) -> usize {
        self.bits.capacity()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted.
    ///
    /// May reallocate if the current capacity is not sufficient.
    pub fn reserve(&mut self, additional: usize) {
        let required_len = self.len.saturating_add(additional);
        if self.capacity() >= required_len {
            return;
        }

        let bits_per_word = <W as traits::Word>::BITS;
        let required_total_bits = required_len.saturating_mul(self.bit_width);
        let required_words = (required_total_bits + bits_per_word - 1) / bits_per_word;
        
        // +1 for the padding word.
        let required_word_capacity = required_words + 1;
        if required_word_capacity > self.word_capacity() {
            let additional_words = required_word_capacity.saturating_sub(self.bits.len());
            self.bits.reserve(additional_words);
        }
    }

    /// Resizes the vector in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the vector is extended by the
    /// difference, with each additional slot filled with `value`.
    /// If `new_len` is less than `len`, the vector is simply truncated.
    ///
    /// This implementation is optimized for performance on large extensions by
    /// writing bits in batches rather than calling `push` repeatedly.
    pub fn resize(&mut self, new_len: usize, value: T) {
        if new_len > self.len {
            // --- Extend the vector (Optimized Path) ---
            let additional = new_len - self.len;
            self.reserve(additional);

            let value_w = <T as Storable<W>>::into_word(value);
            let bits_per_word = <W as traits::Word>::BITS;
            let limit = if self.bit_width < bits_per_word { W::ONE << self.bit_width } else { W::max_value() };

            if self.bit_width < bits_per_word && value_w >= limit {
                panic!("Value {:?} does not fit in the configured bit_width of {}", value_w, self.bit_width);
            }

            // Ensure the underlying Vec has enough *initialized* words to write into.
            let required_total_bits = new_len * self.bit_width;
            let required_words = (required_total_bits + bits_per_word - 1) / bits_per_word;
            let required_vec_len = required_words + 1; // +1 for padding
            if self.bits.len() < required_vec_len {
                self.bits.resize(required_vec_len, W::ZERO);
            }

            // Write the new values in a loop. This is much faster than calling
            // `push` repeatedly as it avoids redundant capacity checks.
            for i in self.len..new_len {
                // SAFETY: We have already reserved and resized, so the indices are valid.
                unsafe {
                    self.set_unchecked(i, value_w);
                }
            }
            self.len = new_len;
        } else {
            // --- Truncate the vector (Simple Path) ---
            self.len = new_len;
        }
    }

    /// Shrinks the capacity of the vector as much as possible.
    pub fn shrink_to_fit(&mut self) {
        let min_word_len = if self.len == 0 {
            0
        } else {
            let bits_per_word = <W as traits::Word>::BITS;
            let required_total_bits = self.len.saturating_mul(self.bit_width);
            let required_words = (required_total_bits + bits_per_word - 1) / bits_per_word;
            // +1 for the padding word.
            required_words + 1
        };

        if self.bits.len() > min_word_len {
             self.bits.truncate(min_word_len);
        }
        self.bits.shrink_to_fit();
    }
}

// This block implements mutable, in-place operations for `FixedVec` instances
// that have a mutable backend (e.g., `Vec<W>` or `&mut [W]`).
impl<T, W, E, B> FixedVec<T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{

    /// Returns a mutable proxy for an element at a given index.
    ///
    /// This allows for syntax like `*vec.at_mut(i).unwrap() = new_value;`.
    /// The proxy holds a temporary copy of the value, and writes it back into
    /// the vector when it is dropped.
    ///
    /// Returns `None` if the index is out of bounds.
    pub fn at_mut(&mut self, index: usize) -> Option<MutProxy<T, W, E, B>> {
        if index >= self.len {
            return None;
        }
        Some(MutProxy::new(self, index))
    }

    /// Sets the value of an element at a given index.
    pub fn set(&mut self, index: usize, value: T) {
        assert!(index < self.len, "Index out of bounds: expected index < {}, got {}", self.len, index);
        
        let value_w = <T as Storable<W>>::into_word(value);
        let bits_per_word = <W as traits::Word>::BITS;

        let limit = if self.bit_width < bits_per_word {
            W::ONE << self.bit_width
        } else {
            W::max_value()
        };

        if self.bit_width < bits_per_word && value_w >= limit {
            panic!("Value {:?} does not fit in the configured bit_width of {}", value_w, self.bit_width);
        }
        
        unsafe { self.set_unchecked(index, value_w) };
    }

    /// Sets the value of an element at a given index without bounds checking.
    unsafe fn set_unchecked(&mut self, index: usize, value_w: W) {
        let bits_per_word = <W as traits::Word>::BITS;
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / bits_per_word;
        let bit_offset = bit_pos % bits_per_word;
        
        let limbs = self.bits.as_mut();

        if E::IS_LITTLE {
            if bit_offset + self.bit_width <= bits_per_word {
                let word = &mut limbs[word_index];
                *word &= !(self.mask << bit_offset);
                *word |= value_w << bit_offset;
            } else {
                let (left, right) = limbs.split_at_mut(word_index + 1);
                let low_word = &mut left[word_index];
                let high_word = &mut right[0];
                
                *low_word &= !(self.mask << bit_offset);
                *low_word |= value_w << bit_offset;
                
                let bits_in_high = (bit_offset + self.bit_width) - bits_per_word;
                let high_mask = self.mask >> (self.bit_width - bits_in_high);
                *high_word &= !high_mask;
                *high_word |= value_w >> (self.bit_width - bits_in_high);
            }
        } else { // Big-Endian
            if bit_offset + self.bit_width <= bits_per_word {
                // The value fits within a single word.
                let shift = bits_per_word - self.bit_width - bit_offset;
                let mask = self.mask << shift;
                let word = &mut limbs[word_index];
                *word &= !mask.to_be();
                *word |= (value_w << shift).to_be();
            } else {
                // The value spans two words.
                let (left, right) = limbs.split_at_mut(word_index + 1);
                let high_word = &mut left[word_index];
                let low_word = &mut right[0];
                
                let bits_in_first = bits_per_word - bit_offset;
                let bits_in_second = self.bit_width - bits_in_first;

                // 1. Handle the high word (first word)
                let high_mask = (self.mask >> bits_in_second) << (bits_per_word - bits_in_first - bit_offset);
                let high_value = value_w >> bits_in_second;
                *high_word &= !high_mask.to_be();
                *high_word |= (high_value << (bits_per_word - bits_in_first - bit_offset)).to_be();

                // 2. Handle the low word (second word)
                let low_mask = self.mask << (bits_per_word - bits_in_second);
                let low_value = value_w << (bits_per_word - bits_in_second);
                *low_word &= !low_mask.to_be();
                *low_word |= low_value.to_be();
            }
        }
    }

    /// Returns an iterator over non-overlapping mutable chunks of the vector.
    ///
    /// # Panics
    /// Panics if `chunk_size` is 0.
    pub fn chunks_mut(&mut self, chunk_size: usize) -> iter_mut::ChunksMut<T, W, E, B> {
        iter_mut::ChunksMut::new(self, chunk_size)
    }
}

impl<T, W, E, B, B2> PartialEq<FixedVec<T, W, E, B2>> for FixedVec<T, W, E, B>
where
    T: Storable<W> + PartialEq,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    B2: AsRef<[W]>,
{
    /// Checks for equality between two `FixedVec` instances.
    ///
    /// This comparison is highly efficient as it first checks metadata (`len` and
    /// `bit_width`) and then performs a direct bit-level comparison of the
    /// underlying compressed data buffers.
    fn eq(&self, other: &FixedVec<T, W, E, B2>) -> bool {
        if self.len() != other.len() || self.bit_width() != other.bit_width() {
            return false;
        }
        // If metadata matches, the raw bits must match.
        self.as_limbs() == other.as_limbs()
    }
}

/// Implements `PartialEq` for comparing a `FixedVec` with a standard slice.
///
/// This performs an element-by-element comparison, which is less efficient
/// than comparing two `FixedVec`s directly.
impl<T, W, E, B, T2> PartialEq<&[T2]> for FixedVec<T, W, E, B>
where
    T: Storable<W> + PartialEq<T2>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    T2: Clone,
{
    fn eq(&self, other: &&[T2]) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == *b)
    }
}