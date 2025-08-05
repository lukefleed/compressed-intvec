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

use dsi_bitstream::{prelude::Endianness, traits::{BE, LE}};
use mem_dbg::{MemDbg, MemSize};
use std::{error::Error as StdError, fmt, marker::PhantomData};
use traits::{Storable, Word};

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
    /// An `Option` containing the [`FixedVecView`] if the specified range is
    /// within the bounds of the vector, or `None` otherwise.
    pub fn slice(&self, start: usize, len: usize) -> Option<view::FixedVecView<T, W, E, B>> {
        if start.saturating_add(len) > self.len {
            return None;
        }
        Some(view::FixedVecView::new(self, start..start + len))
    }

    /// Splits the vector into two views at a given index.
    ///
    /// # Arguments
    /// * `mid`: The index at which to split the vector.
    ///
    /// # Returns
    /// An `Option` containing a tuple of two [`FixedVecView`]s if `mid` is
    /// within the bounds of the vector, or `None` otherwise.
    pub fn split_at(&self, mid: usize) -> Option<(view::FixedVecView<T, W, E, B>, view::FixedVecView<T, W, E, B>)> {
        if mid > self.len {
            return None;
        }
        let left = view::FixedVecView::new(self, 0..mid);
        let right = view::FixedVecView::new(self, mid..self.len);
        Some((left, right))
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

// This block implements mutable, in-place operations for `FixedVec` instances
// that have a mutable backend (e.g., `Vec<W>` or `&mut [W]`).
impl<T, W, E, B> FixedVec<T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    /// Sets the value of an element at a given index.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds. Also panics if the provided `value`
    /// does not fit within the configured `bit_width`.
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

        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / bits_per_word;
        let bit_offset = bit_pos % bits_per_word;
        
        let limbs = self.bits.as_mut();

        // This is a read-modify-write operation on the underlying words.
        if E::IS_LITTLE {
            if bit_offset + self.bit_width <= bits_per_word {
                // Fast path: element is within a single word.
                let original_word = &mut limbs[word_index];
                *original_word &= !(self.mask << bit_offset); // Clear the bits
                *original_word |= value_w << bit_offset;     // Set the new bits
            } else {
                // Slow path: element spans two words.
                let (left, right) = limbs.split_at_mut(word_index + 1);
                let low_word = &mut left[word_index];
                let high_word = &mut right[0];

                // Clear and set bits in the low word
                *low_word &= !(self.mask << bit_offset);
                *low_word |= value_w << bit_offset;

                // Clear and set bits in the high word
                let bits_in_high = (bit_offset + self.bit_width) - bits_per_word;
                let high_mask = self.mask >> (self.bit_width - bits_in_high);
                *high_word &= !high_mask;
                *high_word |= value_w >> (self.bit_width - bits_in_high);
            }
        } else {
            // Big-Endian logic is significantly more complex to write manually
            // and is often less performant. For now, we delegate to a simpler get/set pattern.
            // TODO: Implement optimized Big-Endian set logic.
            unimplemented!("Mutable Big-Endian operations are not yet implemented.");
        }
    }
}

use std::ops::{Index, IndexMut};

/// Implements `Index` for read-only bracket access (`vec[i]`).
///
/// Due to Rust's borrowing rules, this trait cannot be implemented in a way
/// that returns a true reference (`&T`) without significant overhead or unsafe
/// practices for a bit-packed structure. The `get()` method is the idiomatic
/// and efficient way to access elements.
impl<T, W, E, B> Index<usize> for FixedVec<T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    type Output = T;

    #[inline]
    fn index(&self, _index: usize) -> &Self::Output {
        // This pattern is an anti-pattern in Rust for types that don't have
        // stable references to return. It involves leaking memory or thread_local
        // storage, both of which are undesirable.
        panic!("Direct indexing that returns a temporary owned value is not supported. Use .get(index) instead.");
    }
}


/// Implements `IndexMut` for mutable bracket access (`vec[i] = new_val`).
///
/// This implementation is intentionally omitted. Returning a mutable proxy object
/// (`&mut T`) that correctly interacts with Rust's borrow checker and lifetimes
/// is highly complex and often requires language features like Generic Associated
/// Types (GATs) to be implemented cleanly and safely.
/// The `set(index, value)` method is the recommended way to mutate elements.
impl<T, W, E, B> IndexMut<usize> for FixedVec<T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    #[inline]
    fn index_mut(&mut self, _index: usize) -> &mut Self::Output {
        unimplemented!("IndexMut is not implemented. Please use the .set(index, value) method for mutable access.");
    }
}

use num_traits::ToPrimitive; // Aggiungi questo in cima al file `mod.rs`

// ... (resto del file)

// This block implements resizing and capacity management methods for `FixedVec`
// instances that have an owned `Vec<W>` backend.
impl<T, W, E> FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W> + ToPrimitive, // Aggiunto ToPrimitive per la conversione a u64
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
        // SAFETY: A new vector is always valid.
        Ok(unsafe { Self::new_unchecked(Vec::new(), 0, bit_width) })
    }

    /// Appends an element to the back of the vector.
    ///
    /// This operation may reallocate the underlying buffer if it is full.
    ///
    /// # Panics
    /// Panics if the `value` does not fit within the configured `bit_width`.
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

        // Ensure there is space for data + 1 padding word.
        if required_words + 1 > self.bits.len() {
            // Reallocate with a growth factor.
            let new_capacity_words = (self.bits.len() * 2).max(required_words + 1);
            self.bits.resize(new_capacity_words, W::ZERO);
        }
        
        // Manual bit-writing logic, similar to `set`.
        let word_index = bit_pos / bits_per_word;
        let bit_offset = bit_pos % bits_per_word;

        if E::IS_LITTLE {
            if bit_offset + self.bit_width <= bits_per_word {
                // Fast path: element fits within a single word.
                self.bits[word_index] |= value_w << bit_offset;
            } else {
                // Slow path: element spans two words.
                self.bits[word_index] |= value_w << bit_offset;
                self.bits[word_index + 1] |= value_w >> (bits_per_word - bit_offset);
            }
        } else {
            // Big-Endian logic
            unimplemented!("push for Big-Endian is not yet implemented.");
        }
        
        self.len += 1;
    }

    /// Removes the last element from the vector and returns it, or `None` if it is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        // get() is safe to unwrap because we checked is_empty()
        let value = self.get(self.len - 1).unwrap(); 
        self.len -= 1;
        Some(value)
    }

    /// Removes all elements from the vector.
    pub fn clear(&mut self) {
        self.len = 0;
        self.bits.fill(W::ZERO);
    }
}