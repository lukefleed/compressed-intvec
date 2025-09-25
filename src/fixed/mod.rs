//! A generic, compressed, and randomly accessible vector with fixed-width encoding.
//!
//! This module provides [`FixedVec`] and its thread-safe counterpart, [`AtomicFixedVec`],
//! two data structures for storing sequences of integers where each element is encoded
//! using the same number of bits. This strategy, known as fixed-width encoding,
//! is suitable for data where values are bounded within a known range, as it
//! allows for O(1) random access by directly calculating the memory location of any element.
//!
//! # Core Concepts
//!
//! The design is generic over four parameters: `FixedVec<T, W, E, B>`.
//! - `T`: The element type as seen by the user (e.g., [`u32`], [`i16`]), constrained by the [`Storable`] trait.
//! - `W`: The underlying unsigned integer type used for storage (e.g., [`u64`], [`usize`]), constrained by the [`Word`] trait.
//! - `E`: The [`Endianness`] (e.g., [`LE`] or [`BE`]) for bit-level operations.
//! - `B`: The backing buffer, which abstracts over ownership. This allows [`FixedVec`] to be an owned container (e.g., `Vec<W>`) or a zero-copy, borrowed view (e.g., `&[W]`).
//!
//! ### Immutability and Zero-Copy Views
//! Immutable access is performed in O(1) time by calculating an element's bit-level offset.
//! Structures like [`FixedVecSlice`](slice::FixedVecSlice) provide zero-copy views into a vector, representing
//! a sub-region of the data without requiring data duplication or new allocations.
//!
//! ### Mutability via Proxy Objects
//! Direct mutable references (`&mut T`) to individual bit-packed elements are not possible, as
//! elements do not align with byte boundaries and may not even exist as discrete entities in memory.
//! Instead, mutability is managed through the proxy object pattern. Methods like [`at_mut`](FixedVec::at_mut) return a temporary proxy ([`MutProxy`]) that holds a decoded copy of the value.
//! Modifications are applied to this copy, and the value is automatically encoded and written
//! back into the vector's bitstream when the proxy object goes out of scope (i.e., is dropped).
//!
//! # Core Data Structures
//!
//! - [`FixedVec`]: The primary implementation for single-threaded contexts.
//! - [`AtomicFixedVec`]: A thread-safe variant for concurrent applications. It provides an
//!   API analogous to Rust's standard atomic types ([`load`](AtomicFixedVec::load), [`store`](AtomicFixedVec::store), [`fetch_add`](AtomicFixedVec::fetch_add), etc.).
//!   It uses lock-free atomic instructions for elements contained within a single machine word and
//!   a fine-grained locking strategy (lock striping) for elements that span word boundaries. This
//!   hybrid approach ensures thread safety for any `bit-width` configuration.
//!
//! # Main Components
//!
//! - [`FixedVec`] and [`AtomicFixedVec`]
//! - [`BitWidth`]: An enum to control the bit-width selection strategy. Options include:
//!     - [`BitWidth::Minimal`]: Selects the minimal bit-width required.
//!     - [`BitWidth::PowerOfTwo`]: Rounds up to the nearest power of two.
//!     - [`BitWidth::Explicit`]: Allows specifying a fixed bit-width.
//! - **Builders**: [`FixedVecBuilder`](builder::FixedVecBuilder) and [`FixedVecFromIterBuilder`](builder::FixedVecFromIterBuilder)
//! - **Slices**: [`FixedVecSlice`](slice::FixedVecSlice) for creating immutable or mutable views.
//!
//! # Examples
//!
//! Create a [`FixedVec`] from a slice of data. The builder will
//! automatically determine the minimal number of bits required.
//!
//! ```
//! use compressed_intvec::fixed::{FixedVec, UFixedVec};
//!
//! // The numbers 0-7 can all be represented in 3 bits.
//! let data: Vec<u32> = (0..8).collect();
//!
//! // The builder infers that `bit_width` should be 3.
//! let vec: UFixedVec<u32> = FixedVec::builder()
//!     .build(&data)
//!     .unwrap();
//!
//! assert_eq!(vec.len(), 8);
//! assert_eq!(vec.bit_width(), 3);
//! assert_eq!(vec.get(5), Some(5));
//! ```
//!
//! ## Storing Signed Integers
//!
//! [`FixedVec`] can store signed integers. The underlying storage uses zig-zag encoding,
//! which maps small negative and positive numbers to small unsigned integers.
//!
//! ```
//! use compressed_intvec::fixed::{FixedVec, SFixedVec};
//!
//! // The values range from -2 to 1. Zig-zag encoding maps these to
//! // unsigned values, so the maximum value is 3, which
//! // requires 2 bits.
//! let data: &[i16] = &[-2, -1, 0, 1];
//! let vec: SFixedVec<i16> = FixedVec::builder().build(data).unwrap();
//!
//! assert_eq!(vec.bit_width(), 2);
//! assert_eq!(vec.get(0), Some(-2));
//! assert_eq!(vec.get(3), Some(1));
//! ```
//!
//! # Implementation Notes
//!
//! To ensure safe and efficient memory access, [`FixedVec`] adds one padding
//! word at the end of its storage buffer. This padding prevents out-of-bounds
//! reads in methods like [`get_unchecked`](FixedVec::get_unchecked) and is a
//! prerequisite for unaligned access with [`get_unaligned_unchecked`](FixedVec::get_unaligned_unchecked).
//! The builders handle this padding automatically. When creating a [`FixedVec`] from raw parts,
//! it is the caller's responsibility to ensure this padding is present.
//!
//! # Common Type Aliases
//!
//! To simplify usage, this crate provides several type aliases for common [`FixedVec`]
//! configurations. In most cases, you should prefer using these aliases over the
//! fully generic `FixedVec<T, W, E, B>` struct.
//!
//! ### General-Purpose Aliases
//!
//! These aliases use [`usize`] for the storage word, which is often the most
//! efficient choice for the target architecture.
//!
//! - [`UFixedVec<T>`]: For unsigned integers (e.g., [`u8`], [`u16`], [`u32`]).
//! - [`SFixedVec<T>`]: For signed integers (e.g., [`i8`], [`i16`], [`i32`]).
//!
//! ### Concrete Aliases for [`u64`]/[`i64`]
//!
//! These aliases are specialized for [`u64`]/[`i64`] elements stored in [`u64`] words:
//!
//! - [`LEFixedVec`]: [`u64`] elements, Little-Endian.
//! - [`BEFixedVec`]: [`u64`] elements, Big-Endian.
//! - [`LESFixedVec`]: [`i64`] elements, Little-Endian.
//! - [`BESFixedVec`]: [`i64`] elements, Big-Endian.
//!
//! The [`atomic`] module provides a similar set of aliases like [`UAtomicFixedVec`]
//! and [`SAtomicFixedVec`].

// Declare and export submodules.
#[macro_use]
pub mod macros;
pub mod builder;
pub mod iter;
pub mod iter_mut;
pub mod parallel;
pub mod proxy;
pub mod slice;
pub mod traits;

pub mod atomic;

// Conditionally compile the serde module.
#[cfg(feature = "serde")]
mod serde;

use dsi_bitstream::{
    prelude::Endianness,
    traits::{BE, LE},
};
use mem_dbg::{MemDbg, MemSize};
use num_traits::ToPrimitive;
use std::{error::Error as StdError, fmt, iter::FromIterator, marker::PhantomData};
use traits::{Storable, Word};

use crate::fixed::{proxy::MutProxy, traits::DefaultParams};

// Re-export atomic aliases for convenience.
pub use atomic::{AtomicFixedVec, SAtomicFixedVec, UAtomicFixedVec};

// Type aliases for common `FixedVec` configurations.

/// A [`FixedVec`] for unsigned integers with a `usize` word and Little-Endian layout.
///
/// This is a convenient alias for a common configuration. The element type `T`
/// can be any unsigned integer that implements [`Storable`], such as `u8`, `u16`,
/// `u32`, [`u64`], `u128`, or `usize`.
pub type UFixedVec<T, B = Vec<usize>> = FixedVec<T, usize, LE, B>;

/// A [`FixedVec`] for signed integers with a `usize` word and Little-Endian layout.
///
/// This alias is suitable for general-purpose use with signed types like `i8`,
/// `i16`, `i32`, [`i64`], `i128`, or `isize`.
pub type SFixedVec<T, B = Vec<usize>> = FixedVec<T, usize, LE, B>;

// --- Concrete Aliases for [`u64`]/[`i64`] elements ---
// These are provided for common use cases and backward compatibility.

/// A [`FixedVec`] for [`u64`] elements with a [`u64`] backend and Little-Endian layout.
pub type LEFixedVec<B = Vec<u64>> = FixedVec<u64, u64, LE, B>;
/// A [`FixedVec`] for [`i64`] elements with a [`u64`] backend and Little-Endian layout.
pub type LESFixedVec<B = Vec<u64>> = FixedVec<i64, u64, LE, B>;

/// A [`FixedVec`] for [`u64`] elements with a [`u64`] backend and Big-Endian layout.
pub type BEFixedVec<B = Vec<u64>> = FixedVec<u64, u64, BE, B>;
/// A [`FixedVec`] for [`i64`] elements with a [`u64`] backend and Big-Endian layout.
pub type BESFixedVec<B = Vec<u64>> = FixedVec<i64, u64, BE, B>;

/// Specifies the strategy for determining the number of bits for each integer.
///
/// This enum controls how the bit width of a [`FixedVec`] is determined during
/// its construction. The choice of strategy involves a trade-off between memory
/// usage and random access performance.
///
/// # Performance Considerations
///
/// - **[`Minimal`](BitWidth::Minimal) vs. [`PowerOfTwo`](BitWidth::PowerOfTwo)**: While [`Minimal`](BitWidth::Minimal) provides the most compact
///   storage, [`PowerOfTwo`](BitWidth::PowerOfTwo) can offer better performance for certain operations.
///   When the `bit_width` is a power of two (e.g., 8, 16, 32) and aligns with
///   word boundaries, some in-place operations like [`map_in_place`](FixedVec::map_in_place) can use a
///   faster, word-at-a-time algorithm.
///
/// - **[`Explicit`](BitWidth::Explicit)**: This is the fastest strategy at construction time, as it
///   avoids the need to iterate through the input data to find the maximum
///   value. Use this when the required bit width is known in advance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BitWidth {
    /// Use the minimum number of bits required to store the largest value.
    ///
    /// This strategy analyzes the input data to find the maximum value and sets
    /// the bit width accordingly. It ensures the most compact memory representation.
    #[default]
    Minimal,

    /// Round the bit width up to the next power of two (e.g., 8, 16, 32).
    ///
    /// This strategy can improve random access performance for some in-place
    /// operations, as they can be implemented more efficiently with bit-shift operations on aligned data.
    PowerOfTwo,

    /// Use a specific number of bits.
    ///
    /// This strategy enforces a user-defined bit width. If any value in the
    /// input data exceeds what can be stored in this many bits, the build
    /// process will fail.
    Explicit(usize),
}

/// Defines errors that can occur during [`FixedVec`] operations.
#[derive(Debug)]
pub enum Error {
    /// A value in the input data is too large to be stored with the configured
    /// bit width.
    ValueTooLarge {
        /// The value that could not be stored.
        value: u128,
        /// The index of the value in the input data.
        index: usize,
        /// The configured number of bits.
        bit_width: usize,
    },
    /// A parameter is invalid for the requested operation.
    ///
    /// This typically occurs if `bit_width` is larger than the storage word size
    /// or if a provided buffer is too small.
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
                "value {value} at index {index} does not fit in {bit_width} bits"
            ),
            Error::InvalidParameters(s) => write!(f, "Invalid parameters: {s}"),
        }
    }
}

impl StdError for Error {}

/// A compressed vector of integers with fixed-width encoding.
///
/// [`FixedVec`] stores a sequence of integers where each element is encoded using
/// the same number of bits. This allows for O(1) random access by calculating
/// the memory location of any element. It is suitable for data where values are
/// bounded within a known range.
///
/// # Type Parameters
///
/// - `T`: The integer type for the elements (e.g., `u32`, `i16`). It must
///   implement the [`Storable`] trait.
/// - `W`: The underlying storage word (e.g., [`u64`], `usize`). It must implement
///   the [`Word`] trait.
/// - `E`: The [`Endianness`] (e.g., [`LE`] or [`BE`]) for bit-level operations.
/// - `B`: The backend storage buffer, such as `Vec<W>` for an owned vector or
///   `&[W]` for a borrowed view.
///
/// For common configurations, several [type aliases] are provided.
///
/// [type aliases]: crate::fixed#type-aliases
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct FixedVec<T: Storable<W>, W: Word, E: Endianness, B: AsRef<[W]> = Vec<W>> {
    /// The underlying storage for the bit-packed data.
    pub(crate) bits: B,
    /// The number of bits used to encode each element.
    pub(crate) bit_width: usize,
    /// A bitmask with the lowest `bit_width` bits set to one.
    pub(crate) mask: W,
    /// The number of elements in the vector.
    pub(crate) len: usize,
    /// Zero-sized markers for the generic type parameters `T`, `W`, and `E`.
    pub(crate) _phantom: PhantomData<(T, W, E)>,
}

// [`FixedVec`] builder implementation.
impl<T, W, E> FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
{
    /// Creates a builder for constructing a [`FixedVec`] from a slice.
    ///
    /// The builder provides methods to customize the vector's properties, such
    /// as the [`BitWidth`] strategy.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, BitWidth, UFixedVec};
    ///
    /// let data: &[u32] = &[10, 20, 30, 40, 50];
    /// let vec: UFixedVec<u32> = FixedVec::builder()
    ///     .bit_width(BitWidth::Minimal)
    ///     .build(data)
    ///     .unwrap();
    ///
    /// assert_eq!(vec.get(1), Some(20));
    /// ```
    pub fn builder() -> builder::FixedVecBuilder<T, W, E> {
        builder::FixedVecBuilder::new()
    }

    /// Creates a builder for constructing a [`FixedVec`] from an iterator.
    ///
    /// This builder is suitable for large datasets provided by an iterator,
    /// as it processes the data in a streaming fashion.
    ///
    /// # Arguments
    ///
    /// * `iter`: An iterator that yields the integer values.
    /// * `bit_width`: The number of bits to use for each element. This must be
    ///   specified, as the builder cannot analyze the data in advance.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let data = 0..100u32;
    /// let vec: UFixedVec<u32> = FixedVec::from_iter_builder(data, 7)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(vec.len(), 100);
    /// assert_eq!(vec.bit_width(), 7);
    /// assert_eq!(vec.get(99), Some(99));
    /// ```
    pub fn from_iter_builder<I: IntoIterator<Item = T>>(
        iter: I,
        bit_width: usize,
    ) -> builder::FixedVecFromIterBuilder<T, W, E, I> {
        builder::FixedVecFromIterBuilder::new(iter, bit_width)
    }

    /// Creates an owned [`FixedVec`] from a slice of data.
    ///
    /// This method is a convenience wrapper around the builder API, using
    /// the default bit width strategy ([`BitWidth::Minimal`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let data: &[u32] = &[10, 20, 30];
    ///
    /// let vec: UFixedVec<u32> = FixedVec::from_slice(data).unwrap();
    ///
    /// assert_eq!(vec.len(), 3);
    /// assert_eq!(vec.bit_width(), 5); // 30 fits in 5 bits
    /// assert_eq!(vec.get(0), Some(10));
    /// ```
    pub fn from_slice(slice: &[T]) -> Result<Self, Error> {
        Self::builder().build(slice)
    }
}

// Core immutable API.
impl<T, W, E, B> FixedVec<T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    /// Creates a [`FixedVec`] from its raw components.
    ///
    /// This constructor allows for the creation of a zero-copy view over an
    /// existing buffer.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::InvalidParameters`] if `bit_width` is larger than
    /// the word size or if the `bits` buffer is too small to hold the specified
    /// number of elements and the required padding.
    ///
    /// # Implementation Notes
    ///
    /// The provided `bits` buffer must contain at least one extra "padding"
    /// word at the end. This padding is essential to prevent out-of-bounds
    /// memory access in methods like `get_unchecked`.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// // 3 elements * 5 bits = 15 bits. This requires 1 data word.
    /// // We need 1 extra word for padding.
    /// let mut buffer = vec![0_usize; 2];
    ///
    /// // Manually encode data into the first word.
    /// // 10 (01010), 20 (10100), 30 (11110)
    /// buffer[0] = 0b11110_10100_01010;
    ///
    /// let vec = UFixedVec::<u32, _>::from_parts(&buffer, 3, 5).unwrap();
    /// assert_eq!(vec.get(0), Some(10));
    /// assert_eq!(vec.get(1), Some(20));
    /// assert_eq!(vec.get(2), Some(30));
    /// ```
    pub fn from_parts(bits: B, len: usize, bit_width: usize) -> Result<Self, Error> {
        if bit_width > <W as traits::Word>::BITS {
            return Err(Error::InvalidParameters(format!(
                "bit_width ({}) cannot be greater than the word size ({})",
                bit_width,
                <W as traits::Word>::BITS
            )));
        }

        let total_bits = len * bit_width;
        let data_words = total_bits.div_ceil(<W as traits::Word>::BITS);

        // Essential safety check: ensure the buffer is large enough for the data
        // AND the 1 padding word required.
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

    /// Returns `true` if the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of bits used to encode each element.
    #[inline]
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// Returns a read-only slice of the underlying storage words.
    #[inline]
    pub fn as_limbs(&self) -> &[W] {
        self.bits.as_ref()
    }

    /// Returns a raw pointer to the start of the underlying buffer and its
    /// length in words.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the buffer is not mutated in a way that
    /// violates the invariants of the [`FixedVec`] while the pointer is active.
    pub fn as_raw_parts(&self) -> (*const W, usize) {
        let slice = self.bits.as_ref();
        (slice.as_ptr(), slice.len())
    }

    /// Creates a [`FixedVec`] from its raw components without performing checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the following invariants are met:
    /// 1. The `bits` buffer must be large enough to hold `len * bit_width` bits.
    /// 2. The `bits` buffer must have at least one extra padding word at the end
    ///    to prevent out-of-bounds reads during access.
    /// 3. `bit_width` must not be greater than the number of bits in `W`.
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

    /// Returns the element at the specified index, or `None` if the index is
    /// out of bounds.
    ///
    /// This operation is O(1).
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let data: &[u32] = &[10, 20, 30];
    /// let vec: UFixedVec<u32> = FixedVec::builder().build(data).unwrap();
    ///
    /// assert_eq!(vec.get(1), Some(20));
    /// assert_eq!(vec.get(3), None);
    /// ```
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Returns the element at the specified index without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    /// The `index` must be less than the vector's `len`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        debug_assert!(index < self.len);

        let bits_per_word = <W as traits::Word>::BITS;
        // Optimization: if bit_width matches word size, access is trivial.
        if self.bit_width == bits_per_word {
            let val = *self.as_limbs().get_unchecked(index);
            // Apply endianness correction if needed.
            let final_val = if E::IS_BIG { val.to_be() } else { val };
            return <T as Storable<W>>::from_word(final_val);
        }

        // Calculate the bit position and word index for the element.
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / bits_per_word;
        let bit_offset = bit_pos % bits_per_word;

        let limbs = self.as_limbs();
        let final_word: W;

        // The logic is specialized for endianness to maximize performance.
        if E::IS_LITTLE {
            // Fast path: the element is fully contained within a single word.
            if bit_offset + self.bit_width <= bits_per_word {
                final_word = (*limbs.get_unchecked(word_index) >> bit_offset) & self.mask;
            } else {
                // Slow path: the element spans two words.
                // Read the low part from the first word and the high part from the next.
                let low = *limbs.get_unchecked(word_index) >> bit_offset;
                let high = *limbs.get_unchecked(word_index + 1) << (bits_per_word - bit_offset);
                final_word = (low | high) & self.mask;
            }
        } else {
            // Big-Endian logic requires byte-swapping the words before extraction.
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

    /// Returns the element at `index` using unaligned memory access.
    ///
    /// This method can be significantly faster for random access. It performs a
    /// single, potentially unaligned read of a [`Word`] and extracts the value.
    ///
    /// # Performance
    ///
    /// This method is generally the fastest way to perform random reads on
    /// Little-Endian architectures, especially when the `bit_width` is not a
    /// power of two.
    ///
    /// - [`get_unchecked`](Self::get_unchecked): May require reading one or two separate machine words
    ///   and combining them with bit shifts. This is fast if the data is already
    ///   in the CPU cache.
    /// - [`get_unaligned_unchecked`](Self::get_unaligned_unchecked): Performs a single memory read that may
    ///   cross word boundaries. It often results in fewer instructions and better throughput than the two
    ///   separate reads of [`get_unchecked`](Self::get_unchecked), especially in memory-bound scenarios.
    ///
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    /// The [`FixedVec`] must have been constructed with sufficient padding (at
    /// least one [`Word`]) to guarantee that the unaligned read does not go
    /// past the allocated buffer. This padding is guaranteed by the default
    /// builders.
    ///
    /// # Implementation Notes
    ///
    /// For Big-Endian systems, or for `bit_width` values that are too large for a
    /// single unaligned read to be sufficient (i.e., `bit_width > W::BITS - 7`),
    /// this method falls back to the standard [`get_unchecked`](Self::get_unchecked) implementation.
    #[inline(always)]
    pub unsafe fn get_unaligned_unchecked(&self, index: usize) -> T {
        debug_assert!(index < self.len);

        if E::IS_LITTLE {
            let bits_per_word = <W as Word>::BITS;
            if self.bit_width == bits_per_word {
                return self.get_unchecked(index);
            }

            // A single unaligned read of a `Word` is only guaranteed to be correct
            // if the value spans at most `W::BITS` bits from the start of the read.
            // Since the value can start at any bit offset (0-7) within the first
            // byte, the condition is `bit_rem + bit_width <= W::BITS`. The worst-case
            // `bit_rem` is 7, so we must have `7 + bit_width <= W::BITS`.
            // If `bit_width > W::BITS - 7`, we fall back to the safe, two-read implementation.
            if self.bit_width > bits_per_word.saturating_sub(7) {
                return self.get_unchecked(index);
            }

            let bit_pos = index * self.bit_width;
            let byte_pos = bit_pos / 8;
            let bit_rem = bit_pos % 8;

            let limbs_ptr = self.as_limbs().as_ptr() as *const u8;

            // Perform an unaligned read from the calculated byte position.
            let word: W = (limbs_ptr.add(byte_pos) as *const W).read_unaligned();
            let extracted_word = word >> bit_rem;

            <T as Storable<W>>::from_word(extracted_word & self.mask)
        } else {
            // For Big-Endian, the logic for unaligned reads is complex and
            // architecture-dependent. Fall back to the standard `get_unchecked`.
            self.get_unchecked(index)
        }
    }

    /// Returns an iterator over the elements of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let data: &[u32] = &[10, 20, 30];
    /// let vec: UFixedVec<u32> = FixedVec::builder().build(data).unwrap();
    /// let mut iter = vec.iter();
    ///
    /// for (i, value) in iter.enumerate() {
    ///     assert_eq!(Some(value), vec.get(i));
    /// }
    /// ```
    pub fn iter(&self) -> iter::FixedVecIter<'_, T, W, E, B> {
        iter::FixedVecIter::new(self)
    }

    /// Returns an unchecked iterator over the elements of the vector.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the iterator is not advanced beyond the
    /// vector's length.
    pub unsafe fn iter_unchecked(&self) -> iter::FixedVecUncheckedIter<'_, T, W, E, B> {
        iter::FixedVecUncheckedIter::new(self)
    }

    /// Creates an immutable view (slice) of a sub-region of the vector.
    ///
    /// Returns `None` if the specified range is out of bounds.
    ///
    /// # Arguments
    /// * `start`: The starting index of the slice.
    /// * `len`: The number of elements in the slice.
    pub fn slice(&self, start: usize, len: usize) -> Option<slice::FixedVecSlice<&Self>> {
        if start.saturating_add(len) > self.len {
            return None;
        }
        Some(slice::FixedVecSlice::new(self, start..start + len))
    }

    /// Splits the vector into two immutable views at a given index.
    ///
    /// Returns `None` if `mid` is out of bounds.
    ///
    /// # Arguments
    /// * `mid`: The index at which to split the vector.
    pub fn split_at(
        &self,
        mid: usize,
    ) -> Option<(slice::FixedVecSlice<&Self>, slice::FixedVecSlice<&Self>)> {
        if mid > self.len {
            return None;
        }
        let left = slice::FixedVecSlice::new(self, 0..mid);
        let right = slice::FixedVecSlice::new(self, mid..self.len);
        Some((left, right))
    }

    /// Returns an iterator over non-overlapping chunks of the vector.
    ///
    /// Each chunk is a [`FixedVecSlice`] of length `chunk_size`, except for the
    /// last chunk, which may be shorter.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// [`FixedVecSlice`]: slice::FixedVecSlice
    pub fn chunks(&self, chunk_size: usize) -> iter::Chunks<'_, T, W, E, B> {
        iter::Chunks::new(self, chunk_size)
    }

    /// Returns an iterator over all contiguous windows of length `size`.
    ///
    /// The windows overlap. If the vector is shorter than `size`, the iterator
    /// returns no values.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed_vec;
    ///
    /// let vec = fixed_vec![1u32, 2, 3, 4, 5];
    /// let mut windows = vec.windows(3);
    ///
    /// let slice1 = vec.slice(0, 3).unwrap();
    /// let slice2 = vec.slice(1, 3).unwrap();
    /// let slice3 = vec.slice(2, 3).unwrap();
    ///
    /// assert_eq!(windows.next().unwrap(), slice1);
    /// assert_eq!(windows.next().unwrap(), slice2);
    /// assert_eq!(windows.next().unwrap(), slice3);
    /// assert!(windows.next().is_none());
    /// ```
    pub fn windows(&self, size: usize) -> iter::Windows<'_, T, W, E, B> {
        assert!(size != 0, "window size cannot be zero");
        iter::Windows::new(self, size)
    }

    /// Returns a raw pointer to the storage word containing the start of an element.
    ///
    /// This method returns a pointer to the [`Word`] in the backing buffer where
    /// the data for the element at `index` begins.
    ///
    /// Returns `None` if `index` is out of bounds.
    ///
    /// # Safety
    ///
    /// This method is safe as it only returns a raw pointer. Dereferencing the
    /// pointer is `unsafe`. The caller must ensure that the pointer is not used
    /// after the [`FixedVec`] is dropped or modified.
    pub fn addr_of(&self, index: usize) -> Option<*const W> {
        if index >= self.len {
            return None;
        }

        let bit_pos = index * self.bit_width;
        let word_idx = bit_pos / <W as Word>::BITS;

        let limbs = self.as_limbs();
        if word_idx < limbs.len() {
            Some(limbs.as_ptr().wrapping_add(word_idx))
        } else {
            None
        }
    }

    /// Hints to the CPU to prefetch the data for the element at `index` into cache.
    ///
    /// This method uses an intrinsic to reduce memory latency for predictable
    /// access patterns. It only has an effect on architectures that support it
    /// (e.g., x86, x86-64) and compiles to a no-op on other platforms.
    pub fn prefetch(&self, index: usize) {
        if index >= self.len {
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

            let bit_pos = index * self.bit_width;
            let byte_pos = bit_pos / 8;

            let limbs_ptr = self.as_limbs().as_ptr() as *const i8;

            // SAFETY: Bounds check on `index` ensures `byte_pos` is within the
            // allocated buffer (including padding). The pointer is valid.
            unsafe {
                // Prefetch into all cache levels (a good general-purpose default).
                _mm_prefetch(limbs_ptr.add(byte_pos), _MM_HINT_T0);
            }
        }
    }

    /// Binary searches this vector for a given element.
    ///
    /// If the value is found, returns `Ok(usize)` with the index of the
    /// matching element. If the value is not found, returns `Err(usize)` with
    /// the index where the value could be inserted to maintain order.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let data: &[u32] = &[10, 20, 30, 40, 50];
    /// let vec: UFixedVec<u32> = FixedVec::builder().build(data).unwrap();
    ///
    /// assert_eq!(vec.binary_search(&30), Ok(2));
    /// assert_eq!(vec.binary_search(&35), Err(3));
    /// ```
    pub fn binary_search(&self, value: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
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

    /// Binary searches this vector with a key extraction function.
    ///
    /// This method is useful when searching for a value of a different type
    /// than the elements of the slice. The function `f` is used to extract a
    /// key of type `K` from an element, which is then compared to `key`.
    pub fn binary_search_by_key<K: Ord, F>(&self, key: &K, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> K,
    {
        self.binary_search_by(|probe| f(*probe).cmp(key))
    }

    /// Binary searches this vector with a custom comparison function.
    ///
    /// The comparator function `f` should return an `Ordering` indicating
    /// the relation of a probe element to the value being searched for.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed_vec;
    /// use std::cmp::Ordering;
    ///
    /// let vec = fixed_vec![0u32, 1, 1, 2, 3];
    ///
    /// // Search for a value by comparing it to the probe.
    /// let result = vec.binary_search_by(|probe| probe.cmp(&1));
    /// assert!(matches!(result, Ok(1) | Ok(2)));
    ///
    /// let result_not_found = vec.binary_search_by(|probe| probe.cmp(&4));
    /// assert_eq!(result_not_found, Err(5));
    /// ```
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> std::cmp::Ordering,
    {
        let mut low = 0;
        let mut high = self.len();

        while low < high {
            let mid = low + (high - low) / 2;
            // SAFETY: The loop invariants ensure `mid` is always in bounds.
            let mid_val = unsafe { self.get_unchecked(mid) };

            match f(&mid_val) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => return Ok(mid),
                std::cmp::Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }

    /// Returns the index of the partition point of the vector.
    ///
    /// The vector is partitioned according to the predicate `pred`. This means
    /// all elements for which `pred` returns `true` are on the left of the
    /// partition point, and all elements for which `pred` returns `false` are
    /// on the right.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed_vec;
    ///
    /// let vec = fixed_vec![0u32, 1, 2, 3, 4, 5];
    ///
    /// // Find the partition point for elements `< 3`.
    /// let partition_idx = vec.partition_point(|&x| x < 3);
    /// assert_eq!(partition_idx, 3);
    /// ```
    pub fn partition_point<P>(&self, mut pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
    {
        let mut len = self.len();
        let mut left = 0;

        while len > 0 {
            let half = len / 2;
            let mid = left + half;
            // SAFETY: The loop invariants ensure `mid` is always in bounds.
            let value = unsafe { self.get_unchecked(mid) };

            if pred(&value) {
                left = mid + 1;
                len -= half + 1;
            } else {
                len = half;
            }
        }
        left
    }
}

/// Allows iterating over a borrowed [`FixedVec`] (e.g., `for val in &my_vec`).
impl<'a, T, W, E, B> IntoIterator for &'a FixedVec<T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
{
    type Item = T;
    type IntoIter = iter::FixedVecIter<'a, T, W, E, B>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Allows iterating over an owned [`FixedVec`], consuming it.
impl<T, W, E> IntoIterator for FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W> + 'static,
    W: Word,
    E: Endianness,
{
    type Item = T;
    type IntoIter = iter::FixedVecIntoIter<'static, T, W, E>;

    /// Consumes the vector and returns an iterator over its elements.
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
    /// Creates a [`FixedVec`] by collecting elements from an iterator.
    ///
    /// The bit width is determined automatically using the [`BitWidth::Minimal`]
    /// strategy. This requires collecting the iterator into a temporary `Vec<T>`
    /// to analyze its contents before compression.
    ///
    /// # Memory Usage
    ///
    /// Because this implementation must first collect all items into a temporary
    /// `Vec<T>` to determine the optimal `bit_width`, it may lead to a temporary
    /// peak in memory usage that is roughly double the size of the uncompressed
    /// data.
    ///
    /// For very large datasets where this memory overhead is a concern, it is
    /// recommended to use [`FixedVec::from_iter_builder`] instead. The builder
    /// allows for streaming construction but requires the `bit_width` to be
    /// specified manually.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let data = 0u32..100;
    /// let vec: UFixedVec<u32> = data.collect();
    ///
    /// assert_eq!(vec.len(), 100);
    /// assert_eq!(vec.get(50), Some(50));
    /// ```
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let data: Vec<T> = iter.into_iter().collect();
        Self::builder().build(&data).unwrap()
    }
}

impl<T, W, E> Default for FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
{
    /// Creates an empty [`FixedVec`] with a default `bit_width` of 1.
    fn default() -> Self {
        // SAFETY: An empty vector with a valid bit_width is always safe.
        unsafe { Self::new_unchecked(Vec::new(), 0, 1) }
    }
}

// Methods for owned vectors (`B = Vec<W>`).
impl<T, W, E> FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W> + ToPrimitive,
    W: Word,
    E: Endianness,
{
    /// Creates a new, empty [`FixedVec`] with a specified bit width.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::InvalidParameters`] if `bit_width` is greater than
    /// the number of bits in the storage word `W`.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let vec: UFixedVec<u32> = FixedVec::new(8).unwrap();
    /// assert!(vec.is_empty());
    /// assert_eq!(vec.bit_width(), 8);
    /// ```
    pub fn new(bit_width: usize) -> Result<Self, Error> {
        if bit_width > <W as traits::Word>::BITS {
            return Err(Error::InvalidParameters(format!(
                "bit_width ({}) cannot be greater than the word size ({})",
                bit_width,
                <W as traits::Word>::BITS
            )));
        }
        Ok(unsafe { Self::new_unchecked(Vec::new(), 0, bit_width) })
    }

    /// Appends an element to the end of the vector.
    ///
    /// This operation has amortized O(1) complexity.
    ///
    /// # Panics
    ///
    /// Panics if the `value` is too large to be represented by the configured
    /// `bit_width`.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let mut vec: UFixedVec<u32> = FixedVec::new(4).unwrap();
    /// vec.push(10);
    /// vec.push(15);
    ///
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(vec.get(1), Some(15));
    /// ```
    #[inline(always)]
    pub fn push(&mut self, value: T) {
        let value_w = <T as Storable<W>>::into_word(value);

        // Check if the value fits within the configured bit width.
        if (value_w & !self.mask) != W::ZERO {
            panic!(
                "Value {:?} does not fit in the configured bit_width of {}",
                value_w, self.bit_width
            );
        }

        let bits_per_word = <W as traits::Word>::BITS;
        // Grow the underlying buffer if the new element would exceed its bit capacity.
        if (self.len + 1) * self.bit_width > self.bits.len() * bits_per_word {
            self.bits.push(W::ZERO);
        }

        // SAFETY: We have ensured the value fits and the buffer has capacity.
        unsafe {
            self.set_unchecked(self.len, value_w);
        }

        self.len += 1;
    }

    /// Removes the last element from the vector and returns it.
    ///
    /// Returns `None` if the vector is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed_vec;
    ///
    /// let mut vec = fixed_vec![1u32, 2, 3];
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec.len(), 2);
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let value = self.get(self.len - 1).unwrap();
        self.len -= 1;
        Some(value)
    }

    /// Removes all elements from the vector, leaving the capacity unchanged.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Creates a new, empty [`FixedVec`] with a specified bit width and capacity.
    ///
    /// The vector will be able to hold at least `capacity` elements without
    /// reallocating.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::InvalidParameters`] if `bit_width` is greater than
    /// the number of bits in the storage word `W`.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let vec: UFixedVec<u32> = FixedVec::with_capacity(5, 1000).unwrap();
    /// assert!(vec.capacity() >= 1000);
    /// ```
    pub fn with_capacity(bit_width: usize, capacity: usize) -> Result<Self, Error> {
        if bit_width > <W as traits::Word>::BITS {
            return Err(Error::InvalidParameters(format!(
                "bit_width ({}) cannot be greater than the word size ({})",
                bit_width,
                <W as traits::Word>::BITS
            )));
        }
        let bits_per_word = <W as traits::Word>::BITS;
        let total_bits = capacity.saturating_mul(bit_width);
        let num_words = total_bits.div_ceil(bits_per_word);

        let buffer = if capacity == 0 {
            Vec::new()
        } else {
            Vec::with_capacity(num_words + 1) // +1 for padding
        };

        Ok(unsafe { Self::new_unchecked(buffer, 0, bit_width) })
    }

    /// Returns the number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        if self.bit_width == 0 {
            return usize::MAX;
        }
        let word_capacity = self.bits.capacity();
        if word_capacity <= 1 {
            return 0; // Not enough for data + padding.
        }
        // Subtract padding words before calculating element capacity.
        ((word_capacity - 1) * <W as traits::Word>::BITS) / self.bit_width
    }

    /// Returns the capacity of the underlying storage in words.
    pub fn word_capacity(&self) -> usize {
        self.bits.capacity()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows [`usize`].
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let mut vec: UFixedVec<u32> = FixedVec::new(4).unwrap();
    /// vec.reserve(100);
    /// assert!(vec.capacity() >= 100);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        let target_element_capacity = self.len.saturating_add(additional);
        if self.capacity() >= target_element_capacity {
            return;
        }
        let bits_per_word = <W as Word>::BITS;
        let required_total_bits = target_element_capacity.saturating_mul(self.bit_width);
        let required_data_words = required_total_bits.div_ceil(bits_per_word);
        let required_word_capacity = required_data_words + 1; // +1 for padding

        let current_len = self.bits.len();
        if self.bits.capacity() < required_word_capacity {
            self.bits.reserve(required_word_capacity - current_len);
        }
    }

    /// Resizes the vector so that its length is equal to `new_len`.
    ///
    /// If `new_len` is greater than the current length, the vector is extended
    /// by appending `value`. If `new_len` is less than the current length, the
    /// vector is truncated.
    ///
    /// # Panics
    ///
    /// Panics if the `value` used for filling does not fit in the configured
    /// `bit_width`.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let mut vec = UFixedVec::<u32>::new(4).unwrap();
    /// vec.push(1);
    /// vec.push(2);
    ///
    /// vec.resize(4, 10);
    /// assert_eq!(vec.len(), 4);
    /// assert_eq!(vec.get(3), Some(10));
    ///
    /// vec.resize(1, 0);
    /// assert_eq!(vec.len(), 1);
    /// assert_eq!(vec.get(0), Some(1));
    /// ```
    #[inline(always)]
    pub fn resize(&mut self, new_len: usize, value: T) {
        if new_len > self.len {
            let value_w = <T as Storable<W>>::into_word(value);

            if (value_w & !self.mask) != W::ZERO {
                panic!(
                    "Value {:?} does not fit in the configured bit_width of {}",
                    value_w, self.bit_width
                );
            }

            let bits_per_word = <W as traits::Word>::BITS;
            let required_total_bits = new_len * self.bit_width;
            let required_data_words = required_total_bits.div_ceil(bits_per_word);
            let required_vec_len = required_data_words.saturating_add(1); // Padding

            if self.bits.len() < required_vec_len {
                self.bits.resize(required_vec_len, W::ZERO);
            }

            for i in self.len..new_len {
                unsafe {
                    self.set_unchecked(i, value_w);
                }
            }
            self.len = new_len;
        } else {
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
            let required_words = required_total_bits.div_ceil(bits_per_word);
            required_words + 1 // +1 for padding
        };

        if self.bits.len() > min_word_len {
            self.bits.truncate(min_word_len);
        }
        self.bits.shrink_to_fit();
    }

    /// Removes and returns the element at `index`, shifting all elements after
    /// it to the left.
    ///
    /// This operation is O(n), where n is the number of elements after `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Complexity
    ///
    /// This operation has a complexity of O(L), where L is the number of bits
    /// that need to be shifted. In the worst case (removing the first element),
    /// this is proportional to the total number of bits in the vector
    /// (`self.len() * self.bit_width()`).
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "remove: index out of bounds");

        let value_to_return = self.get(index).unwrap();

        let start_bit = index * self.bit_width;
        let end_bit = self.len * self.bit_width;
        let total_bits_to_shift = end_bit - (start_bit + self.bit_width);

        if total_bits_to_shift > 0 {
            self.shift_bits_left(start_bit, self.bit_width, total_bits_to_shift);
        }

        self.len -= 1;

        value_to_return
    }

    /// Inserts an element at `index`, shifting all elements after it to the right.
    ///
    /// This operation is O(n), where n is the number of elements after `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds or if the `element` is too large to
    /// be represented by the configured `bit_width`.
    ///
    /// # Complexity
    ///
    /// This operation has a complexity of O(L), where L is the number of bits
    /// that need to be shifted. In the worst case (inserting at the beginning),
    /// this is proportional to the total number of bits in the vector
    /// (`self.len() * self.bit_width()`).
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds or if the `element` is too large to
    /// be represented by the configured `bit_width`.
    pub fn insert(&mut self, index: usize, element: T) {
        assert!(index <= self.len, "insert: index out of bounds");
        let value_w = <T as Storable<W>>::into_word(element);
        let bits_per_word = <W as Word>::BITS;
        let limit = if self.bit_width < bits_per_word {
            W::ONE << self.bit_width
        } else {
            W::max_value()
        };
        if self.bit_width < bits_per_word && value_w >= limit {
            panic!(
                "Value {:?} does not fit in the configured bit_width of {}",
                value_w, self.bit_width
            );
        }
        self.reserve(1);
        let start_shift_bit = index * self.bit_width;
        let num_bits_to_move = (self.len - index) * self.bit_width;
        if num_bits_to_move > 0 {
            self.shift_bits_right(start_shift_bit, self.bit_width, num_bits_to_move);
        }
        self.len += 1;
        unsafe {
            self.set_unchecked(index, value_w);
        }
    }

    /// Shifts a range of bits to the left in-place.
    fn shift_bits_left(&mut self, start_bit: usize, shift_amount: usize, num_bits_to_move: usize) {
        if num_bits_to_move == 0 {
            return;
        }

        let bits_per_word = <W as Word>::BITS;

        // Fast path for word-aligned shifts, using `copy_within`.
        if shift_amount % bits_per_word == 0 {
            let start_write_word = start_bit / bits_per_word;
            let start_read_word = (start_bit + shift_amount) / bits_per_word;
            let num_words_to_move = num_bits_to_move.div_ceil(bits_per_word);

            if start_read_word < self.bits.len() {
                let read_end = (start_read_word + num_words_to_move).min(self.bits.len());
                self.bits
                    .copy_within(start_read_word..read_end, start_write_word);
            }
            return;
        }

        // Slow path for unaligned shifts (word-at-a-time).
        let shift_rem = shift_amount % bits_per_word;
        let inv_shift_rem = bits_per_word - shift_rem;

        let start_write_bit = start_bit;
        let end_write_bit = start_bit + num_bits_to_move;

        let start_write_word = start_write_bit / bits_per_word;
        let end_write_word = (end_write_bit - 1) / bits_per_word;

        for write_word_idx in start_write_word..=end_write_word {
            // Fetch the source data, which may span two source words.
            let read_bit = write_word_idx * bits_per_word + shift_rem;
            let read_word_idx = read_bit / bits_per_word;

            let low_part = self.bits.get(read_word_idx).copied().unwrap_or(W::ZERO) >> shift_rem;
            let high_part =
                self.bits.get(read_word_idx + 1).copied().unwrap_or(W::ZERO) << inv_shift_rem;

            let value_to_write = low_part | high_part;

            // Create a mask for the bits we are about to modify.
            let mut mask = W::max_value();
            if write_word_idx == start_write_word {
                mask &= W::max_value() << (start_write_bit % bits_per_word);
            }
            if write_word_idx == end_write_word {
                let end_offset = end_write_bit % bits_per_word;
                if end_offset != 0 {
                    mask &= (W::ONE << end_offset).wrapping_sub(W::ONE);
                }
            }

            self.bits[write_word_idx] =
                (self.bits[write_word_idx] & !mask) | (value_to_write & mask);
        }
    }

    /// Shifts a range of bits to the right in-place.
    fn shift_bits_right(&mut self, start_bit: usize, shift_amount: usize, num_bits_to_move: usize) {
        if num_bits_to_move == 0 {
            return;
        }

        let bits_per_word = <W as Word>::BITS;

        // Ensure the vector has enough capacity and is resized to accommodate the shift.
        let required_end_bit = start_bit + shift_amount + num_bits_to_move;
        let required_words = required_end_bit.div_ceil(bits_per_word);
        let required_vec_len = required_words.saturating_add(1); // +1 for padding
        if self.bits.len() < required_vec_len {
            self.bits.resize(required_vec_len, W::ZERO);
        }

        // Fast path for word-aligned shifts.
        if shift_amount % bits_per_word == 0 {
            let start_read_word = start_bit / bits_per_word;
            let start_write_word = (start_bit + shift_amount) / bits_per_word;
            let num_words_to_move = num_bits_to_move.div_ceil(bits_per_word);

            if start_read_word + num_words_to_move <= self.bits.len() {
                self.bits.copy_within(
                    start_read_word..start_read_word + num_words_to_move,
                    start_write_word,
                );
            }
        } else {
            // Slow path for unaligned shifts (iterating from right to left).
            let word_shift = shift_amount / bits_per_word;
            let shift_rem = shift_amount % bits_per_word;
            let inv_shift_rem = bits_per_word - shift_rem;

            let start_write_bit = start_bit + shift_amount;
            let end_write_bit = start_write_bit + num_bits_to_move;

            let start_write_word = start_write_bit / bits_per_word;
            let end_write_word = (end_write_bit - 1) / bits_per_word;

            for write_word_idx in (start_write_word..=end_write_word).rev() {
                let read_word_idx = write_word_idx - word_shift;

                // Fetch source data, which may span two words.
                let high_part =
                    self.bits.get(read_word_idx).copied().unwrap_or(W::ZERO) << shift_rem;
                let low_part = if read_word_idx > 0 {
                    self.bits.get(read_word_idx - 1).copied().unwrap_or(W::ZERO) >> inv_shift_rem
                } else {
                    W::ZERO
                };
                let value_to_write = low_part | high_part;

                // Create a mask for the bits we are about to modify.
                let mut mask = W::max_value();
                if write_word_idx == start_write_word {
                    mask &= W::max_value() << (start_write_bit % bits_per_word);
                }
                if write_word_idx == end_write_word {
                    let end_offset = end_write_bit % bits_per_word;
                    if end_offset != 0 {
                        mask &= (W::ONE << end_offset).wrapping_sub(W::ONE);
                    }
                }

                self.bits[write_word_idx] =
                    (self.bits[write_word_idx] & !mask) | (value_to_write & mask);
            }
        }

        // Zero out the vacated bits at the beginning of the shifted region.
        let mut clear_bit = start_bit;
        let end_clear_bit = start_bit + shift_amount;

        while clear_bit < end_clear_bit {
            let word_idx = clear_bit / bits_per_word;
            let offset = clear_bit % bits_per_word;
            let bits_to_clear = (bits_per_word - offset).min(end_clear_bit - clear_bit);

            let mask = if bits_to_clear == bits_per_word {
                W::max_value()
            } else {
                ((W::ONE << bits_to_clear).wrapping_sub(W::ONE)) << offset
            };

            if word_idx < self.bits.len() {
                self.bits[word_idx] &= !mask;
            }
            clear_bit += bits_to_clear;
        }
    }

    /// Removes an element at `index` and returns it, replacing it with the last
    /// element of the vector.
    ///
    /// This operation is O(1) but does not preserve the order of elements.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn swap_remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "swap_remove: index out of bounds");

        if index == self.len - 1 {
            self.pop().unwrap()
        } else {
            let old_val = unsafe { self.get_unchecked(index) };
            let last_val = self.pop().unwrap(); // `pop` already decrements len
            self.set(index, last_val);
            old_val
        }
    }

    /// Appends an element to the vector, returning an error if the value doesn't fit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ValueTooLarge`] if the `value` cannot be represented
    /// by the configured `bit_width`.
    pub fn try_push(&mut self, value: T) -> Result<(), Error> {
        let value_w = <T as Storable<W>>::into_word(value);
        let bits_per_word = <W as traits::Word>::BITS;

        let limit = if self.bit_width < bits_per_word {
            W::ONE << self.bit_width
        } else {
            W::max_value()
        };

        if self.bit_width < bits_per_word && value_w >= limit {
            return Err(Error::ValueTooLarge {
                value: value_w.to_u128().unwrap(),
                index: self.len,
                bit_width: self.bit_width,
            });
        }

        self.push(value);
        Ok(())
    }

    /// Extends the vector with the elements from a slice.
    ///
    /// # Panics
    ///
    /// Panics if any value in `other` does not fit within the `bit_width`.
    pub fn extend_from_slice(&mut self, other: &[T]) {
        if other.is_empty() {
            return;
        }

        self.reserve(other.len());

        // Pre-validate all values to ensure atomicity.
        let bits_per_word = <W as traits::Word>::BITS;
        let limit = if self.bit_width < bits_per_word {
            W::ONE << self.bit_width
        } else {
            W::max_value()
        };
        if self.bit_width < bits_per_word {
            for (i, &value) in other.iter().enumerate() {
                let value_w = <T as Storable<W>>::into_word(value);
                if value_w >= limit {
                    panic!(
                        "Value at index {} of slice ({:?}) does not fit in the configured bit_width of {}",
                        i, value_w, self.bit_width
                    );
                }
            }
        }

        let old_len = self.len;
        let new_len = old_len + other.len();

        // Ensure the underlying Vec has enough initialized words to write into.
        let required_total_bits = new_len * self.bit_width;
        let required_data_words = required_total_bits.div_ceil(bits_per_word);
        let required_vec_len = required_data_words.saturating_add(1); // Padding
        if self.bits.len() < required_vec_len {
            self.bits.resize(required_vec_len, W::ZERO);
        }

        // Write the new values.
        for (i, &value) in other.iter().enumerate() {
            // SAFETY: We have reserved, resized, and validated the data.
            unsafe {
                self.set_unchecked(old_len + i, <T as Storable<W>>::into_word(value));
            }
        }

        self.len = new_len;
    }
}

// Mutable in-place operations.
impl<T, W, E, B> FixedVec<T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    /// Returns a mutable proxy for an element at `index`.
    ///
    /// This allows for syntax like `*vec.at_mut(i).unwrap() = new_value;`.
    ///
    /// Returns `None` if the index is out of bounds.
    pub fn at_mut(&mut self, index: usize) -> Option<MutProxy<'_, T, W, E, B>> {
        if index >= self.len {
            return None;
        }
        Some(MutProxy::new(self, index))
    }

    /// Returns a mutable slice of the underlying storage words.
    ///
    /// # Safety
    ///
    /// Modifying the returned slice is logically unsafe. Any change to the bits
    /// can violate the invariants of the [`FixedVec`], leading to panics or
    /// incorrect results on subsequent method calls.
    pub unsafe fn as_mut_limbs(&mut self) -> &mut [W] {
        self.bits.as_mut()
    }

    /// Sets the value of the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds or if `value` is too large to be
    /// represented by the configured `bit_width`.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec, BitWidth};
    ///
    /// let data: &[u32] = &[10, 20, 30];
    /// let mut vec: UFixedVec<u32> = FixedVec::builder().bit_width(BitWidth::Explicit(7)).build(data).unwrap();
    ///
    /// vec.set(1, 99);
    /// assert_eq!(vec.get(1), Some(99));
    /// ```
    pub fn set(&mut self, index: usize, value: T) {
        assert!(
            index < self.len,
            "Index out of bounds: expected index < {}, got {}",
            self.len,
            index
        );

        let value_w = <T as Storable<W>>::into_word(value);
        let bits_per_word = <W as traits::Word>::BITS;

        let limit = if self.bit_width < bits_per_word {
            W::ONE << self.bit_width
        } else {
            W::max_value()
        };

        if self.bit_width < bits_per_word && value_w >= limit {
            panic!(
                "Value {:?} does not fit in the configured bit_width of {}",
                value_w, self.bit_width
            );
        }

        unsafe { self.set_unchecked(index, value_w) };
    }

    /// Sets the value of the element at `index` without bounds or value checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is within bounds and that `value_w`
    /// fits within the configured `bit_width`. Failure to do so will result in
    /// data corruption or a panic.
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, index: usize, value_w: W) {
        let bits_per_word = <W as traits::Word>::BITS;
        if self.bit_width == bits_per_word {
            *self.bits.as_mut().get_unchecked_mut(index) = if E::IS_LITTLE {
                value_w
            } else {
                value_w.to_be()
            };
            return;
        }

        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / bits_per_word;
        let bit_offset = bit_pos % bits_per_word;

        let limbs = self.bits.as_mut();

        if E::IS_LITTLE {
            // Fast path: the value fits entirely within a single word.
            if bit_offset + self.bit_width <= bits_per_word {
                let mut word = *limbs.get_unchecked(word_index);
                // Clear the target bits and then OR the new value.
                word &= !(self.mask << bit_offset);
                word |= value_w << bit_offset;
                *limbs.get_unchecked_mut(word_index) = word;
            } else {
                let remaining_bits_in_first_word = bits_per_word - bit_offset;
                let (left, right) = limbs.split_at_mut_unchecked(word_index + 1);
                let mut low_word_val = *left.get_unchecked(word_index);
                let low_mask = (<W as num_traits::NumCast>::from(1u8).unwrap() << bit_offset)
                    .wrapping_sub(<W as num_traits::NumCast>::from(1u8).unwrap());
                low_word_val &= low_mask;
                low_word_val |= value_w << bit_offset;
                *left.get_unchecked_mut(word_index) = low_word_val;

                let mut high_word_val = *right.get_unchecked(0);
                high_word_val &= !(self.mask >> remaining_bits_in_first_word);
                high_word_val |= value_w >> remaining_bits_in_first_word;
                *right.get_unchecked_mut(0) = high_word_val;
            }
        } else {
            // Big-Endian set logic.
            if bit_offset + self.bit_width <= bits_per_word {
                let shift = bits_per_word - self.bit_width - bit_offset;
                let mask = self.mask << shift;
                let word = limbs.get_unchecked_mut(word_index);
                *word &= !mask.to_be();
                *word |= (value_w << shift).to_be();
            } else {
                let (left, right) = limbs.split_at_mut_unchecked(word_index + 1);
                let high_word = left.get_unchecked_mut(word_index);
                let low_word = right.get_unchecked_mut(0);

                let bits_in_first = bits_per_word - bit_offset;
                let bits_in_second = self.bit_width - bits_in_first;

                let high_mask =
                    (self.mask >> bits_in_second) << (bits_per_word - bits_in_first - bit_offset);
                let high_value = value_w >> bits_in_second;
                *high_word &= !high_mask.to_be();
                *high_word |= (high_value << (bits_per_word - bits_in_first - bit_offset)).to_be();

                let low_mask = self.mask << (bits_per_word - bits_in_second);
                let low_value = value_w << (bits_per_word - bits_in_second);
                *low_word &= !low_mask.to_be();
                *low_word |= low_value.to_be();
            }
        }
    }

    /// Sets the value of an element, returning an error if the value doesn't fit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ValueTooLarge`] if the `value` cannot be represented
    /// by the configured `bit_width`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn try_set(&mut self, index: usize, value: T) -> Result<(), Error> {
        assert!(index < self.len, "try_set: index out of bounds");

        let value_w = <T as Storable<W>>::into_word(value);
        let bits_per_word = <W as traits::Word>::BITS;

        let limit = if self.bit_width < bits_per_word {
            W::ONE << self.bit_width
        } else {
            W::max_value()
        };

        if self.bit_width < bits_per_word && value_w >= limit {
            return Err(Error::ValueTooLarge {
                value: value_w.to_u128().unwrap(),
                index,
                bit_width: self.bit_width,
            });
        }

        unsafe { self.set_unchecked(index, value_w) };
        Ok(())
    }

    /// Returns an iterator over non-overlapping, mutable chunks of the vector.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    pub fn chunks_mut(&mut self, chunk_size: usize) -> iter_mut::ChunksMut<'_, T, W, E, B> {
        iter_mut::ChunksMut::new(self, chunk_size)
    }

    /// Returns an unchecked mutable iterator over the elements of the vector.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the iterator is not advanced beyond the
    /// vector's length.
    pub unsafe fn iter_mut_unchecked(&mut self) -> iter_mut::IterMutUnchecked<'_, T, W, E, B> {
        iter_mut::IterMutUnchecked::new(self)
    }

    /// Applies a function to all elements in place, without checking if the
    /// returned values fit within the `bit_width`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the function `f` always returns a value
    /// that can be represented by `self.bit_width()` bits. Returning a value
    /// that is too large will result in data corruption.
    pub unsafe fn map_in_place_unchecked<F>(&mut self, mut f: F)
    where
        F: FnMut(T) -> T,
    {
        if E::IS_LITTLE {
            // This path operates on words directly for efficiency.
            let mut word_op = |word: W| -> W {
                let val_t = <T as Storable<W>>::from_word(word);
                let new_val_t = f(val_t);
                <T as Storable<W>>::into_word(new_val_t)
            };
            self.map_in_place_generic_word_op(&mut word_op);
        } else {
            // Fallback for BE, which is more complex to optimize at the word level.
            for i in 0..self.len {
                let old_val_t = self.get_unchecked(i);
                let new_val_t = f(old_val_t);
                self.set_unchecked(i, <T as Storable<W>>::into_word(new_val_t));
            }
        }
    }

    /// Internal worker for `map_in_place_unchecked`, operating on `W`.
    unsafe fn map_in_place_generic_word_op<FW>(&mut self, f_w: &mut FW)
    where
        FW: FnMut(W) -> W,
    {
        let bit_width = self.bit_width;
        if self.len == 0 || bit_width == 0 {
            return;
        }

        let bits_per_word = <W as Word>::BITS;

        // Path for power-of-two bit widths that align with word boundaries.
        if bit_width.is_power_of_two() && bits_per_word % bit_width == 0 {
            let elems_per_word = bits_per_word / bit_width;
            let mask = self.mask;
            let num_full_words = self.len / elems_per_word;

            for word_idx in 0..num_full_words {
                let old_word = *self.bits.as_ref().get_unchecked(word_idx);
                let mut new_word = W::ZERO;
                for i in 0..elems_per_word {
                    let shift = i * bit_width;
                    let old_val_w = (old_word >> shift) & mask;
                    new_word |= f_w(old_val_w) << shift;
                }
                *self.bits.as_mut().get_unchecked_mut(word_idx) = new_word;
            }
            // Process remaining elements individually.
            let start_idx = num_full_words * elems_per_word;
            for i in start_idx..self.len {
                let old_val_t = self.get(i).unwrap();
                let old_val_w = <T as Storable<W>>::into_word(old_val_t);
                self.set_unchecked(i, f_w(old_val_w));
            }
            return;
        }

        // Generic path for non-power-of-two bit widths.
        let limbs = self.bits.as_mut();
        let num_words = (self.len * bit_width).div_ceil(bits_per_word);
        let last_word_idx = num_words.saturating_sub(1);

        let mut write_buffer = W::ZERO;
        let mut read_buffer = *limbs.get_unchecked(0);
        let mut global_bit_offset = 0;

        for word_idx in 0..last_word_idx {
            let lower_word_boundary = word_idx * bits_per_word;
            let upper_word_boundary = lower_word_boundary + bits_per_word;

            while global_bit_offset + bit_width <= upper_word_boundary {
                let offset_in_word = global_bit_offset - lower_word_boundary;
                let element = (read_buffer >> offset_in_word) & self.mask;
                write_buffer |= f_w(element) << offset_in_word;
                global_bit_offset += bit_width;
            }

            let next_word = *limbs.get_unchecked(word_idx + 1);
            let mut new_write_buffer = W::ZERO;

            if upper_word_boundary != global_bit_offset {
                let elem_idx = global_bit_offset / bit_width;
                if elem_idx >= self.len {
                    *limbs.get_unchecked_mut(word_idx) = write_buffer;
                    return;
                }

                let remainder_in_word = upper_word_boundary - global_bit_offset;
                let offset_in_word = global_bit_offset - lower_word_boundary;

                let element = ((read_buffer >> offset_in_word) | (next_word << remainder_in_word))
                    & self.mask;
                let new_element = f_w(element);

                write_buffer |= new_element << offset_in_word;
                new_write_buffer = new_element >> remainder_in_word;

                global_bit_offset += bit_width;
            }

            *limbs.get_unchecked_mut(word_idx) = write_buffer;

            read_buffer = next_word;
            write_buffer = new_write_buffer;
        }

        let lower_word_boundary = last_word_idx * bits_per_word;

        while global_bit_offset < self.len * bit_width {
            let offset_in_word = global_bit_offset - lower_word_boundary;
            let element = (read_buffer >> offset_in_word) & self.mask;
            write_buffer |= f_w(element) << offset_in_word;
            global_bit_offset += bit_width;
        }

        *limbs.get_unchecked_mut(last_word_idx) = write_buffer;
    }

    /// Applies a function to all elements in the vector, modifying them in-place.
    ///
    /// # Panics
    ///
    /// Panics if the function `f` returns a value that does not fit within the
    /// configured `bit_width`.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{FixedVec, BitWidth, UFixedVec};
    ///
    /// // Values up to 9*2=18, requires 5 bits. We must build with enough space.
    /// let initial_data: Vec<u32> = (0..10).collect();
    /// let mut vec: UFixedVec<u32> = FixedVec::builder()
    ///     .bit_width(BitWidth::Explicit(5))
    ///     .build(&initial_data)
    ///     .unwrap();
    ///
    /// vec.map_in_place(|x| x * 2);
    ///
    /// for i in 0..vec.len() {
    ///     assert_eq!(vec.get(i), Some(i as u32 * 2));
    /// }
    /// ```
    pub fn map_in_place<F>(&mut self, mut f: F)
    where
        F: FnMut(T) -> T,
    {
        let bit_width = self.bit_width;
        let limit = if bit_width < <W as Word>::BITS {
            W::ONE << bit_width
        } else {
            W::max_value()
        };

        let safe_f = |value: T| {
            let new_value = f(value);
            let new_value_w = <T as Storable<W>>::into_word(new_value);
            if bit_width < <W as Word>::BITS && new_value_w >= limit {
                panic!(
                    "map_in_place: returned value {new_value_w:?} does not fit in bit_width {bit_width}"
                );
            }
            new_value
        };

        // SAFETY: The `safe_f` wrapper ensures that any value passed to the
        // underlying unsafe function is valid for the vector's bit_width.
        unsafe {
            self.map_in_place_unchecked(safe_f);
        }
    }

    /// Replaces the element at `index` with a new `value`, returning the old value.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds or if `value` does not fit within
    /// the configured `bit_width`.
    pub fn replace(&mut self, index: usize, value: T) -> T {
        assert!(index < self.len, "replace: index out of bounds");

        let old_value = unsafe { self.get_unchecked(index) };
        self.set(index, value);
        old_value
    }

    /// Swaps the elements at indices `a` and `b`.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` are out of bounds.
    pub fn swap(&mut self, a: usize, b: usize) {
        assert!(a < self.len, "swap: index a out of bounds");
        assert!(b < self.len, "swap: index b out of bounds");

        if a == b {
            return;
        }

        unsafe {
            let val_a = self.get_unchecked(a);
            let val_b = self.get_unchecked(b);
            self.set_unchecked(a, <T as Storable<W>>::into_word(val_b));
            self.set_unchecked(b, <T as Storable<W>>::into_word(val_a));
        }
    }

    /// Returns a mutable proxy to the first element, or `None` if empty.
    pub fn first_mut(&mut self) -> Option<MutProxy<'_, T, W, E, B>> {
        if self.is_empty() {
            None
        } else {
            self.at_mut(0)
        }
    }

    /// Returns a mutable proxy to the last element, or `None` if empty.
    pub fn last_mut(&mut self) -> Option<MutProxy<'_, T, W, E, B>> {
        if self.is_empty() {
            None
        } else {
            let len = self.len();
            self.at_mut(len - 1)
        }
    }

    /// Returns the first element of the vector, or `None` if empty.
    pub fn first(&self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(unsafe { self.get_unchecked(0) })
        }
    }

    /// Returns the last element of the vector, or `None` if empty.
    pub fn last(&self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let len = self.len();
            Some(unsafe { self.get_unchecked(len - 1) })
        }
    }

    /// Splits the vector into two mutable slices at `mid`.
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> (
        slice::FixedVecSlice<&mut Self>,
        slice::FixedVecSlice<&mut Self>,
    ) {
        assert!(mid <= self.len, "mid > len in split_at_mut");
        // SAFETY: The two slices are guaranteed not to overlap.
        unsafe {
            let ptr = self as *mut Self;
            let left = slice::FixedVecSlice::new(&mut *ptr, 0..mid);
            let right = slice::FixedVecSlice::new(&mut *ptr, mid..self.len());
            (left, right)
        }
    }

    /// Returns a mutable iterator over the elements of the vector.
    pub fn iter_mut(&mut self) -> iter_mut::IterMut<'_, T, W, E, B> {
        iter_mut::IterMut::new(self)
    }

    /// Rotates the elements of the vector in-place such that the element at `mid`
    /// becomes the first element.
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    pub fn rotate_left(&mut self, mid: usize) {
        assert!(mid <= self.len, "mid > len in rotate_left");
        if self.is_empty() || mid == 0 || mid == self.len {
            return;
        }
        // A simple, correct implementation.
        let mut temp = Vec::with_capacity(mid);
        for i in 0..mid {
            temp.push(unsafe { self.get_unchecked(i) });
        }
        for i in mid..self.len {
            let val = unsafe { self.get_unchecked(i) };
            unsafe { self.set_unchecked(i - mid, <T as Storable<W>>::into_word(val)) };
        }
        for (i, val) in temp.into_iter().enumerate() {
            unsafe { self.set_unchecked(self.len - mid + i, <T as Storable<W>>::into_word(val)) };
        }
    }

    /// Rotates the elements of the vector in-place such that the element at
    /// `len - k` becomes the first element.
    ///
    /// # Panics
    ///
    /// Panics if `k > len`.
    pub fn rotate_right(&mut self, k: usize) {
        assert!(k <= self.len, "k > len in rotate_right");
        if self.is_empty() || k == 0 || k == self.len {
            return;
        }
        self.rotate_left(self.len - k);
    }

    /// Fills the vector with the given value.
    ///
    /// # Panics
    ///
    /// Panics if `value` does not fit in the configured `bit_width`.
    pub fn fill(&mut self, value: T) {
        let value_w = <T as Storable<W>>::into_word(value);
        let bits_per_word = <W as traits::Word>::BITS;
        let limit = if self.bit_width < bits_per_word {
            W::ONE << self.bit_width
        } else {
            W::max_value()
        };
        if self.bit_width < bits_per_word && value_w >= limit {
            panic!(
                "Value {:?} does not fit in the configured bit_width of {}",
                value_w, self.bit_width
            );
        }

        for i in 0..self.len() {
            unsafe { self.set_unchecked(i, value_w) };
        }
    }

    /// Fills the vector with values returned by a closure.
    ///
    /// # Panics
    ///
    /// Panics if the closure returns a value that does not fit in the configured
    /// `bit_width`.
    pub fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut() -> T,
    {
        let bits_per_word = <W as traits::Word>::BITS;
        let limit = if self.bit_width < bits_per_word {
            W::ONE << self.bit_width
        } else {
            W::max_value()
        };

        for i in 0..self.len() {
            let value = f();
            let value_w = <T as Storable<W>>::into_word(value);
            if self.bit_width < bits_per_word && value_w >= limit {
                panic!(
                    "Value {:?} returned by closure does not fit in bit_width {}",
                    value_w, self.bit_width
                );
            }
            unsafe { self.set_unchecked(i, value_w) };
        }
    }

    /// Copies a sequence of elements from a source [`FixedVec`] into this one.
    ///
    /// # Panics
    ///
    /// Panics if the source and destination vectors do not have the same `bit_width`,
    /// or if the source or destination ranges are out of bounds.
    pub fn copy_from_slice(
        &mut self,
        src: &Self,
        src_range: std::ops::Range<usize>,
        dest_index: usize,
    ) {
        assert_eq!(
            self.bit_width, src.bit_width,
            "bit_width mismatch in copy_from_slice"
        );
        assert!(src_range.start <= src_range.end, "source range start > end");
        assert!(src_range.end <= src.len(), "source range out of bounds");
        let len = src_range.len();
        assert!(
            dest_index + len <= self.len(),
            "destination range out of bounds"
        );

        if len == 0 {
            return;
        }

        // Fast path: if bit alignments are the same, we can do word-level copies.
        let bit_width = self.bit_width;
        let bits_per_word = <W as Word>::BITS;
        let src_bit_offset = (src_range.start * bit_width) % bits_per_word;
        let dest_bit_offset = (dest_index * bit_width) % bits_per_word;

        if src_bit_offset == dest_bit_offset {
            let src_word_start = (src_range.start * bit_width) / bits_per_word;
            let dest_word_start = (dest_index * bit_width) / bits_per_word;
            let total_bits_to_copy = len * bit_width;
            let num_words_to_copy = total_bits_to_copy.div_ceil(bits_per_word);

            let src_words = &src.bits.as_ref()[src_word_start..src_word_start + num_words_to_copy];
            let dest_words =
                &mut self.bits.as_mut()[dest_word_start..dest_word_start + num_words_to_copy];

            // If the last word is not fully copied, we need to mask it.
            let residual_bits = total_bits_to_copy % bits_per_word;
            if residual_bits == 0 {
                dest_words.copy_from_slice(src_words);
            } else {
                if num_words_to_copy > 1 {
                    dest_words[..num_words_to_copy - 1]
                        .copy_from_slice(&src_words[..num_words_to_copy - 1]);
                }
                let last_word_mask = (W::ONE << residual_bits).wrapping_sub(W::ONE);
                let dest_last_word = &mut dest_words[num_words_to_copy - 1];
                let src_last_word = &src_words[num_words_to_copy - 1];
                *dest_last_word =
                    (*dest_last_word & !last_word_mask) | (*src_last_word & last_word_mask);
            }
        } else {
            // Slow path: copy element by element if alignments differ.
            if dest_index > src_range.start && dest_index < src_range.end {
                // Copy backwards if ranges overlap and dest is after src.
                for i in (0..len).rev() {
                    let val = unsafe { src.get_unchecked(src_range.start + i) };
                    unsafe {
                        self.set_unchecked(dest_index + i, <T as Storable<W>>::into_word(val))
                    };
                }
            } else {
                for i in 0..len {
                    let val = unsafe { src.get_unchecked(src_range.start + i) };
                    unsafe {
                        self.set_unchecked(dest_index + i, <T as Storable<W>>::into_word(val))
                    };
                }
            }
        }
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
    /// Checks for equality between two [`FixedVec`] instances.
    ///
    /// It first checks `len` and `bit_width`, then the underlying storage.
    fn eq(&self, other: &FixedVec<T, W, E, B2>) -> bool {
        if self.len() != other.len() || self.bit_width() != other.bit_width() {
            return false;
        }
        self.as_limbs() == other.as_limbs()
    }
}

/// Implements `PartialEq` for comparing a [`FixedVec`] with a standard slice.
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

impl<T, W, E> Extend<T> for FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W> + ToPrimitive,
    W: Word,
    E: Endianness,
{
    /// Extends the vector with the contents of an iterator.
    ///
    /// # Panics
    ///
    /// Panics if any value from the iterator does not fit within the
    /// configured `bit_width`.
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();
        self.reserve(lower_bound);
        for item in iter {
            self.push(item);
        }
    }
}

impl<T, W, E> From<FixedVec<T, W, E, Vec<W>>> for FixedVec<T, W, E, Box<[W]>>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
{
    /// Converts a `Vec`-backed [`FixedVec`] into a `Box<[]>`-backed [`FixedVec`].
    fn from(vec: FixedVec<T, W, E, Vec<W>>) -> Self {
        unsafe { Self::new_unchecked(vec.bits.into_boxed_slice(), vec.len, vec.bit_width) }
    }
}

impl<T, W, E> From<FixedVec<T, W, E, Box<[W]>>> for FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
{
    /// Converts a `Box<[]>`-backed [`FixedVec`] into a `Vec`-backed [`FixedVec`].
    fn from(vec: FixedVec<T, W, E, Box<[W]>>) -> Self {
        unsafe { Self::new_unchecked(vec.bits.into_vec(), vec.len, vec.bit_width) }
    }
}

impl<'a, T> TryFrom<&'a [T]> for FixedVec<T, <T as DefaultParams>::W, <T as DefaultParams>::E>
where
    T: Storable<<T as DefaultParams>::W> + DefaultParams,
    <T as DefaultParams>::W: Word,
    <T as DefaultParams>::E: Endianness,
    dsi_bitstream::impls::BufBitWriter<
        <T as DefaultParams>::E,
        dsi_bitstream::impls::MemWordWriterVec<
            <T as DefaultParams>::W,
            Vec<<T as DefaultParams>::W>,
        >,
    >: dsi_bitstream::prelude::BitWrite<<T as DefaultParams>::E, Error = std::convert::Infallible>,
{
    type Error = Error;

    /// Creates a [`FixedVec`] from a slice using [`BitWidth::Minimal`] and default parameters.
    ///
    /// This is a convenience method equivalent to `FixedVec::builder().build(slice)`.
    /// It uses the default [`Word`] ([`usize`]) and [`Endianness`] ([`LE`]) associated with the element type `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::fixed::{UFixedVec, SFixedVec};
    /// use std::convert::TryFrom;
    ///
    /// // For unsigned types
    /// let data_u: &[u32] = &[10, 20, 30];
    /// let vec_u = UFixedVec::<u32>::try_from(data_u).unwrap();
    /// assert_eq!(vec_u.bit_width(), 5);
    ///
    /// // For signed types
    /// let data_s: &[i16] = &[-10, 0, 10];
    /// let vec_s = SFixedVec::<i16>::try_from(data_s).unwrap();
    /// assert_eq!(vec_s.bit_width(), 5);
    /// ```
    fn try_from(slice: &'a [T]) -> Result<Self, Self::Error> {
        Self::builder().bit_width(BitWidth::Minimal).build(slice)
    }
}
