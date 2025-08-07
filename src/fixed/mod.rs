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

// // Conditionally compile the atomic module.
// #[cfg(feature = "atomic")]
// pub mod atomic;

// Conditionally compile the serde module.
#[cfg(feature = "serde")]
mod serde;

use dsi_bitstream::{prelude::Endianness, traits::{BE, LE}};
use mem_dbg::{MemDbg, MemSize};
use std::{error::Error as StdError, fmt, marker::PhantomData, iter::FromIterator};
use traits::{Storable, Word};
use num_traits::ToPrimitive;

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

        // Essential safety check: ensure the buffer is large enough for the data
        // AND the 2 padding words required.
        if bits.as_ref().len() < data_words + 2 {
            return Err(Error::InvalidParameters(format!(
                "The provided buffer is too small. It has {} words, but {} data words + 2 padding words are required.",
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

    /// Returns the raw parts of the vector: a pointer to the start of the
    /// underlying buffer and its length in words.
    ///
    /// The caller must ensure that the buffer is not mutated in a way that
    /// violates the `FixedVec`'s invariants while the pointer is active.
    pub fn as_raw_parts(&self) -> (*const W, usize) {
        let slice = self.bits.as_ref();
        (slice.as_ptr(), slice.len())
    }

    /// Creates a `FixedVec` from its constituent parts, enabling zero-copy views.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// 1. `len * bit_width` is not larger than the number of bits available in `bits`.
    /// 2. The `bits` slice has at least one extra padding word at the end
    ///    to prevent out-of-bounds reads during [`get_unchecked`].
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


    /// Retrieves the element at `index` using potentially unaligned memory access.
    ///
    /// This method is a high-performance alternative to `get_unchecked`. Instead
    /// of potentially performing two memory reads for elements that span word
    /// boundaries, it performs a single (potentially unaligned) read of a `Word`
    /// and extracts the bits with shifts.
    ///
    /// On architectures that handle unaligned reads efficiently (e.g., x86-64),
    /// this can be significantly faster, especially for random access patterns.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is Undefined Behavior.
    /// The `FixedVec` must also have been constructed with sufficient padding
    /// (at least 2 `Word`s) to guarantee that the unaligned read does not go
    /// past the allocated memory buffer. This is guaranteed by the default builders.
    #[inline(always)]
    pub unsafe fn get_unaligned_unchecked(&self, index: usize) -> T {
        debug_assert!(index < self.len);

        if E::IS_LITTLE {
            let bits_per_word = <W as Word>::BITS;
            if self.bit_width == bits_per_word {
                return self.get_unchecked(index);
            }

            let bit_pos = index * self.bit_width;
            let byte_pos = bit_pos / 8;
            let bit_rem = bit_pos % 8;

            let limbs_ptr = self.as_limbs().as_ptr() as *const u8;
            
            let word: W = (limbs_ptr.add(byte_pos) as *const W).read_unaligned();
            let extracted_word = word >> bit_rem;
            
            <T as Storable<W>>::from_word(extracted_word & self.mask)
        } else {
            // For Big-Endian, the logic for unaligned reads is highly complex
            // and architecture-dependent. The standard `get_unchecked` is already
            // heavily optimized for this case. We fall back to it for correctness
            // and robust performance.
            self.get_unchecked(index)
        }
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

    /// Returns an iterator that does not perform bounds checking, in reverse.
    ///
    /// # Safety
    /// The returned iterator is unsafe to use. The caller must ensure that the
    /// iterator's `next_unchecked` method is not called more times than the
    /// length of the vector.
    pub unsafe fn iter_rev_unchecked(&self) -> iter::FixedVecReverseUncheckedIter<T, W, E, B> {
        iter::FixedVecReverseUncheckedIter::new(self)
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

    /// Returns a raw pointer to the storage word containing the start of an element.
    ///
    /// This method provides a way to get a pointer to the underlying memory
    /// where an element's data begins. The pointer points to the start of the
    /// `Word` (e.g., `u64`) in the backing buffer.
    ///
    /// The bit offset of the element's first bit *within* that word can be
    /// calculated as `(index * self.bit_width()) % W::BITS`.
    ///
    /// Returns `None` if `index` is out of bounds.
    ///
    /// # Safety
    ///
    /// This method is safe as it only returns a raw pointer. However,
    /// dereferencing this pointer is `unsafe` and requires extreme care. The
    /// caller must ensure that the pointer is not used after the `FixedVec`
    /// is dropped or modified.
    ///
    /// The pointer is not guaranteed to be aligned to the `Word` size if the
    /// backing buffer `B` is not aligned.
    pub fn addr_of(&self, index: usize) -> Option<*const W> {
        if index >= self.len {
            return None;
        }

        let bit_pos = index * self.bit_width;
        let word_idx = bit_pos / <W as Word>::BITS;
        
        let limbs = self.as_limbs();
        // Check if the calculated word index is valid for the slice.
        if word_idx < limbs.len() {
            // Get a pointer to the start of the slice and offset it.
            // This is safer than `&limbs[word_idx] as *const _`.
            Some(limbs.as_ptr().wrapping_add(word_idx))
        } else {
            // This case should ideally not be reached if len and bit_width are consistent
            // with the buffer size, but it's a safe fallback.
            None
        }
    }

    /// Hints to the CPU to prefetch the data for the element at `index` into the cache.
    ///
    /// Prefetching can significantly improve performance for access patterns with
    /// some degree of predictability (e.g., strided or sequential access), as it
    /// helps to hide memory latency.
    ///
    /// This method is a wrapper around the `_mm_prefetch` intrinsic and will
    /// only have an effect on architectures that support it (like x86 and x86-64).
    /// On other architectures, it compiles to a no-op.
    ///
    /// If `index` is out of bounds, this method does nothing.
    ///
    /// # Arguments
    ///
    /// * `index`: The index of the element to prefetch.
    pub fn prefetch(&self, index: usize) {
        if index >= self.len {
            return;
        }

        // Target only x86 and x86-64 architectures where this intrinsic is available.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

            let bit_pos = index * self.bit_width;
            let byte_pos = bit_pos / 8;
            
            let limbs_ptr = self.as_limbs().as_ptr() as *const i8;

            // SAFETY: We have already performed the bounds check on `index`, which
            // ensures that the calculated `byte_pos` will be within the allocated
            // buffer (including padding). The pointer is valid.
            unsafe {
                // We use _MM_HINT_T0 to indicate a prefetch into all levels of
                // the cache, which is a good general-purpose default.
                _mm_prefetch(limbs_ptr.add(byte_pos), _MM_HINT_T0);
            }
        }
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

impl<T, W, E> Default for FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
{
    /// Creates an empty `FixedVec` with a default `bit_width` of 1.
    ///
    /// A `bit_width` of 1 is chosen as a safe default that can at least
    /// represent the value 0.
    fn default() -> Self {
        // SAFETY: An empty vector with a valid bit_width is always safe.
        unsafe { Self::new_unchecked(Vec::new(), 0, 1) }
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
    ///
    /// This implementation is optimized for performance by using a pre-computed
    /// mask for validation and relying on the amortized O(1) complexity of `Vec::push`.'
    #[inline(always)]
    pub fn push(&mut self, value: T) {
        let value_w = <T as Storable<W>>::into_word(value);

        // --- 1. Optimized Input Validation ---
        // This is significantly faster than calculating a limit via shifts in a loop.
        // It performs a single bitwise AND and a comparison.
        if (value_w & !self.mask) != W::ZERO {
            panic!(
                "Value {:?} does not fit in the configured bit_width of {}",
                value_w, self.bit_width
            );
        }

        // --- 2. Efficient Capacity Management ---
        let bits_per_word = <W as traits::Word>::BITS;
        if (self.len + 1) * self.bit_width > self.bits.len() * bits_per_word {
            self.bits.push(W::ZERO);
        }

        // --- 3. Write Data ---
        unsafe {
            self.set_unchecked(self.len, value_w);
        }

        // --- 4. Update State ---
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
        let num_words = total_bits.div_ceil(bits_per_word);
        
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

    pub fn reserve(&mut self, additional: usize) {
        let target_element_capacity = self.len.saturating_add(additional);
        if self.capacity() >= target_element_capacity { return; }
        let bits_per_word = <W as Word>::BITS;
        let required_total_bits = target_element_capacity.saturating_mul(self.bit_width);
        let required_data_words = required_total_bits.div_ceil(bits_per_word);
        let required_word_capacity = required_data_words + 1;
        
        let current_len = self.bits.len();
        if self.bits.capacity() < required_word_capacity {
             // We want the final capacity to be at least `required_word_capacity`.
             // `reserve` ensures capacity for `len + additional`.
             // So we need to ask for `required_word_capacity - len`.
             self.bits.reserve(required_word_capacity - current_len);
        }
    }

    /// Resizes the vector in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the vector is extended by the
    /// difference, with each additional slot filled with `value`.
    /// If `new_len` is less than `len`, the vector is simply truncated.
    #[inline(always)]
    pub fn resize(&mut self, new_len: usize, value: T) {
        if new_len > self.len {
            let value_w = <T as Storable<W>>::into_word(value);

            // Optimized validation using the pre-computed mask.
            if (value_w & !self.mask) != W::ZERO {
                panic!("Value {:?} does not fit in the configured bit_width of {}", value_w, self.bit_width);
            }

            let bits_per_word = <W as traits::Word>::BITS;
            let required_total_bits = new_len * self.bit_width;
            let required_data_words = required_total_bits.div_ceil(bits_per_word);
            let required_vec_len = required_data_words.saturating_add(2); // Padding

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
            // Truncate the vector
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
            // +1 for the padding word.
            required_words + 1
        };

        if self.bits.len() > min_word_len {
             self.bits.truncate(min_word_len);
        }
        self.bits.shrink_to_fit();
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// This operation is O(n) where n is the number of elements after `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "remove: index out of bounds");

        // 1. Read the value to be returned before we overwrite it.
        let value_to_return = self.get(index).unwrap();

        let start_bit = index * self.bit_width;
        let end_bit = self.len * self.bit_width;
        let total_bits_to_shift = end_bit - (start_bit + self.bit_width);

        if total_bits_to_shift > 0 {
            // 2. Shift the subsequent data to the left.
            self.shift_bits_left(start_bit, self.bit_width, total_bits_to_shift);
        }

        // 3. Update the length.
        self.len -= 1;

        value_to_return
    }

    /// Inserts an element at position `index` within the vector.
    pub fn insert(&mut self, index: usize, element: T) {
        assert!(index <= self.len, "insert: index out of bounds");
        let value_w = <T as Storable<W>>::into_word(element);
        let bits_per_word = <W as Word>::BITS;
        let limit = if self.bit_width < bits_per_word { W::ONE << self.bit_width } else { W::max_value() };
        if self.bit_width < bits_per_word && value_w >= limit {
            panic!("Value {:?} does not fit in the configured bit_width of {}", value_w, self.bit_width);
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

    /// A high-performance helper to shift a range of bits to the left in-place.
    ///
    /// This operation is optimized with a fast path for word-aligned shifts.
    fn shift_bits_left(&mut self, start_bit: usize, shift_amount: usize, num_bits_to_move: usize) {
        if num_bits_to_move == 0 {
            return;
        }

        let bits_per_word = <W as Word>::BITS;

        // --- Fast Path: Word-aligned shift ---
        if shift_amount % bits_per_word == 0 {
            let start_write_word = start_bit / bits_per_word;
            let start_read_word = (start_bit + shift_amount) / bits_per_word;
            let num_words_to_move = num_bits_to_move.div_ceil(bits_per_word);
            
            // Check if there is anything to copy from within the buffer.
            if start_read_word < self.bits.len() {
                let read_end = (start_read_word + num_words_to_move).min(self.bits.len());
                self.bits.copy_within(start_read_word..read_end, start_write_word);
            }
            
            // If the shift moves data from "beyond" the buffer, zero out the rest.
            let words_copied = self.bits.len().saturating_sub(start_read_word).min(num_words_to_move);
            if words_copied < num_words_to_move {
                let zero_start = start_write_word + words_copied;
                let zero_end = (start_write_word + num_words_to_move).min(self.bits.len());
                if zero_start < zero_end {
                    self.bits[zero_start..zero_end].fill(W::ZERO);
                }
            }
            return;
        }
        
        // --- Slow Path: Unaligned shift (word-at-a-time) ---
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
            let high_part = self.bits.get(read_word_idx + 1).copied().unwrap_or(W::ZERO) << inv_shift_rem;

            let value_to_write = low_part | high_part;
            
            // Create a mask for the bits we are about to modify in the destination word.
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

            self.bits[write_word_idx] = (self.bits[write_word_idx] & !mask) | (value_to_write & mask);
        }
    }
    
    /// A high-performance helper to shift a range of bits to the right in-place.
    ///
    /// This operation is optimized with a fast path for word-aligned shifts.
    /// The unaligned path iterates from right to left to avoid data corruption.
    fn shift_bits_right(&mut self, start_bit: usize, shift_amount: usize, num_bits_to_move: usize) {
        if num_bits_to_move == 0 { return; }

        let bits_per_word = <W as Word>::BITS;
        
        // Ensure the vector has enough capacity and is resized to accommodate the shift.
        let required_end_bit = start_bit + shift_amount + num_bits_to_move;
        let required_words = required_end_bit.div_ceil(bits_per_word);
        let required_vec_len = required_words.saturating_add(1); // +1 for padding
        if self.bits.len() < required_vec_len {
            self.bits.resize(required_vec_len, W::ZERO);
        }

        // --- Fast Path: Word-aligned shift ---
        if shift_amount % bits_per_word == 0 {
            let start_read_word = start_bit / bits_per_word;
            let start_write_word = (start_bit + shift_amount) / bits_per_word;
            let num_words_to_move = num_bits_to_move.div_ceil(bits_per_word);
            
            if start_read_word + num_words_to_move <= self.bits.len() {
                self.bits.copy_within(start_read_word..start_read_word + num_words_to_move, start_write_word);
            }
        } else {
            // --- Slow Path: Unaligned shift (from right to left) ---
            let word_shift = shift_amount / bits_per_word;
            let shift_rem = shift_amount % bits_per_word;
            let inv_shift_rem = bits_per_word - shift_rem;

            let start_write_bit = start_bit + shift_amount;
            let end_write_bit = start_write_bit + num_bits_to_move;
            
            let start_write_word = start_write_bit / bits_per_word;
            let end_write_word = (end_write_bit - 1) / bits_per_word;

            for write_word_idx in (start_write_word..=end_write_word).rev() {
                let read_word_idx = write_word_idx - word_shift;
                
                // Fetch source data from two potential source words.
                let high_part = self.bits.get(read_word_idx).copied().unwrap_or(W::ZERO) << shift_rem;
                let low_part = if read_word_idx > 0 {
                    self.bits.get(read_word_idx - 1).copied().unwrap_or(W::ZERO) >> inv_shift_rem
                } else {
                    W::ZERO
                };
                let value_to_write = low_part | high_part;
                
                // Create a mask for the bits we are about to modify in the destination word.
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
                
                self.bits[write_word_idx] = (self.bits[write_word_idx] & !mask) | (value_to_write & mask);
            }
        }
        
        // --- Cleanup: Zero out the vacated bits at the beginning of the shifted region ---
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

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    /// This does not preserve ordering, but is O(1).
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn swap_remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "swap_remove: index out of bounds");

        if index == self.len - 1 {
            // If it's the last element, just pop it.
            self.pop().unwrap()
        } else {
            // SAFETY: bounds have been checked.
            let old_val = unsafe { self.get_unchecked(index) };
            let last_val = self.pop().unwrap(); // pop already decrements len
            self.set(index, last_val);
            old_val
        }
    }

    /// Appends an element to the back of the vector, returning an error if the value doesn't fit.
    ///
    /// # Errors
    ///
    /// Returns an `Error::ValueTooLarge` if the provided `value` cannot be
    /// represented by the configured `bit_width`.
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
                index: self.len, // The index it *would* have
                bit_width: self.bit_width,
            });
        }
        
        // `push` panics on its own check, but we've already done it.
        // To avoid re-checking, we can call a non-panicking inner logic.
        // For simplicity here, we just call the original push.
        self.push(value);
        Ok(())
    }

    /// Extends the vector with the contents of a slice.
    ///
    /// This method is generally more efficient than calling `push` in a loop,
    /// as it can perform a single capacity check and allocation.
    ///
    /// # Panics
    ///
    /// Panics if any value in `other` does not fit within the configured `bit_width`.
    pub fn extend_from_slice(&mut self, other: &[T]) {
        if other.is_empty() {
            return;
        }

        self.reserve(other.len());

        // Pre-validate all values in the slice to ensure atomicity.
        // If any value is invalid, the vector remains unchanged.
        let bits_per_word = <W as traits::Word>::BITS;
        let limit = if self.bit_width < bits_per_word { W::ONE << self.bit_width } else { W::max_value() };
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
        
        // Ensure the underlying Vec has enough *initialized* words to write into.
        let required_total_bits = new_len * self.bit_width;
        let required_data_words = required_total_bits.div_ceil(bits_per_word);
        let required_vec_len = required_data_words.saturating_add(2); // Padding
        if self.bits.len() < required_vec_len {
            self.bits.resize(required_vec_len, W::ZERO);
        }
        
        // Write the new values in an optimized loop.
        for (i, &value) in other.iter().enumerate() {
            // SAFETY: We have already reserved, resized, and validated the data.
            unsafe {
                self.set_unchecked(old_len + i, <T as Storable<W>>::into_word(value));
            }
        }

        self.len = new_len;
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


    /// Returns a mutable slice of the underlying storage words.
    ///
    /// # Safety
    ///
    /// This method is safe, but modifying the returned slice is inherently
    /// unsafe from a logical perspective. Any modification to the bits can
    /// violate the invariants of the `FixedVec`, leading to panic or incorrect
    /// results on subsequent method calls (like `get` or `iter`).
    ///
    /// The caller must ensure that any changes to the slice maintain the
    /// bit-packed structure as expected by the `FixedVec`'s parameters
    /// (`len` and `bit_width`).
    pub fn as_mut_limbs(&mut self) -> &mut [W] {
        self.bits.as_mut()
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
    /// # Safety
    /// The caller must ensure that `index` is within bounds and that `value_w`
    /// fits within the configured `bit_width`.
    pub unsafe fn set_unchecked(&mut self, index: usize, value_w: W) {
        let bits_per_word = <W as traits::Word>::BITS;
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / bits_per_word;
        let bit_offset = bit_pos % bits_per_word;
        
        let limbs = self.bits.as_mut();

        if E::IS_LITTLE {
            if bit_offset + self.bit_width <= bits_per_word {
                // The value fits within a single word.
                let word = &mut limbs[word_index];
                *word &= !(self.mask << bit_offset);
                *word |= value_w << bit_offset;
            } else {
                // The value spans two words.
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

    /// Sets the value of an element, returning an error if the value doesn't fit.
    ///
    /// # Errors
    ///
    /// Returns an `Error::ValueTooLarge` if the provided `value` cannot be
    /// represented by the configured `bit_width`. Panics if `index` is out of bounds.
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
        
        // `set` would panic, but we've pre-flighted the check.
        unsafe { self.set_unchecked(index, value_w) };
        Ok(())
    }

    /// Returns an iterator over non-overlapping mutable chunks of the vector.
    ///
    /// # Panics
    /// Panics if `chunk_size` is 0.
    pub fn chunks_mut(&mut self, chunk_size: usize) -> iter_mut::ChunksMut<T, W, E, B> {
        iter_mut::ChunksMut::new(self, chunk_size)
    }

    /// Applies a function to all elements in place without checking if the
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
        if self.len == 0 || self.bit_width == 0 {
            return;
        }

        let bits_per_word = <W as Word>::BITS;
        let bit_width = self.bit_width;

        // --- Fast Path ---
        // This path is taken when elements are perfectly aligned within words (i.e.,
        // the bit width is a power of two that divides the word size). This allows
        // for a highly optimized loop that processes one word at a time without
        // handling complex boundary-crossing logic.
        if bit_width.is_power_of_two() && bits_per_word % bit_width == 0 {
            let elems_per_word = bits_per_word / bit_width;
            let mask = self.mask;
            let num_full_words = self.len / elems_per_word;

            for word_idx in 0..num_full_words {
                let mut new_word = W::ZERO;
                if E::IS_LITTLE {
                    let old_word = self.bits.as_ref()[word_idx];
                    for i in 0..elems_per_word {
                        let shift = i * bit_width;
                        let old_val_w = (old_word >> shift) & mask;
                        let new_val_w = <T as Storable<W>>::into_word(f(<T as Storable<W>>::from_word(old_val_w)));
                        new_word |= new_val_w << shift;
                    }
                    self.bits.as_mut()[word_idx] = new_word;
                } else { // Big-Endian
                    let old_word = W::from_be(self.bits.as_ref()[word_idx]);
                    for i in 0..elems_per_word {
                        let shift = bits_per_word - (i + 1) * bit_width;
                        let old_val_w = (old_word >> shift) & mask;
                        let new_val_w = <T as Storable<W>>::into_word(f(<T as Storable<W>>::from_word(old_val_w)));
                        new_word |= new_val_w << shift;
                    }
                    self.bits.as_mut()[word_idx] = new_word.to_be();
                }
            }

            // Process any remaining elements that do not fill a complete final word.
            let start_idx = num_full_words * elems_per_word;
            for i in start_idx..self.len {
                self.set_unchecked(i, <T as Storable<W>>::into_word(f(self.get_unchecked(i))));
            }
            return;
        }

        // --- Optimized Generic Path (Little-Endian) ---
        // This path handles any bit width by processing the vector as a stream of bits.
        // It reads elements that may span word boundaries and accumulates the modified
        // results into a local write buffer. The buffer is flushed to memory only when
        // a full word is ready, minimizing memory writes.
        if E::IS_LITTLE {
            let limbs = self.bits.as_mut();
            let num_limbs = (self.len * bit_width).div_ceil(bits_per_word);
            let limbs_ptr = limbs.as_ptr();

            let mut bit_pos = 0;
            let mut write_word_idx = 0;
            
            let mut write_buffer = W::ZERO;
            let mut bits_in_write_buffer = 0;
            
            let mask = self.mask;

            // Prefetch distance (in words) to fetch memory ahead of processing.
            const PREFETCH_DISTANCE: usize = 3;

            for i in 0..self.len {
                // On supported architectures, prefetch the memory word that will likely
                // be needed in a few iterations, hiding memory latency.
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    let next_bit_pos = (i + PREFETCH_DISTANCE) * bit_width;
                    let prefetch_word_idx = next_bit_pos / bits_per_word;
                    if prefetch_word_idx < num_limbs {
                        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                        // SAFETY: The calculated index is within the slice bounds for future iterations.
                        _mm_prefetch(limbs_ptr.add(prefetch_word_idx) as *const i8, _MM_HINT_T0);
                    }
                }

                let word_idx = bit_pos / bits_per_word;
                let bit_offset = bit_pos % bits_per_word;

                // Extract the value. This logic handles both aligned and boundary-crossing elements.
                let val_w = if bit_offset + bit_width <= bits_per_word {
                    // Element is fully contained within a single word.
                    (*limbs.get_unchecked(word_idx) >> bit_offset) & mask
                } else {
                    // Element spans two words. Combine bits from both using a `u128` temporary
                    // to avoid overflow during the shift.
                    let w1 = (*limbs.get_unchecked(word_idx)).to_u128().unwrap();
                    let w2 = (*limbs.get_unchecked(word_idx + 1)).to_u128().unwrap();
                    let combined = w1 | (w2 << bits_per_word);
                    num_traits::cast((combined >> bit_offset) & mask.to_u128().unwrap()).unwrap()
                };

                // Apply the user-provided function.
                let new_val_w = <T as Storable<W>>::into_word(f(<T as Storable<W>>::from_word(val_w)));

                // Accumulate the new value into the local write buffer.
                write_buffer |= new_val_w << bits_in_write_buffer;
                bits_in_write_buffer += bit_width;

                // If the write buffer is full, flush it to memory.
                if bits_in_write_buffer >= bits_per_word {
                    *limbs.get_unchecked_mut(write_word_idx) = write_buffer;
                    write_word_idx += 1;
                    bits_in_write_buffer -= bits_per_word;
                    // Carry over the remaining bits from the new value to the next word's buffer.
                    write_buffer = new_val_w >> (bit_width - bits_in_write_buffer);
                }
                
                bit_pos += bit_width;
            }

            // After the loop, flush any remaining partial data in the write buffer.
            if bits_in_write_buffer > 0 {
                let final_mask = (W::ONE << bits_in_write_buffer).wrapping_sub(W::ONE);
                let word_to_write = limbs.get_unchecked_mut(write_word_idx);
                *word_to_write &= !final_mask;
                *word_to_write |= write_buffer & final_mask;
            }

        } else {
            // Fallback Path for Big Endian. This logic is correct but less optimized
            // than the streaming approach for Little Endian.
            for i in 0..self.len {
                let old_val_t = self.get_unchecked(i);
                let new_val_t = f(old_val_t);
                self.set_unchecked(i, <T as Storable<W>>::into_word(new_val_t));
            }
        }
    }

    /// Applies a function to all elements in the vector, modifying them in-place.
    ///
    /// This method is highly optimized and will use a fast path for bit widths
    /// that are a power of two, performing word-at-a-time modifications. For
    /// other bit widths, it uses a generic but still efficient element-at-a-time
    /// approach.
    ///
    /// # Panics
    ///
    /// Panics if the function `f` returns a value that does not fit within the
    /// configured `bit_width` of the vector.
    ///
    /// # Example
    /// ```
    /// use compressed_intvec::prelude::*;
    ///
    /// // The initial values (0..10) would fit in 4 bits.
    /// // However, the mapped values (up to 9 * 2 = 18) will require 5 bits.
    /// // We must build the vector with enough space for the final results.
    /// let initial_data: Vec<u32> = (0..10).collect();
    /// let mut vec: UFixedVec<u32> = FixedVec::builder()
    ///     .bit_width(BitWidth::Explicit(5))
    ///     .build(&initial_data)
    ///     .unwrap();
    ///
    /// vec.map_in_place(|x| x * 2);
    ///
    /// let expected: Vec<u32> = (0..10).map(|x| x * 2).collect();
    /// assert_eq!(vec, &expected[..]);
    /// ```
    pub fn map_in_place<F>(&mut self, mut f: F)
    where
        F: FnMut(T) -> T,
    {
        // Capture necessary fields from `self` by value before creating the closure.
        // This prevents the closure from borrowing `self`.
        let bit_width = self.bit_width;
        let limit = if bit_width < <W as Word>::BITS {
            W::ONE << bit_width
        } else {
            W::max_value()
        };

        // This closure now captures `bit_width` and `limit` by value, not `&self`.
        let safe_f = |value: T| {
            let new_value = f(value);
            let new_value_w = <T as Storable<W>>::into_word(new_value);
            if bit_width < <W as Word>::BITS && new_value_w >= limit {
                panic!(
                    "map_in_place: returned value {:?} does not fit in the configured bit_width of {}",
                    new_value_w, bit_width
                );
            }
            new_value
        };

        // Now, `self` is not borrowed by the closure, so we can mutably borrow it here.
        // SAFETY: The `safe_f` wrapper ensures that any value passed to the
        // underlying unsafe function is valid for the vector's bit_width.
        unsafe {
            self.map_in_place_unchecked(safe_f);
        }
    }

    /// Replaces the element at a given index with a new value, returning the old value.
    ///
    /// This method first reads the existing value at the specified index, then
    /// overwrites it with the new value, and finally returns the original value.
    ///
    /// # Arguments
    ///
    /// * `index`: The index of the element to replace.
    /// * `value`: The new value to write at the specified index.
    ///
    /// # Returns
    ///
    /// The value that was previously at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds or if `value` does not fit within
    /// the configured `bit_width` of the vector.
    pub fn replace(&mut self, index: usize, value: T) -> T {
        // The assert inside `set` will also panic, but checking early provides a clearer message.
        assert!(index < self.len, "replace: index out of bounds");

        // Since we have already performed the bounds check, it is safe to use
        // the unchecked version for performance.
        let old_value = unsafe { self.get_unchecked(index) };

        // The `set` method handles the value-too-large check and the bit manipulation.
        self.set(index, value);

        old_value
    }

    /// Swaps two elements in the vector.
    ///
    /// # Arguments
    ///
    /// * `a`: The index of the first element.
    /// * `b`: The index of the second element.
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

        // A straightforward and correct implementation reads both values first,
        // then writes them back. This avoids issues where the bit ranges of the
        // two elements might overlap after being written once.
        // SAFETY: Bounds have been checked.
        unsafe {
            let val_a = self.get_unchecked(a);
            let val_b = self.get_unchecked(b);
            self.set_unchecked(a, <T as Storable<W>>::into_word(val_b));
            self.set_unchecked(b, <T as Storable<W>>::into_word(val_a));
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