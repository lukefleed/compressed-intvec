//! A compressed integer vector using variable-length encoding with fast random access.
//!
//! This module provides [`IntVec`], a data structure for storing sequences of
//! integers in a compressed format while retaining efficient random access. It is
//! well-suited for datasets where integer values are non-uniformly distributed,
//! as it uses instantaneous variable-length codes to represent smaller numbers with fewer bits.
//!
//! # Core Concepts
//!
//! ## Variable-Length Encoding
//!
//! Unlike [`FixedVec`], which uses a fixed number of
//! bits for every integer, [`IntVec`] employs **instantaneous codes** (such as
//! Gamma, Delta, or Rice codes) provided by the [`dsi-bitstream`] crate. This
//! approach allows each integer to be encoded with a variable number of bits,
//! typically using shorter codes for smaller or more frequent values. This can
//! lead to significant space savings, especially for data with many small numbers.
//!
//! Signed integers (e.g., `i8`, `i32`) are supported transparently through
//! zig-zag encoding, which maps small negative and positive integers to small
//! unsigned integers.
//!
//! ## Random Access and Sampling
//!
//! A key challenge with variable-length codes is that the location of the *i*-th
//! element cannot be calculated directly. To solve this, [`IntVec`] implements a
//! **sampling mechanism**. It stores the bit position of every *k*-th element in
//! a separate, auxiliary [`FixedVec`]. This parameter, `k`, is the **sampling rate**.
//!
//! To access an element at `index`, [`IntVec`]:
//! 1.  Finds the nearest sample by calculating `index / k`.
//! 2.  Retrieves the bit offset of the start of that sampled block.
//! 3.  Jumps to that offset in the compressed data stream.
//! 4.  Sequentially decodes the remaining `index % k` elements to reach the target.
//!
//! This strategy provides amortized O(1) random access, as the number of
//! sequential decoding steps is bounded by `k`.
//!
//! ### The `k` Trade-off
//!
//! The choice of the sampling rate `k` involves a trade-off:
//! -   **Smaller `k`**: Faster random access (fewer elements to decode per access)
//!     but higher memory usage due to a larger samples table.
//! -   **Larger `k`**: Slower random access but better compression, as the
//!     samples table is smaller.
//!
//! The optimal `k` depends on the specific access patterns of an application.
//!
//! # Design and Immutability
//!
//! [`IntVec`] is **immutable** after creation.
//! Unlike [`FixedVec`], it does not provide methods for
//! in-place modification (e.g., `set`, `push`).
//!
//! If a value in the middle of the compressed bitstream were changed, its new
//! encoded length might be different. For example, changing `5` (which might be
//! 4 bits) to `5000` (which might be 16 bits) would require shifting all
//! subsequent data, invalidating every sample point that follows. The cost of
//! such an operation would be prohibitive, scaling with the length of the vector.
//!
//! For this reason, [`IntVec`] is designed as a write-once, read-many data structure.
//!
//! # Access Strategies and Readers
//!
//! [`IntVec`] provides multiple interfaces for accessing data, each optimized for a
//! different pattern of use.
//!
//! - **[`get()`](IntVec::get)**: For single, infrequent lookups. Each call creates and discards
//!   an internal reader, which incurs overhead if used in a loop.
//!
//! - **[`get_many()`](IntVec::get_many)**: The most efficient method for retrieving a batch of elements
//!   when all indices are known beforehand. It sorts the indices to perform a
//!   single, monotonic scan over the data, minimizing redundant decoding and seek
//!   operations.
//!
//! - **[`IntVecReader`] (Reusable Stateless Reader)**: This reader is created via
//!   [`IntVec::reader()`]. It maintains an internal, reusable bitstream reader,
//!   amortizing its setup cost over multiple calls. Each [`get()`](IntVec::get) call is
//!   **stateless** with respect to position: it performs a full seek to the nearest
//!   sample and decodes forward from there, independently of previous calls. It is
//!   best suited for true random access patterns where indices are sparse and
//!   unpredictable.
//!
//! - **[`IntVecSeqReader`] (Stateful Sequential Reader)**: This reader, created via
//!   [`IntVec::seq_reader()`], is a stateful object optimized for access patterns with
//!   high locality. It maintains an internal cursor of the current decoding position.
//!   - **Fast Path**: If a requested `index` is at or after the cursor's current
//!     position and within the same sample block, the reader simply decodes
//!     forward from its last position, avoiding a costly seek operation.
//!   - **Fallback Path**: If the requested `index` is before the cursor or in a
//!     different sample block, the reader falls back to the standard behavior of
//!     seeking to the nearest sample and decoding from there. This makes it very
//!     efficient for iterating through sorted or clustered indices.
//!
//! # Main Components
//!
//! - [`IntVec`]: The core compressed vector.
//! - [`IntVecBuilder`]: The primary tool for constructing an [`IntVec`] with
//!   custom compression codecs and sampling rates.
//! - [`VariableCodecSpec`]: An enum to specify the compression codec.
//! - [`IntVecReader`]: A reusable, stateless reader for efficient random access.
//! - [`IntVecSeqReader`]: A stateful reader optimized for sequential or localized access patterns.
//! - [`IntVecSlice`]: An immutable, zero-copy view over a portion of the vector.
//!
//! # Examples
//!
//! ## Basic Usage with Unsigned Integers
//!
//! Create a [`UIntVec`] (an alias for `IntVec<u32, LE>`) from a slice of `u32`. The builder will automatically
//! select a suitable codec and use a default sampling rate.
//!
//! ```
//! use compressed_intvec::variable::{IntVec, UIntVec};
//!
//! let data: Vec<u32> = vec![100, 200, 300, 1024];
//! let vec: UIntVec<u32> = IntVec::from_slice(&data).unwrap();
//!
//! assert_eq!(vec.len(), 4);
//! // Accessing an element
//! assert_eq!(vec.get(1), Some(200));
//! ```
//!
//! ## Storing Signed Integers
//!
//! [`IntVec`] handles signed integers, such as [`i16`], by mapping them to unsigned
//! values using zig-zag encoding.
//!
//! ```
//! use compressed_intvec::variable::{IntVec, SIntVec};
//!
//! let data: &[i16] = &[-5, 20, -100, 0, 8];
//! let vec: SIntVec<i16> = IntVec::from_slice(data).unwrap();
//!
//! assert_eq!(vec.len(), 5);
//! assert_eq!(vec.get(0), Some(-5));
//! assert_eq!(vec.get(2), Some(-100));
//! ```
//!
//! ## Manual Codec and Sampling Rate
//!
//! For fine-grained control, use the [`IntVecBuilder`]. Here, we specify a
//! sampling rate of `k=8` and use the `Zeta` code with `k=3`.
//!
//! ```
//! use compressed_intvec::variable::{IntVec, UIntVec, VariableCodecSpec};
//!
//! let data: Vec<u64> = (0..100).map(|i| i * i).collect();
//!
//! let vec: UIntVec<u64> = IntVec::builder(&data)
//!     .k(8) // Set sampling rate
//!     .codec(VariableCodecSpec::Zeta { k: Some(3) }) // Set compression codec
//!     .build()
//!     .unwrap();
//!
//! assert_eq!(vec.get_sampling_rate(), 8);
//! assert_eq!(vec.get(10), Some(100));
//! ```
//!
//! Best performance is achieved when the sampling rate `k` is a power of two. Usually a value of `32` or `16` is a good trade-off between speed and compression ratio.
//! 
//! ## Codec Selection and Performance
//!
//! The choice of compression codec is critical for performance and space efficiency.
//! [`IntVecBuilder`](builder::IntVecBuilder) offers automatic codec selection via 
//! [`VariableCodecSpec::Auto`]. When enabled, the builder analyzes the entire input
//! dataset to find the codec that offers the best compression ratio.
//!
//! This analysis involves calculating the compressed size for the data with
//! approximately 70 different codec configurations. This process introduces a
//! significant, one-time **construction overhead**.
//! 
//! Use [`Auto`](VariableCodecSpec::Auto) for read-heavy workloads where the [`IntVec`] 
//! is built once and then accessed many times. The initial cost is easily amortized by
//! the long-term space savings.
//! 
//! If your application creates many small [`IntVec`]s or accesses them frequently,
//! the repeated cost of analysis can become a performance
//! bottleneck. In such scenarios, it is better to explicitly specify a codec
//! (e.g., [`VariableCodecSpec::Gamma`] or [`VariableCodecSpec::Delta`]) that is known
//! to be a good general-purpose choice for your data.
//! 
//! ```
//! use compressed_intvec::prelude::*;
//! 
//! let data: Vec<u32> = (0..100).collect();
//! 
//! // Create an IntVec with automatic codec selection
//! let vec: UIntVec<u32> = IntVec::builder(&data)
//!     .build()
//!     .unwrap();
//!```
//!  
//! [`dsi-bitstream`]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/

#[macro_use]
mod macros;

pub mod builder;
pub mod codec;
pub mod iter;
#[cfg(feature = "parallel")]
mod parallel;
pub mod reader;
pub mod seq_reader;
#[cfg(feature = "serde")]
pub mod serde;
pub mod slice;
pub mod traits;

pub use self::{codec::VariableCodecSpec, traits::Storable};
use crate::fixed::{Error as FixedVecError, FixedVec};
use dsi_bitstream::{
    codes::params::DefaultReadParams,
    dispatch::StaticCodeRead,
    prelude::{
        BitRead, BitSeek, BufBitReader, BufBitWriter, Codes, CodesRead, CodesWrite, Endianness,
        MemWordReader, MemWordWriterVec,
    },
    traits::{BitWrite, BE, LE},
};
use mem_dbg::{MemDbg, MemSize};
use std::{
    error::Error,
    fmt::{self},
    marker::PhantomData,
};

pub use builder::{IntVecBuilder, IntVecFromIterBuilder};
use iter::{IntVecIntoIter, IntVecIter};
pub use reader::IntVecReader;
pub use seq_reader::IntVecSeqReader;
pub use slice::IntVecSlice;

/// Defines the set of errors that can occur in `IntVec` operations.
#[derive(Debug)]
pub enum IntVecError {
    /// An error occurred during an I/O operation, typically from the underlying
    /// bitstream reader or writer.
    Io(std::io::Error),
    /// A generic error from the `dsi-bitstream` library, often related to
    /// decoding malformed data.
    Bitstream(Box<dyn Error + Send + Sync>),
    /// An error indicating that one or more parameters are invalid for the
    /// requested operation.
    InvalidParameters(String),
    /// An error that occurs during the dynamic dispatch of codec functions.
    CodecDispatch(String),
    /// An error indicating that a provided index is outside the valid bounds
    /// of the vector.
    IndexOutOfBounds(usize),
}

impl fmt::Display for IntVecError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntVecError::Io(e) => write!(f, "I/O error: {}", e),
            IntVecError::Bitstream(e) => write!(f, "Bitstream error: {}", e),
            IntVecError::InvalidParameters(s) => write!(f, "Invalid parameters: {}", s),
            IntVecError::CodecDispatch(s) => write!(f, "Codec dispatch error: {}", s),
            IntVecError::IndexOutOfBounds(index) => write!(f, "Index out of bounds: {}", index),
        }
    }
}

impl Error for IntVecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            IntVecError::Io(e) => Some(e),
            IntVecError::Bitstream(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for IntVecError {
    fn from(e: std::io::Error) -> Self {
        IntVecError::Io(e)
    }
}

impl From<core::convert::Infallible> for IntVecError {
    fn from(_: core::convert::Infallible) -> Self {
        unreachable!()
    }
}

impl From<FixedVecError> for IntVecError {
    fn from(e: FixedVecError) -> Self {
        IntVecError::InvalidParameters(e.to_string())
    }
}

/// A compressed, randomly accessible vector of integers using variable-length encoding.
///
/// `IntVec` achieves compression by using instantaneous codes and enables fast,
/// amortized O(1) random access via a sampling mechanism. See the
/// [module-level documentation](crate::variable) for a detailed explanation.
///
/// # Type Parameters
///
/// - `T`: The integer type for the elements (e.g., `u32`, `i16`). It must
///   implement the [`Storable`] trait.
/// - `E`: The [`Endianness`] of the underlying bitstream (e.g., [`LE`] or [`BE`]).
/// - `B`: The backend storage buffer, such as `Vec<u64>` for an owned vector or
///   `&[u64]` for a borrowed, zero-copy view.
#[derive(Debug, Clone, MemSize, MemDbg)]
pub struct IntVec<T: Storable, E: Endianness, B: AsRef<[u64]> = Vec<u64>> {
    /// The raw, bit-packed compressed data.
    pub(super) data: B,
    /// A `FixedVec` containing the bit offsets of sampled elements.
    pub(super) samples: FixedVec<u64, u64, LE, B>,
    /// The sampling rate `k`. Every `k`-th element's position is stored.
    pub(super) k: usize,
    /// The number of elements in the vector.
    pub(super) len: usize,
    /// The `dsi-bitstream` code used for compression.
    pub(super) encoding: Codes,
    /// Zero-sized markers for the generic type parameters.
    pub(super) _markers: PhantomData<(T, E)>,
}

/// Type alias for the bit writer used internally by `IntVec` builders.
pub(crate) type IntVecBitWriter<E> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;
/// Type alias for the bit reader used internally by `IntVec` accessors.
pub(crate) type IntVecBitReader<'a, E> =
    BufBitReader<E, MemWordReader<u64, &'a [u64]>, DefaultReadParams>;

impl<T: Storable, E: Endianness> IntVec<T, E, Vec<u64>> {
    /// Creates a builder for constructing an owned [`IntVec`] from a slice of data.
    ///
    /// This is the most flexible way to create an [`IntVec`], allowing customization
    /// of the compression codec and sampling rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::variable::{IntVec, UIntVec, VariableCodecSpec};
    ///
    /// let data: &[u32] = &[5, 8, 13, 21, 34];
    /// let vec: UIntVec<u32> = IntVec::builder(data)
    ///     .k(2) // Sample every 2nd element
    ///     .codec(VariableCodecSpec::Delta)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(vec.get(3), Some(21));
    /// ```
    pub fn builder(input: &'_ [T]) -> IntVecBuilder<'_, T, E> {
        IntVecBuilder::new(input)
    }

    /// Creates a builder for constructing an owned [`IntVec`] from an iterator.
    ///
    /// This is useful for large datasets that are generated on the fly.
    pub fn from_iter_builder<I>(iter: I) -> IntVecFromIterBuilder<T, E, I>
    where
        I: IntoIterator<Item = T> + Clone,
    {
        IntVecFromIterBuilder::new(iter)
    }

    /// Consumes the [`IntVec`] and returns its decoded values as a standard `Vec<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::variable::{IntVec, SIntVec};
    ///
    /// let data: &[i32] = &[-10, 0, 10];
    /// let vec: SIntVec<i32> = IntVec::from_slice(data).unwrap();
    /// let decoded_data = vec.into_vec();
    ///
    /// assert_eq!(decoded_data, &[-10, 0, 10]);
    /// ```
    pub fn into_vec(self) -> Vec<T>
    where
        for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.into_iter().collect()
    }

    /// Creates an owned [`IntVec`] from a slice of data using default settings.
    ///
    /// This method uses [`VariableCodecSpec::Auto`] to select a codec and a
    /// default sampling rate of `k=16`.
    pub fn from_slice(slice: &[T]) -> Result<Self, IntVecError>
    where
        for<'a> crate::variable::IntVecBitWriter<E>:
            BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        Self::builder(slice)
            .k(16)
            .codec(VariableCodecSpec::Auto)
            .build()
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> IntVec<T, E, B> {
    /// Creates a new [`IntVec`] from its raw components, enabling zero-copy views.
    ///
    /// This constructor is intended for advanced use cases, such as memory-mapping
    /// a pre-built [`IntVec`] from disk without copying the data.
    ///
    /// # Errors
    ///
    /// Returns an [`IntVecError::InvalidParameters`] if `k` is zero or if the
    /// number of samples is inconsistent with `len` and `k`.
    pub fn from_parts(
        data: B,
        samples_data: B,
        samples_len: usize,
        samples_num_bits: usize,
        k: usize,
        len: usize,
        encoding: Codes,
    ) -> Result<Self, IntVecError> {
        let samples =
            FixedVec::<u64, u64, LE, B>::from_parts(samples_data, samples_len, samples_num_bits)?;

        if k == 0 {
            return Err(IntVecError::InvalidParameters(
                "Sampling rate k cannot be zero".to_string(),
            ));
        }
        let expected_samples = if len == 0 { 0 } else { len.div_ceil(k) };
        if samples.len() != expected_samples {
            return Err(IntVecError::InvalidParameters(format!(
                "Inconsistent number of samples. Expected {}, found {}",
                expected_samples,
                samples.len()
            )));
        }

        Ok(unsafe { Self::new_unchecked(data, samples, k, len, encoding) })
    }

    /// Creates a new [`IntVec`] from its raw parts without performing safety checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all parameters are consistent and valid. The
    /// `samples` must contain the correct bit offsets for the `data` stream,
    /// and `len`, `k`, and `encoding` must accurately describe the layout.
    /// Mismatched parameters will lead to panics or incorrect data retrieval.
    pub(crate) unsafe fn new_unchecked(
        data: B,
        samples: FixedVec<u64, u64, LE, B>,
        k: usize,
        len: usize,
        encoding: Codes,
    ) -> Self {
        Self {
            data,
            samples,
            k,
            len,
            encoding,
            _markers: PhantomData,
        }
    }

    /// Creates a zero-copy, immutable view (a _slice_) of this vector.
    ///
    /// Returns `None` if the specified range is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::variable::{IntVec, UIntVec};
    ///
    /// let data: Vec<u32> = (0..20).collect();
    /// let vec: UIntVec<u32> = IntVec::from_slice(&data).unwrap();
    /// let slice = vec.slice(5, 10).unwrap();
    ///
    /// assert_eq!(slice.len(), 10);
    /// assert_eq!(slice.get(0), Some(5)); // Corresponds to index 5 of the original vec
    /// ```
    pub fn slice(&'_ self, start: usize, len: usize) -> Option<IntVecSlice<'_, T, E, B>> {
        if start.saturating_add(len) > self.len {
            return None;
        }
        Some(IntVecSlice::new(self, start..start + len))
    }

    /// Splits the vector into two immutable slices at a given index.
    ///
    /// Returns `None` if `mid` is out of bounds.
    #[allow(clippy::type_complexity)]
    pub fn split_at(
        &'_ self,
        mid: usize,
    ) -> Option<(IntVecSlice<'_, T, E, B>, IntVecSlice<'_, T, E, B>)> {
        if mid > self.len {
            return None;
        }
        let left = IntVecSlice::new(self, 0..mid);
        let right = IntVecSlice::new(self, mid..self.len);
        Some((left, right))
    }

    /// Returns the number of integers in the vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the sampling rate `k` used during encoding.
    #[inline]
    pub fn get_sampling_rate(&self) -> usize {
        self.k
    }

    /// Returns the number of sample points stored in the vector.
    #[inline]
    pub fn get_num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Returns a reference to the inner `FixedVec` of samples.
    #[inline]
    pub fn samples_ref(&self) -> &FixedVec<u64, u64, LE, B> {
        &self.samples
    }

    /// Returns a read-only slice of the underlying compressed data words (`&[u64]`).
    #[inline]
    pub fn as_limbs(&self) -> &[u64] {
        self.data.as_ref()
    }

    /// Returns the concrete [`Codes`] variant that was used for compression.
    #[inline]
    pub fn encoding(&self) -> Codes {
        self.encoding
    }

    /// Returns a clone of the underlying storage as a `Vec<u64>`.
    pub fn limbs(&self) -> Vec<u64> {
        self.data.as_ref().to_vec()
    }

    /// Returns an iterator over the decompressed values.
    pub fn iter(&'_ self) -> impl Iterator<Item = T> + '_
    where
        for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        IntVecIter::new(self)
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> IntVec<T, E, B>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a reusable, stateless reader for efficient random access.
    ///
    /// This method returns an [`IntVecReader`], a struct that maintains a persistent,
    /// reusable bitstream reader. This amortizes the setup cost across multiple `get`
    /// operations, making it more efficient than calling [`get`](IntVec::get) repeatedly in a loop.
    ///
    /// This reader is **stateless**: it performs a full seek from the nearest sample point for each call,
    /// independently of any previous access.
    ///
    /// # When to use it
    /// Use [`IntVecReader`] for true random access patterns where lookup indices are sparse,
    /// unordered, or not known in advance (e.g., graph traversals, pointer chasing).
    /// For accessing a known set of indices, [`get_many`](IntVec::get_many) is generally superior.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: Vec<u32> = (0..100).rev().collect(); // Data is not sequential
    /// let vec: UIntVec<u32> = IntVec::from_slice(&data).unwrap();
    ///
    /// // Create a reusable reader for multiple random lookups
    /// let mut reader = vec.reader();
    ///
    /// assert_eq!(reader.get(99).unwrap(), Some(0));
    /// assert_eq!(reader.get(0).unwrap(), Some(99));
    /// assert_eq!(reader.get(50).unwrap(), Some(49));
    /// ```
    pub fn reader(&'_ self) -> IntVecReader<'_, T, E, B> {
        IntVecReader::new(self)
    }

    /// Creates a stateful, reusable reader optimized for sequential access.
    ///
    /// This method returns an [`IntVecSeqReader`], which is specifically designed
    /// to take advantage of the vector's internal state, tracking the current decoding position (cursor).
    ///
    /// This statefulness enables a key optimization:
    /// - **Fast Path**: If a requested index is at or after the cursor and within
    ///   the same sample block, the reader decodes forward from its last known
    ///   position. This avoids a costly seek operation.
    /// - **Fallback Path**: If the requested index is before the cursor (requiring a
    ///   backward move) or in a different sample block, the reader falls back to
    ///   the standard behavior of seeking to the nearest sample point.
    ///
    /// # When to use it
    /// Use [`IntVecSeqReader`] when your access pattern has high locality, meaning
    /// indices are primarily increasing and often clustered together. It is ideal
    /// for iterating through a sorted list of indices or for stream-like processing.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: Vec<u32> = (0..100).collect();
    /// let vec: UIntVec<u32> = IntVec::from_slice(&data).unwrap();
    ///
    /// // Create a reader optimized for sequential access
    /// let mut seq_reader = vec.seq_reader();
    ///
    /// // Accessing indices in increasing order is efficient
    /// assert_eq!(seq_reader.get(10).unwrap(), Some(10));
    /// // This next call is fast, as it decodes forward from index 10
    /// assert_eq!(seq_reader.get(15).unwrap(), Some(15));
    ///
    /// // A large jump will trigger a seek to a new sample block
    /// assert_eq!(seq_reader.get(90).unwrap(), Some(90));
    ///
    /// // A backward jump will also trigger a seek
    /// assert_eq!(seq_reader.get(5).unwrap(), Some(5));
    /// ```
    pub fn seq_reader(&'_ self) -> IntVecSeqReader<'_, T, E, B> {
        IntVecSeqReader::new(self)
    }

    /// Returns the element at the specified index, or `None` if the index is
    /// out of bounds.
    ///
    /// This operation is amortized O(1).
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::variable::{IntVec, UIntVec};
    ///
    /// let data: Vec<u32> = (0..100).collect();
    /// let vec: UIntVec<u32> = IntVec::from_slice(&data).unwrap();
    ///
    /// assert_eq!(vec.get(50), Some(50));
    /// assert_eq!(vec.get(100), None);
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
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        let mut reader = self.reader();
        reader.get_unchecked(index)
    }

    /// Retrieves multiple elements from the vector at the specified indices.
    ///
    /// This method is generally more efficient than calling [`get`](Self::get) in a loop, as
    /// it sorts the indices and scans through the compressed data stream once.
    ///
    /// # Errors
    ///
    /// Returns [`IntVecError::IndexOutOfBounds`] if any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::variable::{IntVec, UIntVec};
    ///
    /// let data: Vec<u32> = (0..100).collect();
    /// let vec: UIntVec<u32> = IntVec::from_slice(&data).unwrap();
    ///
    /// let indices = [99, 0, 50];
    /// let values = vec.get_many(&indices).unwrap();
    /// assert_eq!(values, vec![99, 0, 50]);
    /// ```
    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<T>, IntVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        for &index in indices {
            if index >= self.len {
                return Err(IntVecError::IndexOutOfBounds(index));
            }
        }
        // SAFETY: We have just performed the bounds checks.
        Ok(unsafe { self.get_many_unchecked(indices) })
    }

    /// Retrieves multiple elements without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with any out-of-bounds index is undefined behavior.
    #[allow(clippy::uninit_vec)]
    pub unsafe fn get_many_unchecked(&self, indices: &[usize]) -> Vec<T> {
        if indices.is_empty() {
            return Vec::new();
        }
        let mut results = Vec::with_capacity(indices.len());
        // SAFETY: The vector is immediately populated by the sorted access logic below.
        results.set_len(indices.len());

        let mut indexed_indices: Vec<(usize, usize)> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();
        // Sort by the target index to enable efficient sequential scanning.
        indexed_indices.sort_unstable_by_key(|&(idx, _)| idx);

        if self.k.is_power_of_two() {
            // Optimization: use bit-shift for division if k is a power of two.
            let k_exp = self.k.trailing_zeros();
            self.get_many_dsi_inner(
                &indexed_indices,
                &mut results,
                |idx| idx >> k_exp,
                |block| block << k_exp,
            )
            .unwrap();
        } else {
            self.get_many_dsi_inner(
                &indexed_indices,
                &mut results,
                |idx| idx / self.k,
                |block| block * self.k,
            )
            .unwrap();
        }

        results
    }

    /// Internal implementation for `get_many_unchecked`.
    ///
    /// This function takes closures to abstract away the division/multiplication
    /// by `k`, allowing for a bit-shift optimization when `k` is a power of two.
    fn get_many_dsi_inner<F1, F2>(
        &self,
        indexed_indices: &[(usize, usize)],
        results: &mut [T],
        block_of: F1,
        start_of_block: F2,
    ) -> Result<(), IntVecError>
    where
        F1: Fn(usize) -> usize,
        F2: Fn(usize) -> usize,
    {
        let mut reader = self.reader();
        let mut current_decoded_index: usize = 0;

        for &(target_index, original_position) in indexed_indices {
            // Check if we need to jump to a new sample block. This is true if the
            // target index is before our current position, or if it's in a different
            // sample block than the one we're currently in.
            if target_index < current_decoded_index
                || block_of(target_index) != block_of(current_decoded_index.saturating_sub(1))
            {
                let target_sample_block = block_of(target_index);
                // SAFETY: The public-facing `get_many` performs bounds checks.
                let start_bit = unsafe { self.samples.get_unchecked(target_sample_block) };
                reader.reader.set_bit_pos(start_bit)?;
                current_decoded_index = start_of_block(target_sample_block);
            }

            // Sequentially decode elements until we reach our target.
            for _ in current_decoded_index..target_index {
                reader.code_reader.read(&mut reader.reader)?;
            }
            let value = reader.code_reader.read(&mut reader.reader)?;
            // Place the decoded value in its original requested position.
            results[original_position] = Storable::from_word(value);
            current_decoded_index = target_index + 1;
        }
        Ok(())
    }

    /// Retrieves multiple elements from an iterator of indices.
    ///
    /// This is a convenient alternative to [`get_many`](Self::get_many) when the indices are not
    /// already in a slice. It may be less performant as it cannot pre-sort the
    /// indices for optimal access.
    pub fn get_many_from_iter<I>(&self, indices: I) -> Result<Vec<T>, IntVecError>
    where
        I: IntoIterator<Item = usize>,
    {
        let indices_iter = indices.into_iter();
        let (lower_bound, _) = indices_iter.size_hint();
        let mut results = Vec::with_capacity(lower_bound);
        let mut seq_reader = self.seq_reader();

        for index in indices_iter {
            let value = seq_reader
                .get(index)?
                .ok_or(IntVecError::IndexOutOfBounds(index))?;
            results.push(value);
        }

        Ok(results)
    }
}

impl<T: Storable + Ord, E: Endianness, B: AsRef<[u64]>> IntVec<T, E, B>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Binary searches this vector for a given element.
    ///
    /// If the value is found, returns `Ok(usize)` with the index of the
    /// matching element. If the value is not found, returns `Err(usize)` with
    /// the index where the value could be inserted to maintain order.
    ///
    /// # Complexity
    ///
    /// The time complexity of this operation is O(k * log n), where `n` is the
    /// number of elements in the vector and `k` is the sampling rate. This is
    /// because each of the O(log n) probes during the search requires an
    /// element access, which has a cost proportional to `k` in the worst case.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::variable::{IntVec, SIntVec};
    ///
    /// let data: &[i32] = &[-10, 0, 10, 20, 30];
    /// let vec: SIntVec<i32> = IntVec::from_slice(data).unwrap();
    ///
    /// assert_eq!(vec.binary_search(&10), Ok(2));
    /// assert_eq!(vec.binary_search(&15), Err(3));
    /// ```
    pub fn binary_search(&self, value: &T) -> Result<usize, usize> {
        self.binary_search_by(|probe| probe.cmp(value))
    }

    /// Binary searches this vector with a custom comparison function.
    ///
    /// # Complexity
    ///
    /// The time complexity of this operation is O(k * log n), where `n` is the
    /// number of elements in the vector and `k` is the sampling rate.
    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> std::cmp::Ordering,
    {
        let mut low = 0;
        let mut high = self.len();
        let mut reader = self.reader();

        while low < high {
            let mid = low + (high - low) / 2;
            // SAFETY: The loop invariants ensure `mid` is always in bounds.
            let cmp = f(unsafe { reader.get_unchecked(mid) });

            match cmp {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => return Ok(mid),
                std::cmp::Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }

    /// Binary searches this vector with a key extraction function.
    ///
    /// # Complexity
    ///
    /// The time complexity of this operation is O(k * log n), where `n` is the
    /// number of elements in the vector and `k` is the sampling rate.
    #[inline]
    pub fn binary_search_by_key<K: Ord, F>(&self, b: &K, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> K,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> IntoIterator for IntVec<T, E, B>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;
    type IntoIter = IntVecIntoIter<T, E, B>;

    fn into_iter(self) -> Self::IntoIter {
        IntVecIntoIter::new(self)
    }
}

/// An [`IntVec`] for unsigned integers with Little-Endian bit layout.
pub type UIntVec<T> = IntVec<T, LE>;
/// An [`IntVec`] for signed integers with Little-Endian bit layout.
pub type SIntVec<T> = IntVec<T, LE>;
/// An [`IntVec`] for `u64` elements with Big-Endian bit layout.
pub type BEIntVec = IntVec<u64, BE>;
/// An [`IntVec`] for `u64` elements with Little-Endian bit layout.
pub type LEIntVec = IntVec<u64, LE>;
/// An [`IntVec`] for `i64` elements with Big-Endian bit layout.
pub type BESIntVec = IntVec<i64, BE>;
/// An [`IntVec`] for `i64` elements with Little-Endian bit layout.
pub type LESIntVec = IntVec<i64, LE>;

impl<T, E, B, O> PartialEq<O> for IntVec<T, E, B>
where
    T: Storable + PartialEq,
    E: Endianness,
    B: AsRef<[u64]>,
    O: AsRef<[T]>,
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Checks for equality between an `IntVec` and a standard slice.
    ///
    /// The comparison is done by iterating over both and comparing elements
    /// one by one. The overall comparison is not a single atomic operation.
    fn eq(&self, other: &O) -> bool {
        let other_slice = other.as_ref();
        if self.len() != other_slice.len() {
            return false;
        }
        self.iter().zip(other_slice.iter()).all(|(a, b)| a == *b)
    }
}
