//! A compressed integer vector using variable-length encoding with fast random access.
//!
//! This module provides [`VarVec`], a data structure for storing sequences of
//! integers in a compressed format while retaining efficient random access. It is
//! well-suited for datasets where integer values are non-uniformly distributed,
//! as it uses instantaneous variable-length codes to represent smaller numbers with fewer bits.
//!
//! # Core Concepts
//!
//! ## Variable-Length Encoding
//!
//! Unlike [`FixedVec`], which uses a fixed number of
//! bits for every integer, [`VarVec`] employs **instantaneous codes** (such as
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
//! element cannot be calculated directly. To solve this, [`VarVec`] implements a
//! **sampling mechanism**. It stores the bit position of every *k*-th element in
//! a separate, auxiliary [`FixedVec`]. This parameter, `k`, is the **sampling rate**.
//!
//! To access an element at `index`, [`VarVec`]:
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
//! [`VarVec`] is **immutable** after creation.
//! Unlike [`FixedVec`], it does not provide methods for
//! in-place modification (e.g., `set`, `push`).
//!
//! If a value in the middle of the compressed bitstream were changed, its new
//! encoded length might be different. For example, changing `5` (which might be
//! 4 bits) to `5000` (which might be 16 bits) would require shifting all
//! subsequent data, invalidating every sample point that follows. The cost of
//! such an operation would be prohibitive, scaling with the length of the vector.
//!
//! For this reason, [`VarVec`] is designed as a write-once, read-many data structure.
//!
//! # Access Strategies and Readers
//!
//! [`VarVec`] provides multiple interfaces for accessing data, each optimized for a
//! different pattern of use.
//!
//! - **[`get()`](VarVec::get)**: For single, infrequent lookups. Each call creates and discards
//!   an internal reader, which incurs overhead if used in a loop.
//!
//! - **[`get_many()`](VarVec::get_many)**: The most efficient method for retrieving a batch of elements
//!   when all indices are known beforehand. It sorts the indices to perform a
//!   single, monotonic scan over the data, minimizing redundant decoding and seek
//!   operations.
//!
//! - **[`VarVecReader`] (Reusable Stateless Reader)**: This reader is created via
//!   [`VarVec::reader()`]. It maintains an internal, reusable bitstream reader,
//!   amortizing its setup cost over multiple calls. Each [`get()`](VarVec::get) call is
//!   **stateless** with respect to position: it performs a full seek to the nearest
//!   sample and decodes forward from there, independently of previous calls. It is
//!   best suited for true random access patterns where indices are sparse and
//!   unpredictable.
//!
//! - **[`VarVecSeqReader`] (Stateful Sequential Reader)**: This reader, created via
//!   [`VarVec::seq_reader()`], is a stateful object optimized for access patterns with
//!   high locality. It maintains an internal cursor of the current decoding position.
//!   - **Fast Path**: If a requested `index` is at or after the cursor's current
//!     position and within the same sample block, the reader simply decodes
//!     forward from its last position, avoiding a costly seek operation.
//!   - **Fallback Path**: If the requested `index` is before the cursor or in a
//!     different sample block, the reader falls back to the standard behavior of
//!     seeking to the nearest sample and decoding from there. This makes it very
//!     efficient for iterating through sorted or clustered indices.
//!
//! # Migration Notice
//!
//! As of version 0.6.0, all `IntVec*` types have been renamed to `VarVec*` for
//! consistency with the module naming convention (`fixed` → `FixedVec`, `seq` → `SeqVec`,
//! `variable` → `VarVec`). The old names remain available as deprecated type aliases
//! and will continue to work, but using the new names is recommended.
//!
//! # Main Components
//!
//! - [`VarVec`]: The core compressed vector.
//! - [`VarVecBuilder`]: The primary tool for constructing an [`VarVec`] with
//!   custom compression codecs and sampling rates.
//! - [`Codec`]: An enum to specify the compression codec.
//! - [`VarVecReader`]: A reusable, stateless reader for efficient random access.
//! - [`VarVecSeqReader`]: A stateful reader optimized for sequential or localized access patterns.
//! - [`VarVecSlice`]: An immutable, zero-copy view over a portion of the vector.
//!
//! # Examples
//!
//! ## Basic Usage with Unsigned Integers
//!
//! Create a [`UVarVec`] (an alias for `VarVec<u32, LE>`) from a slice of `u32`. The builder will automatically
//! select a suitable codec and use a default sampling rate.
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::variable::{VarVec, UVarVec};
//!
//! let data: Vec<u32> = vec![100, 200, 300, 1024];
//! let vec: UVarVec<u32> = VarVec::from_slice(&data)?;
//!
//! assert_eq!(vec.len(), 4);
//! // Accessing an element
//! assert_eq!(vec.get(1), Some(200));
//! # Ok(())
//! # }
//! ```
//!
//! ## Storing Signed Integers
//!
//! [`VarVec`] handles signed integers, such as [`i16`], by mapping them to unsigned
//! values using zig-zag encoding.
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::variable::{VarVec, SVarVec};
//!
//! let data: &[i16] = &[-5, 20, -100, 0, 8];
//! let vec: SVarVec<i16> = VarVec::from_slice(data)?;
//!
//! assert_eq!(vec.len(), 5);
//! assert_eq!(vec.get(0), Some(-5));
//! assert_eq!(vec.get(2), Some(-100));
//! # Ok(())
//! # }
//! ```
//!
//! ## Manual Codec and Sampling Rate
//!
//! For fine-grained control, use the [`VarVecBuilder`]. Here, we specify a
//! sampling rate of `k=8` and use the `Zeta` code with `k=3`.
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::variable::{VarVec, UVarVec, Codec};
//!
//! let data: Vec<u64> = (0..100).map(|i| i * i).collect();
//!
//! let vec: UVarVec<u64> = VarVec::builder()
//!     .k(8) // Set sampling rate
//!     .codec(Codec::Zeta { k: Some(3) }) // Set compression codec
//!     .build(&data)?;
//!
//! assert_eq!(vec.sampling_rate(), 8);
//! assert_eq!(vec.get(10), Some(100));
//! # Ok(())
//! # }
//! ```
//!
//! Best performance is achieved when the sampling rate `k` is a power of two. Usually a value of `32` or `16` is a good trade-off between speed and compression ratio.
//!
//! ## Codec Selection and Performance
//!
//! The choice of compression codec is critical for performance and space efficiency.
//! [`VarVecBuilder`] offers automatic codec selection via
//! [`Codec::Auto`]. When enabled, the builder analyzes the entire input
//! dataset to find the codec that offers the best compression ratio.
//!
//! This analysis involves calculating the compressed size for the data with
//! approximately 70 different codec configurations. This process introduces a
//! significant, one-time **construction overhead**.
//!
//! Use [`Auto`](Codec::Auto) for read-heavy workloads where the [`VarVec`]
//! is built once and then accessed many times. The initial cost is easily amortized by
//! the long-term space savings.
//!
//! If your application creates many small [`VarVec`]s or accesses them frequently,
//! the repeated cost of analysis can become a performance
//! bottleneck. In such scenarios, it is better to explicitly specify a codec
//! (e.g., [`Gamma`](Codec::Gamma) or [`Delta`](Codec::Delta)) that is known
//! to be a good general-purpose choice for your data.
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::prelude::*;
//!
//! let data: Vec<u32> = (0..100).collect();
//!
//! // Create an VarVec with automatic codec selection
//! let vec: UVarVec<u32> = VarVec::builder()
//!     .build(&data)?;
//! # Ok(())
//! # }
//! ```
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

pub use self::{codec::Codec, traits::Storable};
use crate::fixed::{Error as FixedVecError, FixedVec};
use dsi_bitstream::{
    codes::params::DefaultReadParams,
    dispatch::StaticCodeRead,
    prelude::{
        BitRead, BitSeek, BufBitReader, BufBitWriter, Codes, CodesRead, CodesWrite, Endianness,
        MemWordReader, MemWordWriterVec,
    },
    traits::{BE, BitWrite, LE},
};
use mem_dbg::{DbgFlags, FlatType, MemDbgImpl, MemSize, SizeFlags};
use std::{
    error::Error,
    fmt::{self, Write},
    marker::PhantomData,
};

pub use builder::{VarVecBuilder, VarVecFromIterBuilder};
use iter::{VarVecIntoIter, VarVecIter};
pub use reader::VarVecReader;
pub use seq_reader::VarVecSeqReader;
pub use slice::{VarVecSlice, VarVecSliceIter};

/// Defines the set of errors that can occur in `VarVec` operations.
#[derive(Debug)]
pub enum VarVecError {
    /// An error occurred during an I/O operation, typically from the underlying
    /// bitstream reader or writer.
    Io(std::io::Error),
    /// A generic error from the [`dsi-bitstream`](https://crates.io/crates/dsi-bitstream) library, often related to decoding malformed data.
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

impl fmt::Display for VarVecError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VarVecError::Io(e) => write!(f, "I/O error: {}", e),
            VarVecError::Bitstream(e) => write!(f, "Bitstream error: {}", e),
            VarVecError::InvalidParameters(s) => write!(f, "Invalid parameters: {}", s),
            VarVecError::CodecDispatch(s) => write!(f, "Codec dispatch error: {}", s),
            VarVecError::IndexOutOfBounds(index) => write!(f, "Index out of bounds: {}", index),
        }
    }
}

impl Error for VarVecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            VarVecError::Io(e) => Some(e),
            VarVecError::Bitstream(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for VarVecError {
    fn from(e: std::io::Error) -> Self {
        VarVecError::Io(e)
    }
}

impl From<core::convert::Infallible> for VarVecError {
    fn from(_: core::convert::Infallible) -> Self {
        unreachable!()
    }
}

impl From<FixedVecError> for VarVecError {
    fn from(e: FixedVecError) -> Self {
        VarVecError::InvalidParameters(e.to_string())
    }
}

/// A compressed, randomly accessible vector of integers using variable-length encoding.
///
/// `VarVec` achieves compression by using instantaneous codes and enables fast,
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
#[derive(Debug, Clone)]
pub struct VarVec<T: Storable, E: Endianness, B: AsRef<[u64]> = Vec<u64>> {
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

/// Type alias for the bit writer used internally by `VarVec` builders.
pub(crate) type VarVecBitWriter<E> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;
/// Type alias for the bit reader used internally by `VarVec` accessors.
pub(crate) type VarVecBitReader<'a, E> =
    BufBitReader<E, MemWordReader<u64, &'a [u64], true>, DefaultReadParams>;

impl<T: Storable + 'static, E: Endianness> VarVec<T, E, Vec<u64>> {
    /// Creates a builder for constructing an owned [`VarVec`] from a slice of data.
    ///
    /// This is the most flexible way to create an [`VarVec`], allowing customization
    /// of the compression codec and sampling rate.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::variable::{VarVec, UVarVec, Codec};
    ///
    /// let data: &[u32] = &[5, 8, 13, 21, 34];
    /// let vec: UVarVec<u32> = VarVec::builder()
    ///     .k(2) // Sample every 2nd element
    ///     .codec(Codec::Delta)
    ///     .build(data)?;
    ///
    /// assert_eq!(vec.get(3), Some(21));
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> VarVecBuilder<T, E> {
        VarVecBuilder::new()
    }

    /// Creates a builder for constructing an owned [`VarVec`] from an iterator.
    ///
    /// This is useful for large datasets that are generated on the fly.
    pub fn from_iter_builder<I>(iter: I) -> VarVecFromIterBuilder<T, E, I>
    where
        I: IntoIterator<Item = T> + Clone,
    {
        VarVecFromIterBuilder::new(iter)
    }

    /// Consumes the [`VarVec`] and returns its decoded values as a standard `Vec<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::variable::{VarVec, SVarVec};
    ///
    /// let data: &[i32] = &[-10, 0, 10];
    /// let vec: SVarVec<i32> = VarVec::from_slice(data)?;
    /// let decoded_data = vec.into_vec();
    ///
    /// assert_eq!(decoded_data, &[-10, 0, 10]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_vec(self) -> Vec<T>
    where
        for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.into_iter().collect()
    }

    /// Creates an owned [`VarVec`] from a slice of data using default settings.
    ///
    /// This method uses [`Codec::Auto`] to select a codec and a
    /// default sampling rate of `k=16`.
    pub fn from_slice(slice: &[T]) -> Result<Self, VarVecError>
    where
        for<'a> crate::variable::VarVecBitWriter<E>:
            BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        Self::builder().k(16).codec(Codec::Auto).build(slice)
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> VarVec<T, E, B> {
    /// Creates a new [`VarVec`] from its raw components, enabling zero-copy views.
    ///
    /// This constructor is intended for advanced use cases, such as memory-mapping
    /// a pre-built [`VarVec`] from disk without copying the data.
    ///
    /// # Errors
    ///
    /// Returns an [`VarVecError::InvalidParameters`] if `k` is zero or if the
    /// number of samples is inconsistent with `len` and `k`.
    pub fn from_parts(
        data: B,
        samples_data: B,
        samples_len: usize,
        samples_num_bits: usize,
        k: usize,
        len: usize,
        encoding: Codes,
    ) -> Result<Self, VarVecError> {
        let samples =
            FixedVec::<u64, u64, LE, B>::from_parts(samples_data, samples_len, samples_num_bits)?;

        if k == 0 {
            return Err(VarVecError::InvalidParameters(
                "Sampling rate k cannot be zero".to_string(),
            ));
        }
        let expected_samples = if len == 0 { 0 } else { len.div_ceil(k) };
        if samples.len() != expected_samples {
            return Err(VarVecError::InvalidParameters(format!(
                "Inconsistent number of samples. Expected {}, found {}",
                expected_samples,
                samples.len()
            )));
        }

        Ok(unsafe { Self::new_unchecked(data, samples, k, len, encoding) })
    }

    /// Creates a new [`VarVec`] from its raw parts without performing safety checks.
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
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::variable::{VarVec, UVarVec};
    ///
    /// let data: Vec<u32> = (0..20).collect();
    /// let vec: UVarVec<u32> = VarVec::from_slice(&data)?;
    /// let slice = vec.slice(5, 10).expect("valid slice range");
    ///
    /// assert_eq!(slice.len(), 10);
    /// assert_eq!(slice.get(0), Some(5)); // Corresponds to index 5 of the original vec
    /// # Ok(())
    /// # }
    /// ```
    pub fn slice(&'_ self, start: usize, len: usize) -> Option<VarVecSlice<'_, T, E, B>> {
        if start.saturating_add(len) > self.len {
            return None;
        }
        Some(VarVecSlice::new(self, start..start + len))
    }

    /// Splits the vector into two immutable slices at a given index.
    ///
    /// Returns `None` if `mid` is out of bounds.
    #[allow(clippy::type_complexity)]
    pub fn split_at(
        &'_ self,
        mid: usize,
    ) -> Option<(VarVecSlice<'_, T, E, B>, VarVecSlice<'_, T, E, B>)> {
        if mid > self.len {
            return None;
        }
        let left = VarVecSlice::new(self, 0..mid);
        let right = VarVecSlice::new(self, mid..self.len);
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
    pub fn sampling_rate(&self) -> usize {
        self.k
    }

    /// Returns the number of sample points stored in the vector.
    #[inline]
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Returns the sampling rate `k` used during encoding.
    #[deprecated(since = "0.6.0", note = "renamed to `sampling_rate`; use `sampling_rate` instead")]
    #[inline]
    pub fn get_sampling_rate(&self) -> usize {
        self.sampling_rate()
    }

    /// Returns the number of sample points stored in the vector.
    #[deprecated(
        since = "0.6.0",
        note = "renamed to `num_samples`; use `num_samples` instead"
    )]
    #[inline]
    pub fn get_num_samples(&self) -> usize {
        self.num_samples()
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
        for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        VarVecIter::new(self)
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> VarVec<T, E, B>
where
    for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a reusable, stateless reader for efficient random access.
    ///
    /// This method returns an [`VarVecReader`], a struct that maintains a persistent,
    /// reusable bitstream reader. This amortizes the setup cost across multiple `get`
    /// operations, making it more efficient than calling [`get`](VarVec::get) repeatedly in a loop.
    ///
    /// This reader is **stateless**: it performs a full seek from the nearest sample point for each call,
    /// independently of any previous access.
    ///
    /// # When to use it
    /// Use [`VarVecReader`] for true random access patterns where lookup indices are sparse,
    /// unordered, or not known in advance (e.g., graph traversals, pointer chasing).
    /// For accessing a known set of indices, [`get_many`](VarVec::get_many) is generally superior.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: Vec<u32> = (0..100).rev().collect(); // Data is not sequential
    /// let vec: UVarVec<u32> = VarVec::from_slice(&data)?;
    ///
    /// // Create a reusable reader for multiple random lookups
    /// let mut reader = vec.reader();
    ///
    /// assert_eq!(reader.get(99)?, Some(0));
    /// assert_eq!(reader.get(0)?, Some(99));
    /// assert_eq!(reader.get(50)?, Some(49));
    /// # Ok(())
    /// # }
    /// ```
    pub fn reader(&'_ self) -> VarVecReader<'_, T, E, B> {
        VarVecReader::new(self)
    }

    /// Creates a stateful, reusable reader optimized for sequential access.
    ///
    /// This method returns an [`VarVecSeqReader`], which is specifically designed
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
    /// Use [`VarVecSeqReader`] when your access pattern has high locality, meaning
    /// indices are primarily increasing and often clustered together. It is ideal
    /// for iterating through a sorted list of indices or for stream-like processing.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: Vec<u32> = (0..100).collect();
    /// let vec: UVarVec<u32> = VarVec::from_slice(&data)?;
    ///
    /// // Create a reader optimized for sequential access
    /// let mut seq_reader = vec.seq_reader();
    ///
    /// // Accessing indices in increasing order is efficient
    /// assert_eq!(seq_reader.get(10)?, Some(10));
    /// // This next call is fast, as it decodes forward from index 10
    /// assert_eq!(seq_reader.get(15)?, Some(15));
    ///
    /// // A large jump will trigger a seek to a new sample block
    /// assert_eq!(seq_reader.get(90)?, Some(90));
    ///
    /// // A backward jump will also trigger a seek
    /// assert_eq!(seq_reader.get(5)?, Some(5));
    /// # Ok(())
    /// # }
    /// ```
    pub fn seq_reader(&'_ self) -> VarVecSeqReader<'_, T, E, B> {
        VarVecSeqReader::new(self)
    }

    /// Returns the element at the specified index, or `None` if the index is
    /// out of bounds.
    ///
    /// This operation is amortized O(1).
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::variable::{VarVec, UVarVec};
    ///
    /// let data: Vec<u32> = (0..100).collect();
    /// let vec: UVarVec<u32> = VarVec::from_slice(&data)?;
    ///
    /// assert_eq!(vec.get(50), Some(50));
    /// assert_eq!(vec.get(100), None);
    /// # Ok(())
    /// # }
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
        unsafe { reader.get_unchecked(index) }
    }

    /// Retrieves multiple elements from the vector at the specified indices.
    ///
    /// This method is generally more efficient than calling [`get`](Self::get) in a loop, as
    /// it sorts the indices and scans through the compressed data stream once.
    ///
    /// # Errors
    ///
    /// Returns [`VarVecError::IndexOutOfBounds`] if any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::variable::{VarVec, UVarVec};
    ///
    /// let data: Vec<u32> = (0..100).collect();
    /// let vec: UVarVec<u32> = VarVec::from_slice(&data)?;
    ///
    /// let indices = [99, 0, 50];
    /// let values = vec.get_many(&indices)?;
    /// assert_eq!(values, vec![99, 0, 50]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<T>, VarVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        for &index in indices {
            if index >= self.len {
                return Err(VarVecError::IndexOutOfBounds(index));
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
        unsafe { results.set_len(indices.len()) };

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
    ) -> Result<(), VarVecError>
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
    pub fn get_many_from_iter<I>(&self, indices: I) -> Result<Vec<T>, VarVecError>
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
                .ok_or(VarVecError::IndexOutOfBounds(index))?;
            results.push(value);
        }

        Ok(results)
    }
}

impl<T: Storable + Ord, E: Endianness, B: AsRef<[u64]>> VarVec<T, E, B>
where
    for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
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
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::variable::{VarVec, SVarVec};
    ///
    /// let data: &[i32] = &[-10, 0, 10, 20, 30];
    /// let vec: SVarVec<i32> = VarVec::from_slice(data)?;
    ///
    /// assert_eq!(vec.binary_search(&10), Ok(2));
    /// assert_eq!(vec.binary_search(&15), Err(3));
    /// # Ok(())
    /// # }
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

impl<T: Storable, E: Endianness, B: AsRef<[u64]> + MemSize + FlatType> MemSize for VarVec<T, E, B> {
    fn mem_size_rec(&self, flags: SizeFlags, _refs: &mut mem_dbg::HashMap<usize, usize>) -> usize {
        // Start with the stack size of the struct itself.
        let mut total_size = core::mem::size_of::<Self>();
        // Add the heap-allocated memory for the `data` field.
        total_size += self.data.mem_size(flags) - core::mem::size_of::<B>();
        // Add the heap-allocated memory for the `samples` field's internal buffer.
        total_size +=
            self.samples.mem_size(flags) - core::mem::size_of::<FixedVec<u64, u64, LE, B>>();
        total_size
    }
}

// A local wrapper for `dsi_bitstream::codes::Codes` to override its `MemDbgImpl`.
// This is necessary because the derived implementation for `Codes` is incorrect
// and cannot be fixed due to the orphan rule.
struct CodeWrapper<'a>(&'a Codes);

impl mem_dbg::MemSize for CodeWrapper<'_> {
    fn mem_size_rec(&self, _flags: mem_dbg::SizeFlags, _refs: &mut mem_dbg::HashMap<usize, usize>) -> usize {
        core::mem::size_of_val(self.0)
    }
}

impl mem_dbg::MemDbgImpl for CodeWrapper<'_> {
    // Override the top-level display function for this type.
    fn _mem_dbg_depth_on(
        &self,
        writer: &mut impl core::fmt::Write,
        total_size: usize,
        max_depth: usize,
        prefix: &mut String,
        field_name: Option<&str>,
        is_last: bool,
        padded_size: usize,
        flags: DbgFlags,
        _dbg_refs: &mut mem_dbg::HashSet<usize>,
    ) -> core::fmt::Result {
        if prefix.len() > max_depth {
            return Ok(());
        }

        let real_size = self.mem_size(flags.to_size_flags());
        let mut buffer = String::new(); // Use a temp buffer to format the size part.

        // Replicate the size and percentage formatting from the default `MemDbgImpl`.
        if flags.contains(DbgFlags::HUMANIZE) {
            let (value, uom) = mem_dbg::humanize_float(real_size);
            if uom == " B" {
                let _ = write!(&mut buffer, "{:>5}  B ", real_size);
            } else {
                let precision = if value.abs() >= 100.0 {
                    1
                } else if value.abs() >= 10.0 {
                    2
                } else {
                    3
                };
                let _ = write!(&mut buffer, "{0:>4.1$} {2} ", value, precision, uom);
            }
        } else {
            let align = mem_dbg::n_of_digits(total_size);
            let _ = write!(&mut buffer, "{:>align$} B ", real_size, align = align);
        }

        if flags.contains(DbgFlags::PERCENTAGE) {
            let percentage = if total_size == 0 {
                100.0
            } else {
                100.0 * real_size as f64 / total_size as f64
            };
            let _ = write!(&mut buffer, "{:>6.2}% ", percentage);
        }

        // Write the formatted size string with colors if enabled.
        if flags.contains(DbgFlags::COLOR) {
            writer.write_fmt(format_args!("{}", mem_dbg::color(real_size)))?;
        }
        writer.write_str(&buffer)?;
        if flags.contains(DbgFlags::COLOR) {
            writer.write_fmt(format_args!("{}", mem_dbg::reset_color()))?;
        }

        // Write the tree structure part.
        if !prefix.is_empty() {
            writer.write_str(&prefix[2..])?;
            writer.write_char(if is_last { '╰' } else { '├' })?;
            writer.write_char('╴')?;
        }

        if let Some(field_name) = field_name {
            writer.write_fmt(format_args!("{}", field_name))?;
        }

        // This is the custom part: print the `Debug` format of the enum.
        if flags.contains(DbgFlags::TYPE_NAME) {
            if flags.contains(DbgFlags::COLOR) {
                writer.write_fmt(format_args!("{}", mem_dbg::type_color()))?;
            }
            writer.write_fmt(format_args!(": {:?}", self.0))?;
            if flags.contains(DbgFlags::COLOR) {
                writer.write_fmt(format_args!("{}", mem_dbg::reset_color()))?;
            }
        }

        // Correctly calculate and print padding.
        let padding = padded_size - core::mem::size_of_val(self.0);
        if padding != 0 {
            writer.write_fmt(format_args!(" [{}B]", padding))?;
        }

        writer.write_char('\n')?;
        Ok(())
    }

    // It's a leaf node in the display tree, so no recursion is needed.
    fn _mem_dbg_rec_on(
        &self,
        _writer: &mut impl core::fmt::Write,
        _total_size: usize,
        _max_depth: usize,
        _prefix: &mut String,
        _is_last: bool,
        _flags: DbgFlags,
        _dbg_refs: &mut mem_dbg::HashSet<usize>,
    ) -> core::fmt::Result {
        Ok(())
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]> + MemDbgImpl + FlatType> MemDbgImpl for VarVec<T, E, B> {
    fn _mem_dbg_rec_on(
        &self,
        writer: &mut impl core::fmt::Write,
        total_size: usize,
        max_depth: usize,
        prefix: &mut String,
        _is_last: bool,
        flags: DbgFlags,
        _dbg_refs: &mut mem_dbg::HashSet<usize>,
    ) -> core::fmt::Result {
        // Manually display each field, ensuring correct tree structure.
        self.data._mem_dbg_depth_on(
            writer,
            total_size,
            max_depth,
            prefix,
            Some("data"),
            false,
            core::mem::size_of_val(&self.data),
            flags,
            _dbg_refs,
        )?;
        self.samples._mem_dbg_depth_on(
            writer,
            total_size,
            max_depth,
            prefix,
            Some("samples"),
            false,
            core::mem::size_of_val(&self.samples),
            flags,
            _dbg_refs,
        )?;
        self.k._mem_dbg_depth_on(
            writer,
            total_size,
            max_depth,
            prefix,
            Some("k"),
            false,
            core::mem::size_of_val(&self.k),
            flags,
            _dbg_refs,
        )?;
        self.len._mem_dbg_depth_on(
            writer,
            total_size,
            max_depth,
            prefix,
            Some("len"),
            false,
            core::mem::size_of_val(&self.len),
            flags,
            _dbg_refs,
        )?;

        // Use the custom wrapper to correctly display the `encoding` field.
        let code_wrapper = CodeWrapper(&self.encoding);
        code_wrapper._mem_dbg_depth_on(
            writer,
            total_size,
            max_depth,
            prefix,
            Some("encoding"),
            false, // Not the last field.
            core::mem::size_of_val(&self.encoding),
            flags,
            _dbg_refs,
        )?;

        self._markers._mem_dbg_depth_on(
            writer,
            total_size,
            max_depth,
            prefix,
            Some("_markers"),
            true, // This is the last field.
            core::mem::size_of_val(&self._markers),
            flags,
            _dbg_refs,
        )?;
        Ok(())
    }
}

impl<T: Storable + 'static, E: Endianness + 'static> IntoIterator for VarVec<T, E, Vec<u64>>
where
    for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;
    type IntoIter = VarVecIntoIter<T, E>;

    fn into_iter(self) -> Self::IntoIter {
        VarVecIntoIter::new(self)
    }
}

/// An [`VarVec`] for unsigned integers with Little-Endian bit layout.
pub type UVarVec<T> = VarVec<T, LE>;
/// An [`VarVec`] for signed integers with Little-Endian bit layout.
pub type SVarVec<T> = VarVec<T, LE>;
/// An [`VarVec`] for [`u64`] elements with Big-Endian bit layout.
pub type BEVarVec = VarVec<u64, BE>;
/// An [`VarVec`] for [`u64`] elements with Little-Endian bit layout.
pub type LEVarVec = VarVec<u64, LE>;
/// An [`VarVec`] for [`i64`] elements with Big-Endian bit layout.
pub type BESVarVec = VarVec<i64, BE>;
/// An [`VarVec`] for [`i64`] elements with Little-Endian bit layout.
pub type LESVarVec = VarVec<i64, LE>;

// ============================================================================
// Deprecated Type Aliases for Backward Compatibility
// ============================================================================
// As of version 0.3.0, all `IntVec*` types have been renamed to `VarVec*`
// for consistency with the module naming convention. These deprecated aliases
// maintain backward compatibility and will be removed in a future major version.

/// Deprecated alias for [`VarVec`]. Use [`VarVec`] instead.
///
/// # Deprecation Notice
///
/// This type has been renamed to [`VarVec`] for consistency with the module
/// naming convention (`variable` → `VarVec`). This alias will be removed in
/// a future major release.
#[deprecated(since = "0.6.0", note = "renamed to `VarVec`; use `VarVec` instead")]
pub type IntVec<T, E, B = Vec<u64>> = VarVec<T, E, B>;

/// Deprecated alias for [`VarVecBuilder`]. Use [`VarVecBuilder`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `VarVecBuilder`; use `VarVecBuilder` instead"
)]
pub type IntVecBuilder<T, E> = VarVecBuilder<T, E>;

/// Deprecated alias for [`VarVecFromIterBuilder`]. Use [`VarVecFromIterBuilder`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `VarVecFromIterBuilder`; use `VarVecFromIterBuilder` instead"
)]
pub type IntVecFromIterBuilder<T, E, I> = VarVecFromIterBuilder<T, E, I>;

/// Deprecated alias for [`VarVecReader`]. Use [`VarVecReader`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `VarVecReader`; use `VarVecReader` instead"
)]
pub type IntVecReader<'a, T, E, B> = VarVecReader<'a, T, E, B>;

/// Deprecated alias for [`VarVecSeqReader`]. Use [`VarVecSeqReader`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `VarVecSeqReader`; use `VarVecSeqReader` instead"
)]
pub type IntVecSeqReader<'a, T, E, B> = VarVecSeqReader<'a, T, E, B>;

/// Deprecated alias for [`VarVecSlice`]. Use [`VarVecSlice`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `VarVecSlice`; use `VarVecSlice` instead"
)]
pub type IntVecSlice<'a, T, E, B> = VarVecSlice<'a, T, E, B>;

/// Deprecated alias for [`VarVecIter`]. Use [`VarVecIter`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `VarVecIter`; use `VarVecIter` instead"
)]
pub type IntVecIter<'a, T, E, B> = VarVecIter<'a, T, E, B>;

/// Deprecated alias for [`VarVecIntoIter`]. Use [`VarVecIntoIter`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `VarVecIntoIter`; use `VarVecIntoIter` instead"
)]
pub type IntVecIntoIter<T, E> = VarVecIntoIter<T, E>;

/// Deprecated alias for [`VarVecError`]. Use [`VarVecError`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `VarVecError`; use `VarVecError` instead"
)]
pub type IntVecError = VarVecError;

/// Deprecated alias for [`VarVecSliceIter`]. Use [`VarVecSliceIter`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `VarVecSliceIter`; use `VarVecSliceIter` instead"
)]
pub type IntVecSliceIter<'a, T, E, B> = VarVecSliceIter<'a, T, E, B>;

// Deprecated convenience aliases
/// Deprecated alias for [`UVarVec`]. Use [`UVarVec`] instead.
#[deprecated(since = "0.6.0", note = "renamed to `UVarVec`; use `UVarVec` instead")]
pub type UIntVec<T> = UVarVec<T>;

/// Deprecated alias for [`SVarVec`]. Use [`SVarVec`] instead.
#[deprecated(since = "0.6.0", note = "renamed to `SVarVec`; use `SVarVec` instead")]
pub type SIntVec<T> = SVarVec<T>;

/// Deprecated alias for [`BEVarVec`]. Use [`BEVarVec`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `BEVarVec`; use `BEVarVec` instead"
)]
pub type BEIntVec = BEVarVec;

/// Deprecated alias for [`LEVarVec`]. Use [`LEVarVec`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `LEVarVec`; use `LEVarVec` instead"
)]
pub type LEIntVec = LEVarVec;

/// Deprecated alias for [`BESVarVec`]. Use [`BESVarVec`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `BESVarVec`; use `BESVarVec` instead"
)]
pub type BESIntVec = BESVarVec;

/// Deprecated alias for [`LESVarVec`]. Use [`LESVarVec`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `LESVarVec`; use `LESVarVec` instead"
)]
pub type LESIntVec = LESVarVec;

/// Deprecated alias for [`Codec`]. Use [`Codec`] instead.
#[deprecated(since = "0.6.0", note = "renamed to `Codec`; use `Codec` instead")]
#[allow(deprecated)]
pub use self::codec::VariableCodecSpec;

// Deprecated internal type aliases
/// Deprecated alias for [`VarVecBitReader`]. Use [`VarVecBitReader`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `VarVecBitReader`; use `VarVecBitReader` instead"
)]
#[allow(dead_code)]
pub(crate) type IntVecBitReader<'a, E> = VarVecBitReader<'a, E>;

/// Deprecated alias for [`VarVecBitWriter`]. Use [`VarVecBitWriter`] instead.
#[deprecated(
    since = "0.6.0",
    note = "renamed to `VarVecBitWriter`; use `VarVecBitWriter` instead"
)]
#[allow(dead_code)]
pub(crate) type IntVecBitWriter<E> = VarVecBitWriter<E>;

impl<T, E, B, O> PartialEq<O> for VarVec<T, E, B>
where
    T: Storable + PartialEq,
    E: Endianness,
    B: AsRef<[u64]>,
    O: AsRef<[T]>,
    for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Checks for equality between an [`VarVec`] and a standard slice.
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
