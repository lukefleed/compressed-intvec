//! # A compressed, randomly accessible vector of `u64` integers.
//!
//! This module provides the core implementation of [`IntVec`], a data structure
//! designed for space-efficient storage and fast random access of `u64` integer
//! sequences. It achieves compression by leveraging a variety of instantaneous
//! codes from the [`dsi-bitstream`] crate, which encode integers into a
//! variable-length bitstream.
//!
//! ## Core Functionality
//!
//! - **Compression**: Employs codecs like Gamma (γ), Delta (δ), and Zeta (ζ) for
//!   skewed data, and a highly efficient [`FixedLength`] encoding for uniform data with a small range
//! - **Random Access**: For variable-length codes, it uses a sampling mechanism
//!   to provide fast random access. The sampling rate, `k`, determines the
//!   trade-off between access speed and memory overhead. For [`FixedLength`]
//!   encoding, access is a true O(1) operation.
//! - **Flexible Construction**: Provides a builder API that can construct an
//!   [`IntVec`] from a slice (with automatic codec selection) or an iterator (for
//!   large datasets, requiring manual parameter specification).
//! - **High-Performance Lookups**: Offers optimized methods for various access
//!   patterns, including a reusable [`IntVecReader`] for dynamic lookups, and
//!   efficient batch methods like [`get_many`] and [`par_get_many`].
//!
//! The main struct, [`IntVec`], is generic over [`Endianness`], allowing
//! to choose between Little-Endian ([`LEIntVec`]) and Big-Endian ([`BEIntVec`])
//! representations to optimize for specific hardware architectures.
//!
//! ## Example
//!
//! ```rust
//! use compressed_intvec::prelude::*;
//!
//! // A small vector of integers to be compressed.
//! let data: &[u64] = &[40, 200, 0, 50, 13, 90, 1023];
//!
//! // Use the builder to create an IntVec.
//! // `CodecSpec::Auto` will analyze the data and select the best codec.
//! let intvec = LEIntVec::builder(data)
//!     .k(2) // Use a small sampling rate for this vector.
//!     .codec(CodecSpec::Auto)
//!     .build()
//!.unwrap();
//!
//! // Verify the length and access some elements.
//! assert_eq!(intvec.len(), data.len());
//! assert_eq!(intvec.get(1), Some(200));
//! assert_eq!(intvec.get(6), Some(1023));
//! ```
//!
//! Or alternatively, we can use a fixed-length encoding:
//!
//! ```rust
//! use compressed_intvec::prelude::*;
//!
//! // A small vector of integers to be compressed.
//! let data: &[u64] = &[40, 200, 0, 50, 13, 90, 1023];
//!
//! // Use the builder to create an IntVec with fixed-length encoding.
//! // Using `None` for `num_bits` will automatically select the best bit width (in this case, 10 bits).
//! let intvec = LEIntVec::builder(data)
//!    .codec(CodecSpec::FixedLength { num_bits: None })
//!    .build()
//!    .unwrap();
//!
//! // Verify the length and access some elements.
//! assert_eq!(intvec.len(), data.len());
//! assert_eq!(intvec.get(1), Some(200));
//! assert_eq!(intvec.get(6), Some(1023));
//! ```
//!
//! [`dsi-bitstream`]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/
//! [`Endianness`]: dsi_bitstream::prelude::Endianness
//! [`get_many`]: IntVec::get_many
//! [`par_get_many`]: IntVec::par_get_many
//! [`FixedLength`]: crate::codec_spec::CodecSpec::FixedLength
use super::codec_spec::{resolve_codec, CodecSpec, Encoding};
use dsi_bitstream::{
    codes::params::DefaultReadParams,
    dispatch::{CodesRead, FuncCodeReader, StaticCodeRead},
    impls::{MemWordReader, MemWordWriterVec},
    prelude::*,
};
use mem_dbg::{DbgFlags, MemDbg, MemDbgImpl, MemSize, SizeFlags};
use rayon::slice::ParallelSliceMut;
use std::{error::Error, fmt, marker::PhantomData};

// Declare and export submodules.
mod builder;
mod iter;
#[cfg(feature = "parallel")]
mod parallel;
mod reader;
#[cfg(feature = "serde")]
mod serde;

pub use builder::{IntVecBuilder, IntVecFromIterBuilder};
pub use iter::IntVecIter;
pub use reader::IntVecReader;

/// Defines the set of errors that can occur in `IntVec` operations.
#[derive(Debug)]
pub enum IntVecError {
    /// An error occurred during an I/O operation, typically forwarded from the
    /// underlying bitstream library.
    Io(std::io::Error),
    /// A generic error originating from the `dsi-bitstream` library.
    Bitstream(Box<dyn Error + Send + Sync>),
    /// An error indicating that the provided parameters are invalid for the
    /// requested operation, such as a sampling rate of `0` or an impossible
    /// codec configuration.
    InvalidParameters(String),
    /// An error during the dispatch of a compression or decompression function,
    /// usually because a codec is not supported by the dispatcher.
    CodecDispatch(String),
    /// An error indicating that a requested index is outside the valid bounds
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

/// An enum to hold sample offsets, dynamically choosing `u32` or `u64`
/// to save space on smaller vectors. This is only used for bit-level encodings.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub(crate) enum Samples {
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl Samples {
    /// Get the offset at a given index.
    pub(crate) fn get(&self, index: usize) -> Option<u64> {
        match self {
            Samples::U32(v) => v.get(index).map(|&x| x as u64),
            Samples::U64(v) => v.get(index).copied(),
        }
    }
    /// Get the number of samples.
    pub(crate) fn len(&self) -> usize {
        match self {
            Samples::U32(v) => v.len(),
            Samples::U64(v) => v.len(),
        }
    }
}

// Manual impl to handle the enum variants correctly for MemDbg.
impl MemDbgImpl for Samples {
    fn _mem_dbg_rec_on(
        &self,
        writer: &mut impl fmt::Write,
        total_size: usize,
        max_depth: usize,
        prefix: &mut String,
        is_last: bool,
        flags: DbgFlags,
    ) -> fmt::Result {
        match self {
            Samples::U32(v) => {
                v._mem_dbg_rec_on(writer, total_size, max_depth, prefix, is_last, flags)
            }
            Samples::U64(v) => {
                v._mem_dbg_rec_on(writer, total_size, max_depth, prefix, is_last, flags)
            }
        }
    }
}

impl MemSize for Samples {
    fn mem_size(&self, flags: SizeFlags) -> usize {
        match self {
            Samples::U32(v) => v.mem_size(flags),
            Samples::U64(v) => v.mem_size(flags),
        }
    }
}

/// A compressed, randomly accessible vector of `u64` integers.
///
/// [`IntVec`] uses instantaneous codes from the [`dsi-bitstream`] crate to compress a
/// vector of `u64` integers. To provide efficient random access, it stores sample
/// points of the underlying bitstream at regular intervals, defined by the sampling
/// rate `k`. This creates a trade-off: a smaller `k` results in faster random
/// access but higher memory overhead, while a larger `k` reduces memory usage at
/// the cost of slower access. For [`FixedLength`] encoding, no samples are needed
/// as access is already O(1).
///
/// The most convenient way to create an [`IntVec`] is through its [builder](IntVec::builder),
/// which allows for easy configuration of the sampling rate and compression codec,
/// including automatic parameter selection.
///
/// The generic parameter `E` specifies the [`Endianness`] of the underlying bitstream.
/// For convenience, the type aliases [`LEIntVec`] and [`BEIntVec`] are provided for
/// little-endian and big-endian configurations, respectively.
///
/// # Example
///
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// // A small vector of integers to be compressed.
/// let data: &[u64] = &[40, 200, 0, 50, 13];
///
/// // Use the builder to create a Little-Endian IntVec.
/// // We let the builder automatically select the best codec for the data.
/// let intvec = LEIntVec::builder(data)
///     .k(2) // Use a small sampling rate for this tiny vector.
///     .codec(CodecSpec::Auto)
///     .build()
///     .unwrap();
///
/// // Verify the length and access some elements.
/// assert_eq!(intvec.len(), data.len());
/// assert_eq!(intvec.get(1), Some(200));
/// assert_eq!(intvec.get(2), Some(0));
/// ```
///
/// [`dsi-bitstream`]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/
/// [`FixedLength`]: crate::codec_spec::CodecSpec::FixedLength
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct IntVec<E: Endianness> {
    /// The raw compressed data, stored as a `Vec<u64>`.
    pub(super) data: Vec<u64>,
    /// Bit offsets of sampled elements. This is `Some` for bit-level encodings
    /// and `None` for [`FixedLength`] encoding.
    pub(super) samples: Option<Samples>,
    /// The sampling rate `k`, which determines the interval between samples.
    /// This is `Some` for bit-level encodings and `None` for [`FixedLength`].
    pub(super) k: Option<usize>,
    /// The number of elements in the vector.
    pub(super) len: usize,
    /// The concrete encoding strategy used for compression.
    pub(super) encoding: Encoding,
    /// A zero-sized marker for the endianness type parameter.
    pub(super) endian: PhantomData<E>,
}

/// Type alias for the writer used internally by IntVec.
pub(crate) type IntVecBitWriter<E> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;
/// Type alias for the reader used internally by IntVec.
pub(crate) type IntVecBitReader<'a, E> =
    BufBitReader<E, MemWordReader<u64, &'a Vec<u64>>, DefaultReadParams>;

impl<E> IntVec<E>
where
    E: Endianness,
{
    /// Returns a builder for creating an [`IntVec`] from a slice (`&[u64]`).
    ///
    /// This is the most common and flexible way to create an [`IntVec`].
    /// The builder can automatically select the best codec parameters by analyzing
    /// the data if [`CodecSpec::Auto`] is used.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: &[u64] = &[40, 12, 30, 50, 13];
    ///
    /// // Build with a specific codec and sampling rate.
    /// let intvec = LEIntVec::builder(data)
    ///     .codec(CodecSpec::Gamma)
    ///     .k(16)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(intvec.len(), data.len());
    /// assert_eq!(intvec.get(2), Some(30));
    /// ```
    pub fn builder(input: &[u64]) -> IntVecBuilder<E> {
        IntVecBuilder::new(input)
    }

    /// Returns a builder for creating an [`IntVec`] from an iterator.
    ///
    /// This builder is designed for scenarios where the data is generated on-the-fly
    /// or is too large to fit into memory as a `Vec<u64>`.
    ///
    /// # Limitations
    /// This builder **requires** that codec parameters be specified manually.
    /// Automatic parameter selection ([`CodecSpec::Auto`] or `None` variants)
    /// is not supported because the builder cannot pre-analyze the data.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// // Build from a range iterator with a fixed-bit-width codec.
    /// let intvec = LEIntVec::from_iter_builder(0..1000_u64)
    ///     .codec(CodecSpec::FixedLength { num_bits: Some(10) }) // 1000 requires 10 bits
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(intvec.len(), 1000);
    /// assert_eq!(intvec.get(999), Some(999));
    /// ```
    pub fn from_iter_builder<I: IntoIterator<Item = u64>>(iter: I) -> IntVecFromIterBuilder<E, I> {
        IntVecFromIterBuilder::new(iter)
    }
}

impl<E> IntVec<E>
where
    E: Endianness,
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a stateful, reusable [`IntVecReader`] for this vector.
    ///
    /// A reader maintains its own internal state, including a bitstream reader instance.
    /// This makes it suitable for scenarios where multiple, non-sequential [`get`]
    /// operations are needed, and the indices are determined on-the-fly (e.g., in a
    /// loop where the next lookup depends on the result of the previous one).
    /// Using a shared reader avoids the overhead of creating a new one for each [`get`] call.
    ///
    /// # Performance
    ///
    /// If you have a predefined slice of indices to access, it is recommended
    /// to use [`get_many`](Self::get_many) or [`par_get_many`](Self::par_get_many) instead. Note that [`par_get_many`](Self::par_get_many) may be slower than [`get_many`](Self::get_many) for small vectors due to the overhead of parallelization.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    /// let data: &[u64] = &[10, 20, 30, 40];
    /// let intvec = LEIntVec::builder(data).build().unwrap();
    ///
    /// // Create a reusable reader.
    /// let mut reader = intvec.reader();
    ///
    /// // Perform multiple lookups using the same reader.
    /// assert_eq!(reader.get(1).unwrap(), Some(20));
    /// assert_eq!(reader.get(3).unwrap(), Some(40));
    /// assert_eq!(reader.get(0).unwrap(), Some(10));
    /// ```
    ///
    /// [`get`]: Self::get
    pub fn reader(&self) -> IntVecReader<E> {
        IntVecReader::new(self)
    }

    /// Retrieves the element at the specified index.
    ///
    /// This method provides random access to the compressed data. The exact mechanism
    /// and performance depend on the compression scheme used to create the [`IntVec`].
    ///
    /// # Implementation Notes
    ///
    /// The access strategy is determined by the type of encoding employed:
    ///
    /// - **For fixed-width integer encoding**: Access is an O(1) operation.
    ///   Each integer occupies an identical, predetermined number of bits. The bit
    ///   position of any element can be calculated directly (`index * bit_width`),
    ///   and the value is read from that precise offset in the bitstream. This is
    ///   the most performant access method.
    ///
    /// - **For variable-length, bit-level codes** (e.g., Gamma, Delta, Rice):
    ///   Since each compressed integer can have a different bit length, the position
    ///   of an element cannot be calculated arithmetically. To overcome this, this
    ///   structure uses a sampling strategy. The access process is as follows:
    ///   1. The algorithm identifies which pre-defined block of `k` elements contains
    ///      the target `index`.
    ///   2. It retrieves the starting bit offset of that block from an auxiliary
    ///      "samples" data structure.
    ///   3. The underlying bitstream reader seeks to this stored offset.
    ///   4. From there, it decodes elements sequentially until it reaches the
    ///      target `index` within the block.
    ///
    /// This makes the access time dependent on the sampling rate `k`, as up to `k-1`
    /// elements may need to be decoded for each lookup.
    ///
    /// # Performance
    ///
    /// This method is convenient for single, isolated lookups. However, **it is
    /// inefficient to call `get` repeatedly in a loop**, as each call
    /// creates and discards a new internal bitstream reader, incurring significant overhead.
    ///
    /// For scenarios involving multiple lookups, consider these alternatives:
    ///
    /// - **Batch Access**: If you have a predefined collection of indices, use
    ///   [`get_many`](Self::get_many) or [`par_get_many`](Self::par_get_many).
    ///   These methods are optimized for batch operations, often by sorting the
    ///   indices to perform a single, efficient forward scan over the data, which
    ///   minimizes seeks and redundant decoding.
    ///
    /// - **Dynamic Access**: If lookup indices are generated on-the-fly (e.g.,
    ///   the next lookup depends on the result of the previous one), create a single,
    ///   reusable [`IntVecReader`] via the [`reader()`](Self::reader) method. Calling [`get`]
    ///   on this reader instance amortizes its setup cost across all lookups.
    ///
    /// # Returns
    /// - `Some(u64)` if the `index` is within bounds.
    /// - `None` if the `index` is out of bounds.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    /// let data: &[u64] = &[10, 20, 30, 40];
    /// let intvec = LEIntVec::builder(data).build().unwrap();
    ///
    /// // Retrieve a single element.
    /// assert_eq!(intvec.get(2), Some(30));
    ///
    /// // An out-of-bounds index returns None.
    /// assert_eq!(intvec.get(99), None);
    /// ```
    ///
    /// [`get`]: Self::get
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }
        let mut reader = self.reader();
        reader.get(index).ok().flatten()
    }

    /// Retrieves multiple elements at the specified indices in a highly efficient way.
    ///
    /// This method is optimized for batched random access. For bit-level encodings,
    /// it sorts the requested indices and decodes the elements in a single forward
    /// pass, minimizing seeks and redundant decoding. For [`FixedLength`] encoding,
    /// it reads each element directly.
    ///
    /// # Arguments
    /// * `indices`: A slice of indices to retrieve.
    ///
    /// # Returns
    /// A `Result` containing a `Vec<u64>` with the retrieved values in the same
    /// order as the input `indices`, or an [`IntVecError`] if any index is out of bounds.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    /// let intvec = LEIntVec::from_iter_builder(0..1000u64)
    ///     .codec(CodecSpec::FixedLength{ num_bits: Some(10) }) // 1000 fits in 10 bits
    ///     .build()
    ///     .unwrap();
    ///
    /// let access_indices = [0, 999, 500, 250];
    ///
    /// let values = intvec.get_many(&access_indices).unwrap();
    /// assert_eq!(values, vec![0, 999, 500, 250]);
    /// ```
    /// [`FixedLength`]: crate::codec_spec::CodecSpec::FixedLength
    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<u64>, IntVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        for &index in indices {
            if index >= self.len {
                return Err(IntVecError::IndexOutOfBounds(index));
            }
        }

        let mut results = vec![0; indices.len()];

        match self.encoding {
            Encoding::Dsi(code) => {
                // With bit-level  encoding, k and samples are guaranteed to be Some.
                let k = self.k.unwrap();
                let samples = self.samples.as_ref().unwrap();
                let mut indexed_indices: Vec<(usize, usize)> = indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| (idx, i))
                    .collect();
                indexed_indices.par_sort_unstable_by_key(|&(idx, _)| idx);
                let mut reader = self.reader();
                let code_reader = FuncCodeReader::new(code)
                    .map_err(|e| IntVecError::CodecDispatch(e.to_string()))?;
                let mut current_decoded_index = 0;

                for &(target_index, original_position) in &indexed_indices {
                    let sample_index = target_index / k;
                    let start_bit = samples.get(sample_index).unwrap();
                    let start_sample_index = sample_index * k;

                    if current_decoded_index > target_index
                        || sample_index * k > current_decoded_index
                    {
                        reader.reader.set_bit_pos(start_bit)?;
                        current_decoded_index = start_sample_index;
                    }

                    for _ in current_decoded_index..target_index {
                        code_reader.read(&mut reader.reader)?;
                    }

                    let value = code_reader.read(&mut reader.reader)?;
                    results[original_position] = value;
                    current_decoded_index = target_index + 1;
                }
            }
            Encoding::Fixed { num_bits } => {
                let mut reader = self.reader();
                for (i, &target_index) in indices.iter().enumerate() {
                    let bit_pos = target_index as u64 * num_bits as u64;
                    reader.reader.set_bit_pos(bit_pos)?;
                    results[i] = reader.reader.read_bits(num_bits)?;
                }
            }
        }

        Ok(results)
    }

    /// Consumes the [`IntVec`] and returns a `Vec<u64>` containing all decompressed values.
    ///
    /// This method performs a full, sequential decompression of the vector's contents
    /// into a standard in-memory `Vec<u64>`. It is functionally equivalent to
    /// `self.iter().collect()`.
    ///
    /// # Performance
    ///
    /// For the specific task of full decompression, this sequential method is often
    /// more performant than using its parallel counterpart (e.g., `par_iter().collect()`).
    /// The sequential iterator can better leverage CPU caches and avoids the overhead
    /// of thread management and result aggregation, which can become the bottleneck,
    /// especially when the decoding logic is not computationally intensive.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    /// let data: &[u64] = &[10, 129, 3, 40];
    /// let intvec = LEIntVec::builder(data).build().unwrap();
    ///
    /// // Decompress the entire vector back into a standard Vec.
    /// let decompressed_data = intvec.into_vec();
    /// assert_eq!(decompressed_data, data);
    /// ```
    pub fn into_vec(self) -> Vec<u64> {
        self.iter().collect()
    }

    /// Returns a clone of the underlying storage (`Vec<u64>`) holding the compressed bitstream.
    ///
    /// This method provides low-level access to the raw compressed data.
    pub fn limbs(&self) -> Vec<u64> {
        self.data.clone()
    }

    /// Returns the number of integers in the underlying compressed sequence
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    /// let data: &[u64] = &[10, 20, 30, 40];
    /// let intvec = LEIntVec::builder(data).build().unwrap();
    /// assert_eq!(intvec.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the sampling rate `k` used during encoding, if applicable.
    ///
    /// This value determines the trade-off between random access speed and memory
    /// overhead for bit-level codecs.
    ///
    /// # Returns
    /// - `Some(usize)`: The sampling rate `k` for bit-level encodings.
    /// - `None`: If [`FixedLength`] encoding was used, as it does not require sampling.
    ///
    /// [`FixedLength`]: crate::codec_spec::CodecSpec::FixedLength
    pub fn get_sampling_rate(&self) -> Option<usize> {
        self.k
    }

    /// Returns the number of sample points stored in the vector.
    ///
    /// For bit-level encodings, this is approximately `len / k`.
    /// For [`FixedLength`] encoding, this will be `0`.
    ///
    /// # Returns
    /// - `usize`: The number of sample points, or `0` for [`FixedLength`] encoding.
    ///
    /// [`FixedLength`]: crate::codec_spec::CodecSpec::FixedLength
    ///
    pub fn get_num_samples(&self) -> usize {
        self.samples.as_ref().map_or(0, |s| s.len())
    }

    /// Returns an iterator over the decompressed `u64` values.
    ///
    /// The iterator provides sequential, forward-only access to the data,
    /// decompressing elements on the fly. See [`IntVecIter`] for more details.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    /// let data: &[u64] = &[10, 20, 30, 40];
    /// let intvec = LEIntVec::builder(data).build().unwrap();
    ///
    /// let collected: Vec<u64> = intvec.iter().collect();
    /// assert_eq!(collected, data);
    /// ```
    pub fn iter(&self) -> IntVecIter<E> {
        IntVecIter::new(self)
    }

    /// Returns the concrete `Encoding` variant that was used for compression.
    ///
    /// This is useful for inspecting which codec and parameters were chosen,
    /// especially when using automatic parameter selection.
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }
}

/// A type alias for an [`IntVec`] with Big-Endian ([`BE`]) bitstream encoding.
///
/// Big-endian is the byte order used in many networking protocols and on certain
/// CPU architectures. The choice of endianness can impact performance.
///
/// While using the native endianness of the host machine is typically the most
/// performant option, the optimal choice can be influenced by the availability of
/// specific low-level CPU instructions (e.g., for counting leading/trailing zeros
/// or finding the first set bit). Depending on the architecture and the specific
/// compression code being used, a non-native endianness may unexpectedly yield
/// better performance.
///
/// # Example
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[u64] = &[40, 12, 13, 50, 13];
///
/// // Create a Big-Endian IntVec using the type alias.
/// let be_intvec = BEIntVec::builder(data).build().unwrap();
///
/// assert_eq!(be_intvec.len(), 5);
/// assert_eq!(be_intvec.get(2), Some(13));
/// ```
pub type BEIntVec = IntVec<BE>;

/// A type alias for an [`IntVec`] with Little-Endian ([`LE`]) bitstream encoding.
///
/// Little-endian is the native byte order for most modern commodity CPUs,
/// including x86 and ARM, which can often lead to the best performance for
/// bitstream operations.
///
/// However, the optimal choice is not always straightforward. Performance depends
/// on the interplay between the compression algorithm and the efficiency of
/// low-level bit manipulation instructions on the target hardware, which may
/// favor one endianness over the other in specific cases. Benchmarking is the
/// best way to determine the optimal choice for a given workload.
///
/// # Example
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[u64] = &[40, 12, 13, 50, 13];
///
/// // Create a Little-Endian IntVec using the type alias.
/// let le_intvec = LEIntVec::builder(data).build().unwrap();
///
/// assert_eq!(le_intvec.len(), 5);
/// assert_eq!(le_intvec.get(2), Some(13));
/// ```
pub type LEIntVec = IntVec<LE>;
