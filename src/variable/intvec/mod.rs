//! # A compressed, randomly accessible vector of `u64` integers.
//!
//! This module provides the core implementation of [`IntVec`], a data structure
//! designed for space-efficient storage and fast random access of `u64` integer
//! sequences. It achieves compression by leveraging a variety of instantaneous,
//! variable-length codes from the [`dsi-bitstream`] crate.
//!
//! ## Core Functionality
//!
//! - **Compression**: Employs codecs like Gamma (γ), Delta (δ), and Zeta (ζ) for
//!   data with skewed distributions.
//! - **Fast Random Access**: Uses a sampling mechanism to provide fast random
//!   access. The sampling rate, `k`, determines the trade-off between access
//!   speed and memory overhead.
//! - **Flexible Construction**: Provides a builder API that can construct an
//!   [`IntVec`] from a slice or iterator, with support for automatic codec selection.
//! - **High-Performance Lookups**: Offers optimized methods for various access
//!   patterns, including a reusable [`IntVecReader`] for dynamic lookups, and
//!   efficient batch methods like [`get_many`] and [`par_get_many`].
//!
//! The main struct, [`IntVec`], is generic over [`Endianness`], allowing
//! users to choose between Little-Endian ([`LEIntVec`]) and Big-Endian ([`BEIntVec`])
//! representations to optimize for specific hardware architectures.
//!
//! ## Example
//!
//! ```rust
//! use compressed_intvec::prelude::*;
//!
//! // A small vector of integers to be compressed.
//! let data: &[u64] = &;
//!
//! // Use the builder to create an IntVec.
//! // `VariableCodecSpec::Auto` will analyze the data and select the best DSI codec.
//! let intvec = LEIntVec::builder(&data)
//!     .k(2) // Use a small sampling rate for this vector.
//!     .codec(VariableCodecSpec::Auto)
//!     .build()
//!     .unwrap();
//!
//! // Verify the length and access some elements.
//! assert_eq!(intvec.len(), data.len());
//! assert_eq!(intvec.get(1), Some(200));
//! assert_eq!(intvec.get(6), Some(1023));
//! ```
//!
//! ## Tip for Efficient Access
//!
//! Use `k` as a power of two! It's faster when accessing sequentially close indices.
//! Values like 16 and 32 usually provide a good balance between memory usage and access speed.
//!
//! [`dsi-bitstream`]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/
//! [`Endianness`]: dsi_bitstream::prelude::Endianness
//! [`get_many`]: IntVec::get_many
//! [`par_get_many`]: IntVec::par_get_many
use dsi_bitstream::{
    codes::params::DefaultReadParams,
    prelude::{
        BitRead, BitSeek, BufBitReader, BufBitWriter, Codes, CodesRead, Endianness, MemWordReader,
        MemWordWriterVec, StaticCodeRead,
    },
    traits::{BE, LE},
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
mod seq_reader;
#[cfg(feature = "serde")]
mod serde;

pub use builder::{IntVecBuilder, IntVecFromIterBuilder};
pub use iter::IntVecIter;
pub use reader::IntVecReader;
pub use seq_reader::IntVecSeqReader;

/// Defines the set of errors that can occur in `IntVec` operations.
#[derive(Debug)]
pub enum IntVecError {
    /// An error occurred during an I/O operation, typically forwarded from the
    /// underlying bitstream library.
    Io(std::io::Error),
    /// A generic error originating from the `dsi-bitstream` library.
    Bitstream(Box<dyn Error + Send + Sync>),
    /// An error indicating that the provided parameters are invalid for the
    /// requested operation, such as a sampling rate of `0`.
    InvalidParameters(String),
    /// An error during the dispatch of a compression or decompression function.
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
/// to save space on smaller vectors.
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
/// the cost of slower access.
///
/// The most convenient way to create an [`IntVec`] is through its [builder](IntVec::builder),
/// which allows for easy configuration of the sampling rate and compression codec,
/// including automatic parameter selection.
///
/// The generic parameter `E` specifies the [`Endianness`] of the underlying bitstream.
/// For convenience, the type aliases [`LEIntVec`] and [`BEIntVec`] are provided for
/// little-endian and big-endian configurations, respectively.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct IntVec<E: Endianness> {
    /// The raw compressed data, stored as a `Vec<u64>`.
    pub(super) data: Vec<u64>,
    /// Bit offsets of sampled elements.
    pub(super) samples: Option<Samples>,
    /// The sampling rate `k`, which determines the interval between samples.
    pub(super) k: Option<usize>,
    /// The number of elements in the vector.
    pub(super) len: usize,
    /// The concrete `dsi-bitstream` code used for compression.
    pub(super) encoding: Codes,
    /// A zero-sized marker for the endianness type parameter.
    pub(super) endian: PhantomData<E>,
}

/// Type alias for the writer used internally by `IntVec`.
pub(crate) type IntVecBitWriter<E> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;
/// Type alias for the reader used internally by `IntVec`.
pub(crate) type IntVecBitReader<'a, E> =
    BufBitReader<E, MemWordReader<u64, &'a Vec<u64>>, DefaultReadParams>;

impl<E: Endianness> IntVec<E> {
    /// Returns a builder for creating an [`IntVec`] from a slice of data.
    ///
    /// This method is generic over `AsRef<[u64]>`, so it can accept `&[u64]`,
    /// `Vec<u64>`, etc.
    pub fn builder<T: AsRef<[u64]> + ?Sized>(input: &T) -> IntVecBuilder<E> {
        IntVecBuilder::new(input.as_ref())
    }

    /// Returns a builder for creating an [`IntVec`] from an iterator.
    ///
    /// # Limitations
    /// This builder **requires** that codec parameters be specified manually.
    pub fn from_iter_builder<I: IntoIterator<Item = u64>>(iter: I) -> IntVecFromIterBuilder<E, I> {
        IntVecFromIterBuilder::new(iter)
    }
}

impl<E: Endianness> IntVec<E>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a stateful, reusable [`IntVecReader`] for this vector.
    pub fn reader(&self) -> IntVecReader<E> {
        IntVecReader::new(self)
    }

    /// Creates a stateful, reusable [`IntVecSeqReader`] for this vector.
    pub fn seq_reader(&self) -> IntVecSeqReader<E> {
        IntVecSeqReader::new(self)
    }

    /// Retrieves the element at the specified index.
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
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(
            index < self.len,
            "Index out of bounds: index was {} but length was {}",
            index,
            self.len
        );
        let mut reader = self.reader();
        reader.get_unchecked(index)
    }

    /// Retrieves multiple elements at the specified indices in a highly efficient way.
    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<u64>, IntVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        for &index in indices {
            if index >= self.len {
                return Err(IntVecError::IndexOutOfBounds(index));
            }
        }
        // SAFETY: We have pre-checked the bounds of all indices.
        Ok(unsafe { self.get_many_unchecked(indices) })
    }

    /// Retrieves multiple elements at the specified indices without bounds checking.
    ///
    /// # Safety
    /// Calling this method with any out-of-bounds index is undefined behavior.
    pub unsafe fn get_many_unchecked(&self, indices: &[usize]) -> Vec<u64> {
        if indices.is_empty() {
            return Vec::new();
        }
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

        let mut results = vec![0; indices.len()];
        let mut indexed_indices: Vec<(usize, usize)> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();
        indexed_indices.par_sort_unstable_by_key(|&(idx, _)| idx);

        let k = self.k.unwrap();

        if k.is_power_of_two() {
            let k_exp = k.trailing_zeros();
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
                |idx| idx / k,
                |block| block * k,
            )
            .unwrap();
        }

        results
    }

    /// Inner helper function for DSI-based `get_many` to avoid code duplication.
    fn get_many_dsi_inner<F1, F2>(
        &self,
        indexed_indices: &[(usize, usize)],
        results: &mut [u64],
        block_of: F1,
        start_of_block: F2,
    ) -> Result<(), IntVecError>
    where
        F1: Fn(usize) -> usize,
        F2: Fn(usize) -> usize,
    {
        let samples = self.samples.as_ref().unwrap();
        let mut reader = self.reader();
        let mut current_decoded_index: usize = 0;

        for &(target_index, original_position) in indexed_indices {
            if target_index < current_decoded_index
                || block_of(target_index) != block_of(current_decoded_index.saturating_sub(1))
            {
                let target_sample_block = block_of(target_index);
                let start_bit = samples.get(target_sample_block).unwrap();
                reader.reader.set_bit_pos(start_bit)?;
                current_decoded_index = start_of_block(target_sample_block);
            }

            for _ in current_decoded_index..target_index {
                reader.code_reader.read(&mut reader.reader)?;
            }
            let value = reader.code_reader.read(&mut reader.reader)?;
            results[original_position] = value;
            current_decoded_index = target_index + 1;
        }

        Ok(())
    }

    /// Retrieves multiple elements from an iterator of indices.
    pub fn get_many_from_iter<I>(&self, indices: I) -> Result<Vec<u64>, IntVecError>
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

    /// Consumes the [`IntVec`] and returns a `Vec<u64>`.
    pub fn into_vec(self) -> Vec<u64> {
        self.iter().collect()
    }

    /// Returns a clone of the underlying storage (`Vec<u64>`).
    pub fn limbs(&self) -> Vec<u64> {
        self.data.clone()
    }

    /// Returns the number of integers in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the sampling rate `k` used during encoding.
    pub fn get_sampling_rate(&self) -> Option<usize> {
        self.k
    }

    /// Returns the number of sample points stored in the vector.
    pub fn get_num_samples(&self) -> usize {
        self.samples.as_ref().map_or(0, |s| s.len())
    }

    /// Returns an iterator over the decompressed `u64` values.
    pub fn iter(&self) -> IntVecIter<E> {
        IntVecIter::new(self)
    }

    /// Returns the concrete `Codes` variant that was used for compression.
    pub fn encoding(&self) -> Codes {
        self.encoding
    }
}

/// A type alias for an [`IntVec`] with Big-Endian ([`BE`]) bitstream encoding.
pub type BEIntVec = IntVec<BE>;

/// A type alias for an [`IntVec`] with Little-Endian ([`LE`]) bitstream encoding.
pub type LEIntVec = IntVec<LE>;
