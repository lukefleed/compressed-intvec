//! A compressed vector of variable-length sequences with indexed access.
//!
//! This module provides [`SeqVec`], a data structure for storing multiple
//! integer sequences in a single compressed bitstream. Each sequence is
//! accessed by its index (rank), and all elements within a sequence are
//! decoded together.
//!
//! # Core Concepts
//!
//! ## Use Case
//!
//! [`SeqVec`] is designed for scenarios where:
//!
//! - Data is naturally organized as many variable-length sequences.
//! - Access patterns retrieve entire sequences rather than individual elements.
//! - Memory overhead of per-sequence pointers and padding must be minimized.
//!
//! A common application is representing **adjacency lists** in a compressed
//! graph, where each node's neighbors form a sequence.
//!
//! ## Differences from [`VarVec`]
//!
//! | Aspect | [`VarVec`] | [`SeqVec`] |
//! |--------|-----------|------------|
//! | Access unit | Single element | Entire sequence |
//! | Index meaning | Element position | Sequence rank |
//! | Sampling | Periodic (every k elements) | At sequence boundaries |
//! | Primary operation | `get(i) → T` | `get(i) → Iterator<T>` |
//!
//! ## Compression
//!
//! Like [`VarVec`], [`SeqVec`] uses instantaneous variable-length codes (Gamma,
//! Delta, Zeta, etc.) from the [`dsi-bitstream`] crate. All sequences are
//! concatenated into a single compressed bitstream, with a [`FixedVec`] index
//! storing the bit offset of each sequence's start.
//!
//! ## Sequence Length
//!
//! Sequence lengths are **not stored by default**. The iterator for a sequence
//! terminates when the current bit position reaches the start of the next
//! sequence. This means:
//!
//! - Retrieving a sequence is O(length) for decoding — unavoidable.
//! - Computing sequence length requires full iteration unless lengths are stored.
//!
//! You can opt-in to storing explicit lengths via
//! [`SeqVecBuilder::store_lengths`](crate::seq::SeqVecBuilder::store_lengths).
//! When enabled, O(1) length queries become available via
//! [`SeqVec::sequence_len`](crate::seq::SeqVec::sequence_len), and decoding
//! can avoid the end-bit check in hot loops.
//!
//! ## Immutability
//!
//! [`SeqVec`] is **immutable** after creation. Variable-length encoding makes
//! in-place modification impractical, as changing one element could shift all
//! subsequent data.
//!
//! # Main Components
//!
//! - [`SeqVec`]: The core compressed sequence vector.
//! - [`SeqVecBuilder`]: Builder for constructing a [`SeqVec`] with custom codec.
//! - [`SeqIter`]: Zero-allocation iterator over elements of a single sequence.
//! - [`SeqVecIter`]: Iterator over all sequences, yielding [`SeqIter`] instances.
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::seq::{SeqVec, LESeqVec};
//!
//! let sequences: &[&[u32]] = &[
//!     &[1, 2, 3],
//!     &[10, 20],
//!     &[100, 200, 300, 400],
//!     &[], // Empty sequences are supported
//! ];
//!
//! let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
//!
//! assert_eq!(vec.num_sequences(), 4);
//!
//! // Access a sequence by index
//! let seq1: Vec<u32> = vec.get(1).unwrap().collect();
//! assert_eq!(seq1, vec![10, 20]);
//! #     Ok(())
//! # }
//! ```
//!
//! ## Custom Codec
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::seq::{SeqVec, LESeqVec, Codec};
//!
//! let sequences: Vec<Vec<u64>> = vec![
//!     vec![1, 1, 1, 2, 3],
//!     vec![100, 200, 300],
//! ];
//!
//! let vec: LESeqVec<u64> = SeqVec::builder()
//!     .codec(Codec::Zeta { k: Some(3) })
//!     .build(&sequences)?;
//! #     Ok(())
//! # }
//! ```
//!
//! [`VarVec`]: crate::variable::VarVec
//! [`FixedVec`]: crate::fixed::FixedVec
//! [`dsi-bitstream`]: https://crates.io/crates/dsi-bitstream

mod builder;
mod iter;
mod macros;
#[cfg(feature = "parallel")]
mod parallel;
mod reader;
#[cfg(feature = "serde")]
mod serde;
mod slice;

pub use builder::{SeqVecBuilder, SeqVecFromIterBuilder};
pub use iter::{SeqIter, SeqVecIntoIter, SeqVecIter};
pub use reader::SeqVecReader;
pub use slice::SeqVecSlice;

// Re-export codec spec for convenience.
pub use crate::variable::codec::Codec;

// Re-export deprecated alias for backward compatibility.
#[allow(deprecated)]
pub use crate::variable::VariableCodecSpec;

use crate::common::codec_reader::CodecReader;
use crate::fixed::{Error as FixedVecError, FixedVec};
use crate::variable::traits::Storable;
use dsi_bitstream::{
    dispatch::{Codes, CodesRead, StaticCodeRead},
    impls::{BufBitWriter, MemWordWriterVec},
    prelude::{BE, BitRead, BitSeek, BitWrite, CodesWrite, Endianness, LE},
};
use iter::SeqVecBitReader;
use mem_dbg::{DbgFlags, FlatType, MemDbgImpl, MemSize, SizeFlags};
use std::marker::PhantomData;
use std::{error::Error, fmt};

/// Errors that can occur when working with [`SeqVec`].
#[derive(Debug)]
pub enum SeqVecError {
    /// An I/O error from bitstream operations.
    Io(std::io::Error),
    /// An error from the bitstream library during encoding or decoding.
    Bitstream(Box<dyn Error + Send + Sync>),
    /// Invalid parameters were provided during construction.
    InvalidParameters(String),
    /// An error during codec resolution or dispatch.
    CodecDispatch(String),
    /// The requested sequence index is out of bounds.
    IndexOutOfBounds(usize),
}

impl fmt::Display for SeqVecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeqVecError::Io(e) => write!(f, "I/O error: {}", e),
            SeqVecError::Bitstream(e) => write!(f, "Bitstream error: {}", e),
            SeqVecError::InvalidParameters(s) => write!(f, "Invalid parameters: {}", s),
            SeqVecError::CodecDispatch(s) => write!(f, "Codec dispatch error: {}", s),
            SeqVecError::IndexOutOfBounds(idx) => {
                write!(f, "Sequence index out of bounds: {}", idx)
            }
        }
    }
}

impl Error for SeqVecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            SeqVecError::Io(e) => Some(e),
            SeqVecError::Bitstream(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SeqVecError {
    fn from(e: std::io::Error) -> Self {
        SeqVecError::Io(e)
    }
}

impl From<core::convert::Infallible> for SeqVecError {
    fn from(_: core::convert::Infallible) -> Self {
        unreachable!()
    }
}

impl From<FixedVecError> for SeqVecError {
    fn from(e: FixedVecError) -> Self {
        SeqVecError::InvalidParameters(e.to_string())
    }
}

/// Type alias for the bit writer used internally by [`SeqVec`] builders.
pub(crate) type SeqVecBitWriter<E> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;

/// A compressed, indexed vector of integer sequences.
///
/// `SeqVec` stores multiple sequences of integers in a single compressed
/// bitstream, with an auxiliary index for O(1) access to each sequence by
/// its rank. This is ideal for representing collections of variable-length
/// sequences with minimal memory overhead.
///
/// See the [module-level documentation](self) for detailed usage information.
///
/// # Type Parameters
///
/// * `T` - The element type (e.g., `u32`, `i16`). Must implement [`Storable`].
/// * `E` - The [`Endianness`] of the underlying bitstream ([`LE`] or [`BE`]).
/// * `B` - The backing buffer type, enabling owned (`Vec<u64>`) or borrowed
///   (`&[u64]`) storage for zero-copy operations.
#[derive(Debug, Clone)]
pub struct SeqVec<T: Storable, E: Endianness, B: AsRef<[u64]> = Vec<u64>> {
    /// The compressed bitstream containing all sequences concatenated.
    data: B,
    /// Bit offsets marking the start of each sequence. Contains N+1 elements
    /// where N is the number of sequences. The final element is a sentinel
    /// containing the total bit length.
    /// Uses the same endianness `E` as the struct for design consistency.
    bit_offsets: FixedVec<u64, u64, E, B>,
    /// Optional per-sequence lengths stored in a compact fixed-width vector.
    /// Uses `u64` (architecture-independent) to ensure portability across 32-bit
    /// and 64-bit systems. Accessor methods return `usize` via safe casting.
    seq_lengths: Option<FixedVec<u64, u64, E, Vec<u64>>>,
    /// The compression codec used for all elements.
    encoding: Codes,
    /// Zero-sized markers for the generic type parameters.
    _markers: PhantomData<(T, E)>,
}

// --- Type Aliases ---

/// A [`SeqVec`] with little-endian bit ordering.
pub type LESeqVec<T, B = Vec<u64>> = SeqVec<T, LE, B>;

/// A [`SeqVec`] with big-endian bit ordering.
pub type BESeqVec<T, B = Vec<u64>> = SeqVec<T, BE, B>;

/// A [`SeqVec`] for unsigned integers with little-endian bit ordering.
///
/// Use this type alias for sequences of unsigned integer values stored with
/// little-endian bit ordering.
pub type USeqVec<T, B = Vec<u64>> = SeqVec<T, LE, B>;

/// A [`SeqVec`] for signed integers with little-endian bit ordering.
///
/// Signed integers are transparently encoded using zig-zag encoding via the
/// [`Storable`] trait.
pub type SSeqVec<T, B = Vec<u64>> = SeqVec<T, LE, B>;

/// A [`SeqVec`] for signed integers with big-endian bit ordering.
///
/// Signed integers are transparently encoded using zig-zag encoding via the
/// [`Storable`] trait.
pub type BESSeqVec<T, B = Vec<u64>> = SeqVec<T, BE, B>;

/// A [`SeqVec`] for signed integers with little-endian bit ordering.
///
/// This is an alias for [`SSeqVec`], provided for consistency with the naming
/// pattern `{Endianness}{SignedUnsigned}SeqVec`.
///
/// Signed integers are transparently encoded using zig-zag encoding via the
/// [`Storable`] trait.
pub type LESSeqVec<T, B = Vec<u64>> = SeqVec<T, LE, B>;

/// Type alias for a tuple of two [`SeqVecSlice`] references.
pub type SeqVecSlicePair<'a, T, E, B> = (SeqVecSlice<'a, T, E, B>, SeqVecSlice<'a, T, E, B>);

// --- Construction (Owned) ---

impl<T: Storable + 'static, E: Endianness> SeqVec<T, E, Vec<u64>> {
    /// Creates a [`SeqVec`] from a slice of slices using default settings.
    ///
    /// This method uses [`Codec::Auto`] to select an optimal codec
    /// based on the data distribution.
    ///
    /// # Errors
    ///
    /// Returns a [`SeqVecError`] if codec resolution or encoding fails.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// assert_eq!(vec.num_sequences(), 3);
    /// #     Ok(())
    /// # }
    /// ```
    pub fn from_slices<S: AsRef<[T]>>(sequences: &[S]) -> Result<Self, SeqVecError>
    where
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        Self::builder().codec(Codec::Auto).build(sequences)
    }

    /// Consumes the [`SeqVec`] and returns all sequences as a `Vec<Vec<T>>`.
    ///
    /// This method decodes the entire compressed data, allocating a separate
    /// vector for each sequence. The operation requires time proportional to the
    /// total number of elements across all sequences.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[10, 20, 30]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// let decoded: Vec<Vec<u32>> = vec.into_vecs();
    /// assert_eq!(decoded, vec![vec![1, 2], vec![10, 20, 30]]);
    /// #     Ok(())
    /// # }
    /// ```
    pub fn into_vecs(self) -> Vec<Vec<T>>
    where
        for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.iter().map(|seq_iter| seq_iter.collect()).collect()
    }
}

// --- Construction (Generic Buffer) ---

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVec<T, E, B> {
    /// Creates a [`SeqVec`] from raw components for zero-copy views.
    ///
    /// This constructor is intended for advanced use cases such as memory-mapping
    /// a pre-built [`SeqVec`] from disk or creating a borrowed view.
    ///
    /// # Arguments
    ///
    /// * `data` - The compressed bitstream buffer.
    /// * `bit_offsets_data` - The buffer containing the bit offsets index.
    /// * `bit_offsets_len` - Number of entries in the bit offsets index (N+1).
    /// * `bit_offsets_num_bits` - Bit width of each entry in the offsets index.
    /// * `encoding` - The codec used for compression.
    ///
    /// # Errors
    ///
    /// Returns [`SeqVecError::InvalidParameters`] if:
    /// * `bit_offsets_len` is zero (must have at least the sentinel entry).
    /// * The underlying [`FixedVec`] construction fails.
    ///
    /// # Safety Considerations
    ///
    /// The caller must ensure that:
    /// * `data` contains valid compressed data encoded with `encoding`.
    ///   Invalid data will cause panics or incorrect results during decoding.
    /// * `bit_offsets_data` contains monotonically increasing bit positions.
    ///   Unsorted offsets will cause out-of-order or corrupted sequence retrieval.
    /// * The sentinel value (final bit offset) does not exceed the total bits
    ///   in `data`. Violations cause panics during decoding.
    /// * All bit positions fall within valid boundaries of the bitstream.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    /// use compressed_intvec::fixed::FixedVec;
    /// use dsi_bitstream::prelude::LE;
    /// use dsi_bitstream::impls::{BufBitWriter, MemWordWriterVec};
    /// use dsi_bitstream::prelude::{BitWrite, CodesWrite};
    ///
    /// // Create a simple vector using high-level API
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// // In a real zero-copy scenario, these would come from disk/memory-map
    /// let data = vec.as_limbs().to_vec();
    /// let offsets_ref = vec.bit_offsets_ref();
    ///
    /// // Verify structure is sound before reconstruction
    /// assert_eq!(vec.num_sequences(), 2);
    /// #     Ok(())
    /// # }
    /// ```
    pub fn from_parts(
        data: B,
        bit_offsets_data: B,
        bit_offsets_len: usize,
        bit_offsets_num_bits: usize,
        encoding: Codes,
    ) -> Result<Self, SeqVecError> {
        if bit_offsets_len == 0 {
            return Err(SeqVecError::InvalidParameters(
                "bit_offsets must have at least one entry (the sentinel)".to_string(),
            ));
        }

        let bit_offsets = FixedVec::<u64, u64, E, B>::from_parts(
            bit_offsets_data,
            bit_offsets_len,
            bit_offsets_num_bits,
        )?;

        Ok(Self {
            data,
            bit_offsets,
            seq_lengths: None,
            encoding,
            _markers: PhantomData,
        })
    }

    /// Creates a [`SeqVec`] from raw components with optional stored lengths.
    ///
    /// Use this variant when you have pre-computed sequence lengths from an
    /// earlier encoding. The `seq_lengths` parameter must be consistent with
    /// `bit_offsets` when provided (element count must equal `num_sequences()`).
    ///
    /// # Errors
    ///
    /// Returns [`SeqVecError::InvalidParameters`] if the number of lengths does
    /// not equal the number of sequences.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// // Typical usage: start with high-level API
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// // Verify structure
    /// assert!(!vec.has_stored_lengths());
    /// assert_eq!(vec.num_sequences(), 2);
    /// #     Ok(())
    /// # }
    /// ```
    pub fn from_parts_with_lengths(
        data: B,
        bit_offsets_data: B,
        bit_offsets_len: usize,
        bit_offsets_num_bits: usize,
        seq_lengths: Option<FixedVec<u64, u64, E, Vec<u64>>>,
        encoding: Codes,
    ) -> Result<Self, SeqVecError> {
        if bit_offsets_len == 0 {
            return Err(SeqVecError::InvalidParameters(
                "bit_offsets must have at least one entry (the sentinel)".to_string(),
            ));
        }

        if let Some(lengths) = &seq_lengths
            && lengths.len() + 1 != bit_offsets_len
        {
            return Err(SeqVecError::InvalidParameters(
                "seq_lengths length must match number of sequences".to_string(),
            ));
        }

        let bit_offsets = FixedVec::<u64, u64, E, B>::from_parts(
            bit_offsets_data,
            bit_offsets_len,
            bit_offsets_num_bits,
        )?;

        Ok(Self {
            data,
            bit_offsets,
            seq_lengths,
            encoding,
            _markers: PhantomData,
        })
    }

    /// Creates a [`SeqVec`] from pre-built components without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure all components are consistent and valid. Mismatched
    /// parameters will lead to panics or incorrect data retrieval.
    #[inline]
    pub unsafe fn from_parts_unchecked(
        data: B,
        bit_offsets: FixedVec<u64, u64, E, B>,
        encoding: Codes,
    ) -> Self {
        Self {
            data,
            bit_offsets,
            seq_lengths: None,
            encoding,
            _markers: PhantomData,
        }
    }

    /// Creates a [`SeqVec`] from pre-built components with optional lengths
    /// without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure all components are consistent and valid.
    #[inline]
    pub unsafe fn from_parts_with_lengths_unchecked(
        data: B,
        bit_offsets: FixedVec<u64, u64, E, B>,
        seq_lengths: Option<FixedVec<u64, u64, E, Vec<u64>>>,
        encoding: Codes,
    ) -> Self {
        Self {
            data,
            bit_offsets,
            seq_lengths,
            encoding,
            _markers: PhantomData,
        }
    }
}

// --- Query Methods ---

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVec<T, E, B> {
    /// Returns the number of sequences stored.
    ///
    /// This is O(1) as it is derived from the bit offsets index length.
    #[inline(always)]
    pub fn num_sequences(&self) -> usize {
        // bit_offsets has N+1 entries for N sequences (always at least the sentinel).
        self.bit_offsets.len() - 1
    }

    /// Returns `true` if there are no sequences stored.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.num_sequences() == 0
    }

    /// Returns the compression codec used for encoding.
    #[inline(always)]
    pub fn encoding(&self) -> Codes {
        self.encoding
    }

    /// Returns a reference to the underlying compressed data buffer.
    #[inline(always)]
    pub fn as_limbs(&self) -> &[u64] {
        self.data.as_ref()
    }

    /// Returns a reference to the bit offsets index.
    #[inline(always)]
    pub fn bit_offsets_ref(&self) -> &FixedVec<u64, u64, E, B> {
        &self.bit_offsets
    }

    /// Returns `true` if explicit sequence lengths were stored at construction time.
    #[inline(always)]
    pub fn has_stored_lengths(&self) -> bool {
        self.seq_lengths.is_some()
    }

    /// Returns the length of sequence `index` if explicit lengths are stored.
    ///
    /// Returns `None` if `index` is out of bounds or if lengths were not
    /// stored at construction time. Use [`has_stored_lengths`](Self::has_stored_lengths)
    /// to distinguish between these cases.
    ///
    /// If lengths are stored, this query completes in O(1) time. Otherwise,
    /// determining sequence length requires full iteration. Store lengths at
    /// construction time via [`SeqVecBuilder::store_lengths`](crate::seq::SeqVecBuilder::store_lengths)
    /// when O(1) length queries are needed.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// // Without storing lengths, returns None
    /// assert_eq!(vec.sequence_len(0), None);
    /// assert!(!vec.has_stored_lengths());
    /// #     Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn sequence_len(&self, index: usize) -> Option<usize> {
        if index >= self.num_sequences() {
            return None;
        }

        self.seq_lengths
            .as_ref()
            .map(|lengths| unsafe { lengths.get_unchecked(index) as usize })
    }

    /// Returns the total number of bits in the compressed data.
    ///
    /// This is the sentinel value at the end of the bit offsets index.
    #[inline(always)]
    pub fn total_bits(&self) -> u64 {
        // bit_offsets always has at least one sentinel entry by construction invariant.
        unsafe { self.bit_offsets.get_unchecked(self.bit_offsets.len() - 1) }
    }

    /// Returns the bit offset where sequence `index` starts in the compressed data.
    ///
    /// Returns `None` if `index >= num_sequences()`. This is useful for
    /// understanding the compression footprint or verifying sequence boundaries.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// // First sequence always starts at bit 0
    /// assert_eq!(vec.sequence_start_bit(0), Some(0));
    /// // Second sequence starts after the first
    /// assert!(vec.sequence_start_bit(1).is_some());
    /// // Out of bounds
    /// assert_eq!(vec.sequence_start_bit(2), None);
    /// #     Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn sequence_start_bit(&self, index: usize) -> Option<u64> {
        if index >= self.num_sequences() {
            return None;
        }
        // SAFETY: bounds checked above.
        Some(unsafe { self.bit_offsets.get_unchecked(index) })
    }

    /// Returns the bit offset where sequence `index` starts, without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index < num_sequences()`.
    #[inline(always)]
    pub unsafe fn sequence_start_bit_unchecked(&self, index: usize) -> u64 {
        debug_assert!(
            index < self.num_sequences(),
            "index {} out of bounds for {} sequences",
            index,
            self.num_sequences()
        );
        unsafe { self.bit_offsets.get_unchecked(index) }
    }

    /// Returns the bit offset immediately after sequence `index` ends.
    ///
    /// This is equivalent to the start bit of the next sequence, or the total
    /// bit length for the final sequence. Useful for calculating per-sequence
    /// compression footprint.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// let start = vec.sequence_start_bit(0).unwrap();
    /// let end = unsafe { vec.sequence_end_bit_unchecked(0) };
    /// println!("Sequence 0 occupies {} bits", end - start);
    /// #     Ok(())
    /// # }
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must ensure `index < num_sequences()`.
    #[inline]
    pub unsafe fn sequence_end_bit_unchecked(&self, index: usize) -> u64 {
        debug_assert!(
            index < self.num_sequences(),
            "index {} out of bounds for {} sequences",
            index,
            self.num_sequences()
        );
        unsafe { self.bit_offsets.get_unchecked(index + 1) }
    }
}

// --- Access Methods ---

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVec<T, E, B>
where
    for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Returns an iterator over the elements of sequence `index`.
    ///
    /// Returns `None` if `index >= num_sequences()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// let first: Vec<u32> = vec.get(0).unwrap().collect();
    /// assert_eq!(first, vec![1, 2, 3]);
    /// #     Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<SeqIter<'_, T, E>> {
        if index >= self.num_sequences() {
            return None;
        }
        // SAFETY: bounds checked above.
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Returns an iterator over the elements of sequence `index` without
    /// bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with `index >= num_sequences()` is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> SeqIter<'_, T, E> {
        debug_assert!(
            index < self.num_sequences(),
            "index {} out of bounds for {} sequences",
            index,
            self.num_sequences()
        );

        let start_bit = unsafe { self.sequence_start_bit_unchecked(index) };
        let end_bit = unsafe { self.sequence_end_bit_unchecked(index) };
        let len = self
            .seq_lengths
            .as_ref()
            .map(|lengths| unsafe { lengths.get_unchecked(index) as usize });

        SeqIter::new_with_len(self.data.as_ref(), start_bit, end_bit, self.encoding, len)
    }

    /// Returns the elements of sequence `index` as a newly allocated `Vec`.
    ///
    /// Returns `None` if `index >= num_sequences()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[10, 20, 30]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// assert_eq!(vec.decode_vec(0), Some(vec![10, 20, 30]));
    /// assert_eq!(vec.decode_vec(1), None);
    /// #     Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn decode_vec(&self, index: usize) -> Option<Vec<T>> {
        if index >= self.num_sequences() {
            return None;
        }

        // SAFETY: Bounds check has been performed.
        Some(unsafe { self.decode_vec_unchecked(index) })
    }

    /// Returns the elements of sequence `index` as a newly allocated `Vec`
    /// without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with `index >= num_sequences()` is undefined behavior.
    #[inline(always)]
    pub unsafe fn decode_vec_unchecked(&self, index: usize) -> Vec<T> {
        unsafe { self.get_unchecked(index).collect() }
    }

    /// Decodes sequence `index` into the provided buffer.
    ///
    /// The buffer is cleared before use. Returns the number of elements
    /// decoded, or `None` if `index >= num_sequences()`.
    ///
    /// This method is more efficient than [`decode_vec`](Self::decode_vec) when
    /// reusing a buffer across multiple calls.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// let mut buf = Vec::new();
    /// assert_eq!(vec.decode_into(0, &mut buf), Some(2));
    /// assert_eq!(buf, vec![1, 2]);
    ///
    /// // Buffer is reused (cleared internally).
    /// assert_eq!(vec.decode_into(1, &mut buf), Some(3));
    /// assert_eq!(buf, vec![3, 4, 5]);
    /// #     Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn decode_into(&self, index: usize, buf: &mut Vec<T>) -> Option<usize> {
        if index >= self.num_sequences() {
            return None;
        }

        // SAFETY: Bounds check has been performed.
        Some(unsafe { self.decode_into_unchecked(index, buf) })
    }

    /// Decodes sequence `index` into the provided buffer without bounds checking.
    ///
    /// # Performance
    ///
    /// This method constructs a new bitstream reader and codec dispatcher on each
    /// call. For repeated random access, prefer [`reader()`](Self::reader) which
    /// reuses internal state across calls:
    ///
    /// ```no_run
    /// # use compressed_intvec::seq::{SeqVec, LESeqVec};
    /// # let vec: LESeqVec<u32> = SeqVec::from_slices(&[&[1u32][..]]).unwrap();
    /// let mut reader = vec.reader();
    /// let mut buf = Vec::new();
    /// for i in 0..vec.num_sequences() {
    ///     reader.decode_into(i, &mut buf);
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// Calling this method with `index >= num_sequences()` is undefined behavior.
    #[inline(always)]
    pub unsafe fn decode_into_unchecked(&self, index: usize, buf: &mut Vec<T>) -> usize {
        let start_bit = unsafe { self.sequence_start_bit_unchecked(index) };

        buf.clear();

        // Create reader and codec dispatcher once, then decode all elements
        // directly into the buffer without creating an intermediate SeqIter.
        // This avoids iterator overhead and enables better compiler optimization.
        let mut reader =
            SeqVecBitReader::<E>::new(dsi_bitstream::impls::MemWordReader::new_inf(self.data.as_ref()));
        let _ = reader.set_bit_pos(start_bit);
        let code_reader = CodecReader::new(self.encoding);

        if let Some(lengths) = &self.seq_lengths {
            let count = unsafe { lengths.get_unchecked(index) as usize };
            self.decode_counted(&mut reader, &code_reader, buf, count);
        } else {
            let end_bit = unsafe { self.sequence_end_bit_unchecked(index) };
            self.decode_until(&mut reader, &code_reader, buf, end_bit);
        }

        buf.len()
    }

    /// Decodes a known number of elements into `buf`.
    #[inline(always)]
    fn decode_counted<'a>(
        &self,
        reader: &mut SeqVecBitReader<'a, E>,
        code_reader: &CodecReader<'a, E>,
        buf: &mut Vec<T>,
        count: usize,
    ) {
        buf.reserve(count);
        for _ in 0..count {
            let word = code_reader.read(reader).unwrap();
            buf.push(T::from_word(word));
        }
    }

    /// Decodes elements until the reader reaches `end_bit`.
    ///
    /// Pre-allocates an estimated capacity based on the bit range to reduce
    /// reallocations. The estimate assumes ~4 bits per element, which is
    /// reasonable for common codecs like Delta with values in the 1-10k range.
    #[inline(always)]
    fn decode_until<'a>(
        &self,
        reader: &mut SeqVecBitReader<'a, E>,
        code_reader: &CodecReader<'a, E>,
        buf: &mut Vec<T>,
        end_bit: u64,
    ) {
        let start_bit = reader.bit_pos().unwrap();
        let bit_range = end_bit.saturating_sub(start_bit);
        // Estimate ~4 bits per element; clamp to at least 1 to avoid zero-capacity.
        let estimate = (bit_range / 4).max(1) as usize;
        buf.reserve(estimate);

        while reader.bit_pos().unwrap() < end_bit {
            let word = code_reader.read(reader).unwrap();
            buf.push(T::from_word(word));
        }
    }

    /// Returns an iterator over all sequences.
    ///
    /// Each element of the returned iterator is a [`SeqIter`] for the
    /// corresponding sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3], &[4, 5, 6]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// for (i, seq) in vec.iter().enumerate() {
    ///     println!("Sequence {}: {:?}", i, seq.collect::<Vec<_>>());
    /// }
    /// #     Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn iter(&self) -> SeqVecIter<'_, T, E, B> {
        SeqVecIter::new(
            self.data.as_ref(),
            &self.bit_offsets,
            self.seq_lengths.as_ref(),
            self.encoding,
            self.num_sequences(),
        )
    }

    /// Creates a reusable reader for convenient random access to sequences.
    ///
    /// The returned [`SeqVecReader`] provides a convenient interface for
    /// performing multiple sequence retrievals. While the current implementation
    /// is a thin wrapper, it serves as a natural extension point for future
    /// optimizations such as position tracking or caching.
    ///
    /// ## Performance Considerations
    ///
    /// - **Zero-copy iteration**: Returned iterators borrow directly from the
    ///   compressed data without intermediate allocations.
    /// - **Stateless operation**: Each call to [`get`](Self::get) is independent and creates a fresh [`SeqIter`].
    /// - **Convenience methods**: The reader provides [`decode_vec`](SeqVecReader::decode_vec)
    ///   and [`decode_into`](SeqVecReader::decode_into) for common patterns.
    ///
    /// ## When to Use
    ///
    /// Use a reader when:
    /// - You prefer a consistent interface for multiple accesses.
    /// - You want to use convenience methods like `decode_vec` or `decode_into`.
    /// - Future stateful optimizations would benefit your access pattern.
    ///
    /// For single queries or simple iteration, direct calls to [`get`](Self::get)
    /// or [`iter`](Self::iter) are equally efficient.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5], &[6]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// let mut reader = vec.reader();
    ///
    /// // Perform multiple random accesses efficiently
    /// assert_eq!(reader.decode_vec(2), Some(vec![6]));
    /// assert_eq!(reader.decode_vec(0), Some(vec![1, 2]));
    /// if let Some(seq) = reader.decode_vec(1) {
    ///     for value in seq {
    ///         assert!(value <= 5);
    ///     }
    /// }
    /// #     Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn reader(&self) -> SeqVecReader<'_, T, E, B> {
        SeqVecReader::new(self)
    }

    /// Creates a zero-copy slice of a contiguous range of sequences.
    ///
    /// The slice provides a view into `len` sequences starting at `start`,
    /// without copying the underlying compressed data.
    ///
    /// # Arguments
    ///
    /// * `start` - The index of the first sequence in the slice.
    /// * `len` - The number of sequences to include in the slice.
    ///
    /// # Returns
    ///
    /// Returns `Some(SeqVecSlice)` if the range is valid, or `None` if
    /// `start + len` exceeds the number of sequences.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5], &[6], &[7, 8]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// // Create a slice of sequences 1 and 2
    /// let slice = vec.slice(1, 2).unwrap();
    /// assert_eq!(slice.len(), 2);
    ///
    /// // Index 0 of the slice refers to sequence 1 of the original vector
    /// let seq: Vec<u32> = slice.get(0).unwrap().collect();
    /// assert_eq!(seq, vec![3, 4, 5]);
    /// #     Ok(())
    /// # }
    /// ```
    pub fn slice(&self, start: usize, len: usize) -> Option<slice::SeqVecSlice<'_, T, E, B>> {
        if start.saturating_add(len) > self.num_sequences() {
            return None;
        }
        Some(slice::SeqVecSlice::new(self, start..start + len))
    }

    /// Splits the vector into two slices at the specified index.
    ///
    /// Returns a tuple of two slices: the first contains sequences `[0, mid)`
    /// and the second contains sequences `[mid, len)`.
    ///
    /// # Returns
    ///
    /// Returns `Some((left_slice, right_slice))` if `mid <= num_sequences()`,
    /// or `None` if `mid` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1], &[2], &[3], &[4]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// let (left, right) = vec.split_at(2).unwrap();
    /// assert_eq!(left.len(), 2);
    /// assert_eq!(right.len(), 2);
    ///
    /// assert_eq!(left.decode_vec(0), Some(vec![1]));
    /// assert_eq!(right.decode_vec(0), Some(vec![3]));
    /// #     Ok(())
    /// # }
    /// ```
    pub fn split_at(&self, mid: usize) -> Option<SeqVecSlicePair<'_, T, E, B>> {
        if mid > self.num_sequences() {
            return None;
        }
        Some((
            slice::SeqVecSlice::new(self, 0..mid),
            slice::SeqVecSlice::new(self, mid..self.num_sequences()),
        ))
    }
}

// --- MemSize Implementation ---

impl<T: Storable, E: Endianness, B: AsRef<[u64]> + MemSize + FlatType> MemSize for SeqVec<T, E, B> {
    fn mem_size_rec(&self, flags: SizeFlags, _refs: &mut mem_dbg::HashMap<usize, usize>) -> usize {
        let mut total = core::mem::size_of::<Self>();
        // Add heap-allocated memory for the data buffer.
        total += self.data.mem_size(flags) - core::mem::size_of::<B>();
        // Add heap-allocated memory for the bit_offsets index.
        total +=
            self.bit_offsets.mem_size(flags) - core::mem::size_of::<FixedVec<u64, u64, E, B>>();
        // Add heap-allocated memory for optional sequence lengths.
        if let Some(lengths) = &self.seq_lengths {
            total +=
                lengths.mem_size(flags) - core::mem::size_of::<FixedVec<u64, u64, E, Vec<u64>>>();
        }
        total
    }
}

// --- MemDbgImpl Implementation ---

// Wrapper for Codes to provide correct MemDbgImpl, following the pattern in
// variable::VarVec. This is necessary because the derived implementation for
// Codes is incorrect and cannot be fixed due to the orphan rule.
struct CodeWrapper<'a>(&'a Codes);

impl MemSize for CodeWrapper<'_> {
    fn mem_size_rec(&self, _flags: SizeFlags, _refs: &mut mem_dbg::HashMap<usize, usize>) -> usize {
        core::mem::size_of_val(self.0)
    }
}

impl MemDbgImpl for CodeWrapper<'_> {
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
        use core::fmt::Write;

        if prefix.len() > max_depth {
            return Ok(());
        }

        let real_size = self.mem_size(flags.to_size_flags());
        let mut buffer = String::new();

        if flags.contains(DbgFlags::HUMANIZE) {
            let (value, uom) = mem_dbg::humanize_float(real_size);
            if uom == " B" {
                write!(buffer, "{:>4}{}", value as usize, uom)?;
            } else {
                write!(buffer, "{:>4.2}{}", value, uom)?;
            }
        } else {
            write!(buffer, "{:>9}", real_size)?;
        }

        if flags.contains(DbgFlags::PERCENTAGE) {
            let percentage = 100.0 * real_size as f64 / total_size as f64;
            write!(buffer, " {:>6.2}%", percentage)?;
        }

        write!(writer, "{}", buffer)?;
        write!(writer, " {} {}", prefix, if is_last { "╰" } else { "├" })?;

        if let Some(name) = field_name {
            write!(writer, "{}", name)?;
        }

        // Print the Debug format of the enum with type_color() when TYPE_NAME is set.
        if flags.contains(DbgFlags::TYPE_NAME) {
            if flags.contains(DbgFlags::COLOR) {
                write!(writer, "{}", mem_dbg::type_color())?;
            }
            write!(writer, ": {:?}", self.0)?;
            if flags.contains(DbgFlags::COLOR) {
                write!(writer, "{}", mem_dbg::reset_color())?;
            }
        }

        let padding = padded_size - core::mem::size_of_val(self.0);
        if padding != 0 {
            write!(writer, " [{}B]", padding)?;
        }

        writeln!(writer)?;
        Ok(())
    }

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

impl<T: Storable, E: Endianness, B: AsRef<[u64]> + MemDbgImpl + FlatType> MemDbgImpl for SeqVec<T, E, B> {
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
        self.bit_offsets._mem_dbg_depth_on(
            writer,
            total_size,
            max_depth,
            prefix,
            Some("bit_offsets"),
            false,
            core::mem::size_of_val(&self.bit_offsets),
            flags,
            _dbg_refs,
        )?;

        if let Some(lengths) = &self.seq_lengths {
            lengths._mem_dbg_depth_on(
                writer,
                total_size,
                max_depth,
                prefix,
                Some("seq_lengths"),
                false,
                core::mem::size_of_val(lengths),
                flags,
                _dbg_refs,
            )?;
        }

        let code_wrapper = CodeWrapper(&self.encoding);
        code_wrapper._mem_dbg_depth_on(
            writer,
            total_size,
            max_depth,
            prefix,
            Some("encoding"),
            false,
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
            true,
            core::mem::size_of_val(&self._markers),
            flags,
            _dbg_refs,
        )?;
        Ok(())
    }
}

// --- PartialEq Implementation ---

impl<T: Storable + PartialEq, E: Endianness, B: AsRef<[u64]>> PartialEq for SeqVec<T, E, B>
where
    for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn eq(&self, other: &Self) -> bool {
        // Quick check: same number of sequences?
        if self.num_sequences() != other.num_sequences() {
            return false;
        }

        // Compare all sequences element-by-element
        for i in 0..self.num_sequences() {
            // SAFETY: i < num_sequences() by loop invariant
            let self_iter = unsafe { self.get_unchecked(i) };
            let other_iter = unsafe { other.get_unchecked(i) };

            if self_iter.ne(other_iter) {
                return false;
            }
        }

        true
    }
}

// --- decode_many Implementation ---

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVec<T, E, B>
where
    for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Retrieves multiple sequences by their indices.
    ///
    /// This method decodes the requested sequences in sorted order for efficient
    /// sequential access to the bitstream, then returns them in the order
    /// corresponding to the input indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - A slice of sequence indices to retrieve.
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<Vec<T>>)` containing the sequences in the order of `indices`.
    /// - `Err(SeqVecError::IndexOutOfBounds(idx))` if any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[
    ///     &[1, 2, 3],
    ///     &[10, 20],
    ///     &[100, 200, 300],
    ///     &[1000],
    /// ];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// let indices = [3, 0, 2];
    /// let sequences = vec.decode_many(&indices)?;
    /// assert_eq!(sequences, vec![
    ///     vec![1000],
    ///     vec![1, 2, 3],
    ///     vec![100, 200, 300],
    /// ]);
    /// #     Ok(())
    /// # }
    /// ```
    pub fn decode_many(&self, indices: &[usize]) -> Result<Vec<Vec<T>>, SeqVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        // Bounds checking
        for &index in indices {
            if index >= self.num_sequences() {
                return Err(SeqVecError::IndexOutOfBounds(index));
            }
        }

        // SAFETY: We have just performed the bounds checks.
        Ok(unsafe { self.decode_many_unchecked(indices) })
    }

    /// Retrieves multiple sequences without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with any out-of-bounds index is undefined behavior.
    pub unsafe fn decode_many_unchecked(&self, indices: &[usize]) -> Vec<Vec<T>> {
        if indices.is_empty() {
            return Vec::new();
        }

        // Build indexed pairs: (sequence_index, original_position_in_results)
        let mut indexed_indices: Vec<(usize, usize)> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();

        // Sort by sequence index to enable more sequential bitstream access.
        indexed_indices.sort_unstable_by_key(|&(idx, _)| idx);

        // Pre-allocate result vectors. Use exact lengths when available,
        // otherwise estimate capacity from bit ranges (~4 bits per element).
        let mut results: Vec<Vec<T>> = if let Some(ref lengths) = self.seq_lengths {
            indices
                .iter()
                .map(|&idx| {
                    // SAFETY: bounds were checked by the caller.
                    let len = unsafe { lengths.get_unchecked(idx) as usize };
                    Vec::with_capacity(len)
                })
                .collect()
        } else {
            indices
                .iter()
                .map(|&idx| {
                    let start = unsafe { self.sequence_start_bit_unchecked(idx) };
                    let end = unsafe { self.sequence_end_bit_unchecked(idx) };
                    let cap = ((end - start) / 4).max(1) as usize;
                    Vec::with_capacity(cap)
                })
                .collect()
        };

        // Decode in sorted order for compressed data locality.
        let mut reader = self.reader();

        for &(target_index, original_pos) in &indexed_indices {
            let output = &mut results[original_pos];
            let _ = reader.decode_into(target_index, output);
        }

        results
    }

    /// Decodes multiple sequences into a caller-provided output vector.
    ///
    /// The output vector is cleared and resized to match `indices.len()`. Each
    /// sequence is decoded into its corresponding slot, maintaining the order
    /// specified by `indices`. This is more efficient than calling
    /// [`decode_vec`](Self::decode_vec) repeatedly, as the decoding process
    /// traverses the compressed data in sorted order for better cache locality.
    ///
    /// # Errors
    ///
    /// Returns [`SeqVecError::IndexOutOfBounds`] if any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[
    ///     &[1, 2],
    ///     &[10, 20],
    ///     &[100, 200, 300],
    /// ];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences)?;
    ///
    /// let indices = [2, 0, 1];
    /// let mut output = Vec::new();
    /// vec.decode_many_into(&indices, &mut output)?;
    ///
    /// assert_eq!(output, vec![
    ///     vec![100, 200, 300],
    ///     vec![1, 2],
    ///     vec![10, 20],
    /// ]);
    /// #     Ok(())
    /// # }
    /// ```
    pub fn decode_many_into(
        &self,
        indices: &[usize],
        output: &mut Vec<Vec<T>>,
    ) -> Result<(), SeqVecError> {
        if indices.is_empty() {
            output.clear();
            return Ok(());
        }

        for &index in indices {
            if index >= self.num_sequences() {
                return Err(SeqVecError::IndexOutOfBounds(index));
            }
        }

        output.clear();
        output.resize_with(indices.len(), Vec::new);

        // SAFETY: We have just performed the bounds checks.
        unsafe { self.decode_many_into_unchecked(indices, output.as_mut_slice()) };
        Ok(())
    }

    /// Decodes multiple sequences into a caller-provided output slice without
    /// bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with any out-of-bounds index is undefined behavior.
    pub unsafe fn decode_many_into_unchecked(&self, indices: &[usize], output: &mut [Vec<T>]) {
        debug_assert_eq!(indices.len(), output.len());

        if indices.is_empty() {
            return;
        }

        // Build indexed pairs: (sequence_index, original_position_in_output)
        let mut indexed_indices: Vec<(usize, usize)> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();

        // Sort by sequence index to enable more sequential bitstream access.
        indexed_indices.sort_unstable_by_key(|&(idx, _)| idx);

        // Pre-allocate capacities. Use exact lengths when available,
        // otherwise estimate from bit ranges (~4 bits per element).
        if let Some(ref lengths) = self.seq_lengths {
            for (i, &idx) in indices.iter().enumerate() {
                // SAFETY: bounds were checked by the caller.
                let len = unsafe { lengths.get_unchecked(idx) as usize };
                output[i].reserve(len);
            }
        } else {
            for (i, &idx) in indices.iter().enumerate() {
                let start = unsafe { self.sequence_start_bit_unchecked(idx) };
                let end = unsafe { self.sequence_end_bit_unchecked(idx) };
                let cap = ((end - start) / 4).max(1) as usize;
                output[i].reserve(cap);
            }
        }

        // Decode in sorted order for compressed data locality.
        let mut reader = self.reader();

        for &(target_index, original_pos) in &indexed_indices {
            let output_slot = &mut output[original_pos];
            let _ = reader.decode_into(target_index, output_slot);
        }
    }
}

// --- IntoIterator ---

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> IntoIterator for &'a SeqVec<T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = SeqIter<'a, T, E>;
    type IntoIter = SeqVecIter<'a, T, E, B>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Storable + 'static, E: Endianness + 'static> IntoIterator for SeqVec<T, E, Vec<u64>>
where
    for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = SeqIter<'static, T, E>;
    type IntoIter = SeqVecIntoIter<T, E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        SeqVecIntoIter::new(self)
    }
}
