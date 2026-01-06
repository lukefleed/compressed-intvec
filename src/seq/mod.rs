//! A compressed vector of variable-length sequences with indexed access.
//!
//! This module provides [`SeqVec`], a data structure for storing a collection
//! of integer sequences in a compressed format. Each sequence is accessed by
//! its index (rank), and all elements within a sequence are decoded together.
//!
//! # Core Concepts
//!
//! ## Use Case
//!
//! [`SeqVec`] is designed for scenarios where:
//! - You have many small sequences that you want to store compactly.
//! - You always access an entire sequence at a time, not individual elements.
//! - You want to avoid the overhead of separate allocations and padding for
//!   each sequence.
//!
//! A typical application is representing **adjacency lists** in a compressed
//! graph, where each node's neighbors form a sequence.
//!
//! ## Differences from [`IntVec`]
//!
//! | Aspect | [`IntVec`] | [`SeqVec`] |
//! |--------|-----------|------------|
//! | Access unit | Single element | Entire sequence |
//! | Index meaning | Element position | Sequence rank |
//! | Sampling | Periodic (every k elements) | At sequence boundaries |
//! | Primary operation | `get(i) → T` | `get(i) → Iterator<T>` |
//!
//! ## Compression
//!
//! Like [`IntVec`], [`SeqVec`] uses instantaneous variable-length codes (Gamma,
//! Delta, Zeta, etc.) from the [`dsi-bitstream`] crate. All sequences are
//! concatenated into a single compressed bitstream, with a separate index
//! storing the bit offset of each sequence's start.
//!
//! ## Random Access
//!
//! Accessing sequence `i` is O(1) for locating the start (lookup in the bit
//! offsets index) plus O(length) for decoding the elements. The length of a
//! sequence is not stored explicitly; instead, the iterator reads until it
//! reaches the bit offset of the next sequence.
//!
//! # Main Components
//!
//! - [`SeqVec`]: The core compressed sequence vector.
//! - [`SeqVecBuilder`]: Builder for constructing a [`SeqVec`] with custom codec.
//! - [`SeqIter`]: Zero-allocation iterator over elements of a single sequence.
//! - [`SeqVecReader`]: Reusable reader for efficient repeated access.
//! - [`SeqVecSeqReader`]: Stateful reader optimized for sequential access patterns.
//! - [`SeqVecSlice`]: Zero-copy view over a subset of sequences.
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```ignore
//! use compressed_intvec::seq::{SeqVec, LESeqVec};
//!
//! let sequences: &[&[u32]] = &[
//!     &[1, 2, 3],
//!     &[10, 20],
//!     &[100, 200, 300, 400],
//!     &[], // Empty sequences are supported
//! ];
//!
//! let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
//!
//! assert_eq!(vec.num_sequences(), 4);
//!
//! // Access a sequence by index
//! let seq1: Vec<u32> = vec.get(1).unwrap().collect();
//! assert_eq!(seq1, vec![10, 20]);
//!
//! // Iterate over all sequences
//! for (i, seq_iter) in vec.iter().enumerate() {
//!     println!("Sequence {}: {:?}", i, seq_iter.collect::<Vec<_>>());
//! }
//! ```
//!
//! ## Custom Codec
//!
//! ```ignore
//! use compressed_intvec::seq::{SeqVec, LESeqVec};
//! use compressed_intvec::variable::VariableCodecSpec;
//!
//! let sequences: Vec<Vec<u64>> = vec![
//!     vec![1, 1, 1, 2, 3],
//!     vec![100, 200, 300],
//! ];
//!
//! let vec: LESeqVec<u64> = SeqVec::builder()
//!     .codec(VariableCodecSpec::Zeta { k: Some(3) })
//!     .build(&sequences)
//!     .unwrap();
//! ```
//!
//! [`IntVec`]: crate::variable::IntVec
//! [`dsi-bitstream`]: https://crates.io/crates/dsi-bitstream

pub mod builder;
pub mod error;
pub mod iter;
pub mod reader;
pub mod seq_reader;
pub mod slice;

pub use builder::{SeqVecBuilder, SeqVecFromIterBuilder};
pub use error::SeqVecError;
pub use iter::{SeqIter, SeqVecIter};
pub use reader::SeqVecReader;
pub use seq_reader::SeqVecSeqReader;
pub use slice::SeqVecSlice;

use crate::fixed::FixedVec;
use crate::variable::traits::Storable;
use dsi_bitstream::{
    dispatch::{Codes, CodesRead},
    prelude::{BitRead, BitSeek, Endianness, BE, LE},
};
use iter::SeqVecBitReader;
use mem_dbg::{DbgFlags, MemDbgImpl, MemSize, SizeFlags};
use std::marker::PhantomData;

/// A compressed, indexed vector of integer sequences.
///
/// `SeqVec` stores multiple sequences of integers in a single compressed
/// bitstream, with an auxiliary index for O(1) access to each sequence by
/// its rank. This is ideal for representing collections of variable-length
/// sequences (e.g., adjacency lists) with minimal memory overhead.
///
/// See the [module-level documentation](self) for detailed usage information.
///
/// # Type Parameters
///
/// - `T`: The element type (e.g., `u32`, `i16`). Must implement [`Storable`].
/// - `E`: The [`Endianness`] of the underlying bitstream (e.g., [`LE`] or [`BE`]).
/// - `B`: The backing buffer type, enabling owned (`Vec<u64>`) or borrowed
///   (`&[u64]`) storage for zero-copy operations.
#[derive(Debug, Clone)]
pub struct SeqVec<T: Storable, E: Endianness, B: AsRef<[u64]> = Vec<u64>> {
    /// The compressed bitstream containing all sequences concatenated.
    data: B,
    /// Bit offset where each sequence starts. Contains N+1 elements where N is
    /// the number of sequences; the last element is the total number of bits
    /// (sentinel value for computing the last sequence's extent).
    bit_offsets: FixedVec<u64, u64, LE, B>,
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
pub type USeqVec<T, B = Vec<u64>> = SeqVec<T, LE, B>;

/// A [`SeqVec`] for signed integers with little-endian bit ordering.
///
/// Signed integers are transparently encoded using zig-zag encoding.
pub type SSeqVec<T, B = Vec<u64>> = SeqVec<T, LE, B>;

// --- MemSize and MemDbgImpl ---

impl<T: Storable, E: Endianness, B: AsRef<[u64]> + MemSize> MemSize for SeqVec<T, E, B> {
    fn mem_size(&self, flags: SizeFlags) -> usize {
        let mut total = core::mem::size_of::<Self>();
        total += self.data.mem_size(flags) - core::mem::size_of::<B>();
        total +=
            self.bit_offsets.mem_size(flags) - core::mem::size_of::<FixedVec<u64, u64, LE, B>>();
        total
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]> + MemDbgImpl> MemDbgImpl for SeqVec<T, E, B> {
    fn _mem_dbg_rec_on(
        &self,
        writer: &mut impl core::fmt::Write,
        total_size: usize,
        max_depth: usize,
        prefix: &mut String,
        _is_last: bool,
        flags: DbgFlags,
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
        )?;
        Ok(())
    }
}

// --- Construction ---

impl<T: Storable + 'static, E: Endianness> SeqVec<T, E, Vec<u64>> {
    /// Creates a builder for constructing a [`SeqVec`] with custom settings.
    ///
    /// This is the most flexible way to create a [`SeqVec`], allowing
    /// customization of the compression codec.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    /// use compressed_intvec::variable::VariableCodecSpec;
    ///
    /// let sequences = vec![vec![1u32, 2, 3], vec![4, 5]];
    ///
    /// let vec: LESeqVec<u32> = SeqVec::builder()
    ///     .codec(VariableCodecSpec::Delta)
    ///     .build(&sequences)
    ///     .unwrap();
    /// ```
    #[inline]
    pub fn builder() -> SeqVecBuilder<T, E> {
        SeqVecBuilder::new()
    }

    /// Creates a [`SeqVec`] from a slice of slices using default settings.
    ///
    /// This is a convenience method that uses [`VariableCodecSpec::Auto`] to
    /// automatically select the best compression codec.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[4, 5], &[]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// assert_eq!(vec.num_sequences(), 3);
    /// ```
    pub fn from_slices<S>(sequences: &[S]) -> Result<Self, SeqVecError>
    where
        S: AsRef<[T]>,
        for<'a> crate::variable::IntVecBitWriter<E>: dsi_bitstream::prelude::BitWrite<E, Error = core::convert::Infallible>
            + dsi_bitstream::prelude::CodesWrite<E>,
    {
        Self::builder().build(sequences)
    }
}

// --- Core Methods (applicable to all backends) ---

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVec<T, E, B> {
    /// Creates a new [`SeqVec`] from its raw components.
    ///
    /// This constructor is intended for advanced use cases such as memory-mapping
    /// a pre-built [`SeqVec`] from disk without copying data.
    ///
    /// # Errors
    ///
    /// Returns [`SeqVecError::InvalidParameters`] if the `bit_offsets` vector
    /// has fewer than 2 elements (at minimum, we need a start and end offset).
    ///
    /// # Safety Considerations
    ///
    /// The caller must ensure that:
    /// - The `bit_offsets` contain valid, monotonically non-decreasing offsets.
    /// - The `data` buffer contains properly encoded data matching the `encoding`.
    /// - The final element of `bit_offsets` does not exceed `data.len() * 64`.
    pub fn from_parts(
        data: B,
        bit_offsets_data: B,
        bit_offsets_len: usize,
        bit_offsets_bits: usize,
        encoding: Codes,
    ) -> Result<Self, SeqVecError> {
        let bit_offsets = FixedVec::<u64, u64, LE, B>::from_parts(
            bit_offsets_data,
            bit_offsets_len,
            bit_offsets_bits,
        )?;

        if bit_offsets.len() < 1 {
            return Err(SeqVecError::InvalidParameters(
                "bit_offsets must have at least 1 element (sentinel)".to_string(),
            ));
        }

        Ok(Self {
            data,
            bit_offsets,
            encoding,
            _markers: PhantomData,
        })
    }

    /// Creates a new [`SeqVec`] from raw parts without validation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that all parameters are internally consistent.
    /// Invalid parameters will lead to panics or incorrect data retrieval.
    #[inline]
    pub(crate) unsafe fn new_unchecked(
        data: B,
        bit_offsets: FixedVec<u64, u64, LE, B>,
        encoding: Codes,
    ) -> Self {
        Self {
            data,
            bit_offsets,
            encoding,
            _markers: PhantomData,
        }
    }

    /// Returns the number of sequences stored in this vector.
    ///
    /// This is O(1).
    #[inline]
    pub fn num_sequences(&self) -> usize {
        // bit_offsets has N+1 elements for N sequences.
        self.bit_offsets.len().saturating_sub(1)
    }

    /// Returns `true` if this vector contains no sequences.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_sequences() == 0
    }

    /// Returns the compression codec used for this vector.
    #[inline]
    pub fn encoding(&self) -> Codes {
        self.encoding
    }

    /// Returns a reference to the raw compressed data buffer.
    #[inline]
    pub fn as_limbs(&self) -> &[u64] {
        self.data.as_ref()
    }

    /// Returns a reference to the bit offsets index.
    #[inline]
    pub fn bit_offsets_ref(&self) -> &FixedVec<u64, u64, LE, B> {
        &self.bit_offsets
    }

    /// Returns the bit offset where sequence `index` starts.
    ///
    /// Returns `None` if `index >= num_sequences()`.
    #[inline]
    pub fn sequence_start_bit(&self, index: usize) -> Option<u64> {
        if index >= self.num_sequences() {
            return None;
        }
        Some(unsafe { self.bit_offsets.get_unchecked(index) })
    }

    /// Returns the bit offset where sequence `index` ends (exclusive).
    ///
    /// Returns `None` if `index >= num_sequences()`.
    #[inline]
    pub fn sequence_end_bit(&self, index: usize) -> Option<u64> {
        if index >= self.num_sequences() {
            return None;
        }
        Some(unsafe { self.bit_offsets.get_unchecked(index + 1) })
    }

    /// Returns the total number of bits used for compressed data.
    ///
    /// This is the sentinel value at the end of the bit offsets index.
    #[inline]
    pub fn total_bits(&self) -> u64 {
        if self.bit_offsets.is_empty() {
            0
        } else {
            unsafe { self.bit_offsets.get_unchecked(self.bit_offsets.len() - 1) }
        }
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
    /// # Performance
    ///
    /// This method creates a new bitstream reader on each call. For repeated
    /// access, consider using [`reader()`](Self::reader) or
    /// [`seq_reader()`](Self::seq_reader) instead.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let first: Vec<u32> = vec.get(0).unwrap().collect();
    /// assert_eq!(first, vec![1, 2, 3]);
    /// ```
    #[inline]
    pub fn get(&self, index: usize) -> Option<SeqIter<'_, T, E, B>> {
        if index >= self.num_sequences() {
            return None;
        }
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Returns an iterator over the elements of sequence `index` without
    /// bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with `index >= num_sequences()` is undefined behavior.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> SeqIter<'_, T, E, B> {
        debug_assert!(
            index < self.num_sequences(),
            "index {} out of bounds for {} sequences",
            index,
            self.num_sequences()
        );

        let start_bit = self.bit_offsets.get_unchecked(index);
        let end_bit = self.bit_offsets.get_unchecked(index + 1);

        SeqIter::new(self.data.as_ref(), start_bit, end_bit, self.encoding)
    }

    /// Returns the elements of sequence `index` as a newly allocated `Vec`.
    ///
    /// Returns `None` if `index >= num_sequences()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[10, 20, 30]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// assert_eq!(vec.get_vec(0), Some(vec![10, 20, 30]));
    /// assert_eq!(vec.get_vec(1), None);
    /// ```
    #[inline]
    pub fn get_vec(&self, index: usize) -> Option<Vec<T>> {
        self.get(index).map(|iter| iter.collect())
    }

    /// Decodes sequence `index` into the provided buffer.
    ///
    /// The buffer is cleared before use. Returns the number of elements
    /// decoded, or `None` if `index >= num_sequences()`.
    ///
    /// This method is more efficient than [`get_vec`](Self::get_vec) when
    /// reusing a buffer across multiple calls.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let mut buf = Vec::new();
    /// assert_eq!(vec.get_into(0, &mut buf), Some(2));
    /// assert_eq!(buf, vec![1, 2]);
    ///
    /// assert_eq!(vec.get_into(1, &mut buf), Some(3));
    /// assert_eq!(buf, vec![3, 4, 5]);
    /// ```
    #[inline]
    pub fn get_into(&self, index: usize, buf: &mut Vec<T>) -> Option<usize> {
        let iter = self.get(index)?;
        buf.clear();
        buf.extend(iter);
        Some(buf.len())
    }

    /// Returns an iterator over all sequences in this vector.
    ///
    /// Each element of the returned iterator is a [`SeqIter`] for the
    /// corresponding sequence.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3], &[4, 5, 6]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// for (i, seq) in vec.iter().enumerate() {
    ///     println!("Sequence {}: {:?}", i, seq.collect::<Vec<_>>());
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> SeqVecIter<'_, T, E, B> {
        SeqVecIter::new(
            self.data.as_ref(),
            &self.bit_offsets,
            self.encoding,
            self.num_sequences(),
        )
    }

    /// Creates a reusable, stateless reader for efficient repeated access.
    ///
    /// The reader maintains an internal bitstream reader that is reused across
    /// calls, avoiding the overhead of creating a new reader for each access.
    /// However, each access performs an independent seek operation.
    ///
    /// Use this when accessing sequences in an unpredictable order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4], &[5, 6]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let mut reader = vec.reader();
    /// assert_eq!(reader.get_vec(2), Some(vec![5, 6]));
    /// assert_eq!(reader.get_vec(0), Some(vec![1, 2]));
    /// ```
    #[inline]
    pub fn reader(&self) -> SeqVecReader<'_, T, E, B> {
        SeqVecReader::new(self)
    }

    /// Creates a stateful reader optimized for sequential access patterns.
    ///
    /// This reader tracks its current position and can decode forward without
    /// seeking when accessing consecutive or nearby sequences. This is more
    /// efficient than [`reader()`](Self::reader) when sequences are accessed
    /// in increasing order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4], &[5, 6]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let mut reader = vec.seq_reader();
    /// // Sequential access is optimized
    /// assert_eq!(reader.get_vec(0), Some(vec![1, 2]));
    /// assert_eq!(reader.get_vec(1), Some(vec![3, 4])); // No seek needed
    /// assert_eq!(reader.get_vec(2), Some(vec![5, 6])); // No seek needed
    /// ```
    #[inline]
    pub fn seq_reader(&self) -> SeqVecSeqReader<'_, T, E, B> {
        SeqVecSeqReader::new(self)
    }

    /// Creates an immutable view over a range of sequences.
    ///
    /// Returns `None` if `start + len > num_sequences()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1], &[2], &[3], &[4], &[5]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let slice = vec.slice(1, 3).unwrap(); // Sequences 1, 2, 3
    /// assert_eq!(slice.num_sequences(), 3);
    /// assert_eq!(slice.get_vec(0), Some(vec![2])); // Original index 1
    /// ```
    #[inline]
    pub fn slice(&self, start: usize, len: usize) -> Option<SeqVecSlice<'_, T, E, B>> {
        if start.saturating_add(len) > self.num_sequences() {
            return None;
        }
        Some(SeqVecSlice::new(self, start, len))
    }

    /// Splits this vector into two non-overlapping slices at the given index.
    ///
    /// Returns `None` if `mid > num_sequences()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1], &[2], &[3], &[4]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let (left, right) = vec.split_at(2).unwrap();
    /// assert_eq!(left.num_sequences(), 2);  // Sequences 0, 1
    /// assert_eq!(right.num_sequences(), 2); // Sequences 2, 3
    /// ```
    #[inline]
    pub fn split_at(
        &self,
        mid: usize,
    ) -> Option<(SeqVecSlice<'_, T, E, B>, SeqVecSlice<'_, T, E, B>)> {
        if mid > self.num_sequences() {
            return None;
        }
        let left = SeqVecSlice::new(self, 0, mid);
        let right = SeqVecSlice::new(self, mid, self.num_sequences() - mid);
        Some((left, right))
    }
}

// --- IntoIterator ---

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> IntoIterator for &'a SeqVec<T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = SeqIter<'a, T, E, B>;
    type IntoIter = SeqVecIter<'a, T, E, B>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
