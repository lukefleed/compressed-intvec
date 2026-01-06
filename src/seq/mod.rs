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
//! concatenated into a single compressed bitstream, with a [`FixedVec`] index
//! storing the bit offset of each sequence's start.
//!
//! ## Sequence Length
//!
//! Sequence lengths are **not stored explicitly**. The iterator for a sequence
//! terminates when the current bit position reaches the start of the next
//! sequence. This means:
//!
//! - Retrieving a sequence is O(length) for decoding — unavoidable.
//! - Computing sequence length requires full iteration. If O(1) length queries
//!   are critical, consider caching lengths externally.
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
//! ```
//!
//! ## Custom Codec
//!
//! ```ignore
//! use compressed_intvec::seq::{SeqVec, LESeqVec, VariableCodecSpec};
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
//! [`FixedVec`]: crate::fixed::FixedVec
//! [`dsi-bitstream`]: https://crates.io/crates/dsi-bitstream

mod builder;
mod iter;

pub use builder::{SeqVecBuilder, SeqVecFromIterBuilder};
pub use iter::{SeqIter, SeqVecIter};

// Re-export codec spec for convenience.
pub use crate::variable::codec::VariableCodecSpec;

use crate::fixed::FixedVec;
use crate::variable::traits::Storable;
use dsi_bitstream::{
    dispatch::{Codes, CodesRead},
    impls::{BufBitWriter, MemWordWriterVec},
    prelude::{BitRead, BitSeek, BitWrite, CodesWrite, Endianness, BE, LE},
};
use iter::SeqVecBitReader;
use mem_dbg::{DbgFlags, MemDbgImpl, MemSize, SizeFlags};
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
/// Signed integers are transparently encoded using zig-zag encoding via the
/// [`Storable`] trait.
pub type SSeqVec<T, B = Vec<u64>> = SeqVec<T, LE, B>;

// --- MemSize Implementation ---

impl<T: Storable, E: Endianness, B: AsRef<[u64]> + MemSize> MemSize for SeqVec<T, E, B> {
    fn mem_size(&self, flags: SizeFlags) -> usize {
        let mut total = core::mem::size_of::<Self>();
        // Add heap-allocated memory for the data buffer.
        total += self.data.mem_size(flags) - core::mem::size_of::<B>();
        // Add heap-allocated memory for the bit_offsets index.
        total +=
            self.bit_offsets.mem_size(flags) - core::mem::size_of::<FixedVec<u64, u64, LE, B>>();
        total
    }
}

// --- MemDbgImpl Implementation ---

// Wrapper for Codes to provide correct MemDbgImpl, following the pattern in
// variable::IntVec. This is necessary because the derived implementation for
// Codes is incorrect and cannot be fixed due to the orphan rule.
struct CodeWrapper<'a>(&'a Codes);

impl MemSize for CodeWrapper<'_> {
    fn mem_size(&self, _flags: SizeFlags) -> usize {
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
    ) -> core::fmt::Result {
        use core::fmt::Write;

        if prefix.len() > max_depth {
            return Ok(());
        }

        let real_size = self.mem_size(flags.to_size_flags());
        let mut buffer = String::new();

        if flags.contains(DbgFlags::HUMANIZE) {
            let (value, uom) = mem_dbg::humanize_float(real_size as f64);
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
            if flags.contains(DbgFlags::COLOR) {
                write!(writer, "{}", mem_dbg::yellow())?;
            }
            write!(writer, "{}", name)?;
            if flags.contains(DbgFlags::COLOR) {
                write!(writer, "{}", mem_dbg::reset_color())?;
            }
            write!(writer, ": {:?}", self.0)?;
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
    ) -> core::fmt::Result {
        Ok(())
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

// --- Construction (Owned) ---

impl<T: Storable + 'static, E: Endianness> SeqVec<T, E, Vec<u64>> {
    /// Creates a [`SeqVec`] from a slice of slices using default settings.
    ///
    /// This method uses [`VariableCodecSpec::Auto`] to select an optimal codec
    /// based on the data distribution.
    ///
    /// # Errors
    ///
    /// Returns a [`SeqVecError`] if codec resolution or encoding fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// assert_eq!(vec.num_sequences(), 3);
    /// ```
    pub fn from_slices<S: AsRef<[T]>>(sequences: &[S]) -> Result<Self, SeqVecError>
    where
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        Self::builder()
            .codec(VariableCodecSpec::Auto)
            .build(sequences)
    }

    /// Consumes the [`SeqVec`] and returns all sequences as a `Vec<Vec<T>>`.
    ///
    /// This method decodes all data, allocating a new vector for each sequence.
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
    /// * `bit_offsets_data` contains valid monotonically increasing bit positions.
    /// * The sentinel value does not exceed the total bits in `data`.
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

        let bit_offsets = FixedVec::<u64, u64, LE, B>::from_parts(
            bit_offsets_data,
            bit_offsets_len,
            bit_offsets_num_bits,
        )?;

        Ok(Self {
            data,
            bit_offsets,
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
}

// --- Query Methods ---

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVec<T, E, B> {
    /// Returns the number of sequences stored.
    ///
    /// This is O(1) as it is derived from the bit offsets index length.
    #[inline]
    pub fn num_sequences(&self) -> usize {
        // bit_offsets has N+1 entries for N sequences.
        self.bit_offsets.len().saturating_sub(1)
    }

    /// Returns `true` if there are no sequences stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_sequences() == 0
    }

    /// Returns the compression codec used for encoding.
    #[inline]
    pub fn encoding(&self) -> Codes {
        self.encoding
    }

    /// Returns a reference to the underlying compressed data buffer.
    #[inline]
    pub fn as_limbs(&self) -> &[u64] {
        self.data.as_ref()
    }

    /// Returns a reference to the bit offsets index.
    #[inline]
    pub fn bit_offsets_ref(&self) -> &FixedVec<u64, u64, LE, B> {
        &self.bit_offsets
    }

    /// Returns the total number of bits in the compressed data.
    ///
    /// This is the sentinel value at the end of the bit offsets index.
    #[inline]
    pub fn total_bits(&self) -> u64 {
        if self.bit_offsets.is_empty() {
            0
        } else {
            // SAFETY: We verified the index is not empty.
            unsafe { self.bit_offsets.get_unchecked(self.bit_offsets.len() - 1) }
        }
    }

    /// Returns the bit offset where sequence `index` starts.
    ///
    /// Returns `None` if `index >= num_sequences()`.
    #[inline]
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
    #[inline]
    pub unsafe fn sequence_start_bit_unchecked(&self, index: usize) -> u64 {
        debug_assert!(
            index < self.num_sequences(),
            "index {} out of bounds for {} sequences",
            index,
            self.num_sequences()
        );
        self.bit_offsets.get_unchecked(index)
    }

    /// Returns the bit offset immediately after sequence `index` ends.
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
        self.bit_offsets.get_unchecked(index + 1)
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
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> SeqIter<'_, T, E> {
        debug_assert!(
            index < self.num_sequences(),
            "index {} out of bounds for {} sequences",
            index,
            self.num_sequences()
        );

        let start_bit = self.sequence_start_bit_unchecked(index);
        let end_bit = self.sequence_end_bit_unchecked(index);

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
    /// // Buffer is reused (cleared internally).
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

    /// Returns an iterator over all sequences.
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
