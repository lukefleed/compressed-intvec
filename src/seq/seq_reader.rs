//! A stateful reader for efficient, sequential access into a [`SeqVec`].
//!
//! This module provides [`SeqVecSeqReader`], a reusable reader that is
//! specifically optimized for access patterns that are sequential or have a high
//! degree of locality (i.e., sequence indices are strictly increasing or close
//! to each other).
//!
//! # Performance
//!
//! [`SeqVecSeqReader`] maintains internal state for the current decoding position.
//! The primary optimization is in the [`get_into`](SeqVecSeqReader::get_into)
//! method, which:
//!
//! 1. **Reuses the internal bitstream reader** across multiple sequence accesses.
//! 2. **Tracks the current bit position** to avoid unnecessary seeks when accessing
//!    sequences that are contiguous or nearby in the bitstream.
//! 3. **Reuses the codec dispatcher** ([`CodecReader`]) to amortize setup costs.
//!
//! ## Seek Avoidance Optimization
//!
//! The seek-skipping behavior (avoiding `set_bit_pos()` when the reader is
//! already at the target position) provides **negligible benefit in practice**.
//! Benchmark analysis shows:
//!
//! - A seek operation costs ~3 nanoseconds.
//! - Typical decode times range from 45–450 nanoseconds per sequence.
//! - The seek overhead represents **0.7% to 6.7%** of total time, and often
//!   less due to instruction-level parallelism.
//! - Across sequential, clustered, and random access patterns, throughput
//!   differs by **less than 2%** compared to [`SeqVecReader`].
//!
//! This optimization is only meaningfully beneficial for extremely short
//! sequences (1–2 elements), which are atypical in real-world graph or
//! adjacency list workloads.
//!
//! The type is retained for **API consistency** with the `variable` module
//! and as a building block for potential future optimizations that might
//! provide greater benefit.
//!
//! ## Comparison with [`SeqVecReader`]
//!
//! Unlike [`SeqVecReader`] which is stateless and creates a fresh [`SeqIter`]
//! for each access, [`SeqVecSeqReader`] maintains persistent state. The benefit
//! is most pronounced when:
//!
//! - Using [`get_into`](SeqVecSeqReader::get_into) to materialize sequences
//!   into a reusable buffer.
//! - Accessing sequences in index order or with high spatial locality.
//! - The bit offsets of consecutive sequences are close together.
//!
//! For random access with no locality, the performance is similar to
//! [`SeqVecReader`].
//!
//! [`SeqVec`]: crate::seq::SeqVec
//! [`SeqVecReader`]: crate::seq::SeqVecReader
//! [`SeqIter`]: crate::seq::SeqIter
//! [`CodecReader`]: crate::common::codec_reader::CodecReader

use super::{iter::SeqVecBitReader, SeqIter, SeqVec};
use crate::common::codec_reader::{CodecReader, IntVecBitReader};
use crate::variable::traits::Storable;
use dsi_bitstream::{
    dispatch::CodesRead,
    prelude::{BitRead, BitSeek, Endianness},
};

/// A stateful, sequential reader for a [`SeqVec`] optimized for forward access.
///
/// This reader is created by the [`seq_reader`](super::SeqVec::seq_reader)
/// method. It maintains an internal cursor corresponding to the last-accessed
/// sequence's ending position, making it highly efficient for sequential or
/// mostly-forward access patterns.
///
/// ## API Overview
///
/// - [`get`](Self::get): Returns a [`SeqIter`] over the requested sequence.
///   This provides API uniformity but does not benefit from state tracking.
///   Prefer [`get_into`](Self::get_into) for performance-critical paths.
///
/// - [`get_into`](Self::get_into): **Performance-optimized method.** Decodes
///   the sequence directly into a provided buffer, reusing the internal reader
///   and avoiding seeks when possible.
///
/// - [`get_vec`](Self::get_vec): Convenience method that allocates a new
///   `Vec<T>` for each sequence. For repeated access, prefer [`get_into`](Self::get_into)
///   with a reusable buffer.
///
/// ## Performance Characteristics
///
/// When accessing sequences in increasing index order (or nearby indices),
/// [`get_into`](Self::get_into) can avoid the overhead of seeking by continuing
/// from the current reader position. This is particularly effective when:
///
/// - Bit offsets are consecutive or close together.
/// - The reader is already positioned at or near the target sequence.
///
/// For random access patterns, performance degrades gracefully to match
/// [`SeqVecReader`](super::SeqVecReader).
///
/// ## Examples
///
/// ```
/// use compressed_intvec::seq::{SeqVec, LESeqVec};
///
/// let sequences: &[&[u32]] = &[
///     &[1, 2, 3],
///     &[10, 20, 30, 40],
///     &[100],
/// ];
/// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
///
/// // Create a reader optimized for sequential access
/// let mut seq_reader = vec.seq_reader();
/// let mut buffer = Vec::new();
///
/// // Accessing sequences in order is highly efficient
/// seq_reader.get_into(0, &mut buffer).unwrap();
/// assert_eq!(buffer, &[1, 2, 3]);
///
/// seq_reader.get_into(1, &mut buffer).unwrap(); // Continues from previous position
/// assert_eq!(buffer, &[10, 20, 30, 40]);
///
/// seq_reader.get_into(2, &mut buffer).unwrap(); // No seek required
/// assert_eq!(buffer, &[100]);
/// ```
///
/// [`SeqVec`]: crate::seq::SeqVec
/// [`SeqIter`]: crate::seq::SeqIter
pub struct SeqVecSeqReader<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Immutable reference to the parent SeqVec.
    seqvec: &'a SeqVec<T, E, B>,
    /// The stateful, reusable bitstream reader.
    ///
    /// We use `IntVecBitReader` (which is a type alias for the same underlying
    /// reader type as `SeqVecBitReader`) for consistency with the common module.
    reader: IntVecBitReader<'a, E>,
    /// The hybrid dispatcher that handles codec reading robustly.
    code_reader: CodecReader<'a, E>,
    /// The current bit position of the reader. This is used to determine whether
    /// a seek is required or if we can continue decoding forward.
    current_bit_pos: u64,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVecSeqReader<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new `SeqVecSeqReader`.
    pub(super) fn new(seqvec: &'a SeqVec<T, E, B>) -> Self {
        // Instantiate the hybrid dispatcher. This will not panic, as it falls
        // back to a slower but universally compatible method if the codec's
        // parameters are not supported by the fast path.
        let code_reader = CodecReader::new(seqvec.encoding);

        Self {
            seqvec,
            reader: IntVecBitReader::new(dsi_bitstream::impls::MemWordReader::new(
                seqvec.data.as_ref(),
            )),
            code_reader,
            current_bit_pos: 0,
        }
    }

    /// Retrieves an iterator over sequence `index`, or `None` if out of bounds.
    ///
    /// This method provides API uniformity with [`SeqVecReader::get`](super::SeqVecReader::get)
    /// and [`SeqVec::get`](super::SeqVec::get), but **does not benefit** from
    /// the reader's internal state tracking. It creates a fresh [`SeqIter`]
    /// for the sequence, identical to calling [`SeqVec::get`](super::SeqVec::get)
    /// directly.
    ///
    /// For performance-critical sequential access, use [`get_into`](Self::get_into)
    /// instead, which reuses the internal reader and can avoid seeks.
    ///
    /// ## Note on State Usage
    ///
    /// This method does **not** benefit from the reader's internal state. It creates
    /// a fresh [`SeqIter`] with its own reader, equivalent to calling
    /// [`SeqVec::get`](crate::seq::SeqVec::get) directly. The purpose of this method
    /// is to provide API consistency with [`get_into`](Self::get_into). For stateful
    /// access that reuses the internal reader and avoids seeks, use [`get_into`](Self::get_into)
    /// or [`get_vec`](Self::get_vec).
    ///
    /// ## Returns
    ///
    /// - `Some(SeqIter)` if the index is valid.
    /// - `None` if the index is out of bounds.
    ///
    /// ## Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[10, 20, 30], &[100, 200]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let mut reader = vec.seq_reader();
    ///
    /// // Returns a lazy iterator (does not use internal state)
    /// let iter = reader.get(0).unwrap();
    /// assert_eq!(iter.collect::<Vec<_>>(), vec![10, 20, 30]);
    /// ```
    #[inline]
    pub fn get(&mut self, index: usize) -> Option<SeqIter<'a, T, E>> {
        // Delegate to SeqVec's get method. This does not use our internal state,
        // but provides API consistency. For stateful access that reuses the internal
        // reader and avoids seeks, use get_into() instead.
        self.seqvec.get(index)
    }

    /// Reads sequence `index` into `buf`, reusing the internal reader.
    ///
    /// This is the **primary performance-optimized method** of [`SeqVecSeqReader`].
    /// It exploits the reader's state to avoid seeks when accessing sequences
    /// sequentially or with high locality.
    ///
    /// ## Behavior
    ///
    /// 1. Clears the provided buffer.
    /// 2. If the reader is not positioned at the sequence start, seeks to `start_bit`.
    /// 3. Decodes all elements of the sequence into the buffer.
    /// 4. Updates internal state for the next access.
    ///
    /// ## Performance
    ///
    /// When accessing sequences in increasing order (or when the reader is already
    /// positioned at the target), this method can avoid the seek operation entirely.
    /// For example, if accessing sequences `[10, 11, 12]` in order and their
    /// bit offsets are consecutive, only the first access performs a seek.
    ///
    /// ## Returns
    ///
    /// - `Some(n)` where `n` is the number of elements in the sequence.
    /// - `None` if the index is out of bounds.
    ///
    /// ## Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let mut reader = vec.seq_reader();
    /// let mut buffer = Vec::new();
    ///
    /// // Sequential access reuses the reader efficiently
    /// reader.get_into(0, &mut buffer).unwrap();
    /// assert_eq!(buffer, &[1, 2, 3]);
    ///
    /// reader.get_into(1, &mut buffer).unwrap();
    /// assert_eq!(buffer, &[10, 20]);
    ///
    /// reader.get_into(2, &mut buffer).unwrap();
    /// assert_eq!(buffer, &[100]);
    /// ```
    #[inline]
    pub fn get_into(&mut self, index: usize, buf: &mut Vec<T>) -> Option<usize> {
        let start_bit = self.seqvec.sequence_start_bit(index)?;

        // Clear the buffer for fresh data. The allocation is preserved.
        buf.clear();

        // Only seek if we're not already positioned at the sequence start.
        // This is the core optimization: for sequential access, we're often
        // already at the right position (current_bit_pos == start_bit).
        if self.current_bit_pos != start_bit {
            self.reader.set_bit_pos(start_bit).unwrap();
        }

        if let Some(lengths) = &self.seqvec.seq_lengths {
            let count = unsafe { lengths.get_unchecked(index) } as usize;
            buf.reserve(count);
            for _ in 0..count {
                let word = self.code_reader.read(&mut self.reader).unwrap();
                buf.push(T::from_word(word));
            }
        } else {
            // SAFETY: If start_bit is Some, then index is valid, so index+1 is a
            // valid index into bit_offsets (which has N+1 elements).
            let end_bit = unsafe { self.seqvec.sequence_end_bit_unchecked(index) };

            // Hot loop: Decode elements until we reach the sequence boundary.
            // Use a local variable for position tracking to enable LLVM to keep
            // the comparison in registers. The loop condition becomes a simple
            // register-to-register comparison, minimizing pipeline pressure.
            let mut current_pos = start_bit;
            while current_pos < end_bit {
                let word = self.code_reader.read(&mut self.reader).unwrap();
                buf.push(T::from_word(word));
                current_pos = self.reader.bit_pos().unwrap();
            }
            self.current_bit_pos = current_pos;
        }

        // For sequences with stored lengths, update current_bit_pos after decoding.
        if self.seqvec.seq_lengths.is_none() {
            // current_bit_pos was already updated in the else branch above.
        } else {
            self.current_bit_pos = self.reader.bit_pos().unwrap();
        }

        Some(buf.len())
    }

    /// Convenience: returns sequence `index` as a newly allocated `Vec<T>`.
    ///
    /// This method provides a simpler API than [`get_into`](Self::get_into),
    /// but allocates a fresh vector for each call. For repeated access with
    /// high performance requirements, prefer [`get_into`](Self::get_into) with
    /// a reusable buffer to avoid allocations.
    ///
    /// ## Returns
    ///
    /// - `Some(Vec<T>)` containing the sequence elements.
    /// - `None` if the index is out of bounds.
    ///
    /// ## Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let mut reader = vec.seq_reader();
    ///
    /// let seq0 = reader.get_vec(0).unwrap();
    /// assert_eq!(seq0, vec![1, 2, 3]);
    ///
    /// let seq1 = reader.get_vec(1).unwrap();
    /// assert_eq!(seq1, vec![10, 20]);
    /// ```
    #[inline]
    pub fn get_vec(&mut self, index: usize) -> Option<Vec<T>> {
        // Perform single bounds check upfront.
        let start_bit = self.seqvec.sequence_start_bit(index)?;

        // Estimate capacity based on bit span. Assume at least 8 bits per element
        // on average. Common variable-length codecs (gamma, delta, rice) typically
        // use 8-16 bits per element for moderate values in adjacency lists.
        let end_bit = unsafe { self.seqvec.sequence_end_bit_unchecked(index) };
        let bit_count = (end_bit - start_bit) as usize;
        let estimated_capacity = (bit_count / 8).max(1);

        let mut buf = Vec::with_capacity(estimated_capacity);
        buf.clear(); // Ensure clean state (though vec is fresh).

        // Only seek if not already positioned at the sequence start.
        if self.current_bit_pos != start_bit {
            self.reader.set_bit_pos(start_bit).unwrap();
        }

        if let Some(lengths) = &self.seqvec.seq_lengths {
            let count = unsafe { lengths.get_unchecked(index) } as usize;
            buf.reserve(count);
            for _ in 0..count {
                let word = self.code_reader.read(&mut self.reader).unwrap();
                buf.push(T::from_word(word));
            }
        } else {
            // Inline decoding logic with pre-computed offsets to avoid redundant
            // bounds check that would occur if we called get_into().
            let mut current_pos = start_bit;
            while current_pos < end_bit {
                let word = self.code_reader.read(&mut self.reader).unwrap();
                buf.push(T::from_word(word));
                current_pos = self.reader.bit_pos().unwrap();
            }
            self.current_bit_pos = current_pos;
        }

        // Update position if lengths were stored.
        if self.seqvec.seq_lengths.is_some() {
            self.current_bit_pos = self.reader.bit_pos().unwrap();
        }

        Some(buf)
    }
}
