//! A reader for efficient, repeated random access into a [`SeqVec`].
//!
//! This module provides [`SeqVecReader`], a reusable, stateful reader designed
//! to provide a convenient interface for performing multiple random sequence
//! lookups with optimized reader reuse.
//!
//! # Stateful Design
//!
//! [`SeqVecReader`] maintains an internal bitstream reader and codec dispatcher,
//! enabling efficient reuse across multiple sequence accesses. This design mirrors
//! [`IntVecReader`](crate::variable::IntVecReader) in the `variable` module.
//!
//! - **`get()`**: Returns a fresh [`SeqIter`] for lazy decoding (unavoidable due
//!   to Rust's borrowing rules — the iterator must own its reader).
//! - **`get_into()`**: Decodes directly into a buffer using the internal reader,
//!   avoiding iterator overhead.
//!
//! # Comparison with `SeqVecSeqReader`
//!
//! | Reader | Stateful | `get()` | `get_into()` | Use Case |
//! |--------|----------|---------|-------------|----------|
//! | `SeqVecReader` | Yes (always seeks) | Fresh `SeqIter` | Reuses reader | **Random access** |
//! | `SeqVecSeqReader` | Yes (may skip seeks) | Fresh `SeqIter` | Reuses reader + position | **Sequential access** |
//!
//! [`SeqVec`]: crate::seq::SeqVec
//! [`SeqIter`]: crate::seq::SeqIter

use super::{iter::SeqVecBitReader, SeqIter, SeqVec};
use crate::common::codec_reader::{CodecReader, IntVecBitReader};
use crate::variable::traits::Storable;
use dsi_bitstream::{
    dispatch::{CodesRead, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};

/// A stateful reader for a `SeqVec` that provides convenient random sequence
/// access with optimized reader reuse.
///
/// This reader is created by the [`SeqVec::reader`](super::SeqVec::reader)
/// method. It provides a convenient interface for performing multiple random
/// sequence lookups, with internal reader reuse for efficiency.
///
/// ## Design Rationale
///
/// Unlike the stateless [`SeqVec`] accessors, `SeqVecReader` maintains an
/// internal [`IntVecBitReader`] and [`CodecReader`] that are reused across
/// multiple accesses. This design mirrors [`IntVecReader`](crate::variable::IntVecReader)
/// in the `variable` module.
///
/// The distinction between `get()` and `get_into()` reflects the constraints
/// of Rust's borrowing rules:
///
/// - **`get()`**: Returns a [`SeqIter`] that owns its own bitstream reader.
///   This allows lazy decoding and multiple iterators to coexist, but does not
///   benefit from reader reuse.
///
/// - **`get_into()`**: Decodes directly into a provided buffer using the
///   internal reader. This bypasses [`SeqIter`] creation and reuses the reader,
///   providing better performance for buffer-filling patterns.
///
/// # Examples
///
/// ```
/// use compressed_intvec::seq::{SeqVec, USeqVec};
///
/// let sequences: &[&[u32]] = &[
///     &[10, 20, 30],
///     &[100, 200],
///     &[1000, 2000, 3000, 4000],
/// ];
/// let vec: USeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
///
/// // Create a reusable reader
/// let mut reader = vec.reader();
///
/// // Perform multiple random reads with optimized get_into()
/// let mut buffer = Vec::new();
/// reader.get_into(2, &mut buffer).unwrap();
/// assert_eq!(buffer, vec![1000, 2000, 3000, 4000]);
///
/// // Or use get() for lazy iteration
/// let seq0: Vec<u32> = reader.get(0).unwrap().collect();
/// assert_eq!(seq0, vec![10, 20, 30]);
/// ```
pub struct SeqVecReader<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// A reference to the parent `SeqVec`.
    seqvec: &'a SeqVec<T, E, B>,
    /// The reusable bitstream reader for decoding sequences.
    reader: IntVecBitReader<'a, E>,
    /// The hybrid codec reader for efficient element decoding.
    code_reader: CodecReader<'a, E>,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVecReader<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new `SeqVecReader`.
    #[inline]
    pub(super) fn new(seqvec: &'a SeqVec<T, E, B>) -> Self {
        let reader = IntVecBitReader::new(dsi_bitstream::impls::MemWordReader::new(
            seqvec.data.as_ref(),
        ));
        let code_reader = CodecReader::new(seqvec.encoding);
        Self {
            seqvec,
            reader,
            code_reader,
        }
    }

    /// Retrieves an iterator over the sequence at `index`, or `None` if out of
    /// bounds.
    ///
    /// This method performs a bounds check and then creates a [`SeqIter`] that
    /// will decode the sequence lazily. The iterator owns its own bitstream
    /// reader, so multiple iterators can exist simultaneously.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u64]] = &[&[1, 2, 3], &[10, 20]];
    /// let vec: LESeqVec<u64> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let mut reader = vec.reader();
    /// assert_eq!(reader.get(1).unwrap().sum::<u64>(), 30);
    /// assert!(reader.get(2).is_none()); // Out of bounds
    /// ```
    #[inline]
    pub fn get(&self, index: usize) -> Option<SeqIter<'a, T, E>> {
        if index >= self.seqvec.num_sequences() {
            return None;
        }
        // SAFETY: The bounds check has been performed.
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Retrieves an iterator over the sequence at `index` without bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    /// The caller must ensure that `index < self.seqvec.num_sequences()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[5, 10, 15], &[20, 25]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let mut reader = vec.reader();
    /// let seq: Vec<u32> = unsafe { reader.get_unchecked(0) }.collect();
    /// assert_eq!(seq, vec![5, 10, 15]);
    /// ```
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> SeqIter<'a, T, E> {
        debug_assert!(
            index < self.seqvec.num_sequences(),
            "Index out of bounds: index was {} but length was {}",
            index,
            self.seqvec.num_sequences()
        );

        // Retrieve bit boundaries for this sequence. SAFETY: Caller guarantees
        // that `index` is in bounds, which implies that `index` and `index + 1`
        // are valid indices into the bit_offsets vector (which has N+1 elements).
        let start_bit = self.seqvec.sequence_start_bit_unchecked(index);
        let end_bit = self.seqvec.sequence_end_bit_unchecked(index);

        // Create a fresh iterator for this sequence. The iterator owns its own
        // bitstream reader, so it can outlive this function call.
        SeqIter::new(
            self.seqvec.data.as_ref(),
            start_bit,
            end_bit,
            self.seqvec.encoding,
        )
    }

    /// Retrieves the sequence at `index` as a `Vec<T>`, or `None` if out of
    /// bounds.
    ///
    /// This is a convenience method that allocates a new vector and collects all
    /// elements from the sequence iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let mut reader = vec.reader();
    /// assert_eq!(reader.get_vec(0), Some(vec![1, 2, 3]));
    /// assert_eq!(reader.get_vec(2), None);
    /// ```
    #[inline]
    pub fn get_vec(&self, index: usize) -> Option<Vec<T>> {
        self.get(index).map(|iter| {
            // Estimate capacity: assume ~4 bits per element on average
            // (reasonable for common codecs like Gamma, Delta, Rice).
            let bit_range = unsafe {
                self.seqvec.sequence_end_bit_unchecked(index)
                    - self.seqvec.sequence_start_bit_unchecked(index)
            };
            let estimated_capacity = (bit_range / 4).max(1) as usize;
            let mut buf = Vec::with_capacity(estimated_capacity);
            buf.extend(iter);
            buf
        })
    }

    /// Retrieves the sequence at `index` into the provided buffer, returning the
    /// number of elements decoded.
    ///
    /// The buffer is cleared before decoding. This method is useful for reusing
    /// allocations across multiple sequence retrievals.
    ///
    /// Returns `None` if `index` is out of bounds.
    ///
    /// This implementation reuses the internal bitstream reader and codec
    /// dispatcher, avoiding the overhead of creating a temporary [`SeqIter`].
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20, 30, 40]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let mut reader = vec.reader();
    /// let mut buffer = Vec::new();
    ///
    /// // Decode first sequence
    /// let count = reader.get_into(0, &mut buffer).unwrap();
    /// assert_eq!(count, 3);
    /// assert_eq!(buffer, vec![1, 2, 3]);
    ///
    /// // Reuse buffer for second sequence
    /// let count = reader.get_into(1, &mut buffer).unwrap();
    /// assert_eq!(count, 4);
    /// assert_eq!(buffer, vec![10, 20, 30, 40]);
    /// ```
    #[inline]
    pub fn get_into(&mut self, index: usize, buf: &mut Vec<T>) -> Option<usize> {
        if index >= self.seqvec.num_sequences() {
            return None;
        }

        // SAFETY: Bounds check has been performed.
        let start_bit = unsafe { self.seqvec.sequence_start_bit_unchecked(index) };
        let end_bit = unsafe { self.seqvec.sequence_end_bit_unchecked(index) };

        buf.clear();

        // Always seek to the start position (random access pattern).
        let _ = self.reader.set_bit_pos(start_bit);

        // Decode all elements in the sequence using the reusable reader and codec dispatcher.
        // Track current position separately since we can't rely on reader.bit_pos().
        let mut current_pos = start_bit;
        while current_pos < end_bit {
            let word = self.code_reader.read(&mut self.reader).unwrap();
            buf.push(T::from_word(word));
            // Update position estimate after read
            current_pos = self.reader.bit_pos().unwrap_or(current_pos);
        }

        Some(buf.len())
    }
}
