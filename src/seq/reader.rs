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
//! [`VarVecReader`](crate::variable::VarVecReader) in the `variable` module.
//!
//! - **`decode_into()`**: Decodes directly into a buffer using the internal reader,
//!   avoiding iterator overhead.
//!
//! [`SeqVec`]: crate::seq::SeqVec

use super::{SeqVec, iter::SeqVecBitReader};
use crate::common::codec_reader::{CodecReader, VarVecBitReader};
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
/// Unlike the stateless [`crate::seq::SeqVec`] accessors, `SeqVecReader` maintains an
/// internal `VarVecBitReader` and `CodecReader` that are reused across
/// multiple accesses. This design mirrors [`VarVecReader`](crate::variable::VarVecReader)
/// in the `variable` module.
///
/// The reader exposes only stateful, allocation-aware APIs that benefit from
/// internal reader reuse. For lazy iteration, use [`SeqVec::get`](crate::seq::SeqVec::get)
/// directly.
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
/// // Perform multiple random reads with optimized decode_into()
/// let mut buffer = Vec::new();
/// reader.decode_into(2, &mut buffer).unwrap();
/// assert_eq!(buffer, vec![1000, 2000, 3000, 4000]);
///
/// // Or use SeqVec::get() for lazy iteration
/// let seq0: Vec<u32> = vec.get(0).unwrap().collect();
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
    reader: VarVecBitReader<'a, E>,
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
        let reader = VarVecBitReader::new(dsi_bitstream::impls::MemWordReader::new(
            seqvec.data.as_ref(),
        ));
        let code_reader = CodecReader::new(seqvec.encoding);
        Self {
            seqvec,
            reader,
            code_reader,
        }
    }

    /// Retrieves the sequence at `index` as a `Vec<T>`, or `None` if out of
    /// bounds.
    ///
    /// This method reuses the internal bitstream reader and codec dispatcher,
    /// providing better performance than collecting a [`SeqVec::get`]
    /// iterator into a vector. For optimal memory allocation, ensure the builder option
    /// [`SeqVecBuilder::store_lengths`](crate::seq::SeqVecBuilder::store_lengths)
    /// was used during construction.
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
    /// assert_eq!(reader.decode_vec(0), Some(vec![1, 2, 3]));
    /// assert_eq!(reader.decode_vec(2), None);
    /// ```
    #[inline]
    pub fn decode_vec(&mut self, index: usize) -> Option<Vec<T>> {
        let mut buf = Vec::new();
        self.decode_into(index, &mut buf).map(|_| buf)
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
    /// dispatcher, avoiding the overhead of creating a temporary iterator.
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
    /// let count = reader.decode_into(0, &mut buffer).unwrap();
    /// assert_eq!(count, 3);
    /// assert_eq!(buffer, vec![1, 2, 3]);
    ///
    /// // Reuse buffer for second sequence
    /// let count = reader.decode_into(1, &mut buffer).unwrap();
    /// assert_eq!(count, 4);
    /// assert_eq!(buffer, vec![10, 20, 30, 40]);
    /// ```
    #[inline]
    pub fn decode_into(&mut self, index: usize, buf: &mut Vec<T>) -> Option<usize> {
        if index >= self.seqvec.num_sequences() {
            return None;
        }

        // SAFETY: Bounds check has been performed.
        Some(unsafe { self.decode_into_unchecked(index, buf) })
    }

    /// Retrieves the sequence at `index` into the provided buffer without
    /// bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    #[inline]
    pub unsafe fn decode_into_unchecked(&mut self, index: usize, buf: &mut Vec<T>) -> usize {
        let start_bit = unsafe { self.seqvec.sequence_start_bit_unchecked(index) };

        buf.clear();

        // Always seek to the start position (random access pattern).
        let _ = self.reader.set_bit_pos(start_bit);

        if let Some(lengths) = &self.seqvec.seq_lengths {
            let count = unsafe { lengths.get_unchecked(index) };
            buf.reserve(count);
            for _ in 0..count {
                let word = self.code_reader.read(&mut self.reader).unwrap();
                buf.push(T::from_word(word));
            }
        } else {
            let end_bit = unsafe { self.seqvec.sequence_end_bit_unchecked(index) };

            // Decode all elements in the sequence until we reach the end boundary.
            // For VarVecBitReader backed by MemWordReader, bit_pos() is infallible.
            while self.reader.bit_pos().unwrap() < end_bit {
                let word = self.code_reader.read(&mut self.reader).unwrap();
                buf.push(T::from_word(word));
            }
        }

        buf.len()
    }
}
