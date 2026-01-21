//! Parallel operations for [`SeqVec`].
//!
//! This module provides parallel implementations for [`SeqVec`] operations,
//! enabled by the `parallel` feature flag. These methods are built on the
//! [Rayon] library and are designed to leverage multi-core architectures to
//! accelerate sequence retrieval and decompression.
//!
//! # API Overview
//!
//! | Method | Allocation | Returns | Best For |
//! |--------|------------|---------|----------|
//! | [`par_iter`] | `Vec<T>` per sequence | `impl ParallelIterator<Item = Vec<T>>` | Retaining decoded sequences |
//! | [`par_for_each`] | None | `Vec<R>` | Consumptive ops (sum, count, fold) |
//! | [`par_into_vecs`] | `Vec<T>` per sequence | `Vec<Vec<T>>` | Bulk decode with ownership transfer |
//! | [`par_decode_many`] | `Vec<T>` per index | `Vec<Vec<T>>` | Sparse random access |
//!
//! # Performance Considerations
//!
//! Parallel iteration introduces thread dispatch overhead and reduces cache
//! locality compared to sequential iteration. For small datasets or very fast codecs, sequential [`iter`](super::SeqVec::iter)
//! is typically faster.
//!
//! # SeqVec-Specific Methods
//!
//! The [`par_for_each`] family of methods is unique to [`SeqVec`] and does not
//! exist in [`FixedVec`](crate::fixed::FixedVec) or [`VarVec`](crate::variable::VarVec).
//! This is because:
//!
//! - `FixedVec::par_iter()` and `VarVec::par_iter()` yield `T` directly, so
//!   standard Rayon combinators (`.map()`, `.for_each()`) provide zero-allocation
//!   processing.
//! - `SeqVec::par_iter()` must yield `Vec<T>` (materialized sequences) due to
//!   Rayon's ownership model. The `par_for_each` family provides a zero-allocation
//!   path by passing [`SeqIter`] directly to the closure.
//!
//! [Rayon]: https://github.com/rayon-rs/rayon
//! [`SeqVec`]: super::SeqVec
//! [`SeqIter`]: super::iter::SeqIter
//! [`par_iter`]: super::SeqVec::par_iter
//! [`par_for_each`]: super::SeqVec::par_for_each
//! [`par_into_vecs`]: super::SeqVec::par_into_vecs
//! [`par_decode_many`]: super::SeqVec::par_decode_many

use super::iter::SeqIter;
use super::{SeqVec, SeqVecBitReader, SeqVecError};
use crate::variable::traits::Storable;
use dsi_bitstream::dispatch::CodesRead;
use dsi_bitstream::prelude::{BitRead, BitSeek, Endianness};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

#[cfg(feature = "parallel")]
impl<T, E, B> SeqVec<T, E, B>
where
    T: Storable + Send + Sync,
    E: Endianness + Send + Sync,
    B: AsRef<[u64]> + Send + Sync,
    for<'a> SeqVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>
        + Send,
{
    /// Returns a parallel iterator over all sequences, materializing each as `Vec<T>`.
    ///
    /// This method uses Rayon to decompress and collect all sequences in
    /// parallel. Each sequence is fully decoded into a `Vec<T>` by its
    /// assigned thread before being yielded.
    ///
    /// # Performance
    ///
    /// Parallelization is beneficial when the dataset contains enough sequences and those
    /// sequences are sufficiently large to amortize thread overhead.
    ///
    /// For consumptive operations (sum, count, fold) where the decoded data
    /// is not retained, prefer [`par_for_each`](Self::par_for_each) which
    /// avoids allocation overhead.
    ///
    /// For small datasets or very fast codecs, the sequential [`iter`](Self::iter)
    /// method may be faster due to better cache locality.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    /// use rayon::prelude::*;
    ///
    /// let sequences: &[&[u32]] = &[
    ///     &[1, 2, 3],
    ///     &[10, 20],
    ///     &[100, 200, 300],
    /// ];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// // Collect all sequences in parallel
    /// let all_sequences: Vec<Vec<u32>> = vec.par_iter().collect();
    ///
    /// assert_eq!(all_sequences.len(), 3);
    /// assert_eq!(all_sequences[0], vec![1, 2, 3]);
    /// # }
    /// ```
    pub fn par_iter(&self) -> impl ParallelIterator<Item = Vec<T>> + '_ {
        let num_sequences = self.num_sequences();

        (0..num_sequences).into_par_iter().map_init(
            || self.reader(),
            move |reader, i| {
                // Pre-allocate buffer when sequence length is known.
                let capacity = self
                    .seq_lengths
                    .as_ref()
                    .map(|l| unsafe { l.get_unchecked(i) })
                    .unwrap_or(0);

                let mut buf = Vec::with_capacity(capacity);
                reader.decode_into(i, &mut buf).unwrap();
                buf
            },
        )
    }

    /// Applies a function to each sequence in parallel without materialization.
    ///
    /// Unlike [`par_iter`](Self::par_iter), this method does not allocate a
    /// `Vec<T>` for each sequence. Instead, the closure receives a streaming
    /// [`SeqIter`] directly, enabling zero-allocation parallel processing.
    ///
    /// # Performance
    ///
    /// This method is optimal when:
    /// - The operation is purely consumptive (fold, sum, count, predicate check)
    /// - Memory allocation overhead is significant relative to decode time
    /// - Sequences are short enough that materialization cost matters
    ///
    /// For operations that need to retain sequence data (collect, sort, store),
    /// use [`par_iter`](Self::par_iter) instead.
    ///
    /// # Type Parameters
    ///
    /// - `F`: Closure type that processes a [`SeqIter`] and produces a result
    /// - `R`: Result type produced by the closure, must be [`Send`]
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// // Sum each sequence without allocating intermediate Vecs
    /// let sums: Vec<u64> = vec.par_for_each(|seq| seq.map(|v| v as u64).sum());
    /// assert_eq!(sums, vec![6, 30, 100]);
    ///
    /// // Count elements per sequence
    /// let counts: Vec<usize> = vec.par_for_each(|seq| seq.count());
    /// assert_eq!(counts, vec![3, 2, 1]);
    ///
    /// // Check if any sequence contains a value > 50
    /// let has_large: Vec<bool> = vec.par_for_each(|mut seq| seq.any(|v| v > 50));
    /// assert_eq!(has_large, vec![false, false, true]);
    /// # }
    /// ```
    ///
    /// # Comparison with [`par_iter`](Self::par_iter)
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    /// use rayon::prelude::*;
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// // par_iter: allocates Vec<T> per sequence, then sums
    /// let sums_alloc: Vec<u64> = vec
    ///     .par_iter()
    ///     .map(|s| s.iter().map(|&v| v as u64).sum())
    ///     .collect();
    ///
    /// // par_for_each: zero allocation, sums directly from iterator
    /// let sums_noalloc: Vec<u64> = vec.par_for_each(|seq| seq.map(|v| v as u64).sum());
    ///
    /// assert_eq!(sums_alloc, sums_noalloc);
    /// # }
    /// ```
    pub fn par_for_each<F, R>(&self, f: F) -> Vec<R>
    where
        F: Fn(SeqIter<'_, T, E>) -> R + Sync + Send,
        R: Send,
    {
        let num_sequences = self.num_sequences();
        let data = self.data.as_ref();
        let bit_offsets = &self.bit_offsets;
        let seq_lengths = self.seq_lengths.as_ref();
        let encoding = self.encoding;

        (0..num_sequences)
            .into_par_iter()
            .map(|i| {
                // SAFETY: i < num_sequences by construction of the range.
                let start_bit = unsafe { bit_offsets.get_unchecked(i) };
                let end_bit = unsafe { bit_offsets.get_unchecked(i + 1) };
                let len = seq_lengths.map(|l| unsafe { l.get_unchecked(i) });

                let iter = SeqIter::new_with_len(data, start_bit, end_bit, encoding, len);
                f(iter)
            })
            .collect()
    }

    /// Applies a function to each sequence in parallel and reduces results.
    ///
    /// This is a convenience method combining [`par_for_each`](Self::par_for_each)
    /// with a parallel reduction. Useful when the final result is a single
    /// aggregated value rather than a collection.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// // Total sum across all sequences
    /// let total: u64 = vec.par_for_each_reduce(
    ///     |seq| seq.map(|v| v as u64).sum::<u64>(),
    ///     || 0u64,
    ///     |a, b| a + b,
    /// );
    /// assert_eq!(total, 136); // 6 + 30 + 100
    ///
    /// // Maximum element across all sequences
    /// let max: u32 = vec.par_for_each_reduce(
    ///     |seq| seq.max().unwrap_or(0),
    ///     || 0u32,
    ///     |a, b| a.max(b),
    /// );
    /// assert_eq!(max, 100);
    /// # }
    /// ```
    pub fn par_for_each_reduce<F, R, ID, OP>(&self, f: F, identity: ID, op: OP) -> R
    where
        F: Fn(SeqIter<'_, T, E>) -> R + Sync + Send,
        R: Send,
        ID: Fn() -> R + Sync + Send,
        OP: Fn(R, R) -> R + Sync + Send,
    {
        let num_sequences = self.num_sequences();
        let data = self.data.as_ref();
        let bit_offsets = &self.bit_offsets;
        let seq_lengths = self.seq_lengths.as_ref();
        let encoding = self.encoding;

        (0..num_sequences)
            .into_par_iter()
            .map(|i| {
                let start_bit = unsafe { bit_offsets.get_unchecked(i) };
                let end_bit = unsafe { bit_offsets.get_unchecked(i + 1) };
                let len = seq_lengths.map(|l| unsafe { l.get_unchecked(i) });

                let iter = SeqIter::new_with_len(data, start_bit, end_bit, encoding, len);
                f(iter)
            })
            .reduce(identity, op)
    }

    /// Consumes the [`SeqVec`] and decodes all sequences into separate vectors
    /// in parallel.
    ///
    /// This method is a parallel version of [`into_vecs`](super::SeqVec::into_vecs),
    /// leveraging Rayon to decompress multiple sequences concurrently. Each
    /// sequence is fully decompressed by its assigned thread.
    ///
    /// # Performance
    ///
    /// Parallelization is beneficial when:
    /// - The dataset is large enough to amortize thread overhead
    /// - Sequences are reasonably sized
    ///
    /// For small datasets or very fast codecs, the sequential [`into_vecs`]
    /// method may be faster due to better cache locality.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[
    ///     &[1, 2, 3],
    ///     &[10, 20],
    ///     &[100, 200, 300],
    /// ];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// // Decode all sequences in parallel
    /// let all_sequences: Vec<Vec<u32>> = vec.par_into_vecs();
    ///
    /// assert_eq!(all_sequences.len(), 3);
    /// assert_eq!(all_sequences[0], vec![1, 2, 3]);
    /// assert_eq!(all_sequences[1], vec![10, 20]);
    /// # }
    /// ```
    ///
    /// [`into_vecs`]: super::SeqVec::into_vecs
    pub fn par_into_vecs(self) -> Vec<Vec<T>> {
        let num_sequences = self.num_sequences();
        let seqvec = &self;

        (0..num_sequences)
            .into_par_iter()
            .map_init(
                || seqvec.reader(),
                move |reader, i| {
                    // Pre-allocate buffer when sequence length is known.
                    let capacity = seqvec
                        .seq_lengths
                        .as_ref()
                        .map(|l| unsafe { l.get_unchecked(i) })
                        .unwrap_or(0);

                    let mut buf = Vec::with_capacity(capacity);
                    reader.decode_into(i, &mut buf).unwrap();
                    buf
                },
            )
            .collect()
    }

    /// Retrieves multiple sequences in parallel.
    ///
    /// This method uses Rayon to parallelize the retrieval of multiple sequences
    /// by index. It is particularly useful when accessing a large subset of
    /// sequences that are not contiguous.
    ///
    /// # Errors
    ///
    /// Returns [`SeqVecError::IndexOutOfBounds`] if any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[
    ///     &[1, 2, 3],
    ///     &[10, 20],
    ///     &[100, 200, 300],
    ///     &[1000],
    /// ];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let indices = [3, 0, 2];
    /// let sequences = vec.par_decode_many(&indices).unwrap();
    /// assert_eq!(sequences.len(), 3);
    /// assert_eq!(sequences[0], vec![1000]);  // Index 3
    /// assert_eq!(sequences[1], vec![1, 2, 3]); // Index 0
    /// assert_eq!(sequences[2], vec![100, 200, 300]); // Index 2
    /// # }
    /// ```
    pub fn par_decode_many(&self, indices: &[usize]) -> Result<Vec<Vec<T>>, SeqVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        let num_sequences = self.num_sequences();

        // Bounds checking
        for &index in indices {
            if index >= num_sequences {
                return Err(SeqVecError::IndexOutOfBounds(index));
            }
        }

        // SAFETY: We have pre-checked the bounds of all indices.
        Ok(unsafe { self.par_decode_many_unchecked(indices) })
    }

    /// Retrieves multiple sequences in parallel without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with any out-of-bounds index in the `indices` slice
    /// is undefined behavior. In debug builds, assertions will panic.
    pub unsafe fn par_decode_many_unchecked(&self, indices: &[usize]) -> Vec<Vec<T>> {
        #[cfg(debug_assertions)]
        {
            let num_sequences = self.num_sequences();
            for &index in indices {
                debug_assert!(
                    index < num_sequences,
                    "Index out of bounds: index was {} but num_sequences was {}",
                    index,
                    num_sequences
                );
            }
        }

        if indices.is_empty() {
            return Vec::new();
        }

        let mut results = vec![Vec::new(); indices.len()];

        results.par_iter_mut().enumerate().for_each_init(
            || self.reader(),
            |reader, (original_pos, result)| {
                let target_index = indices[original_pos];

                // Pre-allocate when sequence length is known.
                if let Some(lengths) = &self.seq_lengths {
                    let capacity = unsafe { lengths.get_unchecked(target_index) };
                    result.reserve(capacity);
                }

                // SAFETY: bounds are guaranteed by the caller.
                reader.decode_into(target_index, result).unwrap();
            },
        );

        results
    }

    /// Applies a function to selected sequences in parallel without materialization.
    ///
    /// This is the sparse-access equivalent of [`par_for_each`](Self::par_for_each),
    /// allowing zero-allocation processing of a subset of sequences by index.
    ///
    /// # Errors
    ///
    /// Returns [`SeqVecError::IndexOutOfBounds`] if any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100], &[1000, 2000]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// // Sum only sequences at indices 0 and 2
    /// let sums = vec.par_for_each_many(&[0, 2], |seq| seq.map(|v| v as u64).sum::<u64>()).unwrap();
    /// assert_eq!(sums, vec![6, 100]);
    /// # }
    /// ```
    pub fn par_for_each_many<F, R>(&self, indices: &[usize], f: F) -> Result<Vec<R>, SeqVecError>
    where
        F: Fn(SeqIter<'_, T, E>) -> R + Sync + Send,
        R: Send,
    {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        let num_sequences = self.num_sequences();

        // Bounds checking
        for &index in indices {
            if index >= num_sequences {
                return Err(SeqVecError::IndexOutOfBounds(index));
            }
        }

        let data = self.data.as_ref();
        let bit_offsets = &self.bit_offsets;
        let seq_lengths = self.seq_lengths.as_ref();
        let encoding = self.encoding;

        let results = indices
            .par_iter()
            .map(|&i| {
                // SAFETY: bounds checked above.
                let start_bit = unsafe { bit_offsets.get_unchecked(i) };
                let end_bit = unsafe { bit_offsets.get_unchecked(i + 1) };
                let len = seq_lengths.map(|l| unsafe { l.get_unchecked(i) });

                let iter = SeqIter::new_with_len(data, start_bit, end_bit, encoding, len);
                f(iter)
            })
            .collect();

        Ok(results)
    }
}

