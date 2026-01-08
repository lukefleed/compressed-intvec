//! Parallel operations for [`SeqVec`].
//!
//! This module provides parallel implementations for [`SeqVec`] operations,
//! enabled by the `parallel` feature flag. These methods are built on the
//! [Rayon] library and are designed to leverage multi-core architectures to
//! accelerate sequence retrieval and decompression.
//!
//! [Rayon]: https://github.com/rayon-rs/rayon
//! [`SeqVec`]: super::SeqVec

use super::{SeqVec, SeqVecBitReader, SeqVecError};
use crate::variable::traits::Storable;
use dsi_bitstream::dispatch::CodesRead;
use dsi_bitstream::prelude::{BitRead, BitSeek, Endianness};
use rayon::prelude::*;

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
    /// Returns a parallel iterator over all sequences.
    ///
    /// This method uses Rayon to decompress and iterate over all sequences in
    /// parallel. Each sequence is fully decompressed by its assigned thread.
    ///
    /// # Performance
    ///
    /// Parallelization is beneficial when:
    /// - The dataset is large enough to amortize thread overhead.
    /// - Sequences are reasonably sized.
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
    /// let all_sequences: Vec<Vec<u32>> = vec
    ///     .par_iter()
    ///     .collect();
    ///
    /// assert_eq!(all_sequences.len(), 3);
    /// # }
    /// ```
    pub fn par_iter(&self) -> impl ParallelIterator<Item = Vec<T>> + '_ {
        (0..self.num_sequences()).into_par_iter().map(move |i| {
            // SAFETY: i < num_sequences() by loop invariant
            unsafe { self.get_unchecked(i).collect() }
        })
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
    /// let sequences = vec.par_get_many_sequences(&indices).unwrap();
    /// assert_eq!(sequences.len(), 3);
    /// # }
    /// ```
    pub fn par_get_many_sequences(&self, indices: &[usize]) -> Result<Vec<Vec<T>>, SeqVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        // Bounds checking
        for &index in indices {
            if index >= self.num_sequences() {
                return Err(SeqVecError::IndexOutOfBounds(index));
            }
        }

        // SAFETY: We have pre-checked the bounds of all indices.
        Ok(unsafe { self.par_get_many_sequences_unchecked(indices) })
    }

    /// Retrieves multiple sequences in parallel without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with any out-of-bounds index in the `indices` slice
    /// is undefined behavior. In debug builds, assertions will panic.
    pub unsafe fn par_get_many_sequences_unchecked(&self, indices: &[usize]) -> Vec<Vec<T>> {
        #[cfg(debug_assertions)]
        {
            for &index in indices {
                debug_assert!(
                    index < self.num_sequences(),
                    "Index out of bounds: index was {} but num_sequences was {}",
                    index,
                    self.num_sequences()
                );
            }
        }

        if indices.is_empty() {
            return Vec::new();
        }

        let mut results = vec![Vec::new(); indices.len()];

        results.par_iter_mut().enumerate().for_each_init(
            || self.seq_reader(),
            |reader, (original_pos, result)| {
                let target_index = indices[original_pos];
                // SAFETY: bounds are guaranteed by the caller.
                reader.get_into(target_index, result).unwrap();
            },
        );

        results
    }
}
