//! Parallel operations for [`VarVec`].
//!
//! This module provides parallel implementations for [`VarVec`] operations,
//! enabled by the `parallel` feature flag. These methods are built on the
//! [Rayon] library and are designed to leverage multi-core architectures to
//! accelerate data decompression and access.
//!
//! [Rayon]: https://github.com/rayon-rs/rayon
//! [`VarVec`]: crate::variable::VarVec

use super::{VarVec, VarVecBitReader, VarVecError, traits::Storable};
use dsi_bitstream::{
    dispatch::{CodesRead, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};
use rayon::prelude::{IntoParallelIterator, ParallelIterator, ParallelSlice};

impl<T, E, B> VarVec<T, E, B>
where
    T: Storable + Send + Sync,
    E: Endianness + Send + Sync,
    B: AsRef<[u64]> + Send + Sync,
    for<'a> VarVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>
        + Send,
{
    /// Returns a parallel iterator over the decompressed values.
    ///
    /// This method uses Rayon to decompress the entire vector in parallel. It
    /// can provide a significant speedup on multi-core systems, especially when
    /// using a computationally intensive compression codec.
    ///
    /// # Performance
    ///
    /// For the specific task of full decompression, this parallel version is not
    /// always faster than the sequential [`iter`](super::VarVec::iter). If the
    /// decoding operation is very fast (e.g., with `VByte` encoding), the
    /// operation can be limited by memory bandwidth. In such cases, the
    /// sequential iterator's better use of CPU caches may outperform this
    /// parallel version.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::variable::{VarVec, UVarVec};
    /// use rayon::prelude::*;
    ///
    /// let data: Vec<u32> = (0..1000).collect();
    /// let vec: UVarVec<u32> = VarVec::from_slice(&data)?;
    ///
    /// // Use the parallel iterator to compute the sum in parallel
    /// let sum: u32 = vec.par_iter().sum();
    ///
    /// assert_eq!(sum, (0..1000).sum());
    /// # }
    /// # Ok(())
    /// # }
    /// ```
    pub fn par_iter(&self) -> impl ParallelIterator<Item = T> + '_ {
        let k = self.k;
        let len = self.len;
        let num_samples = self.samples.len();
        let num_threads = rayon::current_num_threads();
        // Divide the sample blocks among the available threads.
        let chunk_size = num_samples.div_ceil(num_threads).max(1);
        let num_chunks = num_samples.div_ceil(chunk_size);
        // Compute once outside the closure to avoid redundant per-chunk checks.
        let k_is_pow2 = k.is_power_of_two();
        let k_exp = k.trailing_zeros();

        (0..num_chunks).into_par_iter().flat_map(move |chunk_idx| {
            use crate::common::codec_reader::CodecReader;

            let start_sample_idx = chunk_idx * chunk_size;
            let end_sample_idx = (start_sample_idx + chunk_size).min(num_samples);

            // Pre-calculate the total number of elements in this chunk to
            // allocate exactly once, avoiding multiple reallocations.
            let chunk_start_elem = if k_is_pow2 {
                start_sample_idx << k_exp
            } else {
                start_sample_idx * k
            };
            let chunk_end_elem = if k_is_pow2 {
                (end_sample_idx << k_exp).min(len)
            } else {
                (end_sample_idx * k).min(len)
            };
            let expected_count = chunk_end_elem - chunk_start_elem;

            let mut bit_reader = VarVecBitReader::<E>::new(
                dsi_bitstream::impls::MemWordReader::new_inf(self.data.as_ref()),
            );
            let mut values = Vec::with_capacity(expected_count);
            let code_reader = CodecReader::new(self.encoding);

            // Each thread decodes its assigned range of sample blocks.
            for sample_idx in start_sample_idx..end_sample_idx {
                let (start_elem_index, end_elem_index) = if k_is_pow2 {
                    (
                        sample_idx << k_exp,
                        ((sample_idx + 1) << k_exp).min(len),
                    )
                } else {
                    (sample_idx * k, ((sample_idx + 1) * k).min(len))
                };

                // SAFETY: `sample_idx` is bounded by `num_samples`, which equals
                // `self.samples.len()`, so the index is valid.
                unsafe {
                    bit_reader
                        .set_bit_pos(self.samples.get_unchecked(sample_idx))
                        .unwrap();
                }

                for _ in start_elem_index..end_elem_index {
                    let word = code_reader.read(&mut bit_reader).unwrap();
                    values.push(Storable::from_word(word));
                }
            }
            values.into_par_iter()
        })
    }

    /// Retrieves multiple elements from a slice of indices in parallel.
    ///
    /// This method uses Rayon to parallelize random access. It works by creating
    /// a separate [`VarVecReader`](super::VarVecReader) for each thread and
    /// distributing the lookup work among them.
    ///
    /// # Errors
    ///
    /// Returns [`VarVecError::IndexOutOfBounds`] if any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::variable::{VarVec, SVarVec};
    ///
    /// let data: Vec<i64> = (0..1000).map(|x| x * -1).collect();
    /// let vec: SVarVec<i64> = VarVec::from_slice(&data)?;
    ///
    /// let indices = [500, 10, 999, 0, 250];
    /// let values = vec.par_get_many(&indices)?;
    ///
    /// assert_eq!(values, vec![-500, -10, -999, 0, -250]);
    /// # }
    /// # Ok(())
    /// # }
    /// ```
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<T>, VarVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        for &index in indices {
            if index >= self.len {
                return Err(VarVecError::IndexOutOfBounds(index));
            }
        }

        // SAFETY: We have pre-checked the bounds of all indices.
        Ok(unsafe { self.par_get_many_unchecked(indices) })
    }

    /// Retrieves multiple elements in parallel without bounds checking.
    ///
    /// This method sorts the indices for monotonic scanning within each thread,
    /// similar to how the sequential [`get_many_unchecked`](super::VarVec::get_many_unchecked)
    /// works but partitioned across threads. This avoids random seeks per
    /// element and exploits sequential decoding locality.
    ///
    /// # Safety
    ///
    /// Calling this method with any out-of-bounds index in the `indices` slice
    /// is undefined behavior. In debug builds, an assertion will panic.
    pub unsafe fn par_get_many_unchecked(&self, indices: &[usize]) -> Vec<T> {
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

        if indices.is_empty() {
            return Vec::new();
        }

        // Create indexed pairs (target_index, original_position) and sort by
        // target_index for monotonic scanning.
        let mut indexed_indices: Vec<(usize, usize)> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();
        indexed_indices.sort_unstable_by_key(|&(idx, _)| idx);

        let mut results: Vec<T> = vec![Storable::from_word(0); indices.len()];

        // Wrapper for concurrent non-overlapping writes using `UnsafeCell`
        // to correctly express interior mutability to the compiler.
        // SAFETY: Each `original_position` is unique (derived from enumerate()),
        // so no two threads write to the same index.
        use std::cell::UnsafeCell;

        struct SyncSlice<'a, T>(UnsafeCell<&'a mut [T]>);
        // SAFETY: Non-overlapping writes from different threads are safe because
        // each thread writes to distinct indices.
        unsafe impl<T: Send> Sync for SyncSlice<'_, T> {}
        impl<T> SyncSlice<'_, T> {
            #[inline]
            unsafe fn write(&self, index: usize, value: T) {
                // SAFETY: The caller guarantees non-overlapping indices.
                // UnsafeCell allows interior mutability without aliasing violations.
                let slice = unsafe { &mut *self.0.get() };
                let ptr = slice.as_mut_ptr();
                unsafe { ptr.add(index).write(value) };
            }
        }

        let sync_results = SyncSlice(UnsafeCell::new(&mut results));

        // Partition the sorted indices among threads. Each thread gets a
        // contiguous chunk of the sorted array, so it scans monotonically.
        let num_threads = rayon::current_num_threads();
        let chunk_size = indexed_indices.len().div_ceil(num_threads).max(1);

        indexed_indices
            .par_chunks(chunk_size)
            .for_each(|chunk| {
                let mut seq_reader = self.seq_reader();
                for &(target_index, original_position) in chunk {
                    // SAFETY: bounds are guaranteed by the caller.
                    let value = unsafe { seq_reader.get_unchecked(target_index) };
                    // SAFETY: `original_position` is a unique index from
                    // 0..indices.len(), so no two threads write the same slot.
                    unsafe { sync_results.write(original_position, value) };
                }
            });

        results
    }
}
