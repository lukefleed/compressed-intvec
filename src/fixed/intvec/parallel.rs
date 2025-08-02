//! # `FixedVec` Parallel Implementations
//!
//! This module provides parallel implementations for [`FixedVec`] operations,
//! enabled by the `parallel` feature flag. These methods leverage the [Rayon]
//! library to accelerate data decompression and access.
//!
//! When the `simd` feature is also enabled, the batch access methods are
//! further accelerated using SIMD instructions for byte-aligned bit-widths
//! (8, 16, 32, 64), combining thread-level and data-level parallelism for
//! maximum throughput.
//!
//! [Rayon]: https://docs.rs/rayon/latest/rayon/
//!
//! [Rayon]: https://docs.rs/rayon/latest/rayon/

use super::{FixedVec, FixedVecError};
use dsi_bitstream::prelude::Endianness;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[cfg(feature = "simd")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

#[cfg(feature = "parallel")]
impl<E: Endianness + Send + Sync> FixedVec<E, Vec<u64>> {
    /// Returns a parallel iterator over the decompressed `u64` values.
    ///
    /// This operation is "embarrassingly parallel" for `FixedVec` and scales
    /// linearly with the number of available cores.
    pub fn par_iter(&self) -> impl ParallelIterator<Item = u64> + '_ {
        (0..self.len)
            .into_par_iter()
            // Each parallel task now calls the highly optimized direct-access get_unchecked.
            .map(move |i| unsafe { self.get_unchecked(i) })
    }

    /// Retrieves multiple elements in parallel.
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<u64>, FixedVecError> {
        for &index in indices {
            if index >= self.len {
                return Err(FixedVecError::IndexOutOfBounds(index));
            }
        }
        // SAFETY: We have pre-checked the bounds of all indices.
        Ok(unsafe { self.par_get_many_unchecked(indices) })
    }

    /// Retrieves multiple elements in parallel without bounds checking.
    pub unsafe fn par_get_many_unchecked(&self, indices: &[usize]) -> Vec<u64> {
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

        let mut results = vec![0; indices.len()];

        #[cfg(feature = "simd")]
        {
            match self.num_bits() {
                8 | 16 | 32 | 64 => {
                    // Create (original_pos, index) pairs to allow reordering.
                    let mut indexed_indices: Vec<(usize, usize)> =
                        indices.iter().copied().enumerate().collect();

                    // Parallel sort by index is efficient and sets up for finding runs.
                    indexed_indices.par_sort_unstable_by_key(|&(_, idx)| idx);

                    // Process in parallel chunks and collect partial results.
                    let partial_results: Vec<Vec<(usize, u64)>> = indexed_indices
                        .par_chunks(4096)
                        .map(|chunk| {
                            let mut chunk_results = Vec::new();
                            let mut i = 0;
                            while i < chunk.len() {
                                let (_original_pos_start, run_start_index) = chunk[i];
                                let mut j = i + 1;
                                while j < chunk.len() && chunk[j].1 == chunk[j - 1].1 + 1 {
                                    j += 1;
                                }
                                let run_len = j - i;
                                let run_slice = &chunk[i..j];

                                if run_len > 4 {
                                    let mut temp_run_results = vec![0; run_len];
                                    unsafe {
                                        super::simd::gather_simd(
                                            self,
                                            run_start_index,
                                            &mut temp_run_results,
                                        );
                                    }
                                    for (k, &(original_pos, _)) in run_slice.iter().enumerate() {
                                        chunk_results.push((original_pos, temp_run_results[k]));
                                    }
                                } else {
                                    for &(original_pos, idx) in run_slice {
                                        chunk_results.push((original_pos, self.get_unchecked(idx)));
                                    }
                                }
                                i = j;
                            }
                            chunk_results
                        })
                        .collect();

                    // Merge partial results back into the results vector.
                    for chunk_results in partial_results {
                        for (original_pos, value) in chunk_results {
                            results[original_pos] = value;
                        }
                    }
                    return results;
                }
                _ => {}
            }
        }

        // Fallback implementation.
        results
            .par_iter_mut()
            .enumerate()
            .for_each(|(original_pos, res_val)| {
                *res_val = self.get_unchecked(indices[original_pos]);
            });
        results
    }
}
