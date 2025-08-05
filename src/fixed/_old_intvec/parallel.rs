//! # `FixedVec` Parallel Implementations
//!
//! This module provides parallel implementations for [`FixedVec`] operations,
//! enabled by the `parallel` feature flag. These methods leverage the [Rayon]
//! library to accelerate data decompression and access.
//!
//! [Rayon]: https://docs.rs/rayon/latest/rayon/

use super::{FixedVec, FixedVecError};
use dsi_bitstream::prelude::Endianness;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

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
    ///
    /// This method is designed for maximum throughput on large batches of
    /// randomly distributed indices on multi-core systems. It uses a simple and
    /// effective "embarrassingly parallel" strategy: the input `indices` slice
    /// is partitioned, and each thread performs lookups for its assigned partition
    /// independently.
    ///
    /// This approach avoids the high cost of sorting and is ideal for workloads
    /// where index access patterns are unpredictable.
    ///
    /// In debug builds, this method will panic if any index is out of bounds.
    ///
    /// # Safety
    /// Calling this method with any out-of-bounds index is undefined behavior
    /// in release builds.
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

        results
            .par_iter_mut()
            .enumerate()
            .for_each(|(original_pos, res_val)| {
                // Each thread performs a scalar lookup for its assigned indices.
                // The `get_unchecked` call is thread-safe as it is read-only.
                *res_val = self.get_unchecked(indices[original_pos]);
            });

        results
    }
}
