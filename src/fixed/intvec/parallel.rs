//! # `FixedVec` Parallel Implementations
//!
//! This module provides parallel implementations for [`FixedVec`] operations,
//! enabled by the `parallel` feature flag. These methods leverage the [Rayon]
//! library to accelerate data decompression and access.
//!
//! For `FixedVec`, access is an O(1) arithmetic operation, making parallelization
//! extremely effective and scalable ("embarrassingly parallel").
//!
//! [Rayon]: https://docs.rs/rayon/latest/rayon/

use super::{FixedVec, FixedVecError};
use dsi_bitstream::prelude::Endianness;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[cfg(feature = "parallel")]
impl<E> FixedVec<E>
where
    E: Endianness + Send + Sync,
{
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
    ///
    /// This method parallelizes the lookups over the provided `indices` slice.
    /// Each lookup is an independent O(1) operation.
    ///
    // # Returns
    /// A `Result` containing a `Vec<u64>` with the retrieved values, or a
    /// [`FixedVecError`] if any index is out of bounds.
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
    /// In debug builds, this method will panic if any index is out of bounds.
    ///
    /// # Safety
    ///
    /// Calling this method with any out-of-bounds index is undefined behavior in release builds.
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

        let mut results = vec![0; indices.len()];
        results.par_iter_mut().enumerate().for_each(|(i, val)| {
            // SAFETY: The caller guarantees that the index is in bounds (or we debug_asserted it).
            *val = self.get_unchecked(indices[i]);
        });
        results
    }
}
