//! # `FixedVec` Parallel Implementations
//!
//! This module provides parallel implementations for [`FixedVec`] operations,
//! enabled by the `parallel` feature flag. These methods leverage the [Rayon]
//! library to accelerate data decompression and access.
//!
//! [Rayon]: https://docs.rs/rayon/latest/rayon/

#![cfg(feature = "parallel")]

use crate::fixed::{
    traits::{Storable, Word},
    Error as FixedVecError, FixedVec,
};
use dsi_bitstream::prelude::Endianness;
use rayon::prelude::*;

impl<T, W, E, B> FixedVec<T, W, E, B>
where
    T: Storable<W> + Send + Sync,
    W: Word + Sync,
    E: Endianness,
    B: AsRef<[W]> + Sync,
{
    /// Returns a parallel iterator over the decompressed values.
    ///
    /// This operation is "embarrassingly parallel" for `FixedVec` because each
    /// element can be decompressed independently. It scales effectively with
    /// the number of available CPU cores.
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = T> + '_ {
        (0..self.len())
            .into_par_iter()
            .map(move |i| unsafe { self.get_unchecked(i) })
    }

    /// Retrieves multiple elements in parallel.
    ///
    /// This method is designed for maximum throughput on large batches of
    /// randomly distributed indices on multi-core systems.
    ///
    /// # Errors
    /// Returns an error if any index is out of bounds.
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<T>, FixedVecError> {
        // Perform a single bounds check sequentially first.
        if let Some(&index) = indices.iter().find(|&&idx| idx >= self.len()) {
            return Err(FixedVecError::InvalidParameters(format!(
                "Index {} out of bounds for vector of length {}",
                index, self.len
            )));
        }
        // SAFETY: We have pre-checked the bounds of all indices.
        Ok(unsafe { self.par_get_many_unchecked(indices) })
    }

    /// Retrieves multiple elements in parallel without bounds checking.
    ///
    /// This method uses a simple and effective "embarrassingly parallel" strategy:
    /// the input `indices` slice is processed in parallel, and each thread performs
    /// lookups for its assigned partition independently.
    ///
    /// # Safety
    /// Calling this method with any out-of-bounds index is Undefined Behavior.
    pub unsafe fn par_get_many_unchecked(&self, indices: &[usize]) -> Vec<T> {
        if indices.is_empty() {
            return Vec::new();
        }

        // Pre-allocate the results vector to avoid allocations within threads.
        let mut results = Vec::with_capacity(indices.len());
        // SAFETY: We are about to fill this vector completely.
        results.set_len(indices.len());

        results
            .par_iter_mut()
            .zip(indices.par_iter())
            .for_each(|(res_val, &index)| {
                // Each thread performs a scalar lookup for its assigned indices.
                // The `get_unchecked` call is thread-safe as it is read-only.
                *res_val = self.get_unchecked(index);
            });

        results
    }
}