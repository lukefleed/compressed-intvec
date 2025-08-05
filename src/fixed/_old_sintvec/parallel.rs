//! # `SFixedVec` Parallel Implementations
//!
//! This module provides parallel implementations for [`SFixedVec`] operations,
//! enabled by the `parallel` feature flag.

use super::{FixedVecError, SFixedVec};
use dsi_bitstream::prelude::{Endianness, ToInt};
use rayon::prelude::*;

#[cfg(feature = "parallel")]
impl<E: Endianness + Send + Sync> SFixedVec<E, Vec<u64>> {
    /// Returns a parallel iterator over the decompressed `i64` values.
    ///
    /// This method wraps the parallel iterator of the inner `FixedVec` and applies
    /// the inverse ZigZag transformation to each element on the fly in parallel.
    pub fn par_iter(&self) -> impl ParallelIterator<Item = i64> + '_ {
        self.inner
            .par_iter()
            .map(|unsigned_val| unsigned_val.to_int())
    }

    /// Retrieves multiple signed integers in parallel.
    ///
    /// This method leverages the parallel `par_get_many` of the inner `FixedVec`
    /// to fetch the compressed data and then transforms the results back to
    /// signed integers using a parallel map operation.
    ///
    /// If the `simd` feature is also enabled, both the initial data gathering
    /// from the inner `FixedVec` and the final ZigZag decoding step can be
    /// SIMD-accelerated, providing a multi-layered optimization.
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<i64>, FixedVecError> {
        let unsigned_values = self.inner.par_get_many(indices)?;

        // This conversion is fast and parallelized by Rayon. If the `simd` feature
        // is enabled, the underlying `decode_zigzag` method in `SFixedVec` can
        // further accelerate this step using SIMD instructions.
        // For simplicity in the parallel implementation, we use a parallel map,
        // which is already highly efficient.
        let signed_values = unsigned_values
            .into_par_iter()
            .map(|unsigned_val| unsigned_val.to_int())
            .collect();
        Ok(signed_values)
    }

    /// Retrieves multiple signed integers in parallel without bounds checking.
    ///
    /// In debug builds, this method will panic if any index is out of bounds.
    ///
    /// # Safety
    /// Calling this method with any out-of-bounds index is undefined behavior in release builds.
    pub unsafe fn par_get_many_unchecked(&self, indices: &[usize]) -> Vec<i64> {
        // 1. Call the underlying `par_get_many_unchecked` on the inner FixedVec.
        //    This operation is already fully optimized with both Rayon and SIMD (if enabled).
        let unsigned_values = self.inner.par_get_many_unchecked(indices);

        // 2. Perform the inverse ZigZag transformation in parallel using Rayon.
        //    This is highly efficient as the `to_int` operation is trivial and
        //    the workload is embarrassingly parallel.
        unsigned_values.into_par_iter().map(ToInt::to_int).collect()
    }
}
