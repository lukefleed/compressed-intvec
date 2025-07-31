//! # `SFixedVec` Parallel Implementations
//!
//! This module provides parallel implementations for [`SFixedVec`] operations,
//! enabled by the `parallel` feature flag.

use super::{FixedVecError, SFixedVec};
use dsi_bitstream::prelude::{Endianness, ToInt};
use rayon::prelude::*;

#[cfg(feature = "parallel")]
impl<E> SFixedVec<E>
where
    E: Endianness + Send + Sync,
{
    /// Returns a parallel iterator over the decompressed `i64` values.
    ///
    /// This method wraps the parallel iterator of the inner `FixedVec` and applies
    /// the inverse ZigZag transformation to each element on the fly.
    pub fn par_iter(&self) -> impl ParallelIterator<Item = i64> + '_ {
        self.inner
            .par_iter()
            .map(|unsigned_val| unsigned_val.to_int())
    }

    /// Retrieves multiple signed integers in parallel.
    ///
    /// This method leverages the parallel `par_get_many` of the inner `FixedVec`
    /// to fetch the compressed data and then transforms the results back to
    /// signed integers.
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<i64>, FixedVecError> {
        let unsigned_values = self.inner.par_get_many(indices)?;
        // This conversion is fast and can be parallelized.
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
        self.inner
            .par_get_many_unchecked(indices)
            .into_par_iter()
            .map(ToInt::to_int)
            .collect()
    }
}
