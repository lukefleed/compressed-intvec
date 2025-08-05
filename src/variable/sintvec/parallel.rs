//! # `SIntVec` Parallel Implementations
//!
//! This module provides parallel implementations for [`SIntVec`] operations,
//! enabled by the `parallel` feature flag. These methods are built on top of the
//! parallel implementations of the inner [`IntVec`] and leverage the [Rayon]
//! library to exploit parallelism in access patterns.
//!
//! [`SIntVec`]: crate::variable::sintvec::SIntVec
//! [`IntVec`]: crate::variable::intvec::IntVec
//! [`par_iter`]: crate::variable::sintvec::SIntVec::par_iter
//! [`par_get_many`]: crate::variable::sintvec::SIntVec::par_get_many

use super::{IntVecError, SIntVec};
use crate::variable::intvec::IntVecBitReader;
use dsi_bitstream::prelude::{BitRead, BitSeek, CodesRead, Endianness, ToInt};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "parallel")]
impl<E, B> SIntVec<E, B>
where
    E: Endianness + Send + Sync,
    B: AsRef<[u64]> + Send + Sync,
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>
        + Send,
{
    /// Returns a parallel iterator over the decompressed `i64` values.
    ///
    /// This method wraps the parallel iterator of the inner `IntVec` and applies
    /// the inverse ZigZag transformation to each element on the fly.
    ///
    /// See [`IntVec::par_iter`](crate::variable::intvec::IntVec::par_iter) for a detailed
    /// discussion of performance characteristics.
    pub fn par_iter(&self) -> impl ParallelIterator<Item = i64> + '_ {
        self.inner
            .par_iter()
            .map(|unsigned_val| unsigned_val.to_int())
    }

    /// Retrieves multiple signed integers in parallel.
    ///
    /// This method delegates to [`IntVec::par_get_many`] and applies the inverse
    /// ZigZag transformation to the results in parallel.
    ///
    /// See [`IntVec::par_get_many`](crate::variable::intvec::IntVec::par_get_many) for a detailed
    /// discussion of performance characteristics.
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<i64>, IntVecError> {
        let unsigned_values = self.inner.par_get_many(indices)?;
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
        let unsigned_values = self.inner.par_get_many_unchecked(indices);
        unsigned_values
            .into_par_iter()
            .map(|unsigned_val| unsigned_val.to_int())
            .collect()
    }
}