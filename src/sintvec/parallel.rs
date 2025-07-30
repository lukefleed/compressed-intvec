//! # `SIntVec` Parallel Implementations
//!
//! This module provides parallel implementations for [`SIntVec`] operations,
//! enabled by the `parallel` feature flag. These methods are built on top of the
//! parallel implementations of the inner [`IntVec`] and leverage the [Rayon]
//! library to exploit parallelism in access patterns.
//!
//! ## Implementation
//!
//! Both [`par_iter`] and [`par_get_many`] are implemented as lightweight
//! wrappers around their [`IntVec`] counterparts. They delegate the heavy
//! lifting of parallel decompression and access to the inner `IntVec` and then
//! apply the inverse ZigZag transformation ([`ToInt`]) to the resulting `u64`
//! values. Since this transformation is a trivial bitwise operation, the
//! performance characteristics and trade-offs of these methods are identical
//! to those of the underlying `IntVec`'s parallel methods.
//!
//! [Rayon]: https://docs.rs/rayon/latest/rayon/
//! [`SIntVec`]: crate::sintvec::SIntVec
//! [`IntVec`]: crate::intvec::IntVec
//! [`par_iter`]: crate::sintvec::SIntVec::par_iter
//! [`par_get_many`]: crate::sintvec::SIntVec::par_get_many
//! [`ToInt`]: dsi_bitstream::prelude::ToInt

use super::{IntVecError, SIntVec};
use dsi_bitstream::prelude::{Endianness, ToInt};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "parallel")]
impl<E> SIntVec<E>
where
    E: Endianness + Send + Sync,
    for<'a> crate::intvec::IntVecBitReader<'a, E>: dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
        + dsi_bitstream::dispatch::CodesRead<E>
        + dsi_bitstream::prelude::BitSeek<Error = core::convert::Infallible>
        + Send,
{
    /// Returns a parallel iterator over the decompressed `i64` values.
    ///
    /// This method wraps the parallel iterator of the inner `IntVec` and applies
    /// the inverse ZigZag transformation to each element on the fly.
    ///
    /// See [`IntVec::par_iter`](crate::intvec::IntVec::par_iter) for a detailed
    /// discussion of performance characteristics. The overhead of the `to_int`
    /// mapping is negligible.
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
    /// See [`IntVec::par_get_many`](crate::intvec::IntVec::par_get_many) for a detailed
    /// discussion of performance characteristics. The overhead of the final
    /// `to_int` mapping is negligible.
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<i64>, IntVecError> {
        let unsigned_values = self.inner.par_get_many(indices)?;
        // This conversion is fast and can be parallelized.
        let signed_values = unsigned_values
            .into_par_iter()
            .map(|unsigned_val| unsigned_val.to_int())
            .collect();
        Ok(signed_values)
    }
}
