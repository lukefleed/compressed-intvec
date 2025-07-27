//! Parallel implementations for `SIntVec`, enabled by the `parallel` feature flag.
//!
//! This module provides parallel equivalents for iterating and accessing elements
//! in an `SIntVec`, leveraging the Rayon library to exploit multi-core architectures.

use super::{IntVecError, SIntVec};
use dsi_bitstream::prelude::{Endianness, ToInt};
use rayon::prelude::ParallelIterator;

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
    /// # Performance
    /// The performance characteristics are largely inherited from the underlying
    /// [`IntVec::par_iter`](crate::intvec::IntVec::par_iter). The additional `to_int` mapping
    /// is a trivial bitwise operation with negligible overhead. Therefore, the same
    /// trade-offs apply: this parallel iterator is most beneficial for
    /// computationally expensive codecs, while the sequential version may be faster
    /// for simple codecs where memory bandwidth is the bottleneck.
    ///
    /// # Example
    /// ```rust
    /// # #[cfg(feature = "parallel")]
    /// # {
    /// use compressed_intvec::prelude::*;
    /// use rayon::prelude::ParallelIterator;
    ///
    /// let data: &[i64] = &[-10, 20, -30, -40, 50];
    /// let sintvec = LESIntVec::builder(data)
    ///     .codec(CodecSpec::Gamma)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Decompress the entire vector in parallel.
    /// let collected: Vec<i64> = sintvec.par_iter().collect();
    /// assert_eq!(collected, data);
    /// # }
    /// ```
    pub fn par_iter(&self) -> impl ParallelIterator<Item = i64> + '_ {
        self.inner
            .par_iter()
            .map(|unsigned_val| unsigned_val.to_int())
    }

    /// Retrieves multiple signed integers in parallel.
    ///
    /// This method leverages the parallel `par_get_many` of the inner `IntVec`
    /// to fetch the compressed data and then transforms the results back to
    /// signed integers.
    ///
    /// # Implementation Notes
    /// The decompression and random access work is performed in parallel by the
    /// inner `IntVec`. Once the `u64` (ZigZag-encoded) values are retrieved, this
    /// method performs a fast, sequential pass to apply the inverse ZigZag
    /// transformation. This final conversion step is extremely lightweight and
    /// does not typically impact overall performance.
    ///
    /// The performance trade-offs of this method are therefore identical to those
    /// of the underlying [`IntVec::par_get_many`](crate::intvec::IntVec::par_get_many).
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: &[i64] = &[-10, 20, -30, -40, 50, 60];
    /// let sintvec = LESIntVec::builder(data)
    ///     .codec(CodecSpec::Gamma)
    ///     .build()
    ///     .unwrap();
    ///
    /// let indices_to_get = vec![0, 2, 4];
    /// let values = sintvec.par_get_many(&indices_to_get).unwrap();
    ///
    /// // The results are returned in the same order as the requested indices.
    /// assert_eq!(values, vec![-10, -30, 50]);
    /// ```
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<i64>, IntVecError> {
        let unsigned_values = self.inner.par_get_many(indices)?;
        // This conversion is fast and can be done sequentially after parallel fetch.
        let signed_values = unsigned_values
            .into_iter()
            .map(|unsigned_val| unsigned_val.to_int())
            .collect();
        Ok(signed_values)
    }
}
