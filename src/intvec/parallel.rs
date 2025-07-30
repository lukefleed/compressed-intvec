//! # Parallel Implementations
//!
//! This module provides parallel implementations for [`IntVec`] operations,
//! enabled by the `parallel` feature flag. These methods are built on top of the
//! [Rayon] library and are designed to leverage multi-core architectures to
//! accelerate data decompression and access.
//!
//! ## Provided Functionality
//!
//! - [`par_iter`]: A parallel iterator for full-vector decompression. This can
//!   provide significant speedups for computationally intensive codecs but may
//!   be outperformed by the sequential [iterator][`IntVec::iter`] for simpler codecs where memory
//!   bandwidth is the limiting factor.
//!
//! - [`par_get_many`]: A method for parallel batch lookups. It parallelizes the
//!   retrieval of elements at specified indices, trading some redundant work
//!   for higher throughput on multi-core systems.
//!
//! The effectiveness of these parallel methods depends on the workload, the
//! chosen compression scheme ([`CodecSpec`]), and the underlying hardware. They are most
//! beneficial for large datasets where the computational cost of decoding is
//! a significant factor.
//!
//! [Rayon]: https://docs.rs/rayon/latest/rayon/
//! [`IntVec`]: crate::intvec::IntVec
//! [`par_iter`]: crate::intvec::IntVec::par_iter
//! [`par_get_many`]: crate::intvec::IntVec::par_get_many
//! [`IntVec::iter`]: crate::intvec::IntVec::iter
//! [`CodecSpec`]: crate::codec_spec::CodecSpec

use super::{Encoding, IntVec, IntVecBitReader, IntVecError};
use dsi_bitstream::{
    dispatch::{CodesRead, FuncCodeReader, StaticCodeRead},
    impls::MemWordReader,
    prelude::{BitRead, BitSeek, Endianness},
};
use rayon::{
    iter::Either,
    prelude::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
};

#[cfg(feature = "parallel")]
impl<E> IntVec<E>
where
    E: Endianness + Send + Sync,
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>
        + Send,
{
    /// Returns a parallel iterator over the decompressed `u64` values.
    ///
    /// This method provides a way to decompress the entire vector in parallel,
    /// which can yield significant speedups on multi-core systems, especially
    /// when using computationally intensive compression schemes.
    ///
    /// # Implementation Notes
    /// The parallelization strategy depends on the underlying encoding:
    /// - **For variable-length, bit-level codes**: The implementation uses a
    ///   coarse-grained parallelization approach. The set of sample points is
    ///   divided into chunks, and each thread is assigned a chunk. A thread will
    ///   then sequentially decompress all elements corresponding to its assigned
    ///   sample points.
    /// - **For fixed-width integer encoding**: The problem is embarrassingly parallel.
    ///   The iterator simply parallelizes over the indices `0..len` and computes
    ///   the value for each index independently.
    ///
    /// # Performance
    /// For the specific task of full decompression, this parallel version is not
    /// always faster than the sequential [`iter`](super::IntVec::iter). If the
    /// decoding operation is very fast (e.g., with Gamma or Delta codes), the
    /// operation is often limited by memory bandwidth rather than CPU computation.
    /// In such cases, the sequential iterator's better use of CPU caches can
    /// outperform this parallel version, which incurs overhead from thread
    /// management and work distribution.
    ///
    /// This parallel iterator is most beneficial for computationally expensive codecs.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    /// use rayon::prelude::ParallelIterator;
    ///
    /// // With from_iter_builder, codec parameters must be specified manually.
    /// let intvec = LEIntVec::from_iter_builder(0..1000_u64)
    ///     .codec(CodecSpec::Delta)
    ///     .k(8)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Collect the decompressed values in parallel.
    /// let original_data: Vec<u64> = (0..1000).collect();
    /// let parallel_collected: Vec<u64> = intvec.par_iter().collect();
    ///
    /// assert_eq!(parallel_collected, original_data);
    /// ```
    pub fn par_iter(&self) -> impl ParallelIterator<Item = u64> + '_ {
        match self.encoding {
            Encoding::Dsi(_) => {
                // With DSI encoding, k and samples are guaranteed to be Some.
                let k = self.k.unwrap();
                let samples = self.samples.as_ref().unwrap();
                let num_samples = samples.len();
                let num_threads = rayon::current_num_threads();
                let chunk_size = num_samples.div_ceil(num_threads).max(1);
                let num_chunks = num_samples.div_ceil(chunk_size);

                let iter = (0..num_chunks).into_par_iter().flat_map(move |chunk_idx| {
                    let start_sample_idx = chunk_idx * chunk_size;
                    let end_sample_idx = (start_sample_idx + chunk_size).min(num_samples);
                    let mut bit_reader = IntVecBitReader::<E>::new(MemWordReader::new(&self.data));
                    let mut values = Vec::new();

                    if let Encoding::Dsi(code) = self.encoding {
                        let code_reader = FuncCodeReader::<E, _>::new(code).unwrap();
                        for sample_idx in start_sample_idx..end_sample_idx {
                            let start_elem_index = sample_idx * k;
                            let end_elem_index = ((sample_idx + 1) * k).min(self.len);

                            bit_reader
                                .set_bit_pos(samples.get(sample_idx).unwrap())
                                .unwrap();

                            for _ in start_elem_index..end_elem_index {
                                values.push(code_reader.read(&mut bit_reader).unwrap());
                            }
                        }
                    }
                    values.into_par_iter()
                });
                Either::Left(iter)
            }
            Encoding::Fixed { .. } => {
                let iter = (0..self.len)
                    .into_par_iter()
                    .map(move |i| self.get(i).unwrap());
                Either::Right(iter)
            }
        }
    }

    /// Retrieves multiple elements in parallel.
    ///
    /// This method provides parallel random access to a slice of indices. It is
    /// optimized for scenarios with a large number of lookups on multi-core systems.
    ///
    /// # Implementation Notes
    ///
    /// This method parallelizes lookups over the provided `indices` slice. To
    /// avoid the high cost of creating a new bitstream reader for every single
    /// lookup, it uses [`rayon::for_each_with`] to create a single, reusable
    /// [`IntVecReader`] for each thread. This reader is then used for all lookups
    /// assigned to that thread, amortizing the setup cost.
    ///
    /// This approach differs from the sequential [`get_many`](super::IntVec::get_many),
    /// which sorts the indices to perform a single, monotonic forward scan. The parallel
    /// version avoids this sorting and synchronization overhead, but it may perform
    /// *redundant decoding* if multiple threads request indices within the same
    /// sample block.
    ///
    /// # Performance
    ///
    /// This trade-off (work amplification vs. massive parallelism) is often
    /// favorable on multi-core systems. The throughput gained from parallelism
    /// can outweigh the cost of redundant work, especially when the number of
    /// requested indices is large and they are distributed randomly across the vector.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// let intvec = LEIntVec::from_iter_builder(0..1000_u64)
    ///     .codec(CodecSpec::Delta)
    ///     .k(8)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Retrieve multiple elements in parallel.
    /// let indices = vec![0, 100, 200, 300, 400, 500, 600, 700, 800, 900];
    /// let results = intvec.par_get_many(&indices).unwrap();
    ///
    /// // Verify the results.
    /// assert_eq!(results.len(), indices.len());
    /// for (i, &index) in indices.iter().enumerate() {
    ///    assert_eq!(results[i], intvec.get(index).unwrap());
    /// }
    ///
    /// ```
    ///
    /// [`IntVecReader`]: super::IntVecReader
    /// [`rayon::for_each_with`]: https://docs.rs/rayon/latest/rayon/iter/trait.ParallelIterator.html#method.for_each_with
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<u64>, IntVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        for &index in indices {
            if index >= self.len {
                return Err(IntVecError::IndexOutOfBounds(index));
            }
        }

        let mut results = vec![0; indices.len()];

        results.par_iter_mut().enumerate().for_each_with(
            self,
            |reader, (original_pos, res_val)| {
                let target_index = indices[original_pos];
                // The `get` on the reader is fallible, but we pre-checked bounds,
                // and other errors are unlikely, so we unwrap for performance.
                *res_val = reader.get(target_index).unwrap();
            },
        );

        Ok(results)
    }
}
