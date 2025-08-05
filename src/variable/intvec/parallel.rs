//! # Parallel Implementations
//!
//! This module provides parallel implementations for [`IntVec`] operations,
//! enabled by the `parallel` feature flag. These methods are built on top of the
//! [Rayon] library and are designed to leverage multi-core architectures to
//! accelerate data decompression and access.
//!
//! [`IntVec`]: crate::variable::intvec::IntVec
//! [`par_iter`]: crate::variable::intvec::IntVec::par_iter
//! [`par_get_many`]: crate::variable::intvec::IntVec::par_get_many

use super::{IntVec, IntVecBitReader, IntVecError};
use dsi_bitstream::{
    dispatch::{CodesRead, FuncCodeReader},
    prelude::{BitRead, BitSeek, Endianness},
};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[cfg(feature = "parallel")]
impl<E, B> IntVec<E, B>
where
    E: Endianness + Send + Sync,
    B: AsRef<[u64]> + Send + Sync,
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
    /// # Performance
    /// For the specific task of full decompression, this parallel version is not
    /// always faster than the sequential [`iter`](super::IntVec::iter). If the
    /// decoding operation is very fast, the operation is often limited by memory
    /// bandwidth. In such cases, the sequential iterator's better use of CPU
    /// caches can outperform this parallel version.
    pub fn par_iter(&self) -> impl ParallelIterator<Item = u64> + '_ {
        let k = self.k;
        let num_samples = self.samples.len();
        let num_threads = rayon::current_num_threads();
        let chunk_size = num_samples.div_ceil(num_threads).max(1);
        let num_chunks = num_samples.div_ceil(chunk_size);

        (0..num_chunks).into_par_iter().flat_map(move |chunk_idx| {
            let start_sample_idx = chunk_idx * chunk_size;
            let end_sample_idx = (start_sample_idx + chunk_size).min(num_samples);
            let mut bit_reader = IntVecBitReader::<E>::new(dsi_bitstream::impls::MemWordReader::new(
                self.data.as_ref(),
            ));
            let mut values = Vec::new();
            let code_reader = FuncCodeReader::<E, _>::new(self.encoding).unwrap();

            for sample_idx in start_sample_idx..end_sample_idx {
                let start_elem_index = sample_idx * k;
                let end_elem_index = ((sample_idx + 1) * k).min(self.len);

                bit_reader
                    .set_bit_pos(self.samples.get(sample_idx).unwrap())
                    .unwrap();

                for _ in start_elem_index..end_elem_index {
                    use dsi_bitstream::prelude::StaticCodeRead;
                    values.push(code_reader.read(&mut bit_reader).unwrap());
                }
            }
            values.into_par_iter()
        })
    }

    /// Retrieves multiple elements in parallel.
    ///
    /// This method provides parallel random access to a slice of indices. It is
    /// optimized for scenarios with a large number of lookups on multi-core systems.
    ///
    /// [`IntVecReader`]: super::IntVecReader
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<u64>, IntVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        for &index in indices {
            if index >= self.len {
                return Err(IntVecError::IndexOutOfBounds(index));
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

        if indices.is_empty() {
            return Vec::new();
        }

        let mut results = vec![0; indices.len()];

        results.par_iter_mut().enumerate().for_each_init(
            || self.reader(), // Create a reader for each thread.
            |reader, (original_pos, res_val)| {
                let target_index = indices[original_pos];
                // SAFETY: bounds are guaranteed by the caller.
                *res_val = unsafe { reader.get_unchecked(target_index) };
            },
        );

        results
    }
}