//! Parallel operations for [`IntVec`].
//!
//! This module provides parallel implementations for [`IntVec`] operations,
//! enabled by the `parallel` feature flag. These methods are built on the
//! [Rayon] library and are designed to leverage multi-core architectures to
//! accelerate data decompression and access.
//!
//! [Rayon]: https://github.com/rayon-rs/rayon
//! [`IntVec`]: crate::variable::IntVec

use super::{traits::Storable, IntVec, IntVecBitReader, IntVecError};
use dsi_bitstream::{
    dispatch::{CodesRead, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[cfg(feature = "parallel")]
impl<T, E, B> IntVec<T, E, B>
where
    T: Storable + Send + Sync,
    E: Endianness + Send + Sync,
    B: AsRef<[u64]> + Send + Sync,
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>
        + Send,
{
    /// Returns a parallel iterator over the decompressed values.
    ///
    /// This method uses Rayon to decompress the entire vector in parallel. It
    /// can provide a significant speedup on multi-core systems, especially when
    /// using a computationally intensive compression codec.
    ///
    /// # Performance
    ///
    /// For the specific task of full decompression, this parallel version is not
    /// always faster than the sequential [`iter`](super::IntVec::iter). If the
    /// decoding operation is very fast (e.g., with `VByte` encoding), the
    /// operation can be limited by memory bandwidth. In such cases, the
    /// sequential iterator's better use of CPU caches may outperform this
    /// parallel version.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::variable::{IntVec, UIntVec};
    /// use rayon::prelude::*;
    ///
    /// let data: Vec<u32> = (0..1000).collect();
    /// let vec: UIntVec<u32> = IntVec::from_slice(&data).unwrap();
    ///
    /// // Use the parallel iterator to compute the sum in parallel
    /// let sum: u32 = vec.par_iter().sum();
    ///
    /// assert_eq!(sum, (0..1000).sum());
    /// # }
    /// ```
    pub fn par_iter(&self) -> impl ParallelIterator<Item = T> + '_ {
        let k = self.k;
        let num_samples = self.samples.len();
        let num_threads = rayon::current_num_threads();
        // Divide the sample blocks among the available threads.
        let chunk_size = num_samples.div_ceil(num_threads).max(1);
        let num_chunks = num_samples.div_ceil(chunk_size);

        (0..num_chunks).into_par_iter().flat_map(move |chunk_idx| {
            use crate::common::codec_reader::CodecReader;

            let start_sample_idx = chunk_idx * chunk_size;
            let end_sample_idx = (start_sample_idx + chunk_size).min(num_samples);
            let mut bit_reader = IntVecBitReader::<E>::new(
                dsi_bitstream::impls::MemWordReader::new(self.data.as_ref()),
            );
            let mut values = Vec::new();
            let code_reader = CodecReader::new(self.encoding);

            // Each thread decodes its assigned range of sample blocks.
            for sample_idx in start_sample_idx..end_sample_idx {
                let (start_elem_index, end_elem_index) = if k.is_power_of_two() {
                    let k_exp = k.trailing_zeros();
                    (sample_idx << k_exp, ((sample_idx + 1) << k_exp).min(self.len))
                } else {
                    (sample_idx * k, ((sample_idx + 1) * k).min(self.len))
                };

                unsafe {
                    bit_reader
                        .set_bit_pos(self.samples.get_unchecked(sample_idx))
                        .unwrap();
                }

                for _ in start_elem_index..end_elem_index {
                    let word = code_reader.read(&mut bit_reader).unwrap();
                    values.push(Storable::from_word(word));
                }
            }
            values.into_par_iter()
        })
    }

    /// Retrieves multiple elements from a slice of indices in parallel.
    ///
    /// This method uses Rayon to parallelize random access. It works by creating
    /// a separate [`IntVecReader`](super::IntVecReader) for each thread and
    /// distributing the lookup work among them.
    ///
    /// # Errors
    ///
    /// Returns [`IntVecError::IndexOutOfBounds`] if any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::variable::{IntVec, SIntVec};
    ///
    /// let data: Vec<i64> = (0..1000).map(|x| x * -1).collect();
    /// let vec: SIntVec<i64> = IntVec::from_slice(&data).unwrap();
    ///
    /// let indices = [500, 10, 999, 0, 250];
    /// let values = vec.par_get_many(&indices).unwrap();
    ///
    /// assert_eq!(values, vec![-500, -10, -999, 0, -250]);
    /// # }
    /// ```
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<T>, IntVecError> {
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
    /// # Safety
    ///
    /// Calling this method with any out-of-bounds index in the `indices` slice
    /// is undefined behavior. In debug builds, an assertion will panic.
    pub unsafe fn par_get_many_unchecked(&self, indices: &[usize]) -> Vec<T> {
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

        let mut results = vec![Storable::from_word(0); indices.len()];

        results.par_iter_mut().enumerate().for_each_init(
            || self.reader(), // Create a thread-local reader.
            |reader, (original_pos, res_val)| {
                let target_index = indices[original_pos];
                // SAFETY: bounds are guaranteed by the caller.
                *res_val = unsafe { reader.get_unchecked(target_index) };
            },
        );

        results
    }
}
