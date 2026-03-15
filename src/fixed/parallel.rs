//! # [`FixedVec`] Parallel Operations
//!
//! This module provides parallel implementations for [`FixedVec`] operations,
//! enabled by the `parallel` feature flag. These methods are based on the [Rayon]
//! library.
//!
//! # Examples
//!
//! ```
//! # #[cfg(feature = "parallel")]
//! # {
//! use compressed_intvec::fixed::{FixedVec, UFixedVec};
//! use rayon::prelude::*;
//!
//! let data: Vec<u32> = (0..1000).collect();
//! let vec: UFixedVec<u32> = FixedVec::builder().build(&data).unwrap();
//!
//! // Sum the elements in parallel.
//! let sum: u32 = vec.par_iter().sum();
//! assert_eq!(sum, (0..1000).sum());
//! # }
//! ```
//!
//! [Rayon]: https://docs.rs/rayon/latest/rayon/

use crate::fixed::{
    Error as FixedVecError, FixedVec,
    traits::{Storable, Word},
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
    /// This operation is highly parallelizable because each element can be
    /// decompressed independently from the others.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")]
    /// # {
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    /// use rayon::prelude::*;
    ///
    /// let data: Vec<u32> = (0..100).collect();
    /// let vec: UFixedVec<u32> = FixedVec::builder().build(&data).unwrap();
    ///
    /// let count = vec.par_iter().filter(|&x| x % 2 == 0).count();
    /// assert_eq!(count, 50);
    /// # }
    /// ```
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = T> + '_ {
        // Use with_min_len to batch elements into chunks, reducing
        // per-element overhead from computing bit_pos/word_index/bit_offset.
        let chunk_size = 1024.min(self.len());
        (0..self.len())
            .into_par_iter()
            .with_min_len(chunk_size)
            .map(move |i| unsafe { self.get_unchecked(i) })
    }

    /// Retrieves multiple elements in parallel.
    ///
    /// # Errors
    ///
    /// Returns an error if any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")]
    /// # {
    /// use compressed_intvec::fixed::{FixedVec, UFixedVec};
    ///
    /// let data: Vec<u32> = (0..100).collect();
    /// let vec: UFixedVec<u32> = FixedVec::builder().build(&data).unwrap();
    ///
    /// let indices = vec![10, 50, 99];
    /// let values = vec.par_get_many(&indices).unwrap();
    /// assert_eq!(values, vec![10, 50, 99]);
    ///
    /// // An out-of-bounds index will result in an error.
    /// let invalid_indices = vec![10, 100];
    /// assert!(vec.par_get_many(&invalid_indices).is_err());
    /// # }
    /// ```
    pub fn par_get_many(&self, indices: &[usize]) -> Result<Vec<T>, FixedVecError> {
        // Perform a single bounds check sequentially first for early failure.
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
    /// # Safety
    ///
    /// Calling this method with any out-of-bounds index is Undefined Behavior.
    /// All indices must be less than `self.len()`.
    pub unsafe fn par_get_many_unchecked(&self, indices: &[usize]) -> Vec<T> {
        if indices.is_empty() {
            return Vec::new();
        }

        // Allocate an uninitialized buffer to hold the results. This avoids
        // the need for `T: Default` and prevents zero-initializing the memory.
        let mut results: Vec<std::mem::MaybeUninit<T>> = Vec::with_capacity(indices.len());
        results.resize_with(indices.len(), std::mem::MaybeUninit::uninit);

        results
            .par_iter_mut()
            .zip(indices.par_iter())
            .for_each(|(res_val, &index)| {
                // Each thread performs a scalar lookup for its assigned indices.
                // The `get_unchecked` call is thread-safe as it is read-only.
                unsafe { res_val.write(self.get_unchecked(index)) };
            });

        // SAFETY: The parallel iteration guarantees that all elements in the
        // `results` vector have been initialized. We can therefore safely
        // transmute the `Vec<MaybeUninit<T>>` to a `Vec<T>`.
        unsafe { std::mem::transmute(results) }
    }
}
