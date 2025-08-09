//! # Mutable Iterators
//!
//! This module provides iterators for mutable, sequential access to the
//! elements of a [`FixedVec`].
//!
//! # Examples
//!
//! ## Mutating elements in parallel
//!
//! The [`ChunksMut`] iterator is designed to be compatible with Rayon's
//! `par_bridge` method, allowing for safe, parallel mutation of vector chunks.
//!
//! ```rust
//! # #[cfg(feature = "parallel")]
//! # {
//! use compressed_intvec::fixed::{FixedVec, UFixedVec, BitWidth};
//! use rayon::prelude::*;
//!
//! let data: Vec<u32> = (0..100).collect();
//! let mut vec: UFixedVec<u32> = FixedVec::builder().bit_width(BitWidth::Explicit(8)).build(&data).unwrap();
//!
//! // Use `par_bridge` to process chunks in parallel.
//! vec.chunks_mut(10).par_bridge().for_each(|mut chunk| {
//!     // Each chunk is a `FixedVecSlice` that can be mutated.
//!     for i in 0..chunk.len() {
//!         if let Some(mut proxy) = chunk.at_mut(i) {
//!             *proxy *= 2;
//!         }
//!     }
//! });
//!
//! assert_eq!(vec.get(10), Some(20));
//! assert_eq!(vec.get(99), Some(198));
//! # }
//! ```

use crate::fixed::{
    slice::FixedVecSlice,
    traits::{Storable, Word},
    FixedVec,
};
use dsi_bitstream::prelude::Endianness;
use std::{cmp::min, marker::PhantomData};

/// An iterator over non-overlapping, mutable chunks of a [`FixedVec`].
///
/// This struct is created by the [`chunks_mut`](super::FixedVec::chunks_mut)
/// method. It is designed to be compatible with Rayon's `par_bridge`, allowing
/// for safe parallel mutation of the vector's chunks.
#[derive(Debug)]
pub struct ChunksMut<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    // A raw pointer to the original `FixedVec` to allow mutable access.
    vec_ptr: *mut FixedVec<T, W, E, B>,
    // The total number of elements in the original vector.
    end: usize,
    // The starting index of the next chunk.
    current_pos: usize,
    // The size of each chunk.
    chunk_size: usize,
    // Ensures the iterator's lifetime is tied to the original mutable borrow.
    _phantom: PhantomData<&'a mut FixedVec<T, W, E, B>>,
}

impl<'a, T, W, E, B> ChunksMut<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    /// Creates a new `ChunksMut` iterator.
    pub(super) fn new(vec: &'a mut FixedVec<T, W, E, B>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size cannot be zero");
        let end = vec.len();
        Self {
            vec_ptr: vec as *mut _,
            chunk_size,
            current_pos: 0,
            end,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, W, E, B> Iterator for ChunksMut<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    type Item = FixedVecSlice<&'a mut FixedVec<T, W, E, B>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos >= self.end {
            return None;
        }

        let start = self.current_pos;
        let len = min(self.chunk_size, self.end - start);
        self.current_pos += len;

        // SAFETY:
        // 1. `self.vec_ptr` is a valid pointer to a `FixedVec` for the lifetime 'a.
        // 2. The borrow checker ensures the original `&'a mut FixedVec` is exclusively
        //    borrowed by this iterator for its entire lifetime.
        // 3. The logic here creates a new `&'a mut` reference from the pointer.
        //    This is safe because each call to `next` produces a slice representing
        //    a unique, non-overlapping range of the original vector.
        // 4. The lifetime `'a` is correct because the data pointed to by `vec_ptr`
        //    is guaranteed to live that long.
        let vec_ref = unsafe { &mut *self.vec_ptr };
        let slice = FixedVecSlice::new(vec_ref, start..start + len);

        // The type of `slice` is `FixedVecSlice<&mut FixedVec<...>>`. The lifetime
        // of the inner mutable reference is correctly inferred to be 'a.
        // No transmute is necessary with the correct struct definition.
        Some(slice)
    }
}