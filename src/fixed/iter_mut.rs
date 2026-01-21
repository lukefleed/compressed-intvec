//! # Mutable Iterators
//!
//! This module provides iterators for mutable, sequential access to the
//! elements of a [`FixedVec`].
//!
//! # Provided Iterators
//!
//! - [`IterMut`]: An iterator that yields a mutable proxy for each element.
//! - [`ChunksMut`]: An iterator that yields non-overlapping, mutable slices.
//!
//! # Examples
//!
//! ## Mutating elements in chunks
//!
//! The [`ChunksMut`] iterator allows for processing a vector in mutable,
//! non-overlapping chunks.
//!
//! ```rust
//! use compressed_intvec::fixed::{FixedVec, UFixedVec, BitWidth};
//!
//! let data: Vec<u32> = (0..100).collect();
//! let mut vec: UFixedVec<u32> = FixedVec::builder().bit_width(BitWidth::Explicit(8)).build(&data).unwrap();
//!
//! // Process each chunk sequentially.
//! for mut chunk in vec.chunks_mut(10) {
//!     // Each chunk is a `FixedVecSlice` that can be mutated.
//!     for i in 0..chunk.len() {
//!         if let Some(mut proxy) = chunk.at_mut(i) {
//!             *proxy *= 2;
//!         }
//!     }
//! }
//!
//! assert_eq!(vec.get(10), Some(20));
//! assert_eq!(vec.get(99), Some(198));
//! ```

use crate::fixed::{
    FixedVec,
    proxy::MutProxy,
    slice::FixedVecSlice,
    traits::{Storable, Word},
};
use dsi_bitstream::prelude::Endianness;
use std::{cmp::min, marker::PhantomData};

/// An iterator over non-overlapping, mutable chunks of a [`FixedVec`].
///
/// This struct is created by the [`chunks_mut`](super::FixedVec::chunks_mut)
/// method.
#[derive(Debug)]
pub struct ChunksMut<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    // A raw pointer to the original `FixedVec` is used to allow creating
    // mutable slices without consuming the iterator.
    vec_ptr: *mut FixedVec<T, W, E, B>,
    end: usize,
    current_pos: usize,
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
        // 2. The borrow checker ensures the original `&'a mut FixedVec` is
        //    exclusively borrowed by this iterator for its entire lifetime.
        // 3. Each call to `next` produces a slice for a unique, non-overlapping
        //    range of the original vector, making the mutable borrow safe.
        let vec_ref = unsafe { &mut *self.vec_ptr };
        let slice = FixedVecSlice::new(vec_ref, start..start + len);

        Some(slice)
    }
}

/// A mutable iterator over the elements of a [`FixedVec`].
///
/// This struct is created by the [`iter_mut`](super::FixedVec::iter_mut)
/// method. It yields a [`MutProxy`] for each element.
pub struct IterMut<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    // A raw pointer is used to allow creating proxies without consuming the iterator.
    vec_ptr: *mut FixedVec<T, W, E, B>,
    current_index: usize,
    end_index: usize,
    _phantom: PhantomData<&'a mut FixedVec<T, W, E, B>>,
}

impl<'a, T, W, E, B> IterMut<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    /// Creates a new `IterMut` iterator.
    pub(super) fn new(vec: &'a mut FixedVec<T, W, E, B>) -> Self {
        let len = vec.len();
        Self {
            vec_ptr: vec as *mut _,
            current_index: 0,
            end_index: len,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, W, E, B> Iterator for IterMut<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    type Item = MutProxy<'a, T, W, E, B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.end_index {
            return None;
        }

        // SAFETY: The iterator's lifetime `'a` and bounds checking ensure
        // that the pointer is valid and that we are creating non-overlapping
        // mutable access proxies one at a time.
        let proxy = unsafe { MutProxy::new(&mut *self.vec_ptr, self.current_index) };
        self.current_index += 1;

        Some(proxy)
    }
}

/// An unchecked mutable iterator over the elements of a [`FixedVec`].
///
/// This struct is created by the [`iter_mut_unchecked`](super::FixedVec::iter_mut_unchecked)
/// method. It does not perform any bounds checking.
///
/// # Safety
///
/// The iterator is safe to use only if it is guaranteed that it will not
/// be advanced beyond the end of the vector.
pub struct IterMutUnchecked<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    iter: IterMut<'a, T, W, E, B>,
}

impl<'a, T, W, E, B> IterMutUnchecked<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    /// Creates a new `IterMutUnchecked`.
    pub(super) fn new(vec: &'a mut FixedVec<T, W, E, B>) -> Self {
        Self {
            iter: IterMut::new(vec),
        }
    }

    /// Returns the next mutable proxy without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method when the iterator is exhausted is undefined behavior.
    #[inline]
    pub unsafe fn next_unchecked(&mut self) -> MutProxy<'a, T, W, E, B> {
        unsafe { self.iter.next().unwrap_unchecked() }
    }
}
