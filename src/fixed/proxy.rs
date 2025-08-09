//! # Mutable Access Proxy
//!
//! This module defines [`MutProxy`], a proxy object that enables mutable access
//! to elements within a [`FixedVec`].
//!
//! The proxy holds a temporary copy of an element's value. When the proxy is
//! dropped, its `Drop` implementation writes the (potentially modified) value
//! back into the vector. This allows for an ergonomic, "index-like" mutation
//! syntax.
//!
//! # Examples
//!
//! ```rust
//! use compressed_intvec::fixed::{FixedVec, UFixedVec, BitWidth};
//!
//! let data: &[u32] = &[10, 20, 30];
//! let mut vec: UFixedVec<u32> = FixedVec::builder().bit_width(BitWidth::Explicit(7)).build(data).unwrap();
//!
//! // Get a mutable proxy for the element at index 1.
//! if let Some(mut proxy) = vec.at_mut(1) {
//!     // DerefMut allows us to modify the value.
//!     *proxy = 99;
//! } // The proxy is dropped here, and the new value is written back.
//!
//! assert_eq!(vec.get(1), Some(99));
//! ```

use super::{
    traits::{Storable, Word},
    FixedVec,
};
use dsi_bitstream::prelude::Endianness;
use std::ops::{Deref, DerefMut};

/// A proxy object for mutable access to an element within a [`FixedVec`].
///
/// This struct is returned by [`FixedVec::at_mut`]. It holds a temporary copy
/// of an element's value. When the proxy is dropped, its `Drop` implementation
/// writes the (potentially modified) value back into the parent vector.
pub struct MutProxy<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    /// A mutable reference to the parent vector.
    vec: &'a mut FixedVec<T, W, E, B>,
    /// The index of the element being accessed.
    index: usize,
    /// A temporary copy of the element's value.
    value: T,
}

impl<'a, T, W, E, B> MutProxy<'a, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    /// Creates a new `MutProxy`.
    ///
    /// This is called by `FixedVec::at_mut`. It reads the initial value
    /// from the vector.
    pub(super) fn new(vec: &'a mut FixedVec<T, W, E, B>, index: usize) -> Self {
        let value = vec
            .get(index)
            .expect("Index out of bounds in MutProxy creation");
        Self { vec, index, value }
    }
}

impl<T, W, E, B> Deref for MutProxy<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    type Target = T;

    /// Returns a reference to the temporary value.
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T, W, E, B> DerefMut for MutProxy<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    /// Returns a mutable reference to the temporary value, allowing modification.
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T, W, E, B> Drop for MutProxy<'_, T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]> + AsMut<[W]>,
{
    /// Writes the potentially modified value back to the `FixedVec` when the
    /// proxy goes out of scope.
    fn drop(&mut self) {
        // The `value` field is copied here before being passed to `set`.
        self.vec.set(self.index, self.value);
    }
}
