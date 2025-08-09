//! # Macros for `AtomicFixedVec`
//!
//! This module provides the [`atomic_fixed_vec!`] macro for creating an
//! [`AtomicFixedVec`] with a `vec!`-like syntax. This is convenient for
//! creating vectors in tests or examples.

/// Creates an `AtomicFixedVec` with default parameters.
///
/// This macro simplifies the creation of an `AtomicFixedVec`. It uses a `u64`
/// storage backend and `BitWidth::Minimal` for space efficiency.
///
/// There are two forms of this macro:
///
/// - Create a vector from a list of elements:
///   ```
///   # use compressed_intvec::atomic_fixed_vec;
///   let vec = atomic_fixed_vec![10u32, 20, 30];
///   # assert_eq!(vec.len(), 3);
///   ```
///
/// - Create a vector from a repeated element:
///   ```
///   # use compressed_intvec::atomic_fixed_vec;
///   let vec = atomic_fixed_vec![0i16; 100];
///   # assert_eq!(vec.len(), 100);
///   ```
///
/// # Examples
///
/// ```
/// use compressed_intvec::prelude::*;
/// use compressed_intvec::atomic_fixed_vec;
/// use compressed_intvec::fixed::UAtomicFixedVec;
///
/// // Create a vector from a list of elements. The type is inferred.
/// let vec = atomic_fixed_vec![10u32, 20, 30];
/// assert_eq!(vec.len(), 3);
/// assert_eq!(vec.bit_width(), 5); // 30 requires 5 bits
///
/// // Create a vector from a repeated element.
/// let vec_rep = atomic_fixed_vec![0i16; 100];
/// assert_eq!(vec_rep.len(), 100);
/// assert_eq!(vec_rep.get(50), Some(0));
///
/// // Create an empty vector (type annotation is required).
/// let empty: UAtomicFixedVec<u64> = atomic_fixed_vec![];
/// assert!(empty.is_empty());
/// ```
#[macro_export]
macro_rules! atomic_fixed_vec {
    // Empty vector: `atomic_fixed_vec![]`
    // Requires type annotation from the user.
    () => {
        $crate::fixed::atomic::AtomicFixedVec::builder().build(&[]).unwrap()
    };

    // From list: `atomic_fixed_vec![a, b, c]`
    ($($elem:expr),+ $(,)?) => {
        // Delegate to a helper function to avoid repeating complex bounds.
        $crate::fixed::atomic::macros::from_slice(&[$($elem),+])
    };

    // From element and length: `atomic_fixed_vec![elem; len]`
    ($elem:expr; $len:expr) => {
        // Delegate to a helper function.
        $crate::fixed::atomic::macros::from_repetition($elem, $len)
    };
}

// --- Macro Helper Functions (Not part of the public API) ---

use crate::fixed::atomic::{builder::AtomicFixedVecBuilder, AtomicFixedVec};
use crate::fixed::traits::Storable;
use num_traits::ToPrimitive;

/// A hidden helper function for the `atomic_fixed_vec![...]` macro variant.
///
/// This function is not intended for direct use. It is called by the macro
/// to construct an `AtomicFixedVec` from a slice of elements.
#[doc(hidden)]
pub fn from_slice<T>(slice: &[T]) -> AtomicFixedVec<T>
where
    T: Storable<u64> + ToPrimitive + Copy,
{
    // The builder defaults to `BitWidth::Minimal`.
    AtomicFixedVecBuilder::new().build(slice).unwrap()
}

/// A hidden helper function for the `atomic_fixed_vec![elem; len]` macro variant.
///
/// This function is not intended for direct use. It is called by the macro
/// to construct an `AtomicFixedVec` by repeating an element.
#[doc(hidden)]
pub fn from_repetition<T>(elem: T, len: usize) -> AtomicFixedVec<T>
where
    T: Storable<u64> + ToPrimitive + Copy,
{
    let v = vec![elem; len];
    AtomicFixedVecBuilder::new().build(&v).unwrap()
}
