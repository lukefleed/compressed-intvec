//! # Macros for `AtomicFixedVec`
//!
//! This module provides the [`atomic_fixed_vec!`] macro for creating a
//! [`AtomicFixedVec`] with a `vec!`-like syntax.

/// Creates an `AtomicFixedVec` with default parameters.
///
/// This macro simplifies the creation of `AtomicFixedVec` by using default
/// parameters inferred from the element type and a `u64` storage backend.
/// It uses `BitWidth::Minimal` for space efficiency.
///
/// # Examples
///
/// ```
/// use compressed_intvec::prelude::*;
///
/// // Create a vector from a list of elements. The type is inferred.
/// let vec = atomic_fixed_vec![10u32, 20, 30];
/// assert_eq!(vec.len(), 3);
/// assert_eq!(vec.bit_width(), 5); // 30 requires 5 bits
///
/// // Create a vector from a repeated element.
/// let vec_rep = atomic_fixed_vec![0i16; 100];
/// assert_eq!(vec_rep.len(), 100);
/// ```
#[macro_export]
macro_rules! atomic_fixed_vec {
    // Empty vector: `atomic_fixed_vec![]`
    // Requires type annotation, e.g., `let v: UAtomicFixedVec<u32> = atomic_fixed_vec![];`
    () => {
        $crate::fixed::atomic::AtomicFixedVec::builder().build(&[]).unwrap()
    };

    // From list: `atomic_fixed_vec![a, b, c]`
    ($($elem:expr),+ $(,)?) => {
        // Delegate to the hidden helper function.
        // The compiler infers `T` from the slice `&[$($elem),+]`.
        $crate::fixed::atomic::macros::from_slice(&[$($elem),+])
    };

    // From element and length: `atomic_fixed_vec![elem; len]`
    ($elem:expr; $len:expr) => {
        // Delegate to the hidden helper function.
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
/// to construct an `AtomicFixedVec` from a slice of elements, using the
/// default builder settings (`BitWidth::Minimal`).
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
/// to construct an `AtomicFixedVec` by repeating an element a specified number
/// of times.
#[doc(hidden)]
pub fn from_repetition<T>(elem: T, len: usize) -> AtomicFixedVec<T>
where
    T: Storable<u64> + ToPrimitive + Copy,
{
    let v = vec![elem; len];
    AtomicFixedVecBuilder::new().build(&v).unwrap()
}