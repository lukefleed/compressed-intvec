//! # Macros for `FixedVec`
//!
//! This module provides the [`fixed_vec!`] macro for creating a [`FixedVec`]
//! with a `vec!`-like syntax.

/// Creates a `FixedVec` with default parameters.
///
/// This macro simplifies the creation of `FixedVec` by using default parameters
/// (`usize` for the storage word, `LittleEndian` for byte order) inferred

/// from the element type. It uses `BitWidth::Minimal` for space efficiency.
#[macro_export]
macro_rules! fixed_vec {
    // Empty vector: `fixed_vec![]`
    // Requires type annotation from the user, e.g., `let v: UFixedVec<u32> = fixed_vec![];`
    () => {
        $crate::fixed::FixedVec::builder().build(&[]).unwrap()
    };

    // From list: `fixed_vec![a, b, c]`
    ($($elem:expr),+ $(,)?) => {
        // Delegate to the hidden helper function.
        // The compiler infers `T` from the slice `&[$($elem),+]`.
        $crate::fixed::macros::from_slice(&[$($elem),+])
    };

    // From element and length: `fixed_vec![elem; len]`
    ($elem:expr; $len:expr) => {
        // Delegate to the hidden helper function.
        $crate::fixed::macros::from_repetition($elem, $len)
    };
}

// --- Macro Helper Functions (Not part of the public API) ---

use crate::fixed::{
    builder::FixedVecBuilder,
    traits::{DefaultParams, Storable, Word},
    BitWidth, FixedVec,
};
use dsi_bitstream::prelude::Endianness;
use num_traits::ToPrimitive;

/// A hidden helper function for the `fixed_vec![...]` macro variant.
#[doc(hidden)]
pub fn from_slice<T>(
    slice: &[T],
) -> FixedVec<T, <T as DefaultParams>::W, <T as DefaultParams>::E>
where
    T: DefaultParams + Storable<<T as DefaultParams>::W> + ToPrimitive,
    <T as DefaultParams>::W: Word,
    <T as DefaultParams>::E: Endianness,
    FixedVecBuilder<T, <T as DefaultParams>::W, <T as DefaultParams>::E>: Default,
    // The complex bound for the builder's `build` method.
    for<'a> dsi_bitstream::impls::BufBitWriter<
        <T as DefaultParams>::E,
        dsi_bitstream::impls::MemWordWriterVec<<T as DefaultParams>::W, Vec<<T as DefaultParams>::W>>,
    >: dsi_bitstream::prelude::BitWrite<
        <T as DefaultParams>::E,
        Error = std::convert::Infallible,
    >,
{
    FixedVec::<T, <T as DefaultParams>::W, <T as DefaultParams>::E>::builder()
        .bit_width(BitWidth::Minimal)
        .build(slice)
        .unwrap()
}

/// A hidden helper function for the `fixed_vec![elem; len]` macro variant.
#[doc(hidden)]
pub fn from_repetition<T>(
    elem: T,
    len: usize,
) -> FixedVec<T, <T as DefaultParams>::W, <T as DefaultParams>::E>
where
    T: DefaultParams + Storable<<T as DefaultParams>::W> + ToPrimitive + Clone,
    <T as DefaultParams>::W: Word,
    <T as DefaultParams>::E: Endianness,
    FixedVecBuilder<T, <T as DefaultParams>::W, <T as DefaultParams>::E>: Default,
    // The complex bound for the builder's `build` method.
    for<'a> dsi_bitstream::impls::BufBitWriter<
        <T as DefaultParams>::E,
        dsi_bitstream::impls::MemWordWriterVec<<T as DefaultParams>::W, Vec<<T as DefaultParams>::W>>,
    >: dsi_bitstream::prelude::BitWrite<
        <T as DefaultParams>::E,
        Error = std::convert::Infallible,
    >,
{
    let mut v = Vec::new();
    v.resize(len, elem);
    FixedVec::<T, <T as DefaultParams>::W, <T as DefaultParams>::E>::builder()
        .bit_width(BitWidth::Minimal)
        .build(&v)
        .unwrap()
}