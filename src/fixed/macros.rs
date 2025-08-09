//! # Macros for [`FixedVec`]

/// Creates a [`FixedVec`] with default parameters.
///
/// This macro simplifies the creation of [`FixedVec`] by using default parameters
/// (`usize` for the storage word, `LittleEndian` for byte order) inferred
/// from the element type. It uses [`BitWidth::Minimal`] for space efficiency.
///
/// There are three forms of this macro:
///
/// - Create a vector from a list of elements:
///   ```
///   # use compressed_intvec::fixed_vec;
///   let vec = fixed_vec![10u32, 20, 30];
///   # assert_eq!(vec.len(), 3);
///   ```
///
/// - Create a vector from a repeated element:
///   ```
///   # use compressed_intvec::fixed_vec;
///   let vec = fixed_vec![0i16; 100];
///   # assert_eq!(vec.len(), 100);
///   ```
///
/// - Create an empty vector (the element type must be specified):
///   ```
///   # use compressed_intvec::fixed_vec;
///   # use compressed_intvec::fixed::UFixedVec;
///   let vec: UFixedVec<u32> = fixed_vec![];
///   # assert!(vec.is_empty());
///   ```
///
/// # Examples
///
/// ```
/// use compressed_intvec::fixed_vec;
/// use compressed_intvec::fixed::UFixedVec;
///
/// // Create a vector from a list of elements.
/// let vec = fixed_vec![10u32, 20, 30];
/// assert_eq!(vec.len(), 3);
/// assert_eq!(vec.bit_width(), 5); // 30 requires 5 bits
///
/// // Create a vector from a repeated element.
/// let vec_rep = fixed_vec![0i16; 100];
/// assert_eq!(vec_rep.len(), 100);
/// assert_eq!(vec_rep.get(50), Some(0));
///
/// // Create an empty vector (type annotation is required).
/// let empty: UFixedVec<u64> = fixed_vec![];
/// assert!(empty.is_empty());
/// ```
#[macro_export]
macro_rules! fixed_vec {
    // Empty vector: `fixed_vec![]`
    // Requires type annotation from the user.
    () => {
        $crate::fixed::FixedVec::builder().build(&[]).unwrap()
    };

    // From list: `fixed_vec![a, b, c]`
    ($($elem:expr),+ $(,)?) => {
        // Delegate to a helper function to avoid repeating complex bounds.
        $crate::fixed::macros::from_slice(&[$($elem),+])
    };

    // From element and length: `fixed_vec![elem; len]`
    ($elem:expr; $len:expr) => {
        // Delegate to a helper function.
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
///
/// This function is not intended for direct use. It is called by the macro
/// to construct a `FixedVec` from a slice of elements.
#[doc(hidden)]
pub fn from_slice<T>(slice: &[T]) -> FixedVec<T, <T as DefaultParams>::W, <T as DefaultParams>::E>
where
    T: DefaultParams + Storable<<T as DefaultParams>::W> + ToPrimitive,
    <T as DefaultParams>::W: Word,
    <T as DefaultParams>::E: Endianness,
    FixedVecBuilder<T, <T as DefaultParams>::W, <T as DefaultParams>::E>: Default,
    // This complex bound is required by the `build` method of the builder.
    for<'a> dsi_bitstream::impls::BufBitWriter<
        <T as DefaultParams>::E,
        dsi_bitstream::impls::MemWordWriterVec<
            <T as DefaultParams>::W,
            Vec<<T as DefaultParams>::W>,
        >,
    >: dsi_bitstream::prelude::BitWrite<<T as DefaultParams>::E, Error = std::convert::Infallible>,
{
    FixedVec::<T, <T as DefaultParams>::W, <T as DefaultParams>::E>::builder()
        .bit_width(BitWidth::Minimal)
        .build(slice)
        .unwrap()
}

/// A hidden helper function for the `fixed_vec![elem; len]` macro variant.
///
/// This function is not intended for direct use. It is called by the macro
/// to construct a `FixedVec` by repeating an element.
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
    // This complex bound is required by the `build` method of the builder.
    for<'a> dsi_bitstream::impls::BufBitWriter<
        <T as DefaultParams>::E,
        dsi_bitstream::impls::MemWordWriterVec<
            <T as DefaultParams>::W,
            Vec<<T as DefaultParams>::W>,
        >,
    >: dsi_bitstream::prelude::BitWrite<<T as DefaultParams>::E, Error = std::convert::Infallible>,
{
    let mut v = Vec::new();
    v.resize(len, elem);
    FixedVec::<T, <T as DefaultParams>::W, <T as DefaultParams>::E>::builder()
        .bit_width(BitWidth::Minimal)
        .build(&v)
        .unwrap()
}
