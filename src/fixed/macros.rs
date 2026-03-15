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

/// Creates a [`FixedVec`] of signed integers (forces `i64`).
///
/// This macro is similar to [`fixed_vec!`], but automatically casts all
/// elements to `i64`. This ensures that ZigZag encoding is used for
/// signed values.
///
/// # Examples
///
/// ```
/// use compressed_intvec::sfixed_vec;
/// let vec = sfixed_vec![-1, 2, -3];
/// assert_eq!(vec.get(0), Some(-1));
/// ```
#[macro_export]
macro_rules! sfixed_vec {
    // Empty case
    () => {
        $crate::fixed::FixedVec::<i64, usize, dsi_bitstream::prelude::LE>::builder()
            .build(&[])
            .unwrap()
    };

    // From list: `sfixed_vec![a, b, c]`
    ($($elem:expr),+ $(,)?) => {
        $crate::fixed::macros::from_slice(&[$($elem as i64),+])
    };

    // From element and length: `sfixed_vec![elem; len]`
    ($elem:expr; $len:expr) => {
        $crate::fixed::macros::from_repetition($elem as i64, $len)
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
    // Determine bit_width from a single element to avoid materializing an
    // intermediate Vec of `len` elements (which could be very large).
    let word_val = <T as Storable<<T as DefaultParams>::W>>::into_word(elem);
    // Convert to u128 to use leading_zeros, since the Word trait does not
    // directly expose it. This is only called once, not in a hot loop.
    let val_u128 = word_val.to_u128().unwrap_or(0);
    let bit_width = if val_u128 == 0 {
        1
    } else {
        (128 - val_u128.leading_zeros()) as usize
    };

    crate::fixed::builder::FixedVecFromIterBuilder::<
        T,
        <T as DefaultParams>::W,
        <T as DefaultParams>::E,
        _,
    >::new(std::iter::repeat_n(elem, len), bit_width)
    .build()
    .unwrap()
}
