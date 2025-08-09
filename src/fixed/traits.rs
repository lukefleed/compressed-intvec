//! # Core Traits for `FixedVec`
//!
//! This module defines the core traits that enable the generic and unified
//! `FixedVec` architecture. These traits abstract away the details of the
//! underlying storage words and the conversion logic for different element types,
//! allowing `FixedVec` to be highly configurable.

use common_traits::{IntoAtomic, SignedInt, UnsignedInt};
use dsi_bitstream::{
    prelude::{ToInt, ToNat},
    traits::Endianness,
};
use num_traits::{Bounded, NumCast, ToPrimitive};
use std::fmt::Debug;

/// A trait that abstracts over the primitive unsigned integer types that can
/// serve as the storage words in a [`FixedVec`].
///
/// This trait establishes a contract for what constitutes a "machine word"
/// for storage, providing access to its size in bits and requiring the
/// necessary traits for bit-level operations.
pub trait Word:
    UnsignedInt
    + Bounded
    + ToPrimitive
    + dsi_bitstream::traits::Word
    + NumCast
    + Copy
    + Send
    + Sync
    + Debug
    + IntoAtomic
    + 'static
{
    /// The number of bits in this word type (e.g., 64 for `u64`).
    const BITS: usize = std::mem::size_of::<Self>() * 8;
}

/// A macro to implement the `Word` trait for a given list of unsigned integer types.
macro_rules! impl_word_for {
    ($($t:ty),*) => {$(
        impl Word for $t {}
    )*};
}

// Implement `Word` for all standard unsigned integer types.
impl_word_for!(u8, u16, u32, u64, usize);

/// A trait that defines a bidirectional, lossless conversion between a user-facing
/// element type `T` and its storage representation `W`.
///
/// This trait is central to `FixedVec`'s ability to store various integer types
/// in a generic bit buffer.
pub trait Storable<W: Word>: Sized + Copy {
    /// Converts the element into its storage word representation.
    ///
    /// For signed integers, this conversion uses ZigZag encoding to map negative
    /// and positive values to a compact unsigned representation.
    fn into_word(self) -> W;
    /// Converts a storage word representation back into an element.
    ///
    /// For signed integers, this reverses the ZigZag encoding.
    fn from_word(word: W) -> Self;
}

/// Macro to implement `Storable` for unsigned integer types.
///
/// This implementation is a direct, lossless cast between the unsigned
/// element type and the storage word type `W`.
macro_rules! impl_storable_for_unsigned {
    ($($T:ty),*) => {$(
        impl<W> Storable<W> for $T
        where
            W: Word + TryFrom<$T>,
            W: TryInto<$T>,
        {
            #[inline(always)]
            fn into_word(self) -> W {
                self.try_into().unwrap_or_else(|_| panic!("BUG: T -> W conversion failed."))
            }

            #[inline(always)]
            fn from_word(word: W) -> Self {
                word.try_into().unwrap_or_else(|_| {
                    panic!("BUG: W -> T conversion failed. Logic error in FixedVec's bit manipulation.")
                })
            }
        }
    )*};
}

/// Macro to implement `Storable` for signed integer types using ZigZag encoding.
///
/// ZigZag encoding maps signed integers to unsigned integers in a way that is
/// efficient for variable-length encoding but is also used here to ensure that
/// small signed values (positive or negative) map to small unsigned values.
macro_rules! impl_storable_for_signed {
    ($($T:ty),*) => {$(
        impl<W> Storable<W> for $T
        where
            W: Word,
            <$T as SignedInt>::UnsignedInt: TryInto<W>,
            W: TryInto<<$T as SignedInt>::UnsignedInt>,
        {
            #[inline(always)]
            fn into_word(self) -> W {
                self.to_nat().try_into().unwrap_or_else(|_| panic!("BUG: Signed -> Unsigned -> W conversion failed."))
            }

            #[inline(always)]
            fn from_word(word: W) -> Self {
                let unsigned_val = word.try_into().unwrap_or_else(|_| {
                    panic!("BUG: W -> Unsigned conversion failed. Logic error in FixedVec.")
                });
                ToInt::to_int(unsigned_val)
            }
        }
    )*};
}

// Implement `Storable` for all primitive integer types.
impl_storable_for_unsigned!(u8, u16, u32, u64, u128, usize);
impl_storable_for_signed!(i8, i16, i32, i64, i128, isize);

/// A sealed trait to associate an element type `T` with its default storage
/// word `W` and `Endianness` `E`.
///
/// This allows for the creation of convenient type aliases like `UFixedVec<T>`
/// that do not require specifying all generic parameters.
pub trait DefaultParams: Sized {
    /// The default word type for storage (usually `usize`).
    type W: Word;
    /// The default endianness (usually `LittleEndian`).
    type E: Endianness;
}

/// Macro to implement `DefaultParams` for unsigned integer types.
macro_rules! impl_default_params_unsigned {
    ($($T:ty),*) => {$(
        impl DefaultParams for $T {
            type W = usize;
            type E = dsi_bitstream::prelude::LE;
        }
    )*};
}

/// Macro to implement `DefaultParams` for signed integer types.
macro_rules! impl_default_params_signed {
    ($($T:ty),*) => {$(
        impl DefaultParams for $T {
            type W = usize;
            type E = dsi_bitstream::prelude::LE;
        }
    )*};
}

impl_default_params_unsigned!(u8, u16, u32, u64, u128, usize);
impl_default_params_signed!(i8, i16, i32, i64, i128, isize);
