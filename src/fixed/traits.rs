//! # Core Traits
//!
//! This module defines the traits that enable the generic implementation of
//! [`FixedVec`]. These traits abstract the underlying storage words and the
//`! conversion logic for different element types.

use common_traits::{SignedInt, UnsignedInt};
use dsi_bitstream::{prelude::{ToInt, ToNat}, traits::Endianness};
use num_traits::{Bounded, NumCast, ToPrimitive};
use std::fmt::Debug;

/// A trait for primitive unsigned integer types that can be used as storage
/// words in a [`FixedVec`].
///
/// This trait establishes a contract for what constitutes a "machine word" for
/// the underlying bit buffer, providing access to its size in bits and requiring
/// the necessary traits for bitstream operations.
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

// Implement `Word` for all standard unsigned integer types that dsi-bitstream supports.
impl_word_for!(u8, u16, u32, u64, u128, usize);

/// A private module to seal the `Storable` trait implementation details.
mod private {
    use super::{Storable, Word};

    /// The sealed trait that contains the actual conversion logic.
    pub trait SealedStorable<W: Word>: Copy + Sized {
        fn into_word(self) -> W;
        fn from_word(word: W) -> Self;
    }

    impl<T: SealedStorable<W>, W: Word> Storable<W> for T {}
}

/// A trait for types that can be stored in a [`FixedVec`].
///
/// This trait defines a bidirectional, lossless conversion between a user-facing
/// element type `T` and its storage representation `W`. For signed integers,
/// this conversion involves ZigZag encoding to map signed values to unsigned
/// words.
pub trait Storable<W: Word>: private::SealedStorable<W> {
    /// Converts the element into its storage representation.
    #[inline(always)]
    fn into_word(self) -> W {
        <Self as private::SealedStorable<W>>::into_word(self)
    }

    /// Converts a storage word back into the element type.
    #[inline(always)]
    fn from_word(word: W) -> Self {
        <Self as private::SealedStorable<W>>::from_word(word)
    }
}

/// Macro to implement `SealedStorable` for unsigned integer types.
macro_rules! impl_storable_for_unsigned {
    ($($T:ty),*) => {$(
        impl<W> private::SealedStorable<W> for $T
        where
            // Use TryFrom/TryInto for both directions for maximum flexibility.
            W: Word + TryFrom<$T>,
            W: TryInto<$T>,
        {
            #[inline(always)]
            fn into_word(self) -> W {
                // The conversion from a smaller/equal unsigned to a larger/equal Word
                // should not fail.
                self.try_into().unwrap_or_else(|_| panic!("BUG: T -> W conversion failed."))
            }

            #[inline(always)]
            fn from_word(word: W) -> Self {
                // The `get` logic masks the word, so this conversion should not fail.
                word.try_into().unwrap_or_else(|_| {
                    panic!("BUG: W -> T conversion failed. Logic error in FixedVec's bit manipulation.")
                })
            }
        }
    )*};
}

/// Macro to implement `SealedStorable` for signed integer types using ZigZag encoding.
macro_rules! impl_storable_for_signed {
    ($($T:ty),*) => {$(
        impl<W> private::SealedStorable<W> for $T
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

/// A sealed trait to associate an element type `T` with its default optimal
/// storage word `W` and `Endianness` `E`.
pub trait DefaultParams: Sized {
    /// The default word type for storage (usually `usize`).
    type W: Word;
    /// The default endianness (usually `LittleEndian`).
    type E: Endianness;
}

// Implement for all unsigned types
macro_rules! impl_default_params_unsigned {
    ($($T:ty),*) => {$(
        impl DefaultParams for $T {
            type W = usize;
            type E = dsi_bitstream::prelude::LE;
        }
    )*};
}

// Implement for all signed types
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