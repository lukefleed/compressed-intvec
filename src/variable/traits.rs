//! Core traits for the `variable` module.
//!
//! This module defines the [`Storable`] trait, which provides a generic
//! abstraction over the integer types that can be stored in an [`IntVec`].
//!
//! [`IntVec`]: crate::variable::IntVec

use dsi_bitstream::prelude::{ToInt, ToNat};

/// A trait for types that can be stored in a variable-length compressed vector.
///
/// This trait provides a bidirectional, lossless conversion between a user-facing
/// element type (e.g., `i32`, `u16`) and the `u64` representation required by the
/// underlying compression codecs.
///
/// # Zig-Zag Encoding for Signed Integers
///
/// For signed integer types, this trait's implementation automatically applies
/// **Zig-Zag encoding**. This is a transformation that maps signed
/// integers to unsigned integers in a way that is efficient for variable-length
/// compression.
///
/// It works by mapping small positive and negative numbers to small unsigned
/// numbers, as shown below:
///
/// | Original Signed | Zig-Zag Unsigned |
/// |-----------------|------------------|
/// | 0               | 0                |
/// | -1              | 1                |
/// | 1               | 2                |
/// | -2              | 3                |
/// | 2               | 4                |
/// | ...             | ...              |
///
/// This ensures that values close to zero, whether positive or negative, are
/// represented by small unsigned integers, which can then be compressed into
/// very few bits by the variable-length codecs. 
pub trait Storable: Sized + Copy {
    /// Converts the element into its `u64` storage representation.
    ///
    /// For unsigned types, this is a simple cast. For signed types, this
    /// applies Zig-Zag encoding.
    fn to_word(self) -> u64;
    /// Converts a `u64` storage word back into the element type.
    ///
    /// For unsigned types, this is a simple cast. For signed types, this
    /// decodes the Zig-Zag encoded value.
    fn from_word(word: u64) -> Self;
}

macro_rules! impl_storable_for_unsigned {
    ($($T:ty),*) => {$(
        impl Storable for $T {
            #[inline(always)]
            fn to_word(self) -> u64 {
                self as u64
            }

            #[inline(always)]
            fn from_word(word: u64) -> Self {
                word as Self
            }
        }
    )*};
}

macro_rules! impl_storable_for_signed {
    ($($T:ty),*) => {$(
        impl Storable for $T {
            #[inline(always)]
            fn to_word(self) -> u64 {
                self.to_nat().into()
            }

            #[inline(always)]
            fn from_word(word: u64) -> Self {
                ToInt::to_int(word)
                    .try_into()
                    .unwrap_or_else(|_| panic!("Value out of range for type"))
            }
        }
    )*};
}

// Implement `Storable` for all primitive integer types.
impl_storable_for_unsigned!(u8, u16, u32, u64);
impl_storable_for_signed!(i8, i16, i32, i64);
