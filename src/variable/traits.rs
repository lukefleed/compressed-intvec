//! Core traits for the [`variable`] module.
//!
//! This module defines the [`Storable`] trait, a generic abstraction that allows
//! different integer types to be stored in an [`IntVec`]. It handles the necessary
//! conversions to the [`u64`] word representation required by the underlying
//! variable-length compression codecs.
//!
//! [`IntVec`]: crate::variable::IntVec
//! [`variable`]: crate::variable

use dsi_bitstream::prelude::{ToInt, ToNat};

/// A trait for types that can be stored in a variable-length compressed vector.
///
/// This trait provides a bidirectional, lossless conversion between a user-facing
/// element type (e.g., [`i32`], [`u16`]) and a [`u64`] representation. This abstraction
/// is essential for [`IntVec`] to support a variety of integer types while using
/// a common set of compression algorithms that operate on [`u64`].
///
/// # Portability and [`usize`]/[`isize`]
///
/// By default, this trait is implemented only for fixed-size integer types
/// (e.g., [`u8`], [`i16`], [`u32`], [`i64`]). This design choice guarantees that data
/// compressed on one machine architecture (e.g., 64-bit) can be safely
/// decompressed on another (e.g., 32-bit) without data loss or corruption.
///
/// Support for the architecture-dependent types [`usize`] and [`isize`] can be
/// enabled via the `arch-dependent-storable` feature flag.
///
/// ## Feature Flag: `arch-dependent-storable`
///
/// Activating this feature provides [`Storable`] implementations for [`usize`] and
/// [`isize`]. However, this breaks the portability guarantee. An `IntVec<usize>`
/// created on a 64-bit system with values greater than [`u32::MAX`] will cause a
/// panic if it is read on a 32-bit system.
///
/// **Enable this feature only if you are certain that the data will never be
/// shared across platforms with different pointer widths.**
///
/// # Zig-Zag Encoding for Signed Integers
///
/// For signed integer types, the implementation of this trait automatically
/// applies **Zig-Zag encoding**. This is a reversible transformation that maps
/// signed integers to unsigned integers, such that values close to zero (both
/// positive and negative) are mapped to small unsigned integers. This is highly
/// effective for variable-length codes, which use fewer bits to represent smaller
/// numbers.
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
/// [`IntVec`]: crate::variable::IntVec
pub trait Storable: Sized + Copy {
    /// Converts the element into its [`u64`] storage representation.
    ///
    /// For unsigned types, this is a direct cast. For signed types, this
    /// applies Zig-Zag encoding.
    fn to_word(self) -> u64;

    /// Converts a [`u64`] storage word back into the element type.
    ///
    /// For unsigned types, this is a direct cast. For signed types, this
    /// reverses the Zig-Zag encoding.
    ///
    /// # Panics
    ///
    /// This method will panic if the `word` contains a value that is out of
    /// range for the target type. This can occur if the data is corrupted or,
    /// in the case of [`usize`]/[`isize`] with the `arch-dependent-storable` feature,
    /// if the data is read on an architecture with a smaller pointer width than
    /// the one it was written on.
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

// Implement `Storable` for all primitive, fixed-size integer types.
impl_storable_for_unsigned!(u8, u16, u32, u64);
impl_storable_for_signed!(i8, i16, i32, i64);

/// Contains `Storable` implementations for architecture-dependent types.
/// This module is only compiled when the `arch-dependent-storable` feature is enabled.
#[cfg(feature = "arch-dependent-storable")]
mod arch_dependent {
    use super::*;
    use std::convert::TryFrom;

    impl Storable for usize {
        #[inline(always)]
        fn to_word(self) -> u64 {
            self as u64
        }

        #[inline(always)]
        fn from_word(word: u64) -> Self {
            usize::try_from(word).unwrap_or_else(|_| {
                panic!("Value {} out of range for this architecture's usize", word)
            })
        }
    }

    impl Storable for isize {
        #[inline(always)]
        fn to_word(self) -> u64 {
            self.to_nat() as u64
        }

        #[inline(always)]
        fn from_word(word: u64) -> Self {
            let word_as_usize = usize::try_from(word).unwrap_or_else(|_| {
                panic!(
                    "Value {} out of range for this architecture's usize (for isize decoding)",
                    word
                )
            });
            ToInt::to_int(word_as_usize)
        }
    }
}
