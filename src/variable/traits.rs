use dsi_bitstream::prelude::{ToInt, ToNat};

/// A trait for types that can be stored in an [`IntVec`].
///
/// This trait defines a bidirectional, lossless conversion between a user-facing
/// element type `T` and the `u64` storage word required by the DSI codec.
/// For signed integers, this conversion involves ZigZag encoding to map signed
/// values to unsigned words.
pub trait Storable: Sized + Copy {
    /// Converts the element into its `u64` storage representation.
    fn to_word(self) -> u64;
    /// Converts a `u64` storage word back into the element type.
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

// Implement `Storable` for all primitive integer types supported by the variable-length codec.
impl_storable_for_unsigned!(u8, u16, u32, u64);
impl_storable_for_signed!(i8, i16, i32, i64);
