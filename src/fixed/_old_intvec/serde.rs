//! Manual [`serde`] implementation for [`FixedVec`].
//!
//! A manual implementation is adopted to avoid serializing the `mask` field,
//! which is derived data, and to correctly handle generic backends.

#![cfg_attr(docsrs, doc(cfg(feature = "serde")))]

use crate::fixed::intvec::{Endianness, FixedVec};
pub use serde::{Deserialize, Deserializer, Serialize, Serializer}; // Rendi pubblici gli use

/// A private helper struct for serializing and deserializing a [`FixedVec`].
#[derive(Serialize, Deserialize)]
struct FixedVecProxy<B: AsRef<[u64]>> {
    bits: B,
    len: usize,
    num_bits: usize,
}

impl<E: Endianness, B: AsRef<[u64]> + Serialize> Serialize for FixedVec<E, B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let proxy = FixedVecProxy {
            bits: self.bits.as_ref(), // Always serialize as a slice for consistency
            len: self.len,
            num_bits: self.num_bits,
        };
        proxy.serialize(serializer)
    }
}

// We implement Deserialize only for the owned version Vec<u64>
// as deserializing into a borrowed slice `&[u64]` is a more complex
// zero-copy deserialization pattern that is out of scope for now.
impl<'de, E: Endianness> Deserialize<'de> for FixedVec<E, Vec<u64>> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let proxy = FixedVecProxy::<Vec<u64>>::deserialize(deserializer)?;
        // SAFETY: The proxy contains all necessary data, and `new_unchecked`
        // will correctly initialize the struct, including recalculating the mask.
        Ok(unsafe { FixedVec::new_unchecked(proxy.bits, proxy.len, proxy.num_bits) })
    }
}
