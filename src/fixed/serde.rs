//! # Serde Serialization and Deserialization
//!
//! This module provides `serde` implementations for [`FixedVec`] and [`AtomicFixedVec`],
//! enabled by the `serde` feature flag. It uses a "proxy struct" pattern to
//! ensure that the public API remains clean and to handle the specific needs
//! of serialization and deserialization.
//!
//! # Examples
//!
//! ## Serializing and deserializing a [`FixedVec`]
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # #[cfg(feature = "serde")]
//! # {
//! use compressed_intvec::fixed::{FixedVec, UFixedVec};
//!
//! let vec: UFixedVec<u32> = (0..10).collect();
//!
//! // Serialize the vector to a JSON string.
//! let serialized = serde_json::to_string(&vec)?;
//!
//! // Deserialize it back.
//! let deserialized: UFixedVec<u32> = serde_json::from_str(&serialized)?;
//!
//! assert_eq!(vec, deserialized);
//! # }
//! # Ok(())
//! # }
//! ```

use super::{
    atomic::AtomicFixedVec,
    traits::{Storable, Word},
    FixedVec,
};
use dsi_bitstream::prelude::Endianness;
use num_traits::ToPrimitive;
use serde::{
    de,
    ser::{Serializer},
    Deserialize, Deserializer, Serialize,
};
use std::sync::atomic::Ordering;

// --- `FixedVec` Serde Implementation ---

/// A private proxy struct for deserializing [`FixedVec`].
///
/// This struct holds the essential data needed to reconstruct a [`FixedVec`].
/// It is used to decouple the `serde` implementation from the main struct's
/// internal details and type parameters.
#[derive(Deserialize)]
#[serde(bound(deserialize = "B: Deserialize<'de>"))]
struct FixedVecProxy<B> {
    bits: B,
    len: usize,
    bit_width: usize,
}

impl<T, W, E, B> Serialize for FixedVec<T, W, E, B>
where
    T: Storable<W>,
    W: Word,
    E: Endianness,
    B: AsRef<[W]>,
    W: Serialize,
{
    /// Serializes a [`FixedVec`] by creating and serializing a temporary proxy struct.
    /// This pattern correctly handles lifetimes when serializing slice-backed data.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct SerializeProxy<'a, W: Word> {
            bits: &'a [W],
            len: usize,
            bit_width: usize,
        }

        let proxy = SerializeProxy {
            bits: self.bits.as_ref(),
            len: self.len,
            bit_width: self.bit_width,
        };
        proxy.serialize(serializer)
    }
}

impl<'de, T, W, E> Deserialize<'de> for FixedVec<T, W, E, Vec<W>>
where
    T: Storable<W>,
    W: Word + Deserialize<'de>,
    E: Endianness,
{
    /// Deserializes data into an owned [`FixedVec`].
    ///
    /// The implementation first deserializes into a proxy struct and then
    /// validates the data to ensure it can form a valid [`FixedVec`].
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize into the proxy struct. This will produce an owned Vec<W>.
        let proxy = FixedVecProxy::<Vec<W>>::deserialize(deserializer)?;

        // Validate the deserialized data to prevent creating an invalid state.
        let word_size_bits = std::mem::size_of::<W>() * 8;
        if proxy.bit_width > word_size_bits {
            return Err(de::Error::custom(format!(
                "Deserialized bit_width ({}) cannot be greater than the word size ({})",
                proxy.bit_width, word_size_bits
            )));
        }

        let required_bits = proxy.len.saturating_mul(proxy.bit_width);
        let required_data_words = required_bits.div_ceil(word_size_bits);

        // The buffer must be large enough to contain the specified number of elements.
        // We check for `required_data_words` as a minimum, allowing for padding.
        if proxy.bits.len() < required_data_words {
            return Err(de::Error::custom(format!(
                "Deserialized buffer is too small. It has {} words, but at least {} are required.",
                proxy.bits.len(),
                required_data_words
            )));
        }

        // SAFETY: We have validated that the parameters are consistent. The `mask`
        // field will be correctly recalculated by `new_unchecked`.
        Ok(unsafe { FixedVec::new_unchecked(proxy.bits, proxy.len, proxy.bit_width) })
    }
}

// --- `AtomicFixedVec` Serde Implementation ---

/// A private proxy struct for serializing and deserializing [`AtomicFixedVec`].
#[derive(Serialize, Deserialize)]
struct AtomicFixedVecProxy {
    storage: Vec<u64>,
    len: usize,
    bit_width: usize,
}

impl<T> Serialize for AtomicFixedVec<T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    /// Serializes an [`AtomicFixedVec`].
    ///
    /// This is done by atomically loading each element from the `Vec<AtomicU64>`
    /// into a plain `Vec<u64>` and then serializing a proxy struct.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert `Vec<AtomicU64>` to `Vec<u64>` for serialization.
        // Relaxed ordering is sufficient as we just need any consistent atomic view.
        let storage_plain: Vec<u64> = self
            .as_slice()
            .iter()
            .map(|atomic| atomic.load(Ordering::Relaxed))
            .collect();

        let proxy = AtomicFixedVecProxy {
            storage: storage_plain,
            len: self.len(),
            bit_width: self.bit_width(),
        };

        proxy.serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for AtomicFixedVec<T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    /// Deserializes data into an [`AtomicFixedVec`].
    ///
    /// This uses the proxy struct and the internal `new` constructor to safely
    /// reconstruct the atomic vector, including its transient fields like `locks`.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let proxy = AtomicFixedVecProxy::deserialize(deserializer)?;

        // The `new` constructor performs all necessary validation of `bit_width`
        // and `len`, and correctly initializes the lock structure.
        let atomic_vec = AtomicFixedVec::<T>::new(proxy.bit_width, proxy.len)
            .map_err(|err| de::Error::custom(err.to_string()))?;

        let expected_words = atomic_vec.as_slice().len();
        if proxy.storage.len() != expected_words {
            return Err(de::Error::custom(format!(
                "Mismatched storage length: expected {}, got {}",
                expected_words,
                proxy.storage.len()
            )));
        }

        // Fill the newly allocated storage with the deserialized data.
        let storage_slice = atomic_vec.as_slice();
        for (i, val) in proxy.storage.into_iter().enumerate() {
            storage_slice[i].store(val, Ordering::Relaxed);
        }

        Ok(atomic_vec)
    }
}
