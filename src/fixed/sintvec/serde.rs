//! Manual [`serde`] implementation for [`SFixedVec`].
//!
//! This is implemented manually to leverage `serde(transparent)` behavior
//! without using `derive`, which caused issues with generic bounds.

#![cfg_attr(docsrs, doc(cfg(feature = "serde")))]

use crate::fixed::intvec::FixedVec;
use crate::fixed::sintvec::{Endianness, SFixedVec};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl<E: Endianness, B: AsRef<[u64]> + Serialize> Serialize for SFixedVec<E, B> {
    /// Delegates serialization to the inner `FixedVec`.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

impl<'de, E: Endianness> Deserialize<'de> for SFixedVec<E, Vec<u64>> {
    /// Delegates deserialization to the inner `FixedVec`.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(SFixedVec {
            inner: FixedVec::<E, Vec<u64>>::deserialize(deserializer)?,
        })
    }
}
