//! [`serde`] support for [`VarVec`].
//!
//! This module provides `Serialize` and `Deserialize` implementations for
//! [`VarVec`], allowing it to be easily serialized and deserialized with formats
//! like JSON, Bincode, etc. This is enabled by the `serde` feature flag.
//!
//! # Implementation
//!
//! A manual implementation is necessary because `VarVec` uses generic type
//! parameters that don't directly map to serde's derive capabilities. The
//! `Codes` type from `dsi-bitstream` natively supports serde in version 0.9+.
//!
//! # Examples
//!
//! Serializing and deserializing an `VarVec` using `serde_json`:
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # #[cfg(feature = "serde")] {
//! use compressed_intvec::prelude::*;
//!
//! // The `sint_vec!` macro creates a vector of `i64`.
//! // We specify the type of `vec` as `LESVarVec` to match.
//! let vec: LESVarVec = sint_vec![-10, 20, -30, 40, -50];
//!
//! // Serialize the vector to a JSON string
//! let serialized = serde_json::to_string(&vec)?;
//!
//! // Deserialize the JSON string back into an VarVec of the same type.
//! let deserialized: LESVarVec = serde_json::from_str(&serialized)?;
//!
//! // To compare for equality, we can collect both into a standard Vec.
//! // This verifies that the content is identical after the round trip.
//! assert_eq!(vec.iter().collect::<Vec<_>>(), deserialized.iter().collect::<Vec<_>>());
//! # }
//! # Ok(())
//! # }
//! ```
//!
//! [`serde`]: https://serde.rs/
//! [`VarVec`]: crate::variable::VarVec

use super::{traits::Storable, Endianness, VarVec};
use crate::fixed::{FixedVec, LEFixedVec};
use dsi_bitstream::prelude::{Codes, LE};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl<T: Storable, E: Endianness, B: AsRef<[u64]> + Serialize> Serialize for VarVec<T, E, B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct SerializeProxy<'a, BSamples: AsRef<[u64]> + Serialize> {
            data: &'a [u64],
            samples: &'a FixedVec<u64, u64, LE, BSamples>,
            k: usize,
            len: usize,
            encoding: Codes,
        }

        let proxy = SerializeProxy {
            data: self.data.as_ref(),
            samples: &self.samples,
            k: self.k,
            len: self.len,
            encoding: self.encoding,
        };
        proxy.serialize(serializer)
    }
}

/// A helper struct for deserializing an owned `VarVec`.
#[derive(Deserialize)]
#[serde(rename = "VarVec")]
struct VarVecProxy {
    data: Vec<u64>,
    samples: LEFixedVec,
    k: usize,
    len: usize,
    encoding: Codes,
}

impl<'de, T: Storable, E: Endianness> Deserialize<'de> for VarVec<T, E, Vec<u64>> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = VarVecProxy::deserialize(deserializer)?;
        // SAFETY: The deserialized proxy struct contains all necessary components,
        // which are assumed to be consistent as they were serialized together.
        Ok(unsafe {
            VarVec::new_unchecked(
                helper.data,
                helper.samples,
                helper.k,
                helper.len,
                helper.encoding,
            )
        })
    }
}
