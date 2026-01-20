//! [`serde`] support for [`IntVec`].
//!
//! This module provides `Serialize` and `Deserialize` implementations for
//! [`IntVec`], allowing it to be easily serialized and deserialized with formats
//! like JSON, Bincode, etc. This is enabled by the `serde` feature flag.
//!
//! # Implementation
//!
//! A manual implementation is necessary because the underlying `dsi-bitstream::codes::Codes`
//! enum does not implement the `serde` traits. This module uses a serializable
//! "proxy" enum to handle this conversion.
//!
//! # Examples
//!
//! Serializing and deserializing an `IntVec` using `serde_json`:
//!
//! ```
//! # #[cfg(feature = "serde")] {
//! use compressed_intvec::prelude::*;
//!
//! // The `sint_vec!` macro creates a vector of `i64`.
//! // We specify the type of `vec` as `LESIntVec` to match.
//! let vec: LESIntVec = sint_vec![-10, 20, -30, 40, -50];
//!
//! // Serialize the vector to a JSON string
//! let serialized = serde_json::to_string(&vec).unwrap();
//!
//! // Deserialize the JSON string back into an IntVec of the same type.
//! let deserialized: LESIntVec = serde_json::from_str(&serialized).unwrap();
//!
//! // To compare for equality, we can collect both into a standard Vec.
//! // This verifies that the content is identical after the round trip.
//! assert_eq!(vec.iter().collect::<Vec<_>>(), deserialized.iter().collect::<Vec<_>>());
//! # }
//! ```
//!
//! [`serde`]: https://serde.rs/
//! [`IntVec`]: crate::variable::IntVec

use super::{traits::Storable, Endianness, IntVec};
use crate::common::serde::CodesSerde;
use crate::fixed::{FixedVec, LEFixedVec};
use dsi_bitstream::prelude::LE;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl<T: Storable, E: Endianness, B: AsRef<[u64]> + Serialize> Serialize for IntVec<T, E, B> {
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
            encoding: CodesSerde,
        }

        let proxy = SerializeProxy {
            data: self.data.as_ref(),
            samples: &self.samples,
            k: self.k,
            len: self.len,
            encoding: self.encoding.into(),
        };
        proxy.serialize(serializer)
    }
}

/// A helper struct for deserializing an owned `IntVec`.
#[derive(Deserialize)]
#[serde(rename = "IntVec")]
struct IntVecProxy {
    data: Vec<u64>,
    samples: LEFixedVec,
    k: usize,
    len: usize,
    encoding: CodesSerde,
}

impl<'de, T: Storable, E: Endianness> Deserialize<'de> for IntVec<T, E, Vec<u64>> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = IntVecProxy::deserialize(deserializer)?;
        // SAFETY: The deserialized proxy struct contains all necessary components,
        // which are assumed to be consistent as they were serialized together.
        Ok(unsafe {
            IntVec::new_unchecked(
                helper.data,
                helper.samples,
                helper.k,
                helper.len,
                helper.encoding.into(),
            )
        })
    }
}
