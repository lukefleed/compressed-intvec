//! Serialization of [`AtomicFixedVec`] with the `serde` feature, in the format of
//! [`FixedVec`].

use super::{AtomicFixedVec, FixedVec, layout};
use crate::Storable;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::atomic::Ordering::Relaxed;

impl<T: Storable> Serialize for AtomicFixedVec<T> {
    /// Serializes the elements, each loaded atomically with [`Relaxed`] ordering. Concurrent
    /// updates may yield a result containing elements observed in different vector states.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let required = layout::required_words(self.len, self.bit_width)
            .expect("the length of a vector does not overflow");
        let mut words = vec![0; required];
        layout::pack(
            &mut words,
            self.bit_width,
            (0..self.len).map(|index| self.load_word(index, Relaxed)),
        );
        FixedVec::<T>::from_parts(words, self.len, self.bit_width as u32)
            .expect("the words hold the elements")
            .serialize(serializer)
    }
}

impl<'de, T: Storable> Deserialize<'de> for AtomicFixedVec<T> {
    /// Deserializes the packed format of [`FixedVec`] and converts its storage to atomic words.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        FixedVec::<T>::deserialize(deserializer).map(Self::from)
    }
}
