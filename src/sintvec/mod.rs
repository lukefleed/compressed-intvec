//! A compressed, randomly accessible vector of signed `i64` integers.
//!
//! This module provides [`SIntVec`], a specialized vector for compressing signed
//! integer data.
//!
//! # The Challenge of Compressing Signed Integers
//!
//! Standard bit-level compression codes (like Gamma or Delta) are designed for
//! non-negative integers and perform best on small values. A direct `i64 as u64`
//! cast is problematic because small negative numbers (e.g., -1, -2) become very
//! large positive numbers (e.g., `u64::MAX`, `u64::MAX - 1`), which are highly
//! inefficient to compress with these codes.
//!
//! # ZigZag Encoding
//!
//! To solve this, [`SIntVec`] uses a bijective mapping known as **ZigZag encoding**
//! to transform the signed integers into unsigned integers before compression. This
//! transformation maps integers close to zero (both positive and negative) to small
//! positive integers, making them highly compressible. The mapping is as follows:
//!
//! | Original `i64` | Mapped `u64` |
//! | :------------: | :----------: |
//! |       0        |      0       |
//! |      -1        |      1       |
//! |       1        |      2       |
//! |      -2        |      3       |
//! |       2        |      4       |
//! |      ...       |     ...      |
//!
//! This transformation is handled transparently by [`SIntVec`]. All compression,
//! storage, and random access logic is then delegated to an inner [`IntVec`].

use crate::codec_spec::Encoding;
use crate::intvec::{IntVec, IntVecError, IntVecSeqReader};
use dsi_bitstream::prelude::{Endianness, ToInt, BE, LE};
use mem_dbg::{MemDbg, MemSize};

// Declare and export submodules.
mod builder;
mod iter;
#[cfg(feature = "parallel")]
mod parallel;

pub use builder::SIntVecBuilder;
pub use iter::SIntVecIter;

/// A compressed, randomly accessible vector of signed `i64` integers.
///
/// [`SIntVec`] acts as a wrapper around [`IntVec`] that transparently handles the
/// encoding of signed integers (`i64`) into unsigned integers (`u64`) using
/// the ZigZag transformation. This allows for efficient compression of typical
/// signed integer distributions, where values are often clustered around zero.
///
/// All compression logic and storage are delegated to the inner [`IntVec`].
/// This struct exposes an equivalent API but operates on `i64` values. The
/// performance characteristics of its methods are nearly identical to their
/// [`IntVec`] counterparts, with only the negligible overhead of the `to_int`
/// transformation on the final results.
///
/// # Limitations
///
/// Unlike `IntVec`, the [`SIntVecBuilder`] **requires that codec parameters be
/// specified manually**. Automatic parameter selection is not supported because the
/// on-the-fly ZigZag transformation of the data prevents the builder from performing
/// a pre-analysis pass to determine optimal codec parameters.
///
/// # Example
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[i64] = &[-10, 20, -30, 40, -50, 60, -70, 80, -90, 100];
///
/// // SIntVec requires manual codec selection. Let's use Gamma.
/// let sintvec = LESIntVec::builder(data)
///     .codec(CodecSpec::Gamma)
///     .k(4)
///     .build()
///     .unwrap();
///
/// assert_eq!(sintvec.len(), data.len());
/// assert_eq!(sintvec.get(0), Some(-10));
/// assert_eq!(sintvec.get(2), Some(-30));
/// assert_eq!(sintvec.get(3), Some(40));
/// ```
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct SIntVec<E: Endianness> {
    /// The inner `IntVec` that stores the ZigZag-encoded `u64` values.
    inner: IntVec<E>,
}

impl<E: Endianness> SIntVec<E>
where
    for<'a> crate::intvec::IntVecBitReader<'a, E>: dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
        + dsi_bitstream::dispatch::CodesRead<E>
        + dsi_bitstream::prelude::BitSeek<Error = core::convert::Infallible>,
{
    /// Returns a builder for creating an [`SIntVec`] from a slice (`&[i64]`).
    ///
    /// See [`SIntVecBuilder`] for more details.
    pub fn builder(input: &[i64]) -> SIntVecBuilder<E> {
        SIntVecBuilder::new(input)
    }

    /// Retrieves the signed integer at the specified index.
    ///
    /// This method delegates to [`IntVec::get`] and applies the inverse ZigZag
    /// transformation to the result.
    pub fn get(&self, index: usize) -> Option<i64> {
        self.inner.get(index).map(ToInt::to_int)
    }

    /// Retrieves multiple signed integers at the specified indices.
    ///
    /// This method delegates to [`IntVec::get_many`] and applies the inverse ZigZag transformation to the results.
    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<i64>, IntVecError> {
        let unsigned_values = self.inner.get_many(indices)?;
        Ok(unsigned_values.into_iter().map(ToInt::to_int).collect())
    }

    /// Retrieves multiple signed integers from an iterator of indices.
    ///
    /// This method delegates to [`IntVec::get_many_from_iter`] and applies
    /// the inverse ZigZag transformation to the results.
    pub fn get_many_from_iter<I>(&self, indices: I) -> Result<Vec<i64>, IntVecError>
    where
        I: IntoIterator<Item = usize>,
    {
        let unsigned_values = self.inner.get_many_from_iter(indices)?;
        Ok(unsigned_values.into_iter().map(ToInt::to_int).collect())
    }

    /// Returns a stateful, sequential reader for this vector's underlying data.
    ///
    /// This method provides access to the sequential reader of the inner
    /// [`IntVec`]. It is designed for dynamic, stateful access patterns where
    /// indices are mostly sequential, offering significant performance gains by
    /// avoiding unnecessary seeks.
    ///
    /// See [`IntVec::seq_reader`] for detailed documentation on its behavior
    /// and performance characteristics.
    /// ```
    pub fn seq_reader(&self) -> IntVecSeqReader<E> {
        self.inner.seq_reader()
    }

    /// Returns an iterator over the decompressed `i64` values.
    ///
    /// The iterator wraps the inner [`IntVec`]'s iterator and applies the inverse
    /// ZigZag transformation to each value on the fly. See [`SIntVecIter`].
    pub fn iter(&self) -> SIntVecIter<E> {
        SIntVecIter::new(self)
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the underlying `Encoding` used for compression.
    pub fn encoding(&self) -> Encoding {
        self.inner.encoding()
    }

    /// Returns the sampling rate `k` used during encoding, if applicable.
    pub fn get_sampling_rate(&self) -> Option<usize> {
        self.inner.get_sampling_rate()
    }
}

/// A type alias for an [`SIntVec`] with Big-Endian ([`BE`]) bitstream encoding.
///
/// See [`BEIntVec`] for more details on endianness.
///
/// [`BEIntVec`]: crate::intvec::BEIntVec
pub type BESIntVec = SIntVec<BE>;

/// A type alias for an [`SIntVec`] with Little-Endian ([`LE`]) bitstream encoding.
///
/// See [`LEIntVec`] for more details on endianness.
///
/// [`LEIntVec`]: crate::intvec::LEIntVec
pub type LESIntVec = SIntVec<LE>;

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
mod serde_impls {
    use super::{Endianness, IntVec, SIntVec};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl<E: Endianness> Serialize for SIntVec<E>
    where
        IntVec<E>: Serialize,
    {
        /// Serializes the `SIntVec` by delegating to its inner `IntVec`.
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            // Since SIntVec is a transparent wrapper, we just serialize the inner field.
            self.inner.serialize(serializer)
        }
    }

    impl<'de, E: Endianness> Deserialize<'de> for SIntVec<E>
    where
        IntVec<E>: Deserialize<'de>,
    {
        /// Deserializes the `SIntVec` by deserializing its inner `IntVec`.
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            // Deserialize the inner IntVec and wrap it in a new SIntVec.
            let inner = IntVec::<E>::deserialize(deserializer)?;
            Ok(SIntVec { inner })
        }
    }
}
