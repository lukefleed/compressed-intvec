//! # A compressed, randomly accessible vector of signed `i64` integers.
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

use crate::variable::{
    intvec::{IntVec, IntVecBitReader, IntVecError},
    sintvec::{builder::SIntVecFromIterBuilder, iter::SIntVecIter},
};
use dsi_bitstream::prelude::{BitRead, BitSeek, Codes, CodesRead, Endianness, ToInt, BE, LE};
use mem_dbg::{MemDbg, MemSize};

// Declare and export submodules.
mod builder;
mod iter;
#[cfg(feature = "parallel")]
mod parallel;

pub use builder::SIntVecBuilder;

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
/// The [`SIntVecBuilder`] **requires that codec parameters be
/// specified manually**. Automatic parameter selection is not supported because the
/// on-the-fly ZigZag transformation prevents a pre-analysis pass.
///
/// # Example
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[i64] = &;
///
/// // SIntVec requires manual codec selection. Let's use Gamma.
/// let sintvec = LESIntVec::builder(&data)
///     .codec(VariableCodecSpec::Gamma)
///     .k(4)
///     .build()
///     .unwrap();
///
/// assert_eq!(sintvec.len(), data.len());
/// assert_eq!(sintvec.get(0), Some(-10));
/// ```
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent, bound = ""))]
pub struct SIntVec<E: Endianness> {
    /// The inner `IntVec` that stores the ZigZag-encoded `u64` values.
    inner: IntVec<E>,
}

impl<E: Endianness> SIntVec<E>
where
    for<'b> crate::variable::intvec::IntVecBitReader<'b, E>: dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
        + dsi_bitstream::dispatch::CodesRead<E>
        + dsi_bitstream::prelude::BitSeek<Error = core::convert::Infallible>,
{
    /// Returns a builder for creating an [`SIntVec`] from a slice of data.
    ///
    /// This method is generic over `AsRef<[i64]>`, so it can accept `&[i64]`,
    /// `Vec<i64>`, etc. See [`SIntVecBuilder`] for more details.
    pub fn builder<T: AsRef<[i64]> + ?Sized>(input: &T) -> SIntVecBuilder<E> {
        SIntVecBuilder::new(input.as_ref())
    }

    /// Returns a builder for creating a [`SIntVec`] from an iterator.
    pub fn from_iter_builder<I: IntoIterator<Item = i64>>(iter: I) -> SIntVecFromIterBuilder<E, I> {
        SIntVecFromIterBuilder::new(iter)
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the underlying `Codes` variant used for compression.
    pub fn encoding(&self) -> Codes {
        self.inner.encoding()
    }

    /// Returns the sampling rate `k` used during encoding.
    pub fn get_sampling_rate(&self) -> Option<usize> {
        self.inner.get_sampling_rate()
    }

    /// Returns the number of sample points stored in the vector.
    pub fn get_num_samples(&self) -> usize {
        self.inner.get_num_samples()
    }

    /// Returns a clone of the underlying storage (`Vec<u64>`).
    pub fn limbs(&self) -> Vec<u64> {
        self.inner.limbs()
    }
}

impl<E: Endianness> SIntVec<E>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Retrieves the signed integer at the specified index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<i64> {
        self.inner.get(index).map(ToInt::to_int)
    }

    /// Retrieves the signed integer at the specified index without bounds checking.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> i64 {
        self.inner.get_unchecked(index).to_int()
    }

    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<i64>, IntVecError> {
        let unsigned_values = self.inner.get_many(indices)?;
        Ok(unsigned_values.into_iter().map(ToInt::to_int).collect())
    }

    /// Retrieves multiple signed integers at the specified indices without bounds checking.
    ///
    /// # Safety
    /// Calling this method with any out-of-bounds index is undefined behavior.
    pub unsafe fn get_many_unchecked(&self, indices: &[usize]) -> Vec<i64> {
        self.inner
            .get_many_unchecked(indices)
            .into_iter()
            .map(ToInt::to_int)
            .collect()
    }

    /// Retrieves multiple signed integers from an iterator of indices.
    pub fn get_many_from_iter<I>(&self, indices: I) -> Result<Vec<i64>, IntVecError>
    where
        I: IntoIterator<Item = usize>,
    {
        let unsigned_values = self.inner.get_many_from_iter(indices)?;
        Ok(unsigned_values.into_iter().map(ToInt::to_int).collect())
    }

    /// Returns an iterator over the decompressed `i64` values.
    pub fn iter(&self) -> SIntVecIter<E> {
        SIntVecIter::new(self)
    }
}

/// A type alias for an [`SIntVec`] with Big-Endian ([`BE`]) bitstream encoding.
///
/// See [`BEIntVec`] for more details on endianness.
///
/// [`BEIntVec`]: crate::variable::intvec::BEIntVec
pub type BESIntVec = SIntVec<BE>;

/// A type alias for an [`SIntVec`] with Little-Endian ([`LE`]) bitstream encoding.
///
/// See [`LEIntVec`] for more details on endianness.
///
/// [`LEIntVec`]: crate::variable::intvec::LEIntVec
pub type LESIntVec = SIntVec<LE>;
