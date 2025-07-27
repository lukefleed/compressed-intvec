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
use crate::intvec::{IntVec, IntVecError};
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
/// [`SIntVec`] simply provides a convenient API that accepts and returns `i64` values.
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
/// let data: &[i64] = &[-10, 200, 30, -40, 50];
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
/// assert_eq!(sintvec.get(2), Some(30));
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
    /// This builder requires that codec parameters be specified manually because
    /// it transforms the data on-the-fly and cannot perform a pre-analysis pass.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: &[i64] = &[-10, 20, 30, -40, 50];
    ///
    /// // Codec parameters must be fixed.
    /// let sintvec = LESIntVec::builder(data)
    ///     .codec(CodecSpec::Delta)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(sintvec.get(1), Some(20));
    /// ```
    pub fn builder(input: &[i64]) -> SIntVecBuilder<E> {
        SIntVecBuilder::new(input)
    }

    /// Retrieves the signed integer at the specified index.
    ///
    /// This method retrieves the underlying compressed `u64` value from the inner
    /// [`IntVec`] and then applies the inverse ZigZag transformation to restore the
    /// original `i64` value.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: &[i64] = &[-10, 20, 30, -40, 50];
    /// let sintvec = LESIntVec::builder(data).codec(CodecSpec::Gamma).build().unwrap();
    ///
    /// assert_eq!(sintvec.get(0), Some(-10));
    /// assert_eq!(sintvec.get(4), Some(50));
    /// assert_eq!(sintvec.get(99), None); // Out of bounds
    /// ```
    pub fn get(&self, index: usize) -> Option<i64> {
        self.inner.get(index).map(ToInt::to_int)
    }

    /// Returns an iterator over the decompressed `i64` values.
    ///
    /// The iterator wraps the inner [`IntVec`]'s iterator and applies the inverse
    /// ZigZag transformation to each value on the fly.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: &[i64] = &[10, -20, 30, -40, 50];
    /// let sintvec = LESIntVec::builder(data).codec(CodecSpec::Gamma).build().unwrap();
    ///
    /// let collected: Vec<i64> = sintvec.iter().collect();
    /// assert_eq!(collected, data);
    /// ```
    pub fn iter(&self) -> SIntVecIter<E> {
        SIntVecIter::new(self)
    }

    /// Returns the number of elements in the vector.
    /// This is delegated to the inner [`IntVec`].
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the vector contains no elements.
    /// This is delegated to the inner [`IntVec`].
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// let empty_sintvec = LESIntVec::builder(&[]).codec(CodecSpec::Gamma).build().unwrap();
    /// assert!(empty_sintvec.is_empty());
    /// assert_eq!(empty_sintvec.len(), 0);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the underlying `Encoding` used for compression.
    /// This is delegated to the inner [`IntVec`].
    pub fn encoding(&self) -> Encoding {
        self.inner.encoding()
    }

    /// Returns the sampling rate `k` used during encoding, if applicable.
    /// This is delegated to the inner [`IntVec`].
    pub fn get_sampling_rate(&self) -> Option<usize> {
        self.inner.get_sampling_rate()
    }
}

/// A type alias for an [`SIntVec`] with Big-Endian ([`BE`]) bitstream encoding.
///
/// Big-endian is a byte order used in many networking protocols and on certain
/// CPU architectures (e.g., PowerPC, MIPS). The choice of endianness can have
/// a measurable impact on performance.
///
/// While using the native endianness of the host machine is typically the most
/// performant option, the optimal choice can be influenced by the availability of
/// specific low-level CPU instructions for bit manipulation (like counting
/// leading/trailing zeros). Depending on the architecture and compression code,
/// a non-native endianness may unexpectedly yield better performance.
///
/// # Example
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[i64] = &[10, -20, 30, -40, 50];
///
/// // Create a Big-Endian SIntVec using the type alias.
/// // Note that manual codec selection is required for SIntVec.
/// let be_sintvec = BESIntVec::builder(data)
///     .codec(CodecSpec::Delta)
///     .build()
///     .unwrap();
///
/// assert_eq!(be_sintvec.len(), data.len());
/// assert_eq!(be_sintvec.get(1), Some(-20));
/// ```
pub type BESIntVec = SIntVec<BE>;

/// A type alias for an [`SIntVec`] with Little-Endian ([`LE`]) bitstream encoding.
///
/// Little-endian is the native byte order for most modern commodity CPUs,
/// including x86 and ARM architectures. For this reason, it often leads to the
/// best performance for bitstream operations on common hardware.
///
/// However, the optimal choice is not always straightforward. Performance depends
/// on the interplay between the compression algorithm and the efficiency of
// low-level bit manipulation instructions on the target hardware. Benchmarking
/// is the only definitive way to determine the optimal choice for a given
/// workload and architecture.
///
/// # Example
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[i64] = &[10, -20, 30, -40, 50];
/// let sintvec = LESIntVec::builder(data).codec(CodecSpec::Gamma).build().unwrap();
///
/// assert_eq!(sintvec.len(), data.len());
/// assert_eq!(sintvec.get(1), Some(-20));
/// ```
pub type LESIntVec = SIntVec<LE>;
