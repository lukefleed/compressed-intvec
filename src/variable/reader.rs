//! A reader for efficient, repeated random access into an [`IntVec`].
//!
//! This module provides [`IntVecReader`], a reusable reader that is designed to
//! optimize random access performance.
//!
//! # Performance
//!
//! A standard call to [`get`](super::IntVec::get) is convenient, but it
//! creates and discards an internal bitstream reader for each call. When
//! performing many random lookups, this can introduce significant overhead.
//!
//! [`IntVecReader`] avoids this by maintaining a persistent, reusable
//! reader instance. This amortizes the setup cost across multiple `get` operations,
//! making it ideal for access patterns where lookup indices are not known in
//! advance (e.g., graph traversals, pointer chasing).
//!
//! For reading a predefined list of indices, [`get_many`](super::IntVec::get_many)
//! is generally more efficient, as it can pre-sort the indices for a single
//! sequential scan.
//!
//! [`IntVec`]: crate::variable::IntVec

use super::{traits::Storable, IntVec, IntVecBitReader, IntVecError};
use dsi_bitstream::{
    dispatch::{Codes, CodesRead, FuncCodeReader, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};
use std::marker::PhantomData;

/// An internal hybrid dispatcher for reading compression codes.
///
/// This enum acts as a two-stage dispatcher to read values from a
/// compressed bitstream. It is designed to handle all valid codecs supported by
/// [`dsi-bitstream`](https://crates.io/crates/dsi-bitstream), ensuring correctness while maximizing performance.
///
/// ### Dispatch Strategy
///
/// 1.  **Fast Path ([`CodecReader::Fast`])**: For common codecs
///     (e.g., Gamma, Delta, or Zeta with small parameters), [`dsi-bitstream`](https://crates.io/crates/dsi-bitstream)
///     provides pre-compiled function pointers via [`FuncCodeReader`]. This path
///     avoids the overhead of a `match` statement on every read operation, as the
///     correct function is resolved once at creation time.
///
/// 2.  **Slow Path ([`CodecReader::Slow`])**: For codecs with parameters outside of
///     the pre-compiled set (e.g., `Golomb { b: 15 }`), the fast path is not
///     available. In this case, the dispatcher falls back to storing the [`Codes`]
///     enum variant directly. Each read operation then uses a `match` statement to
///     call the appropriate decoding function. While slightly slower due to the
///     runtime dispatch, this ensures that any validly created `IntVec` can be read.
///
/// This hybrid approach guarantees that the reader will never panic due to an
/// "unsupported code" error, which was a critical issue in previous implementations.
/// The `new` constructor automatically selects the appropriate path.
pub(super) enum CodecReader<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Fast-path reader using a pre-resolved function pointer.
    Fast(FuncCodeReader<E, IntVecBitReader<'a, E>>),
    /// Fallback reader using dynamic dispatch on the `Codes` enum.
    Slow(Codes),
    /// Zero-sized marker to carry the generic parameters, ensuring type safety
    /// and allowing the compiler to properly manage lifetimes.
    _Phantom(PhantomData<(&'a B, T)>),
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> CodecReader<'a, T, E, B>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new [`CodecReader`], automatically selecting the fastest available
    /// dispatch path for the given codec.
    ///
    /// This constructor attempts to create a high-performance [`FuncCodeReader`].
    /// If the codec is not supported by the fast path (e.g., it has uncommon
    /// parameters), it falls back to the dynamic dispatch mechanism.
    /// This method will not panic.
    pub(super) fn new(code: Codes) -> Self {
        match FuncCodeReader::new(code) {
            Ok(fast_reader) => Self::Fast(fast_reader),
            Err(_) => Self::Slow(code),
        }
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> StaticCodeRead<E, IntVecBitReader<'a, E>>
    for CodecReader<'a, T, E, B>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    #[inline]
    fn read(&self, reader: &mut IntVecBitReader<'a, E>) -> Result<u64, core::convert::Infallible> {
        match self {
            // If we have a function pointer, call it directly. This is the fast path.
            Self::Fast(func_reader) => func_reader.read(reader),
            // Otherwise, use the slower dynamic dispatch. This is the fallback path.
            Self::Slow(code) => code.read(reader),
            // This variant is never constructed, but is needed for the type system.
            Self::_Phantom(_) => unreachable!(),
        }
    }
}

/// A stateful reader for an `IntVec` that provides fast random access.
///
/// This reader is created by the [`IntVec::reader`](super::IntVec::reader)
/// method. It maintains an internal, reusable bitstream reader, making it highly
/// efficient for performing multiple random lookups where the access pattern is
/// not known ahead of time.
///
/// # Examples
///
/// ```
/// use compressed_intvec::variable::{IntVec, UIntVec};
///
/// let data: Vec<u32> = (0..100).rev().collect(); // Data is not sequential
/// let vec: UIntVec<u32> = IntVec::from_slice(&data).unwrap();
///
/// // Create a reusable reader
/// let mut reader = vec.reader();
///
/// // Perform multiple random reads efficiently
/// assert_eq!(reader.get(99).unwrap(), Some(0));
/// assert_eq!(reader.get(0).unwrap(), Some(99));
/// assert_eq!(reader.get(50).unwrap(), Some(49));
/// ```
pub struct IntVecReader<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// A reference to the parent `IntVec`.
    pub(super) intvec: &'a IntVec<T, E, B>,
    /// The stateful, reusable bitstream reader.
    pub(super) reader: IntVecBitReader<'a, E>,
    /// The hybrid dispatcher that handles codec reading.
    pub(super) code_reader: CodecReader<'a, T, E, B>,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> IntVecReader<'a, T, E, B>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new `IntVecReader`.
    pub(super) fn new(intvec: &'a IntVec<T, E, B>) -> Self {
        let bit_reader = IntVecBitReader::new(dsi_bitstream::impls::MemWordReader::new(
            intvec.data.as_ref(),
        ));
        // This robustly selects the best available dispatch strategy.
        let code_reader = CodecReader::new(intvec.encoding);
        Self {
            intvec,
            reader: bit_reader,
            code_reader,
        }
    }

    /// Retrieves the element at `index`, or `None` if out of bounds.
    ///
    /// This method leverages the stateful nature of the reader to perform efficient
    /// random access by seeking to the nearest preceding sample point and decoding
    /// sequentially from there.
    #[inline]
    pub fn get(&mut self, index: usize) -> Result<Option<T>, IntVecError> {
        if index >= self.intvec.len {
            return Ok(None);
        }
        // SAFETY: The bounds check has been performed.
        let value = unsafe { self.get_unchecked(index) };
        Ok(Some(value))
    }

    /// Retrieves the element at `index` without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    /// The caller must ensure that `index < self.intvec.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&mut self, index: usize) -> T {
        debug_assert!(
            index < self.intvec.len(),
            "Index out of bounds: index was {} but length was {}",
            index,
            self.intvec.len()
        );

        let k = self.intvec.k;
        let sample_index = index / k;
        // SAFETY: The caller guarantees that `index` is in bounds, which implies
        // that `sample_index` is also a valid index into the samples vector.
        let start_bit = self.intvec.samples.get_unchecked(sample_index);
        let start_index = sample_index * k;

        // The underlying bitstream operations are infallible, so unwrap is safe.
        self.reader.set_bit_pos(start_bit).unwrap();

        // Sequentially decode elements from the sample point up to the target index.
        for _ in start_index..index {
            // We use the hybrid dispatcher here. It will either call a function
            // pointer or use a match statement, depending on the codec.
            self.code_reader.read(&mut self.reader).unwrap();
        }
        // Read the target value.
        let word = self.code_reader.read(&mut self.reader).unwrap();
        Storable::from_word(word)
    }
}