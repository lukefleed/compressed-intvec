//! # A compressed, randomly accessible vector of integers.
//!
//! This module provides the core implementation of [`IntVec`], a data structure
//! designed for space-efficient storage and fast random access of integer
//! sequences. It supports both unsigned (`u8`-`u64`) and signed (`i8`-`i64`)
//! integer types.
//!
//! Compression is achieved by leveraging a variety of instantaneous,
//! variable-length codes from the [`dsi-bitstream`] crate. Signed integers are
//! automatically handled using ZigZag encoding via the [`Storable`] trait.
//!
//! The main generic structure is [`IntVec<T, E, B>`], where `T` is the element
//! type, `E` is the endianness, and `B` is the backing storage. For convenience,
//`! several type aliases are provided:
//! - [`UIntVec<T>`]: For unsigned integers with default Little-Endian encoding.
//! - [`SIntVec<T>`]: For signed integers with default Little-Endian encoding.
//! - [`LEIntVec`], [`BEIntVec`]: For `u64` elements with specific endianness.
//!
//! [`dsi-bitstream`]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/

#[macro_use]
pub mod macros;

pub mod builder;
pub mod codec;
pub mod iter;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod reader;
pub mod seq_reader;
#[cfg(feature = "serde")]
pub mod serde;
pub mod slice;
pub mod traits;

pub mod prelude {
    pub use super::builder::{IntVecBuilder, IntVecFromIterBuilder};
    pub use super::codec::VariableCodecSpec;
    pub use super::iter::{IntVecIntoIter, IntVecIter};
    pub use super::reader::IntVecReader;
    pub use super::seq_reader::IntVecSeqReader;
    pub use super::slice::IntVecSlice;
    pub use super::traits::Storable;
    pub use super::{
        BEIntVec, BESIntVec, LEIntVec, LESIntVec, SIntVec, UIntVec, IntVec,
    };
}

use crate::fixed::{Error as FixedVecError, FixedVec};
use self::{codec::VariableCodecSpec, traits::Storable};
use dsi_bitstream::{
    codes::params::DefaultReadParams,
    dispatch::StaticCodeRead,
    prelude::{
        BitRead, BitSeek, BufBitReader, BufBitWriter, Codes, CodesRead, CodesWrite, Endianness,
        MemWordReader, MemWordWriterVec,
    },
    traits::{BitWrite, BE, LE},
};
use mem_dbg::{MemDbg, MemSize};
use std::{error::Error, fmt, marker::PhantomData};

pub use builder::{IntVecBuilder, IntVecFromIterBuilder};
pub use iter::{IntVecIntoIter, IntVecIter};
pub use reader::IntVecReader;
pub use seq_reader::IntVecSeqReader;
pub use slice::IntVecSlice;

/// Defines the set of errors that can occur in `IntVec` operations.
#[derive(Debug)]
pub enum IntVecError {
    /// An error occurred during an I/O operation.
    Io(std::io::Error),
    /// A generic error from the `dsi-bitstream` library.
    Bitstream(Box<dyn Error + Send + Sync>),
    /// An error indicating invalid parameters.
    InvalidParameters(String),
    /// An error during codec function dispatch.
    CodecDispatch(String),
    /// An error for out-of-bounds index access.
    IndexOutOfBounds(usize),
}

impl fmt::Display for IntVecError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntVecError::Io(e) => write!(f, "I/O error: {}", e),
            IntVecError::Bitstream(e) => write!(f, "Bitstream error: {}", e),
            IntVecError::InvalidParameters(s) => write!(f, "Invalid parameters: {}", s),
            IntVecError::CodecDispatch(s) => write!(f, "Codec dispatch error: {}", s),
            IntVecError::IndexOutOfBounds(index) => write!(f, "Index out of bounds: {}", index),
        }
    }
}

impl Error for IntVecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            IntVecError::Io(e) => Some(e),
            IntVecError::Bitstream(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for IntVecError {
    fn from(e: std::io::Error) -> Self {
        IntVecError::Io(e)
    }
}

impl From<core::convert::Infallible> for IntVecError {
    fn from(_: core::convert::Infallible) -> Self {
        unreachable!()
    }
}

impl From<FixedVecError> for IntVecError {
    fn from(e: FixedVecError) -> Self {
        IntVecError::InvalidParameters(e.to_string())
    }
}

/// A compressed, randomly accessible vector of generic integers.
///
/// It uses instantaneous codes for compression and a sampling mechanism for
/// fast random access, configurable via the sampling rate `k`. The element type
/// `T` must implement the [`Storable`] trait for conversion to/from `u64`.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct IntVec<T: Storable, E: Endianness, B: AsRef<[u64]> = Vec<u64>> {
    /// The raw compressed data.
    pub(super) data: B,
    /// Bit offsets of sampled elements.
    pub(super) samples: FixedVec<u64, u64, LE, B>,
    /// The sampling rate `k`.
    pub(super) k: usize,
    /// The number of elements in the vector.
    pub(super) len: usize,
    /// The `dsi-bitstream` code used for compression.
    pub(super) encoding: Codes,
    /// A zero-sized marker for element type, endianness, and backend.
    pub(super) _markers: PhantomData<(T, E)>,
}

/// Type alias for the writer used internally by `IntVec`.
pub(crate) type IntVecBitWriter<E> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;
/// Type alias for the reader used internally by `IntVec`.
pub(crate) type IntVecBitReader<'a, E> =
    BufBitReader<E, MemWordReader<u64, &'a [u64]>, DefaultReadParams>;

impl<T: Storable, E: Endianness> IntVec<T, E, Vec<u64>> {
    /// Returns a builder for creating an owned [`IntVec`] from a slice of data.
    pub fn builder(input: &'_ [T]) -> IntVecBuilder<'_, T, E> {
        IntVecBuilder::new(input)
    }

    /// Returns a builder for creating an owned [`IntVec`] from an iterator.
    pub fn from_iter_builder<I>(iter: I) -> IntVecFromIterBuilder<T, E, I>
    where
        I: IntoIterator<Item = T> + Clone,
    {
        IntVecFromIterBuilder::new(iter)
    }

    /// Consumes the [`IntVec`] and returns its decoded values as a `Vec<T>`.
    pub fn into_vec(self) -> Vec<T>
    where
        for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.into_iter().collect()
    }

    /// Creates an owned `IntVec` directly from a slice of data.
    ///
    /// This is a convenient alias for `IntVec::builder(slice).build()`.
    /// The codec will be automatically determined using `VariableCodecSpec::Auto`,
    /// and a default sampling rate of `k=16` will be used.
    pub fn from_slice(slice: &[T]) -> Result<Self, IntVecError>
    where
        for<'a> crate::variable::IntVecBitWriter<E>:
            BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        Self::builder(slice)
            .k(16)
            .codec(VariableCodecSpec::Auto)
            .build()
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> IntVec<T, E, B> {
    /// Creates a new `IntVec` from its raw parts, enabling zero-copy views.
    pub fn from_parts(
        data: B,
        samples_data: B,
        samples_len: usize,
        samples_num_bits: usize,
        k: usize,
        len: usize,
        encoding: Codes,
    ) -> Result<Self, IntVecError> {
        let samples = FixedVec::<u64, u64, LE, B>::from_parts(
            samples_data,
            samples_len,
            samples_num_bits,
        )?;

        if k == 0 {
            return Err(IntVecError::InvalidParameters(
                "Sampling rate k cannot be zero".to_string(),
            ));
        }
        let expected_samples = if len == 0 { 0 } else { len.div_ceil(k) };
        if samples.len() != expected_samples {
            return Err(IntVecError::InvalidParameters(format!(
                "Inconsistent number of samples. Expected {}, found {}",
                expected_samples,
                samples.len()
            )));
        }

        Ok(unsafe { Self::new_unchecked(data, samples, k, len, encoding) })
    }

    /// Creates a new `IntVec` from its raw parts without safety checks.
    /// # Safety
    /// The caller must ensure all parameters are consistent and valid.
    pub(crate) unsafe fn new_unchecked(
        data: B,
        samples: FixedVec<u64, u64, LE, B>,
        k: usize,
        len: usize,
        encoding: Codes,
    ) -> Self {
        Self {
            data,
            samples,
            k,
            len,
            encoding,
            _markers: PhantomData,
        }
    }

    /// Creates a zero-copy slice of this vector.
    pub fn slice(&'_ self, start: usize, len: usize) -> Option<IntVecSlice<'_, T, E, B>> {
        if start + len > self.len {
            return None;
        }
        Some(IntVecSlice::new(self, start..start + len))
    }

    /// Splits the vector into two slices at a given index.
    pub fn split_at(&'_ self, mid: usize) -> Option<(IntVecSlice<'_, T, E, B>, IntVecSlice<'_, T, E, B>)> {
        if mid > self.len {
            return None;
        }
        let left = IntVecSlice::new(self, 0..mid);
        let right = IntVecSlice::new(self, mid..self.len);
        Some((left, right))
    }

    /// Returns the number of integers in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the sampling rate `k` used during encoding.
    pub fn get_sampling_rate(&self) -> usize {
        self.k
    }

    /// Returns the number of sample points stored in the vector.
    pub fn get_num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Returns a reference to the inner `FixedVec` of samples.
    pub fn samples_ref(&self) -> &FixedVec<u64, u64, LE, B> {
        &self.samples
    }

    /// Returns a zero-copy, read-only slice of the underlying compressed data (`&[u64]`).
    pub fn as_limbs(&self) -> &[u64] {
        self.data.as_ref()
    }

    /// Returns the concrete `Codes` variant that was used for compression.
    pub fn encoding(&self) -> Codes {
        self.encoding
    }

    /// Returns a clone of the underlying storage as a `Vec<u64>`.
    pub fn limbs(&self) -> Vec<u64> {
        self.data.as_ref().to_vec()
    }

    /// Returns an iterator over the decompressed values.
    pub fn iter(&'_ self) -> IntVecIter<'_, T, E, B>
    where
        for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        IntVecIter::new(self)
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> IntVec<T, E, B>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a stateful, reusable [`IntVecReader`] for this vector.
    pub fn reader(&'_ self) -> IntVecReader<'_, T, E, B> {
        IntVecReader::new(self)
    }

    /// Creates a stateful, reusable [`IntVecSeqReader`] for this vector.
    pub fn seq_reader(&'_ self) -> IntVecSeqReader<'_, T, E, B> {
        IntVecSeqReader::new(self)
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        Some(unsafe { self.get_unchecked(index) })
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        let mut reader = self.reader();
        reader.get_unchecked(index)
    }

    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<T>, IntVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        for &index in indices {
            if index >= self.len {
                return Err(IntVecError::IndexOutOfBounds(index));
            }
        }
        Ok(unsafe { self.get_many_unchecked(indices) })
    }

    pub unsafe fn get_many_unchecked(&self, indices: &[usize]) -> Vec<T> {
        if indices.is_empty() {
            return Vec::new();
        }
        let mut results = Vec::with_capacity(indices.len());
        results.set_len(indices.len());

        let mut indexed_indices: Vec<(usize, usize)> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();
        indexed_indices.sort_unstable_by_key(|&(idx, _)| idx);

        if self.k.is_power_of_two() {
            let k_exp = self.k.trailing_zeros();
            self.get_many_dsi_inner(
                &indexed_indices,
                &mut results,
                |idx| idx >> k_exp,
                |block| block << k_exp,
            )
            .unwrap();
        } else {
            self.get_many_dsi_inner(
                &indexed_indices,
                &mut results,
                |idx| idx / self.k,
                |block| block * self.k,
            )
            .unwrap();
        }

        results
    }

    fn get_many_dsi_inner<F1, F2>(
        &self,
        indexed_indices: &[(usize, usize)],
        results: &mut [T],
        block_of: F1,
        start_of_block: F2,
    ) -> Result<(), IntVecError>
    where
        F1: Fn(usize) -> usize,
        F2: Fn(usize) -> usize,
    {
        let mut reader = self.reader();
        let mut current_decoded_index: usize = 0;

        for &(target_index, original_position) in indexed_indices {
            if target_index < current_decoded_index
                || block_of(target_index) != block_of(current_decoded_index.saturating_sub(1))
            {
                let target_sample_block = block_of(target_index);
                let start_bit = unsafe { self.samples.get_unchecked(target_sample_block) };
                reader.reader.set_bit_pos(start_bit)?;
                current_decoded_index = start_of_block(target_sample_block);
            }

            for _ in current_decoded_index..target_index {
                reader.code_reader.read(&mut reader.reader)?;
            }
            let value = reader.code_reader.read(&mut reader.reader)?;
            results[original_position] = Storable::from_word(value);
            current_decoded_index = target_index + 1;
        }
        Ok(())
    }

    pub fn get_many_from_iter<I>(&self, indices: I) -> Result<Vec<T>, IntVecError>
    where
        I: IntoIterator<Item = usize>,
    {
        let indices_iter = indices.into_iter();
        let (lower_bound, _) = indices_iter.size_hint();
        let mut results = Vec::with_capacity(lower_bound);
        let mut seq_reader = self.seq_reader();

        for index in indices_iter {
            let value = seq_reader
                .get(index)?
                .ok_or(IntVecError::IndexOutOfBounds(index))?;
            results.push(value);
        }

        Ok(results)
    }
}

impl<T: Storable + Ord, E: Endianness, B: AsRef<[u64]>> IntVec<T, E, B>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    pub fn binary_search(&self, value: &T) -> Result<usize, usize> {
        self.binary_search_by(|probe| probe.cmp(value))
    }

    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> std::cmp::Ordering,
    {
        let mut low = 0;
        let mut high = self.len();
        let mut reader = self.reader();

        while low < high {
            let mid = low + (high - low) / 2;
            let cmp = f(unsafe { reader.get_unchecked(mid) });

            match cmp {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => return Ok(mid),
                std::cmp::Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }

    #[inline]
    pub fn binary_search_by_key<K: Ord, F>(&self, b: &K, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> K,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }
}

impl<T: Storable, E: Endianness, B: AsRef<[u64]>> IntoIterator for IntVec<T, E, B>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;
    type IntoIter = IntVecIntoIter<T, E, B>;

    fn into_iter(self) -> Self::IntoIter {
        IntVecIntoIter::new(self)
    }
}

pub type UIntVec<T> = IntVec<T, LE>;
pub type SIntVec<T> = IntVec<T, LE>;
pub type BEIntVec = IntVec<u64, BE>;
pub type LEIntVec = IntVec<u64, LE>;
pub type BESIntVec = IntVec<i64, BE>;
pub type LESIntVec = IntVec<i64, LE>;

impl<T, E, B, O> PartialEq<O> for IntVec<T, E, B>
where
    T: Storable + PartialEq,
    E: Endianness,
    B: AsRef<[u64]>,
    O: AsRef<[T]>,
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn eq(&self, other: &O) -> bool {
        let other_slice = other.as_ref();
        if self.len() != other_slice.len() {
            return false;
        }
        self.iter().zip(other_slice.iter()).all(|(a, b)| a == *b)
    }
}
