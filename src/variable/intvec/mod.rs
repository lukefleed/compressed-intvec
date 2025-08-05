//! # A compressed, randomly accessible vector of `u64` integers.
//!
//! This module provides the core implementation of [`IntVec`], a data structure
//! designed for space-efficient storage and fast random access of `u64` integer
//! sequences. It achieves compression by leveraging a variety of instantaneous,
//! variable-length codes from the [`dsi-bitstream`] crate.
//!
//! [`dsi-bitstream`]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/

use crate::{fixed::intvec::FixedVec, prelude::VariableCodecSpec};
use dsi_bitstream::{
    codes::params::DefaultReadParams,
    prelude::{
        BitRead, BitSeek, BufBitReader, BufBitWriter, Codes, CodesRead, CodesWrite, Endianness, MemWordReader, MemWordWriterVec, StaticCodeRead
    },
    traits::{BitWrite, BE, LE},
};
use mem_dbg::{MemDbg, MemSize};
use rayon::slice::ParallelSliceMut;
use std::{error::Error, fmt, marker::PhantomData};

// Declare and export submodules.
mod builder;
mod iter;
#[cfg(feature = "parallel")]
mod parallel;
mod reader;
mod slice;
mod seq_reader;
#[cfg(feature = "serde")]
mod serde;

pub use builder::{IntVecBuilder, IntVecFromIterBuilder};
pub use iter::IntVecIter;
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

/// A compressed, randomly accessible vector of `u64` integers.
///
/// It uses instantaneous codes for compression and a sampling mechanism for
/// fast random access, configurable via the sampling rate `k`.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct IntVec<E: Endianness, B: AsRef<[u64]> = Vec<u64>> {
    /// The raw compressed data.
    pub(super) data: B,
    /// Bit offsets of sampled elements.
    pub(super) samples: FixedVec<LE, B>,
    /// The sampling rate `k`.
    pub(super) k: usize,
    /// The number of elements in the vector.
    pub(super) len: usize,
    /// The `dsi-bitstream` code used for compression.
    pub(super) encoding: Codes,
    /// A zero-sized marker for endianness and backend type.
    pub(super) endian: PhantomData<(E, B)>,
}

/// Type alias for the writer used internally by `IntVec`.
pub(crate) type IntVecBitWriter<E> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;
/// Type alias for the reader used internally by `IntVec`.
pub(crate) type IntVecBitReader<'a, E> =
    BufBitReader<E, MemWordReader<u64, &'a [u64]>, DefaultReadParams>;

impl<E: Endianness> IntVec<E, Vec<u64>> {
    /// Returns a builder for creating an owned [`IntVec`] from a slice.
    pub fn builder<T: AsRef<[u64]> + ?Sized>(input: &T) -> IntVecBuilder<E> {
        IntVecBuilder::new(input.as_ref())
    }

    /// Returns a builder for creating an owned [`IntVec`] from an iterator.
    pub fn from_iter_builder<I: IntoIterator<Item = u64>>(iter: I) -> IntVecFromIterBuilder<E, I> {
        IntVecFromIterBuilder::new(iter)
    }

    /// Consumes the [`IntVec`] and returns its decoded values as a `Vec<u64>`.
    pub fn into_vec(self) -> Vec<u64>
    where
        for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        self.iter().collect()
    }

    /// Creates an owned `IntVec` directly from a slice of data.
    ///
    /// This is a convenient alias for `IntVec::builder(slice).build()`.
    /// The codec will be automatically determined using `VariableCodecSpec::Auto`,
    /// and a default sampling rate of `k=16` will be used.
    /// To specify different parameters, use the builder directly.
    pub fn from_slice<T>(slice: &T) -> Result<Self, IntVecError>
    where
        T: AsRef<[u64]> + ?Sized,
        for<'a> crate::variable::intvec::IntVecBitWriter<E>:
            BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        Self::builder(slice)
            .k(16)
            .codec(VariableCodecSpec::Auto)
            .build()
    }
}

impl<E: Endianness, B: AsRef<[u64]>> IntVec<E, B> {
    /// Creates a new `IntVec` from its raw parts, enabling zero-copy views.
    pub fn from_parts(
        data: B,
        samples_data: B,
        samples_len: usize,
        samples_num_bits: usize,
        k: usize,
        len: usize,
        encoding: Codes,
    ) -> Result<Self, crate::fixed::intvec::FixedVecError> {
        let samples = FixedVec::<LE, B>::from_parts(samples_data, samples_len, samples_num_bits)?;

        // Perform IntVec-specific validation.
        if k == 0 {
            return Err(crate::fixed::intvec::FixedVecError::InvalidParameters(
                "Sampling rate k cannot be zero".to_string(),
            ));
        }
        let expected_samples = if len == 0 { 0 } else { len.div_ceil(k) };
        if samples.len() != expected_samples {
            return Err(crate::fixed::intvec::FixedVecError::InvalidParameters(
                format!(
                    "Inconsistent number of samples. Expected {}, found {}",
                    expected_samples,
                    samples.len()
                ),
            ));
        }

        // SAFETY: All components have been validated.
        Ok(unsafe { Self::new_unchecked(data, samples, k, len, encoding) })
    }

    /// Creates a new `IntVec` from its raw parts without safety checks.
    /// # Safety
    /// The caller must ensure all parameters are consistent and valid.
    pub(crate) unsafe fn new_unchecked(
        data: B,
        samples: FixedVec<LE, B>,
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
            endian: PhantomData,
        }
    }

    /// Creates a zero-copy slice of this vector.
    ///
    /// # Arguments
    /// * `start`: The starting index of the slice.
    /// * `len`: The number of elements in the slice.
    ///
    /// # Returns
    /// An `Option` containing the [`IntVecSlice`] if the specified range is
    /// within the bounds of the vector, or `None` otherwise.
    pub fn slice(&self, start: usize, len: usize) -> Option<IntVecSlice<E, B>> {
        if start + len > self.len {
            return None;
        }
        Some(IntVecSlice::new(self, start..start + len))
    }

    /// Splits the vector into two slices at a given index.
    ///
    /// # Arguments
    /// * `mid`: The index at which to split the vector.
    ///
    // # Returns
    /// An `Option` containing a tuple of two [`IntVecSlice`]s if `mid` is
    /// within the bounds of the vector, or `None` otherwise.
    pub fn split_at(&self, mid: usize) -> Option<(IntVecSlice<E, B>, IntVecSlice<E, B>)> {
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
    pub fn samples_ref(&self) -> &FixedVec<LE, B> {
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

    /// Returns an iterator over the decompressed `u64` values.
    pub fn iter(&self) -> IntVecIter<E, B>
    where
        for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        IntVecIter::new(self)
    }
}

impl<E: Endianness, B: AsRef<[u64]>> IntVec<E, B>
where
    for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a stateful, reusable [`IntVecReader`] for this vector.
    pub fn reader(&self) -> IntVecReader<E, B> {
        IntVecReader::new(self)
    }

    /// Creates a stateful, reusable [`IntVecSeqReader`] for this vector.
    pub fn seq_reader(&self) -> IntVecSeqReader<E, B> {
        IntVecSeqReader::new(self)
    }

    /// Retrieves the element at the specified index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }
        // SAFETY: The bounds check has been performed.
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Retrieves the element at the specified index without bounds checking.
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(
            index < self.len,
            "Index out of bounds: index was {} but length was {}",
            index,
            self.len
        );
        let mut reader = self.reader();
        reader.get_unchecked(index)
    }

    /// Retrieves multiple elements at the specified indices efficiently.
    pub fn get_many(&self, indices: &[usize]) -> Result<Vec<u64>, IntVecError> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        for &index in indices {
            if index >= self.len {
                return Err(IntVecError::IndexOutOfBounds(index));
            }
        }
        // SAFETY: We have pre-checked the bounds of all indices.
        Ok(unsafe { self.get_many_unchecked(indices) })
    }

    /// Retrieves multiple elements at specified indices without bounds checking.
    /// # Safety
    /// Calling this method with any out-of-bounds index is undefined behavior.
    pub unsafe fn get_many_unchecked(&self, indices: &[usize]) -> Vec<u64> {
        if indices.is_empty() {
            return Vec::new();
        }
        #[cfg(debug_assertions)]
        {
            for &index in indices {
                debug_assert!(
                    index < self.len,
                    "Index out of bounds: index was {} but length was {}",
                    index,
                    self.len
                );
            }
        }

        let mut results = vec![0; indices.len()];
        let mut indexed_indices: Vec<(usize, usize)> = indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();
        indexed_indices.par_sort_unstable_by_key(|&(idx, _)| idx);

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

    /// Inner helper function for DSI-based `get_many`.
    fn get_many_dsi_inner<F1, F2>(
        &self,
        indexed_indices: &[(usize, usize)],
        results: &mut [u64],
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
                let start_bit = self.samples.get(target_sample_block).unwrap();
                reader.reader.set_bit_pos(start_bit)?;
                current_decoded_index = start_of_block(target_sample_block);
            }

            for _ in current_decoded_index..target_index {
                reader.code_reader.read(&mut reader.reader)?;
            }
            let value = reader.code_reader.read(&mut reader.reader)?;
            results[original_position] = value;
            current_decoded_index = target_index + 1;
        }

        Ok(())
    }

    /// Retrieves multiple elements from an iterator of indices.
    pub fn get_many_from_iter<I>(&self, indices: I) -> Result<Vec<u64>, IntVecError>
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

    /// Binary searches this vector for a given element.
    pub fn binary_search(&self, value: u64) -> Result<usize, usize> {
        self.binary_search_by(|probe| probe.cmp(&value))
    }

    /// Binary searches this vector with a comparator function.
    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(u64) -> std::cmp::Ordering,
    {
        let mut low = 0;
        let mut high = self.len();
        let mut reader = self.reader();

        while low < high {
            let mid = low + (high - low) / 2;
            // SAFETY: `mid` is always in bounds.
            let cmp = f(unsafe { reader.get_unchecked(mid) });

            match cmp {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Equal => return Ok(mid),
                std::cmp::Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }

    /// Binary searches this vector with a key extraction function.
    #[inline]
    pub fn binary_search_by_key<B1, F>(&self, b: &B1, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(u64) -> B1,
        B1: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }
}

macro_rules! impl_partial_eq_for_uint_slice {
    ($($t:ty),*) => {$(
        impl<E: Endianness, B: AsRef<[u64]>> PartialEq<Vec<$t>> for IntVec<E, B>
        where
            for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
                + CodesRead<E>
                + BitSeek<Error = core::convert::Infallible>,
        {
            fn eq(&self, other: &Vec<$t>) -> bool {
                self.eq(&other[..])
            }
        }

        impl<E: Endianness, B: AsRef<[u64]>> PartialEq<&[$t]> for IntVec<E, B>
        where
            for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
                + CodesRead<E>
                + BitSeek<Error = core::convert::Infallible>,
        {
            fn eq(&self, other: &&[$t]) -> bool {
                self.eq(*other)
            }
        }

        impl<E: Endianness, B: AsRef<[u64]>> PartialEq<[$t]> for IntVec<E, B>
        where
            for<'a> IntVecBitReader<'a, E>: BitRead<E, Error = core::convert::Infallible>
                + CodesRead<E>
                + BitSeek<Error = core::convert::Infallible>,
        {
            fn eq(&self, other: &[$t]) -> bool {
                if self.len() != other.len() {
                    return false;
                }
                self.iter().eq(other.iter().map(|&x| x as u64))
            }
        }
    )*};
}

impl_partial_eq_for_uint_slice!(u8, u16, u32, u64);

/// A type alias for an [`IntVec`] with Big-Endian ([`BE`]) bitstream encoding.
pub type BEIntVec = IntVec<BE>;

/// A type alias for an [`IntVec`] with Little-Endian ([`LE`]) bitstream encoding.
pub type LEIntVec = IntVec<LE>;