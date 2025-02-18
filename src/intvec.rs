//! # Compressed IntVec Module
//!
//! This module provides a compressed vector of integers that leverages bit-level encoding to
//! efficiently store a sequence of unsigned 64-bit integers.
//!
//! ## Overview
//!
//! The core data structure, `IntVec`, maintains a compressed bitstream along with sampling offsets,
//! which enable fast random access to individual elements without the need to decode the entire stream.
//! The module supports two variants based on endianness:
//!
//! - **Big-Endian** (`BEIntVec`)
//! - **Little-Endian** (`LEIntVec`)
//!
//! Both variants work with codecs that implement the [`Codec`] trait, allowing flexible and configurable
//! encoding/decoding strategies. Codecs may optionally accept extra runtime parameters to tune the compression.
//!
//! ## Key Features
//!
//! - **Efficient Storage**: Compresses integer sequences into a compact bitstream.
//! - **Random Access**: Uses periodic sampling (every *k*-th element) to jump-start decompression.
//! - **Generic Codec Support**: Works with any codec implementing the [`Codec`] trait.
//! - **Endian Flexibility**: Supports both big-endian and little-endian representations.
//!
//! ## Components
//!
//! - **`IntVec`**: The main structure containing compressed data, sample offsets, codec parameters, and metadata. You don't need to interact with this directly.
//! - **`BEIntVec` / `LEIntVec`**: Type aliases for endianness-specific versions of `IntVec`.
//! - **Iterators**: `BEIntVecIter` and `LEIntVecIter` decode values on the fly when iterated.
//!
//! ## Usage Examples
//!
//! ### Creating a Big-Endian Compressed Vector
//!
//! ```rust
//! use compressed_intvec::intvec::BEIntVec;
//! use compressed_intvec::codecs::ExpGolombCodec;
//!
//! // Define a vector of unsigned 64-bit integers.
//! let input = vec![1, 5, 3, 1991, 42];
//!
//! // Create a Big-Endian compressed vector using ExpGolombCodec with a parameter (e.g., 3)
//! // and sample every 2 elements.
//! let intvec = BEIntVec::<ExpGolombCodec>::from_with_param(&input, 2, 3);
//!
//! // Retrieve a specific element by its index.
//! let value = intvec.get(3);
//! assert_eq!(value, Some(1991));
//!
//! // Decode the entire compressed vector back to its original form.
//! let decoded = intvec.into_vec();
//! assert_eq!(decoded, input);
//! ```
//!
//! ### Creating a Little-Endian Compressed Vector
//!
//! ```rust
//! use compressed_intvec::intvec::LEIntVec;
//! use compressed_intvec::codecs::GammaCodec;
//!
//! // Define a vector of unsigned 64-bit integers.
//! let input = vec![10, 20, 30, 40, 50];
//!
//! // Create a Little-Endian compressed vector using GammaCodec without extra codec parameters,
//! // sampling every 2 elements.
//! let intvec = LEIntVec::<GammaCodec>::from(&input, 2);
//!
//! assert_eq!(intvec.get(2), Some(30));
//! ```
//!
//! ## Design Details
//!
//! - **Bitstream Storage**: The compressed data is stored as a vector of 64-bit words (`Vec<u64>`).
//! - **Sampling Strategy**: To support fast random access, sample offsets (in bits) are stored for every *k*-th integer.
//! - **Codec Abstraction**: The module is codec-agnostic; any codec conforming to the [`Codec`] trait can be used.
//! - **Endian Handling**: The endianness of the encoding/decoding process is managed through phantom types,
//!   enabling both big-endian and little-endian variants.
//!
//! ## Module Structure and Extensibility
//!
//! The module's API provides constructors (`from_with_param` and `from`), element access (`get`), full
//! decoding (`into_vec`), and iteration (`iter`). It can be extended with new codecs by implementing
//! the [`Codec`] trait for additional compression methods or parameters.
//!
//! ## Error Handling
//!
//! The current implementation assumes that errors in encoding/decoding are exceptional and uses `.unwrap()`
//! in places where failure is unexpected. For production code, you might consider propagating errors
//! instead of panicking.
//!
//! ## Getting Started
//!
//! 1. Choose or implement a codec that satisfies the [`Codec`] trait requirements.
//! 2. Use the provided constructors to compress a vector of integers.
//! 3. Leverage the efficient sampling mechanism for fast random access, or decode the full content when needed.
//!
//! For more details, refer to the documentation of the [`Codec`] trait and the respective codec implementations.
//!

use crate::codecs::Codec;
use dsi_bitstream::prelude::*;
use mem_dbg::{MemDbg, MemSize};
use std::marker::PhantomData;

/// A compressed vector of integers.
///
/// The `IntVec` stores values in a compressed bitstream along with sample offsets for
/// fast random access. The type is generic over an endianness (`E`) and codec (`C`).
///
/// # Type Parameters
///
/// - `E`: Endianness marker (e.g. `BE` or `LE`).
/// - `C`: The codec used for compression. Must implement `Codec<E, MyBitWrite<E>, MyBitRead<E>>`.
///
/// # Fields
///
/// - `data`: The compressed bitstream data.
/// - `samples`: Offsets (in bits) into `data` for every k-th value, used to jump-start decoding.
/// - `k`: The sampling period.
/// - `len`: The total number of integers stored.
/// - `codec_param`: Extra parameters for `C`.
///
/// # Examples
///
/// ```
/// use compressed_intvec::intvec::BEIntVec;
/// use compressed_intvec::codecs::GammaCodec;
///
/// // Create a compressed vector using a codec without extra parameters.
/// let input = vec![1, 2, 3, 4, 5];
/// let intvec = BEIntVec::<GammaCodec>::from(&input, 2);
/// let value = intvec.get(3);
/// assert_eq!(value, Some(4));
/// assert_eq!(intvec.len(), 5);
/// ```
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "C::Params: Serialize",
        deserialize = "C::Params: for<'a> Deserialize<'a>"
    ))
)]
pub struct IntVec<E: Endianness, W: BitWrite<E>, C: Codec<E, W>> {
    pub data: Vec<u64>,
    pub samples: Vec<usize>,
    pub codec: PhantomData<C>,
    pub k: usize,
    pub len: usize,
    pub codec_param: C::Params,
    pub endian: PhantomData<E>,
}
/// Big-endian variant of `IntVec`.
pub type BEIntVec<C> = IntVec<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>, C>;

impl<C: Codec<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>>> BEIntVec<C>
where
    C::Params: Copy,
{
    /// Creates a new `BEIntVec` from a vector of unsigned 64-bit integers.
    ///
    /// Values are encoded with the specified codec parameter.
    ///
    /// # Arguments
    ///
    /// - `input`: The values to be compressed.
    /// - `k`: The sampling rate (every k-th value is stored as a sample).
    /// - `codec_param`: Parameters for the codec.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::intvec::BEIntVec;
    /// use compressed_intvec::codecs::ExpGolombCodec;
    ///
    /// let input = vec![1, 5, 3, 1991, 42];
    /// let intvec = BEIntVec::<ExpGolombCodec>::from_with_param(&input, 2, 3);
    ///
    /// let value = intvec.get(3);
    /// assert_eq!(value, Some(1991));
    /// ```
    pub fn from_with_param(input: &[u64], k: usize, codec_param: C::Params) -> Self {
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = BufBitWriter::<BE, MemWordWriterVec<u64, Vec<u64>>>::new(word_writer);
        let mut samples = Vec::new();
        let mut total_bits = 0;

        for (i, &x) in input.iter().enumerate() {
            if i % k == 0 {
                samples.push(total_bits);
            }
            total_bits += C::encode(&mut writer, x, codec_param).unwrap();
        }

        writer.flush().unwrap();
        let data = writer.into_inner().unwrap().into_inner();

        IntVec {
            data,
            samples,
            codec: PhantomData,
            k,
            len: input.len(),
            codec_param,
            endian: PhantomData,
        }
    }

    /// Retrieves the value at the given index.
    ///
    /// Returns `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::intvec::BEIntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    ///
    /// let input = vec![1, 5, 3, 12, 42];
    /// let intvec = BEIntVec::<GammaCodec>::from(&input, 2);
    /// let value = intvec.get(3);
    /// assert_eq!(value, Some(12));
    /// ```
    ///
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }

        let sample_index = index / self.k;
        let start_bit = self.samples[sample_index];
        let mut reader =
            BufBitReader::<BE, MemWordReader<u64, &Vec<u64>>>::new(MemWordReader::new(&self.data));

        reader.set_bit_pos(start_bit as u64).ok()?;

        let mut value = 0;
        let start_index = sample_index * self.k;
        let param = self.codec_param;
        for _ in start_index..=index {
            value = C::decode(&mut reader, param).ok()?;
        }
        Some(value)
    }

    /// Returns the original vector of integers.
    ///
    /// This operation is expensive as it requires decoding the entire bitstream.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::intvec::BEIntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    ///
    /// let input = vec![43, 12, 5, 1991, 42];
    /// let intvec = BEIntVec::<GammaCodec>::from(&input, 2);
    /// let values = intvec.into_vec();
    /// assert_eq!(values, input);
    /// ```
    ///
    pub fn into_vec(self) -> Vec<u64> {
        let word_reader = MemWordReader::new(&self.data);
        let mut reader = BufBitReader::<BE, MemWordReader<u64, &Vec<u64>>>::new(word_reader);
        let mut values = Vec::with_capacity(self.len);

        for _ in 0..self.len {
            values.push(C::decode(&mut reader, self.codec_param).unwrap());
        }

        values
    }

    /// Returns an iterator over the values stored in the vector.
    pub fn iter(&self) -> BEIntVecIter<C> {
        BEIntVecIter {
            intvec: self,
            index: 0,
        }
    }

    pub fn limbs(&self) -> Vec<u64> {
        self.data.clone()
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Convenience constructor for codecs with no extra runtime parameter.
impl<C: Codec<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>, Params = ()>> BEIntVec<C> {
    pub fn from(input: &[u64], k: usize) -> Self {
        Self::from_with_param(input, k, ())
    }
}

/// Iterator over the values stored in a `BEIntVec`.
/// The iterator decodes values on the fly.
pub struct BEIntVecIter<'a, C: Codec<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>>> {
    intvec: &'a BEIntVec<C>,
    index: usize,
}

impl<C> Iterator for BEIntVecIter<'_, C>
where
    C: Codec<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>>,
    for<'b> C: Codec<BE, BufBitReader<BE, MemWordReader<u64, &'b Vec<u64>>>>,
    <C as Codec<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>>>::Params: Copy,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.intvec.len() {
            return None;
        }
        let value = self.intvec.get(self.index);
        self.index += 1;
        value
    }
}

/// Little-endian variant of `IntVec`.
pub type LEIntVec<C> = IntVec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>, C>;

impl<C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>>> LEIntVec<C>
where
    C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>>,
    C::Params: Copy,
{
    /// Creates a new `LEIntVec` from a vector of unsigned 64-bit integers.
    ///
    /// Values are encoded with the specified codec parameter.
    ///
    /// # Arguments
    ///
    /// - `input`: The values to LE compressed.
    /// - `k`: The sampling rate (every k-th value is stored as a sample).
    /// - `codec_param`: Parameters for the codec.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::intvec::LEIntVec;
    /// use compressed_intvec::codecs::ExpGolombCodec;
    ///
    /// let input = vec![1, 5, 3, 1991, 42];
    /// let intvec = LEIntVec::<ExpGolombCodec>::from_with_param(&input, 2, 3);
    ///
    /// let value = intvec.get(3);
    /// assert_eq!(value, Some(1991));
    /// ```
    pub fn from_with_param(input: &[u64], k: usize, codec_param: C::Params) -> Self {
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = BufBitWriter::<LE, MemWordWriterVec<u64, Vec<u64>>>::new(word_writer);
        let mut samples = Vec::new();
        let mut total_bits = 0;

        for (i, &x) in input.iter().enumerate() {
            if i % k == 0 {
                samples.push(total_bits);
            }
            total_bits += C::encode(&mut writer, x, codec_param).unwrap();
        }

        writer.flush().unwrap();
        let data = writer.into_inner().unwrap().into_inner();

        IntVec {
            data,
            samples,
            codec: PhantomData,
            k,
            len: input.len(),
            codec_param,
            endian: PhantomData,
        }
    }
    /// Retrieves the value at the given index.
    ///
    /// Returns `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::intvec::LEIntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    ///
    /// let input = vec![1, 5, 3, 1991, 42];
    /// let intvec = LEIntVec::<GammaCodec>::from(&input, 2);
    /// let value = intvec.get(3);
    /// assert_eq!(value, Some(1991));
    /// ```
    ///
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }

        let sample_index = index / self.k;
        let start_bit = self.samples[sample_index];
        let mut reader =
            BufBitReader::<LE, MemWordReader<u64, &Vec<u64>>>::new(MemWordReader::new(&self.data));

        reader.set_bit_pos(start_bit as u64).ok()?;

        let mut value = 0;
        let start_index = sample_index * self.k;
        let param = self.codec_param;
        for _ in start_index..=index {
            value = C::decode(&mut reader, param).ok()?;
        }
        Some(value)
    }

    /// Returns the original vector of integers.
    ///
    /// This operation is expensive as it requires decoding the entire bitstream.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::intvec::LEIntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    ///
    /// let input = vec![43, 12, 5, 1991, 42];
    /// let intvec = LEIntVec::<GammaCodec>::from(&input, 2);
    /// let values = intvec.into_vec();
    /// assert_eq!(values, input);
    /// ```
    ///
    pub fn into_vec(self) -> Vec<u64> {
        let word_reader = MemWordReader::new(&self.data);
        let mut reader = BufBitReader::<LE, MemWordReader<u64, &Vec<u64>>>::new(word_reader);
        let mut values = Vec::with_capacity(self.len);

        for _ in 0..self.len {
            values.push(C::decode(&mut reader, self.codec_param).unwrap());
        }

        values
    }

    /// Returns an iterator over the values stored in the vector.
    pub fn iter(&self) -> LEIntVecIter<C> {
        LEIntVecIter {
            intvec: self,
            index: 0,
        }
    }

    pub fn limbs(&self) -> Vec<u64> {
        self.data.clone()
    }

    /// Returns the numLEr of elements in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Convenience constructor for codecs with no extra runtime parameter.
impl<C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>, Params = ()>> LEIntVec<C> {
    pub fn from(input: &[u64], k: usize) -> Self {
        Self::from_with_param(input, k, ())
    }
}

/// Iterator over the values stored in a `LEIntVec`.
/// The iterator decodes values on the fly.
pub struct LEIntVecIter<'a, C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>>> {
    intvec: &'a LEIntVec<C>,
    index: usize,
}

impl<C> Iterator for LEIntVecIter<'_, C>
where
    C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>>,
    for<'b> C: Codec<LE, BufBitReader<LE, MemWordReader<u64, &'b Vec<u64>>>>,
    <C as Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>>>::Params: Copy,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.intvec.len() {
            return None;
        }
        let value = self.intvec.get(self.index);
        self.index += 1;
        value
    }
}
