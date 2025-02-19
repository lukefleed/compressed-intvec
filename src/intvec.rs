//! # Compressed IntVec Module
//!
//! This module delivers an efficient compressed vector for storing sequences of unsigned 64-bit integers.
//! By leveraging bit-level encoding, it minimizes storage space while supporting fast random access.
//!
//! ## Overview
//!
//! The principal data structure, [`IntVec`], encapsulates a compressed bitstream along with sampling offsets.
//! These offsets enable fast, random access to individual elements without decompressing the entire stream.
//! The module offers two variants based on endianness:
//!
//! - **Big-Endian** ([`BEIntVec`])
//! - **Little-Endian** ([`LEIntVec`])
//!
//! Both variants operate with codecs that implement the [`Codec`] trait, allowing for flexible and configurable
//! encoding/decoding strategies. Some codecs even accept extra runtime parameters to fine-tune the compression.
//!
//! > **Note:** The [`IntVec`] structure is generic and you are not supposed to interact with it directly. Use the two endianness-specific types instead.
//!
//! ## Key Features
//!
//! - **Compact Storage:** Compresses sequences of integers into a streamlined bitstream.
//! - **Fast Random Access:** Employs periodic sampling (every *k*-th element) to quickly locate individual elements.
//! - **Flexible Codec Integration:** Compatible with any codec conforming to the [`Codec`] trait.
//! - **Endianness Options:** Provides both big-endian and little-endian formats to suit various interoperability needs.
//!
//! ## Components
//! - **[`IntVec`]:** The core structure holding compressed data, sampling offsets, codec parameters, and metadata.
//!   Direct interaction with this structure is not permitted, and you should use the endianness-specific types instead.
//! - **[`BEIntVec`] / [`LEIntVec`]:** Type aliases for the big-endian and little-endian implementations of [`IntVec`].
//! - **Iterators:** [`BEIntVecIter`] and [`LEIntVecIter`] facilitate on-the-fly decoding as you iterate through the vector.
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
//! // Create a Big-Endian compressed vector using ExpGolombCodec with an extra codec parameter (e.g., 3)
//! // and sample every 2 elements.
//! let intvec = BEIntVec::<ExpGolombCodec>::from_with_param(&input, 2, 3).unwrap();
//!
//! // Retrieve a specific element by its index.
//! let value = intvec.get(3);
//! assert_eq!(value, 1991);
//!
//! // Decompress the entire structure back into a standard vector.
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
//! // Create a Little-Endian compressed vector using GammaCodec without additional codec parameters,
//! // sampling every 2 elements.
//! let intvec = LEIntVec::<GammaCodec>::from(&input, 2).unwrap();
//!
//! // Verify that random access works correctly.
//! assert_eq!(intvec.get(2), 30);
//! ```
//!
//! ## Design Considerations
//!
//! - **Bitstream Representation:** The compressed data is stored as a vector of 64-bit words ([`Vec<u64>`]).
//! - **Sampling Strategy:** To ensure fast random access, sampling offsets are recorded
//!   for every *k*-th integer.
//! - **Codec Abstraction:** The module is codec-agnostic; any codec that implements the [`Codec`] trait may be employed.
//! - **Endianness Management:** Endianness is seamlessly handled via phantom types, supporting both big-endian and
//!   little-endian variants without affecting performance.

use crate::codecs::Codec;
use dsi_bitstream::prelude::*;
use mem_dbg::{MemDbg, MemSize};
use std::marker::PhantomData;

/// A compressed vector of integers.
///
/// The [`IntVec`] stores values in a compressed bitstream along with sample offsets for
/// fast random access. The type is generic over an endianness (`E`) and codec (`C`).
///
/// # Type Parameters
///
/// - `E`: Endianness marker (e.g. [`BE`] or [`LE`]).
/// - `C`: The codec used for compression. Must implement [`Codec<E, MyBitWrite<E>, MyBitRead<E>>`].
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
/// let intvec = BEIntVec::<GammaCodec>::from(&input, 2).unwrap();
/// let value = intvec.get(3);
/// assert_eq!(value, 4);
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
    /// Creates a new [`BEIntVec`] from a vector of unsigned 64-bit integers.
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
    /// let intvec = BEIntVec::<ExpGolombCodec>::from_with_param(&input, 2, 3).unwrap();
    ///
    /// let value = intvec.get(3);
    /// assert_eq!(value, 1991);
    /// ```
    pub fn from_with_param(
        input: &[u64],
        k: usize,
        codec_param: C::Params,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = BufBitWriter::<BE, MemWordWriterVec<u64, Vec<u64>>>::new(word_writer);
        let mut samples = Vec::new();
        let mut total_bits = 0;

        for (i, &x) in input.iter().enumerate() {
            if i % k == 0 {
                samples.push(total_bits);
            }
            total_bits += C::encode(&mut writer, x, codec_param)?;
        }

        writer.flush()?;
        let data = writer.into_inner()?.into_inner();

        Ok(IntVec {
            data,
            samples,
            codec: PhantomData,
            k,
            len: input.len(),
            codec_param,
            endian: PhantomData,
        })
    }

    /// Retrieves the value at the given index.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::intvec::BEIntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    ///
    /// let input = vec![1, 5, 3, 12, 42];
    /// let intvec = BEIntVec::<GammaCodec>::from(&input, 2).unwrap();
    /// let value = intvec.get(3);
    /// assert_eq!(value, 12);
    /// ```
    ///
    #[inline(always)]
    pub fn get(&self, index: usize) -> u64 {
        if index >= self.len {
            panic!("Index {} is out of bounds", index);
        }

        let sample_index = index / self.k;
        let start_bit = self.samples[sample_index];
        let mut reader =
            BufBitReader::<BE, MemWordReader<u64, &Vec<u64>>>::new(MemWordReader::new(&self.data));

        reader.set_bit_pos(start_bit as u64).unwrap();

        let mut value = 0;
        let start_index = sample_index * self.k;
        let param = self.codec_param;
        for _ in start_index..=index {
            value = C::decode(&mut reader, param).unwrap();
        }
        value
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
    /// let intvec = BEIntVec::<GammaCodec>::from(&input, 2).unwrap();
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

    /// Returns an iterator over the decompressed integer values stored in this compressed vector.
    ///
    /// The iterator decodes values on the fly and does not modify the underlying data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use compressed_intvec::intvec::BEIntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    ///
    /// let input = vec![1, 5, 3, 12, 42];
    /// let intvec = BEIntVec::<GammaCodec>::from(&input, 2).unwrap();
    ///
    /// // Iterate over the vector and print each value.
    /// for (i, value) in intvec.iter().enumerate() {
    ///     assert_eq!(value, input[i]);
    /// }
    /// ```
    pub fn iter(&self) -> BEIntVecIter<C> {
        let word_reader = MemWordReader::new(&self.data);
        let reader = BufBitReader::new(word_reader);
        BEIntVecIter {
            intvec: self,
            reader,
            current_index: 0,
        }
    }

    /// Returns a clone of the internal bitstream data as a vector of 64-bit unsigned integers.
    ///
    /// This can be used for debugging or low-level operations where access to the raw
    /// compressed limb data is required.
    pub fn limbs(&self) -> Vec<u64> {
        self.data.clone()
    }

    /// Returns the number of integers stored in the compressed vector.
    ///
    /// This value represents the total count of decompressed integers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Checks whether the compressed vector contains no elements.
    ///
    /// Returns `true` if the vector is empty, and `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Convenience constructor for codecs with no extra runtime parameter.
impl<C: Codec<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>, Params = ()>> BEIntVec<C> {
    pub fn from(input: &[u64], k: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_with_param(input, k, ())
    }
}

/// Iterator over the values stored in a [`BEIntVec`].
/// The iterator decodes values on the fly.
///
/// # Examples
///
/// ```rust
/// use compressed_intvec::intvec::BEIntVec;
/// use compressed_intvec::codecs::GammaCodec;
///
/// // Create a big-endian compressed vector using GammaCodec.
/// let input = vec![1, 5, 3, 1991, 42];
/// let intvec = BEIntVec::<GammaCodec>::from(&input, 2).unwrap();
///
/// // Iterate over the compressed vector.
/// for (i, value) in intvec.iter().enumerate() {
///    assert_eq!(value, input[i]);
/// }
/// ```
pub struct BEIntVecIter<'a, C>
where
    C: Codec<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>>,
{
    intvec: &'a BEIntVec<C>,
    reader: BufBitReader<BE, MemWordReader<u64, &'a Vec<u64>>>,
    current_index: usize,
}

impl<C> Iterator for BEIntVecIter<'_, C>
where
    C: Codec<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>>,
    C::Params: Copy,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.intvec.len {
            return None;
        }

        match C::decode(&mut self.reader, self.intvec.codec_param) {
            Ok(value) => {
                self.current_index += 1;
                Some(value)
            }
            Err(_) => None,
        }
    }
}

/// Little-endian variant of `IntVec`.
pub type LEIntVec<C> = IntVec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>, C>;

impl<C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>>> LEIntVec<C>
where
    C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>>,
    C::Params: Copy,
{
    /// Creates a new [`LEIntVec`] from a vector of unsigned 64-bit integers.
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
    /// use compressed_intvec::intvec::LEIntVec;
    /// use compressed_intvec::codecs::ExpGolombCodec;
    ///
    /// let input = vec![1, 5, 3, 1991, 42];
    /// let intvec = LEIntVec::<ExpGolombCodec>::from_with_param(&input, 2, 3).unwrap();
    ///
    /// let value = intvec.get(3);
    /// assert_eq!(value, 1991);
    /// ```
    pub fn from_with_param(
        input: &[u64],
        k: usize,
        codec_param: C::Params,
    ) -> Result<Self, Box<dyn std::error::Error>> {
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

        Ok(IntVec {
            data,
            samples,
            codec: PhantomData,
            k,
            len: input.len(),
            codec_param,
            endian: PhantomData,
        })
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
    /// let intvec = LEIntVec::<GammaCodec>::from(&input, 2).unwrap();
    /// let value = intvec.get(3);
    /// assert_eq!(value, 1991);
    /// ```
    ///
    #[inline(always)]
    pub fn get(&self, index: usize) -> u64 {
        if index >= self.len {
            panic!("Index {} is out of bounds", index);
        }

        let sample_index = index / self.k;
        let start_bit = self.samples[sample_index];
        let mut reader =
            BufBitReader::<LE, MemWordReader<u64, &Vec<u64>>>::new(MemWordReader::new(&self.data));

        reader.set_bit_pos(start_bit as u64).unwrap();

        let mut value = 0;
        let start_index = sample_index * self.k;
        let param = self.codec_param;
        for _ in start_index..=index {
            value = C::decode(&mut reader, param).unwrap();
        }
        value
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
    /// let intvec = LEIntVec::<GammaCodec>::from(&input, 2).unwrap();
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
        let word_reader = MemWordReader::new(&self.data);
        let reader = BufBitReader::new(word_reader);
        LEIntVecIter {
            intvec: self,
            reader,
            current_index: 0,
        }
    }

    /// Returns a clone of the internal bitstream data as a vector of 64-bit unsigned integers.
    ///
    /// This can be used for debugging or low-level operations where access to the raw
    /// compressed limb data is required.
    pub fn limbs(&self) -> Vec<u64> {
        self.data.clone()
    }

    /// Returns the number of integers stored in the compressed vector.
    ///
    /// This value represents the total count of decompressed integers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Checks whether the compressed vector contains no elements.
    ///
    /// Returns `true` if the vector is empty, and `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Convenience constructor for codecs with no extra runtime parameter.
impl<C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>, Params = ()>> LEIntVec<C> {
    pub fn from(input: &[u64], k: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_with_param(input, k, ())
    }
}

/// Iterator over the values stored in a [`LEIntVec`].
///
/// This iterator decodes values on the fly using a bit‐stream reader.
/// # Examples
///
/// ```rust
/// use compressed_intvec::intvec::LEIntVec;
/// use compressed_intvec::codecs::GammaCodec;
///
/// // Create a little-endian compressed vector using GammaCodec.
/// // Note: Ensure your codec implements the required traits for iteration.
/// let input = vec![10, 20, 30, 40, 50];
/// let intvec = LEIntVec::<GammaCodec>::from(&input, 2).unwrap();
///
/// // Iterate over the compressed vector.
/// for (i, value) in intvec.iter().enumerate() {
///     assert_eq!(value, input[i]);
/// }
/// ```
pub struct LEIntVecIter<'a, C>
where
    C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>>,
{
    intvec: &'a LEIntVec<C>,
    reader: BufBitReader<LE, MemWordReader<u64, &'a Vec<u64>>>,
    current_index: usize,
}

impl<C> Iterator for LEIntVecIter<'_, C>
where
    C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>>,
    C::Params: Copy,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.intvec.len {
            return None;
        }

        match C::decode(&mut self.reader, self.intvec.codec_param) {
            Ok(value) => {
                self.current_index += 1;
                Some(value)
            }
            Err(_) => None,
        }
    }
}
