//! # Compressed IntVec Module
//!
//! This module implements a compressed vector for storing unsigned 64-bit integers
//! in an efficient manner. It leverages a codec to encode the integers into a compact
//! bitstream and uses a sampling strategy to achieve fast random access.
//!
//! ## How It Works
//!
//! The `IntVec` structure compresses a series of `u64` values by encoding them with a
//! specified codec. Every kth element is sampled during encoding, meaning its bit-offset
//! is recorded. This sampling information allows the decoder to jump directly to the
//! vicinity of any desired element, thereby minimizing the number of bits that need to be
//! processed during random access.
//!
//! ## Features
//!
//! - **Efficient Compression:** Reduces storage requirements by encoding data into a compact bitstream.
//! - **Fast Random Access:** Leverages sampled bit-offsets to quickly decode individual elements.
//! - **Iterative Decoding:** Provides an iterator that decodes values on the fly for sequential access.
//!
//! ## Example Usage
//!
//! ```rust
//! use compressed_intvec::intvec::IntVec;
//! use compressed_intvec::codecs::GammaCodec;
//! use dsi_bitstream::traits::BE;
//!
//! let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
//! let sampling_rate = 2;
//! let intvec = IntVec::<BE, _, GammaCodec>::from(&input, sampling_rate).unwrap();
//!
//! // Fast random access using the sampled bit offsets
//! assert_eq!(intvec.get(3), 4);
//!
//! // Decode the entire vector back to its original form
//! let decompressed: Vec<u64> = intvec.into_vec();
//! assert_eq!(decompressed, input);
//! ```
//!
//! ## Type Aliases
//!
//! For convenience, this module provides type aliases for common configurations:
//!
//! - **BEIntVec:** A big-endian version of the compressed vector.
//!
//!   Example:
//!
//!   ```rust
//!   use compressed_intvec::intvec::BEIntVec;
//!   use compressed_intvec::codecs::GammaCodec;
//!
//!   let input = vec![10, 20, 30];
//!   let intvec = BEIntVec::<GammaCodec>::from(&input, 2).unwrap();
//!   ```
//!
//! - **LEIntVec:** A little-endian version of the compressed vector.
//!
//!   Example:
//!
//!   ```rust
//!   use compressed_intvec::intvec::LEIntVec;
//!   use compressed_intvec::codecs::GammaCodec;
//!
//!   let input = vec![9, 8, 7];
//!   let intvec = LEIntVec::<GammaCodec>::from(&input, 2).unwrap();
//!   ```

use crate::codecs::Codec;
use dsi_bitstream::{codes::params::DefaultReadParams, prelude::*};
use mem_dbg::{MemDbg, MemSize};

/// A compressed vector of unsigned 64-bit integers using a specified codec.
///
/// The `IntVec` structure compresses a sequence of `u64` values by encoding them using a
/// provided codec. It supports random access via sampling and iteration over the decompressed
/// values.
///
/// The type parameters are:
/// - `E`: Endianness to use (`BE` or `LE`).
/// - `W`: A bit writer implementing `BitWrite<E>`.
/// - `C`: A codec implementing `Codec<E, W>`.
///
/// # Examples
///
/// ```rust
/// use compressed_intvec::intvec::IntVec;
/// use compressed_intvec::codecs::GammaCodec;
/// use dsi_bitstream::traits::BE;
///
/// let input = vec![1, 2, 3, 4, 5];
/// let k = 2;
/// let intvec = IntVec::<BE, _, GammaCodec>::from(&input, k).unwrap();
/// assert_eq!(intvec.len(), input.len());
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
    data: Vec<u64>,
    samples: Vec<usize>,
    k: usize,
    len: usize,
    codec_param: C::Params,
    codec: std::marker::PhantomData<C>,
    endian: std::marker::PhantomData<E>,
}

impl<E, C> IntVec<E, BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>, C>
where
    E: Endianness,
    C: Codec<E, BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>>,
    C::Params: Copy,
    BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>: BitWrite<E>,
    for<'a> BufBitReader<E, MemWordReader<u64, &'a Vec<u64>>>:
        GammaRead<E> + DeltaRead<E> + ZetaRead<E> + ZetaReadParam<E> + DeltaReadParam<E>,
{
    /// Constructs an `IntVec` from a slice of `u64` values using the provided codec parameters.
    ///
    /// This method encodes the input values using the specified codec. It also builds sampling
    /// information for efficient random access. The parameter `k` determines the sampling rate,
    /// i.e. every kth element's bit position is recorded as a sample.
    ///
    /// # Parameters
    ///
    /// - `input`: A slice of unsigned 64-bit integers to compress.
    /// - `k`: The sampling rate; every kth element will have its bit offset stored.
    /// - `codec_param`: The parameters for the codec used for encoding.
    ///
    /// # Returns
    ///
    /// Returns `Ok(IntVec)` on success or an error if encoding fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::intvec::IntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    /// use dsi_bitstream::traits::BE;
    ///
    /// let input = vec![10, 20, 30, 40];
    /// let k = 2;
    /// let intvec = IntVec::<BE, _, GammaCodec>::from(&input, k).unwrap();
    /// assert_eq!(intvec.len(), input.len());
    /// ```
    pub fn from_with_param(
        input: &[u64],
        k: usize,
        codec_param: C::Params,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = BufBitWriter::<E, MemWordWriterVec<u64, Vec<u64>>>::new(word_writer);
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

        Ok(Self {
            data,
            samples,
            codec: std::marker::PhantomData,
            k,
            len: input.len(),
            codec_param,
            endian: std::marker::PhantomData,
        })
    }

    /// Constructs an `IntVec` from a slice of `u64` values using the default codec parameters.
    ///
    /// This is a convenience method that calls `from_with_param` with `C::Params::default()`.
    ///
    /// # Parameters
    ///
    /// - `input`: A slice of unsigned 64-bit integers to compress.
    /// - `k`: The sampling rate; every kth element will have its bit offset stored.
    ///
    /// # Returns
    ///
    /// Returns `Ok(IntVec)` on success or an error if encoding fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::intvec::IntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    /// use dsi_bitstream::traits::BE;
    ///
    /// let input = vec![5, 10, 15, 20];
    /// let k = 2;
    /// let intvec = IntVec::<BE, _, GammaCodec>::from(&input, k).unwrap();
    /// assert_eq!(intvec.len(), input.len());
    /// ```
    pub fn from(input: &[u64], k: usize) -> Result<Self, Box<dyn std::error::Error>>
    where
        C::Params: Default,
    {
        Self::from_with_param(input, k, C::Params::default())
    }

    /// Retrieves the element at the specified index after decompressing the value.
    ///
    /// This method uses the sampling information to quickly locate the encoded bit-stream
    /// position and decodes the required value.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Parameters
    ///
    /// - `index`: The index of the desired element.
    ///
    /// # Returns
    ///
    /// Returns the `u64` value at the given index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::intvec::IntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    /// use dsi_bitstream::traits::BE;
    ///
    /// let input = vec![3, 6, 9];
    /// let k = 1;
    /// let intvec = IntVec::<BE, _, GammaCodec>::from(&input, k).unwrap();
    /// assert_eq!(intvec.get(1), 6);
    /// ```
    #[inline(always)]
    pub fn get(&self, index: usize) -> u64 {
        if index >= self.len {
            panic!("Index {} is out of bounds", index);
        }

        let sample_index = index / self.k;
        let start_bit = self.samples[sample_index];
        let mut reader =
            BufBitReader::<E, MemWordReader<u64, &'_ Vec<u64>>, DefaultReadParams>::new(
                MemWordReader::new(&self.data),
            );
        reader.skip_bits(start_bit).unwrap();

        let mut value = 0;
        let start_index = sample_index * self.k;
        for _ in start_index..=index {
            value = C::decode(&mut reader, self.codec_param).unwrap();
        }
        value
    }

    /// Consumes the `IntVec` and returns a vector containing all decompressed `u64` values.
    ///
    /// This method sequentially decodes all the values from the compressed representation
    /// into a standard `Vec<u64>`.
    ///
    /// # Returns
    ///
    /// A vector of decompressed values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::intvec::IntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    /// use dsi_bitstream::traits::LE;
    ///
    /// let input = vec![7, 14, 21];
    /// let k = 2;
    /// let intvec = IntVec::<LE, _, GammaCodec>::from(&input, k).unwrap();
    /// let output = intvec.into_vec();
    /// assert_eq!(output, input);
    /// ```
    pub fn into_vec(self) -> Vec<u64> {
        let word_reader = MemWordReader::new(&self.data);
        let mut reader =
            BufBitReader::<E, MemWordReader<u64, &'_ Vec<u64>>, DefaultReadParams>::new(
                word_reader,
            );
        let mut values = Vec::with_capacity(self.len);

        for _ in 0..self.len {
            values.push(C::decode(&mut reader, self.codec_param).unwrap());
        }
        values
    }

    /// Returns the underlying storage (limbs) of the compressed bitstream.
    ///
    /// The limbs are stored as a vector of `u64`, which represents the raw compressed data.
    ///
    /// # Returns
    ///
    /// A clone of the internal vector of limbs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::intvec::IntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    /// use dsi_bitstream::traits::BE;
    ///
    /// let input = vec![2, 4, 6, 8];
    /// let k = 2;
    /// let intvec = IntVec::<BE, _, GammaCodec>::from(&input, k).unwrap();
    /// let limbs = intvec.limbs();
    /// assert!(!limbs.is_empty());
    /// ```
    pub fn limbs(&self) -> Vec<u64> {
        self.data.clone()
    }

    /// Returns the number of elements encoded in the `IntVec`.
    ///
    /// # Returns
    ///
    /// The length (number of original elements) as a usize.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::intvec::IntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    /// use dsi_bitstream::traits::BE;
    ///
    /// let input = vec![1, 2, 3];
    /// let k = 2;
    /// let intvec = IntVec::<BE, _, GammaCodec>::from(&input, k).unwrap();
    /// assert_eq!(intvec.len(), input.len());
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the sampling rate `k` used during encoding.
    ///
    /// This value indicates that every kth element's bit offset was stored for efficient access.
    ///
    /// # Returns
    ///
    /// The sampling rate as usize.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::intvec::IntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    /// use dsi_bitstream::traits::BE;
    ///
    /// let input = vec![5, 10, 15, 20];
    /// let k = 2;
    /// let intvec = IntVec::<BE, _, GammaCodec>::from(&input, k).unwrap();
    /// assert_eq!(intvec.get_sampling_rate(), k);
    /// ```
    pub fn get_sampling_rate(&self) -> usize {
        self.k
    }

    /// Returns the recorded sample bit positions used for random access.
    ///
    /// The returned vector contains the bit-offset positions for every kth element in the original data.
    ///
    /// # Returns
    ///
    /// A vector of sample positions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::intvec::IntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    /// use dsi_bitstream::traits::BE;
    ///
    /// let input = vec![10, 20, 30, 40, 50];
    /// let k = 2;
    /// let intvec = IntVec::<BE, _, GammaCodec>::from(&input, k).unwrap();
    /// let samples = intvec.get_samples();
    /// assert!(!samples.is_empty());
    /// ```
    pub fn get_samples(&self) -> Vec<usize> {
        self.samples.clone()
    }

    /// Checks if the `IntVec` contains no encoded elements.
    ///
    /// # Returns
    ///
    /// `true` if there are no elements, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::intvec::IntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    /// use dsi_bitstream::traits::BE;
    ///
    /// let empty_intvec: IntVec<BE, _, GammaCodec> = IntVec::from(&[], 2).unwrap();
    /// assert!(empty_intvec.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an iterator over the decompressed values contained in the `IntVec`.
    ///
    /// The iterator decodes each value in sequence and yields it.
    ///
    /// # Returns
    ///
    /// An `IntVecIter` instance that implements `Iterator<Item = u64>`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::intvec::IntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    /// use dsi_bitstream::traits::BE;
    ///
    /// let input = vec![1, 3, 5, 7];
    /// let k = 1;
    /// let intvec = IntVec::<BE, _, GammaCodec>::from(&input, k).unwrap();
    /// let mut iter = intvec.iter();
    /// for value in input {
    ///     assert_eq!(iter.next(), Some(value));
    /// }
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> IntVecIter<E, C> {
        let word_reader = MemWordReader::new(&self.data);
        let reader = BufBitReader::<E, MemWordReader<u64, &'_ Vec<u64>>, DefaultReadParams>::new(
            word_reader,
        );
        IntVecIter {
            intvec: self,
            reader,
            current_index: 0,
        }
    }
}

/// An iterator over the values of an `IntVec`.
///
/// The iterator holds a reference to the `IntVec` and uses a bit reader to decode each value on the fly.
///
/// # Examples
///
/// ```rust
/// use compressed_intvec::intvec::IntVec;
/// use compressed_intvec::codecs::GammaCodec;
/// use dsi_bitstream::traits::BE;
///
/// let input = vec![2, 4, 6];
/// let k = 1;
/// let intvec = IntVec::<BE, _, GammaCodec>::from(&input, k).unwrap();
/// let mut iter = intvec.iter();
/// assert_eq!(iter.next(), Some(2));
/// assert_eq!(iter.next(), Some(4));
/// assert_eq!(iter.next(), Some(6));
/// assert_eq!(iter.next(), None);
/// ```
pub struct IntVecIter<'a, E, C>
where
    E: Endianness,
    C: Codec<E, BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>>,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<u64, Vec<u64>>>:
        dsi_bitstream::traits::BitWrite<E>,
{
    intvec: &'a IntVec<E, BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>, C>,
    reader: BufBitReader<E, MemWordReader<u64, &'a Vec<u64>>, DefaultReadParams>,
    current_index: usize,
}

impl<'a, E, C> Iterator for IntVecIter<'a, E, C>
where
    E: Endianness,
    C: Codec<E, BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>>,
    C::Params: Copy,
    BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>: BitWrite<E>,
    BufBitReader<E, MemWordReader<u64, &'a Vec<u64>>, DefaultReadParams>:
        GammaRead<E> + DeltaRead<E> + ZetaRead<E> + ZetaReadParam<E> + DeltaReadParam<E>,
{
    type Item = u64;

    /// Advances the iterator and returns the next decompressed value.
    ///
    /// Returns [`None`] when iteration is finished.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compressed_intvec::intvec::IntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    /// use dsi_bitstream::traits::BE;
    ///
    /// let input = vec![5, 10];
    /// let k = 1;
    /// let intvec = IntVec::<BE, _, GammaCodec>::from(&input, k).unwrap();
    /// let mut iter = intvec.iter();
    /// assert_eq!(iter.next(), Some(5));
    /// assert_eq!(iter.next(), Some(10));
    /// assert_eq!(iter.next(), None);
    /// ```
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

pub type BEIntVec<C> = IntVec<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>, C>;
pub type LEIntVec<C> = IntVec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>, C>;
