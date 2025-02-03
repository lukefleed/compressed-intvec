use std::{error::Error, marker::PhantomData};

use dsi_bitstream::prelude::*;

// Define type aliases for better readability
type MyBitWriter = BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>;
type MyBitReader<'a> = BufBitReader<LE, MemWordReader<u64, &'a Vec<u64>>>;

/// Trait for encoding and decoding values using a variable-length code.
pub trait Codec {
    /// Encodes `value` into the stream represented by `writer`.
    fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>>;

    /// Decodes a value from the stream represented by `reader`.
    fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>>;
}

/// Implementation of the Gamma Codec.
pub struct GammaCodec;

impl Codec for GammaCodec {
    fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_gamma(value)?)
    }

    fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_gamma()?)
    }
}

/// Implementation of the Delta Codec.
pub struct DeltaCodec;

impl Codec for DeltaCodec {
    fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_delta(value)?)
    }

    fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_delta()?)
    }
}

/// Structure that stores a vector of compressed integers.
/// The generic parameter `C` represents the codec used for encoding/decoding.
#[derive(Debug, Clone)]
pub struct IntVec<C: Codec> {
    /// Compressed data (64-bit words)
    data: Vec<u64>,
    /// Sampled indices for quick restoration
    samples: Vec<usize>,
    /// Marker for the codec used
    codec: PhantomData<C>,
    /// Sampling interval
    k: usize,
    /// Number of elements in the original array
    len: usize,
}

impl<C: Codec> IntVec<C> {
    /// Constructs a new `IntVec` from an input vector and a sampling parameter `k`.
    ///
    /// # Errors
    ///
    /// Returns an error if there is a problem writing or flushing the stream.
    pub fn from(input: Vec<u64>, k: usize) -> Result<Self, Box<dyn Error>> {
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = MyBitWriter::new(word_writer);
        let mut samples = Vec::with_capacity((input.len() + k - 1) / k);
        let mut total_bits = 0;

        for (i, &x) in input.iter().enumerate() {
            if i % k == 0 {
                samples.push(total_bits);
            }
            total_bits += C::encode(&mut writer, x)?;
        }
        writer.flush()?;
        let data = writer.into_inner()?.into_inner();

        Ok(IntVec {
            data,
            samples,
            codec: PhantomData,
            k,
            len: input.len(),
        })
    }

    /// Returns the value at the `index` position in the original vector.
    ///
    /// Returns `None` if the index exceeds the length.
    #[inline]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }
        // Find the nearest left sample
        let sample_index = index / self.k;
        let start_bit = self.samples[sample_index];

        let word_reader = MemWordReader::new(&self.data);
        let mut reader = MyBitReader::new(word_reader);

        // Set the reader position to the identified sample
        reader.set_bit_pos(start_bit as u64).ok()?;

        // Decode from the sample until the desired index is reached
        let mut value = 0;
        let start_index = sample_index * self.k;
        for _ in start_index..=index {
            value = C::decode(&mut reader).ok()?;
        }
        Some(value)
    }

    /// Returns the number of stored elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use crate::{DeltaCodec, GammaCodec, IntVec};

    #[test]
    fn test_gamma_codec() {
        // create a random vector with 1000 elements from 0 to 10000
        let input: Vec<u64> = (0..1000).map(|_| rand::random::<u64>() % 10000).collect();
        let compressed_input = IntVec::<GammaCodec>::from(input.clone(), 64).unwrap();

        for i in 0..input.len() {
            assert_eq!(input[i], compressed_input.get(i).unwrap());
        }
    }

    #[test]
    fn test_gamma_codec_empty() {
        let input: Vec<u64> = Vec::new();
        let compressed_input = IntVec::<GammaCodec>::from(input.clone(), 64).unwrap();

        assert_eq!(compressed_input.len(), 0);
        assert_eq!(compressed_input.is_empty(), true);
    }

    #[test]
    fn test_gamma_codec_single_element() {
        let input: Vec<u64> = vec![42];
        let compressed_input = IntVec::<GammaCodec>::from(input.clone(), 64).unwrap();

        assert_eq!(compressed_input.len(), 1);
        assert_eq!(compressed_input.get(0).unwrap(), 42);
    }

    #[test]
    fn test_delta_codec() {
        // create a random vector with 1000 elements from 0 to 10000
        let input: Vec<u64> = (0..1000).map(|_| rand::random::<u64>() % 10000).collect();
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64).unwrap();

        for i in 0..input.len() {
            assert_eq!(input[i], compressed_input.get(i).unwrap());
        }
    }

    #[test]
    fn test_delta_codec_empty() {
        let input: Vec<u64> = Vec::new();
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64).unwrap();

        assert_eq!(compressed_input.len(), 0);
        assert_eq!(compressed_input.is_empty(), true);
    }

    #[test]
    fn test_delta_codec_single_element() {
        let input: Vec<u64> = vec![42];
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64).unwrap();

        assert_eq!(compressed_input.len(), 1);
        assert_eq!(compressed_input.get(0).unwrap(), 42);
    }

    #[test]
    fn test_gamma_codec_sampling() {
        // create a random vector with 1000 elements from 0 to 10000
        let input: Vec<u64> = (0..1000).map(|_| rand::random::<u64>() % 10000).collect();
        let compressed_input = IntVec::<GammaCodec>::from(input.clone(), 64).unwrap();

        for i in 0..input.len() {
            if i % 64 == 0 {
                assert_eq!(input[i], compressed_input.get(i).unwrap());
            }
        }
    }
}
