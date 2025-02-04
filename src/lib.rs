use std::{error::Error, marker::PhantomData};

use dsi_bitstream::prelude::*;
use mem_dbg::{MemDbg, MemSize};

// Define type aliases for better readability
type MyBitWriter = BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>;
type MyBitReader<'a> = BufBitReader<LE, MemWordReader<u64, &'a Vec<u64>>>;

/// Trait for encoding and decoding values using a variable-length code.
pub trait Codec {
    /// Encodes `value` into the stream represented by `writer`.
    fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>>;

    /// Decodes a value from the stream represented by `reader`
    fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>>;
}

/// Implementation of the Gamma Codec.
pub struct GammaCodec;

impl Codec for GammaCodec {
    #[inline(always)]
    fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_gamma(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_gamma()?)
    }
}

/// Implementation of the Delta Codec.
pub struct DeltaCodec;

impl Codec for DeltaCodec {
    #[inline(always)]
    fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_delta(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_delta()?)
    }
}

pub struct ExpGolombCodec;

impl Codec for ExpGolombCodec {
    #[inline(always)]
    fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_exp_golomb(value, 9)?)
    }

    #[inline(always)]
    fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_exp_golomb(9)?)
    }
}

/// Structure that stores a vector of compressed integers.
/// The generic parameter `C` represents the codec used for encoding/decoding.
#[derive(Debug, Clone, MemDbg, MemSize)]
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
        let mut samples = Vec::new();
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
    #[inline(always)]
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
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
