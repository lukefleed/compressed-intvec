use std::marker::PhantomData;

use dsi_bitstream::prelude::*;

/// Trait for encoding and decoding values with a specific variable-length code.
pub trait Codec {
    /// Encodes `value` into the writer.
    fn encode(
        writer: &mut BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>,
        value: u64,
    ) -> std::result::Result<usize, Box<dyn std::error::Error>>;

    /// Decodes a value from the reader.
    fn decode(
        reader: &mut BufBitReader<LE, MemWordReader<u64, Vec<u64>>>,
    ) -> std::result::Result<u64, Box<dyn std::error::Error>>;
}

// ======================================================
// Gamma Encoding Implementation
// ======================================================

pub struct GammaCodec;

impl Codec for GammaCodec {
    fn encode(
        writer: &mut BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>,
        value: u64,
    ) -> std::result::Result<usize, Box<dyn std::error::Error>> {
        Ok(writer.write_gamma(value)?)
    }

    fn decode(
        reader: &mut BufBitReader<LE, MemWordReader<u64, Vec<u64>>>,
    ) -> std::result::Result<u64, Box<dyn std::error::Error>> {
        Ok(reader.read_gamma()?)
    }
}

// ======================================================
// Delta Encoding Implementation
// ======================================================

pub struct DeltaCodec;

impl Codec for DeltaCodec {
    fn encode(
        writer: &mut BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>,
        value: u64,
    ) -> std::result::Result<usize, Box<dyn std::error::Error>> {
        Ok(writer.write_delta(value)?)
    }

    fn decode(
        reader: &mut BufBitReader<LE, MemWordReader<u64, Vec<u64>>>,
    ) -> std::result::Result<u64, Box<dyn std::error::Error>> {
        Ok(reader.read_delta()?)
    }
}

// ======================================================
// Compressed IntVec Structure
// ======================================================

#[derive(Debug, Clone)]
pub struct IntVec<C: Codec> {
    /// Compressed data
    data: Vec<u64>,
    /// Sampled indices
    samples: Vec<usize>,
    /// Codec used to encode the data
    codec: PhantomData<C>,
    /// Sampling rate
    k: usize,
}

impl<C: Codec> IntVec<C> {
    pub fn from(input: Vec<u64>, k: usize) -> Self {
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = BufBitWriter::<LE, MemWordWriterVec<u64, Vec<u64>>>::new(word_writer);
        let mut samples = Vec::new();
        let mut total_bits = 0;

        for (i, &x) in input.iter().enumerate() {
            if i % k == 0 {
                samples.push(total_bits);
            }
            let bits_used = C::encode(&mut writer, x).unwrap();
            total_bits += bits_used;
        }

        writer.flush().unwrap(); // Ensure all bits are flushed
        let data = writer.into_inner().unwrap().into_inner();

        IntVec {
            data,
            samples,
            codec: PhantomData,
            k,
        }
    }

    // Get the value at the given index in the original vector
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.data.len() {
            return None;
        }

        // Trova il sample più vicino a sinistra dell'indice richiesto.
        let sample_index = index / self.k;
        let start = self.samples[sample_index];

        // Crea il reader a partire dai dati compressi.
        let word_reader = MemWordReader::new(self.data.clone());
        let mut reader = BufBitReader::<LE, MemWordReader<u64, Vec<u64>>>::new(word_reader);

        // Imposta la posizione iniziale del reader sul sample.
        reader.set_bit_pos(start as u64).unwrap();

        // A partire dal sample, decodifica i valori fino a raggiungere quello desiderato.
        let mut value = 0;
        // Calcola l'indice nel blocco a partire dal sample.
        let start_index = sample_index * self.k;
        for _ in start_index..=index {
            value = C::decode(&mut reader).ok()?;
        }

        Some(value)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::{DeltaCodec, GammaCodec, IntVec};

    #[test]
    fn test_gamma_codec() {
        // create a random vector with 1000 elements from 0 to 10000
        let input: Vec<u64> = (0..1000).map(|_| rand::random::<u64>() % 10000).collect();
        let compressed_input = IntVec::<GammaCodec>::from(input.clone(), 64);

        for i in 0..input.len() {
            assert_eq!(input[i], compressed_input.get(i).unwrap());
        }
    }

    #[test]
    fn test_gamma_codec_empty() {
        let input: Vec<u64> = Vec::new();
        let compressed_input = IntVec::<GammaCodec>::from(input.clone(), 64);

        assert_eq!(compressed_input.len(), 0);
        assert_eq!(compressed_input.is_empty(), true);
    }

    #[test]
    fn test_gamma_codec_single_element() {
        let input: Vec<u64> = vec![42];
        let compressed_input = IntVec::<GammaCodec>::from(input.clone(), 64);

        assert_eq!(compressed_input.len(), 1);
        assert_eq!(compressed_input.get(0).unwrap(), 42);
    }

    #[test]
    fn test_delta_codec() {
        // create a random vector with 1000 elements from 0 to 10000
        let input: Vec<u64> = (0..1000).map(|_| rand::random::<u64>() % 10000).collect();
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64);

        for i in 0..input.len() {
            assert_eq!(input[i], compressed_input.get(i).unwrap());
        }
    }

    #[test]
    fn test_delta_codec_empty() {
        let input: Vec<u64> = Vec::new();
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64);

        assert_eq!(compressed_input.len(), 0);
        assert_eq!(compressed_input.is_empty(), true);
    }

    #[test]
    fn test_delta_codec_single_element() {
        let input: Vec<u64> = vec![42];
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64);

        assert_eq!(compressed_input.len(), 1);
        assert_eq!(compressed_input.get(0).unwrap(), 42);
    }

    #[test]
    fn test_gamma_codec_sampling() {
        // create a random vector with 1000 elements from 0 to 10000
        let input: Vec<u64> = (0..1000).map(|_| rand::random::<u64>() % 10000).collect();
        let compressed_input = IntVec::<GammaCodec>::from(input.clone(), 64);

        for i in 0..input.len() {
            if i % 64 == 0 {
                assert_eq!(input[i], compressed_input.get(i).unwrap());
            }
        }
    }
}
