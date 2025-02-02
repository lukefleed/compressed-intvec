// use dsi_bitstream::prelude::*;
// use std::error::Error;
// use std::fmt;
// use std::marker::PhantomData;
// use vers_vecs::BitVec;

// /// Trait for encoding and decoding values with a specific variable-length code.
// pub trait Codec {
//     /// Returns the number of bits required to encode `value`.
//     fn encoding_length(value: u64) -> u64;

//     /// Encodes `value` into the writer.
//     fn encode(
//         writer: &mut BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>,
//         value: u64,
//     ) -> Result<()>;

//     /// Decodes a value from the reader.
//     fn decode(reader: &mut BufBitReader<LE, MemWordReader<u64, Vec<u64>>>) -> Result<u64>;
// }

// /// Elias delta codec.
// pub struct DeltaCodec;

// impl Codec for DeltaCodec {
//     fn encoding_length(value: u64) -> u64 {
//         if value == 0 {
//             return 0; // Not allowed, handled during encoding
//         }
//         let l = 64 - value.leading_zeros() - 1;
//         let l_plus_1 = l + 1;
//         let l1 = 64 - l_plus_1.leading_zeros() - 1;
//         l as u64 + 2 * l1 as u64 + 1
//     }

//     fn encode(
//         writer: &mut BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>,
//         value: u64,
//     ) -> Result<()> {
//         writer.write_delta(value)?;
//         Ok(())
//     }

//     fn decode(reader: &mut BufBitReader<LE, MemWordReader<u64, Vec<u64>>>) -> Result<u64> {
//         let value = reader.read_delta()?;
//         Ok(value)
//     }
// }

// /// Elias gamma codec.
// pub struct GammaCodec;

// impl Codec for GammaCodec {
//     fn encoding_length(value: u64) -> u64 {
//         if value == 0 {
//             return 0; // Not allowed, handled during encoding
//         }
//         let l = (64 - value.leading_zeros() - 1) as usize;
//         2 * l as u64 + 1
//     }

//     fn encode(
//         writer: &mut BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>,
//         value: u64,
//     ) -> Result<()> {
//         writer.write_gamma(value)?;
//         Ok(())
//     }

//     fn decode(reader: &mut BufBitReader<LE, MemWordReader<u64, Vec<u64>>>) -> Result<u64> {
//         let value = reader.read_gamma()?;
//         Ok(value)
//     }
// }

// /// A compressed vector of integers using variable-length codes and sampling for fast access.
// /// The vector is compressed using a variable-length code and samples are stored to allow fast
// /// access to elements.
// pub struct VlcVector<C: Codec> {
//     compressed_data: BitVec,
//     sample_pointers: Vec<u64>,
//     len: u64,
//     sample_dens: u32,
//     _codec: PhantomData<C>,
// }

// #[derive(Debug)]
// pub struct VlcVectorError(String);

// impl fmt::Display for VlcVectorError {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "VlcVector error: {}", self.0)
//     }
// }

// impl Error for VlcVectorError {}

// type Result<T> = std::result::Result<T, Box<dyn Error>>;

// impl<C: Codec> VlcVector<C> {
//     /// Constructs a new `VlcVector` from an iterator of values and a sample density.
//     pub fn new<I: IntoIterator<Item = u64>>(values: I, sample_dens: u32) -> Result<Self> {
//         let values: Vec<u64> = values.into_iter().collect();
//         let len = values.len() as u64;
//         if len == 0 {
//             return Ok(Self {
//                 compressed_data: BitVec::new(),
//                 sample_pointers: Vec::new(),
//                 len: 0,
//                 sample_dens,
//                 _codec: PhantomData,
//             });
//         }

//         let sample_dens = sample_dens as usize;
//         if sample_dens == 0 {
//             return Err(Box::new(VlcVectorError("sample_dens must be > 0".into())));
//         }

//         let word_writer = MemWordWriterVec::new(Vec::<u64>::new());
//         let mut writer = BufBitWriter::<LE, _>::new(word_writer);

//         // Encode values and record samples
//         let mut sample_pointers = Vec::new();
//         let mut bit_position = 0u64;
//         for (i, &value) in values.iter().enumerate() {
//             if i % sample_dens == 0 {
//                 sample_pointers.push(bit_position);
//             }
//             let code_value = value + 1;
//             if code_value == 0 {
//                 return Err(Box::new(VlcVectorError("Cannot encode zero".into())));
//             }
//             C::encode(&mut writer, code_value)?;
//             bit_position += C::encoding_length(code_value);
//         }

//         // writer.flush();

//         // Get the written words and create BitVec
//         let written_words = writer.into_inner()?.into_inner();
//         let compressed_data = BitVec::from_limbs(&written_words);

//         Ok(Self {
//             compressed_data,
//             sample_pointers,
//             len,
//             sample_dens: sample_dens as u32,
//             _codec: PhantomData,
//         })
//     }

//     /// Returns the number of elements in the vector.
//     pub fn len(&self) -> u64 {
//         self.len
//     }

//     /// Checks if the vector is empty.
//     pub fn is_empty(&self) -> bool {
//         self.len == 0
//     }

//     /// Accesses the element at the given index.
//     pub fn get(&self, index: usize) -> Result<u64> {
//         if index >= self.len.try_into().unwrap() {
//             return Err(Box::new(VlcVectorError("Index out of bounds".into())));
//         }

//         let sample_dens = self.sample_dens as usize;
//         let sample_index = index / sample_dens;
//         let sample_pos = *self
//             .sample_pointers
//             .get(sample_index)
//             .ok_or_else(|| VlcVectorError("Sample pointer out of bounds".into()))?;

//         let decode_count = (index % sample_dens) + 1;

//         let limbs = self.compressed_data.iter().collect::<Vec<u64>>();
//         let mut reader = BufBitReader::<LE, _>::new(MemWordReader::new(limbs));

//         reader.set_bit_pos(sample_pos)?;

//         let mut value = 0u64;
//         for _ in 0..decode_count {
//             value = C::decode(&mut reader)?;
//         }

//         Ok(value - 1)
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_delta_codec() {
//         let values = vec![0, 1, 2, 3];
//         let vlc = VlcVector::<DeltaCodec>::new(values.clone(), 2).unwrap();
//         for (i, &v) in values.iter().enumerate() {
//             assert_eq!(vlc.get(i).unwrap(), v);
//         }
//     }

//     #[test]
//     fn test_gamma_codec() {
//         let values = vec![0, 1, 2, 3];
//         let vlc = VlcVector::<GammaCodec>::new(values.clone(), 2).unwrap();
//         for (i, &v) in values.iter().enumerate() {
//             assert_eq!(vlc.get(i).unwrap(), v);
//         }
//     }

//     #[test]
//     fn test_sample_dens_larger_than_length() {
//         let values = vec![5, 10, 15];
//         let vlc = VlcVector::<DeltaCodec>::new(values.clone(), 5).unwrap();
//         assert_eq!(vlc.sample_pointers.len(), 1); // Only the first element is sampled
//         assert_eq!(vlc.get(2).unwrap(), 15);
//     }

//     #[test]
//     fn test_empty_vector() {
//         let vlc = VlcVector::<DeltaCodec>::new(vec![], 128).unwrap();
//         assert!(vlc.is_empty());
//     }

//     #[test]
//     fn test_single_element() {
//         let values = vec![42];
//         let vlc = VlcVector::<DeltaCodec>::new(values.clone(), 1).unwrap();
//         assert_eq!(vlc.get(0).unwrap(), 42);
//     }
// }

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
        reader: &mut BufBitReader<LE, MemWordReader<u32, Vec<u32>>>,
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
        writer.write_gamma(value)?;
        Ok(0)
    }

    fn decode(
        reader: &mut BufBitReader<LE, MemWordReader<u32, Vec<u32>>>,
    ) -> std::result::Result<u64, Box<dyn std::error::Error>> {
        let value = reader.read_gamma()?;
        Ok(value)
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
        writer.write_delta(value)?;
        Ok(0)
    }

    fn decode(
        reader: &mut BufBitReader<LE, MemWordReader<u32, Vec<u32>>>,
    ) -> std::result::Result<u64, Box<dyn std::error::Error>> {
        let value = reader.read_delta()?;
        Ok(value)
    }
}

// ======================================================
// Compressed IntVec Structure
// ======================================================

#[derive(Debug, Clone)]
pub struct IntVec<C: Codec> {
    // / Raw compressed data stored as u64 words
    data: Vec<u64>,
    // / Total number of bits in the compressed stream
    total_bits: usize,
    // / Sampled bit positions for random access
    samples: Vec<usize>,
    // / Codec used for encoding/decoding
    codec: C,
    // / Sampling interval (every k-th element is sampled)
    k: usize,
    // / Number of elements in the original vector
    len: usize,
}

impl<C: Codec> IntVec<C> {
    pub fn from(input: Vec<u64>, codec: C, k: usize) -> Self {
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

        writer.flush();
        let data = writer.into_inner().unwrap().into_inner();

        IntVec {
            data,
            total_bits,
            samples,
            codec,
            k,
            len: input.len(),
        }
    }

    //     ======================================================
    //     Random Access
    //     ======================================================

    // Retrieves the i-th element from the compressed vector.
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }

        // Find the nearest sample position
        let sample_index = index / self.k;
        let element_in_sample = index % self.k;
        let Some(&start_bit) = self.samples.get(sample_index) else {
            return None;
        };

        // Convert u64 data to u32 words for the reader
        let data_u32 = unsafe { std::mem::transmute::<_, Vec<u32>>(self.data.clone()) };

        // Initialize a reader starting at the sampled bit position
        let mut reader =
            BufBitReader::<LE, MemWordReader<u32, Vec<u32>>>::new(MemWordReader::new(data_u32));
        reader.read_bits(start_bit).unwrap();

        // Decode elements until we reach the target index
        let mut value = 0;
        for _ in 0..=element_in_sample {
            value = C::decode(&mut reader).unwrap();
        }

        Some(value)
    }

    // ======================================================
    // Utilities
    // ======================================================

    // / Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    // / Checks if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
