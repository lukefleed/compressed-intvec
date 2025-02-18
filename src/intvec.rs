use crate::codecs::Codec;
use dsi_bitstream::prelude::*;
use mem_dbg::{MemDbg, MemSize};
use std::{error::Error, marker::PhantomData};

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
/// use compressed_intvec::BEIntVec;
/// use compressed_intvec::GammaCodec;
///
/// // Create a compressed vector using a codec without extra parameters.
/// let input = vec![1, 2, 3, 4, 5];
/// let intvec = BEIntVec::<GammaCodec>::from(input.clone(), 2).unwrap();
/// let value = intvec.get(3);
/// assert_eq!(value, Some(4));
/// assert_eq!(intvec.len(), 5);
/// ```
#[derive(Debug, Clone, MemDbg, MemSize)]
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
    /// use compressed_intvec::BEIntVec;
    /// use compressed_intvec::codecs::ExpGolombCodec;
    ///
    /// let input = vec![1, 5, 3, 1991, 42];
    /// let intvec = BEIntVec::<ExpGolombCodec>::from_with_param(input, 2, 3).unwrap();
    ///
    /// let value = intvec.get(3);
    /// assert_eq!(value, Some(1991));
    /// ```
    pub fn from_with_param(
        input: Vec<u64>,
        k: usize,
        codec_param: C::Params,
    ) -> Result<Self, Box<dyn Error>> {
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
    /// Returns `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::BEIntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    ///
    /// let input = vec![1, 5, 3, 12, 42];
    /// let intvec = BEIntVec::<GammaCodec>::from(input.clone(), 2).unwrap();
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
    /// use compressed_intvec::BEIntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    ///
    /// let input = vec![43, 12, 5, 1991, 42];
    /// let intvec = BEIntVec::<GammaCodec>::from(input.clone(), 2).unwrap();
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
    pub fn from(input: Vec<u64>, k: usize) -> Result<Self, Box<dyn Error>> {
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
    /// use compressed_intvec::LEIntVec;
    /// use compressed_intvec::codecs::ExpGolombCodec;
    ///
    /// let input = vec![1, 5, 3, 1991, 42];
    /// let intvec = LEIntVec::<ExpGolombCodec>::from_with_param(input, 2, 3).unwrap();
    ///
    /// let value = intvec.get(3);
    /// assert_eq!(value, Some(1991));
    /// ```
    pub fn from_with_param(
        input: Vec<u64>,
        k: usize,
        codec_param: C::Params,
    ) -> Result<Self, Box<dyn Error>> {
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = BufBitWriter::<LE, MemWordWriterVec<u64, Vec<u64>>>::new(word_writer);
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
    /// Returns `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::LEIntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    ///
    /// let input = vec![1, 5, 3, 1991, 42];
    /// let intvec = LEIntVec::<GammaCodec>::from(input.clone(), 2).unwrap();
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
    /// use compressed_intvec::LEIntVec;
    /// use compressed_intvec::codecs::GammaCodec;
    ///
    /// let input = vec![43, 12, 5, 1991, 42];
    /// let intvec = LEIntVec::<GammaCodec>::from(input.clone(), 2).unwrap();
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
    pub fn from(input: Vec<u64>, k: usize) -> Result<Self, Box<dyn Error>> {
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
