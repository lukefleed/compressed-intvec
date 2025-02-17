use std::{error::Error, marker::PhantomData};

use dsi_bitstream::prelude::*;
use mem_dbg::{MemDbg, MemSize};

/// Trait for encoding and decoding values using a variable-length code.
///
/// The trait is generic over an endianness type `E` and abstracts over writing/reading
/// bit-level representations.
///
/// # Type Parameters
///
/// - `E`: Endianness marker (e.g. big-endian `BE` or little-endian `LE`).
/// - `W`: A writer capable of writing bits/words in the specified codec.
///
/// # Associated Types
///
/// - `Params`: The type of extra parameters needed for the codec. For many codecs this is
///   `()`, but some require additional runtime parameters.
pub trait Codec<E: Endianness, W: BitWrite<E>> {
    type Params;

    fn encode(writer: &mut W, value: u64, params: Self::Params) -> Result<usize, Box<dyn Error>>;

    fn decode<R2>(reader: &mut R2, params: Self::Params) -> Result<u64, Box<dyn Error>>
    where
        R2: for<'a> GammaRead<E>
            + DeltaRead<E>
            + ExpGolombRead<E>
            + ZetaRead<E>
            + RiceRead<E>
            + ZetaReadParam<E>
            + DeltaReadParam<E>
            + GammaReadParam<E>;
}

/// GammaCodec: no extra runtime parameter.
///
/// Uses the gamma code for encoding and decoding.
pub struct GammaCodec;

impl<E: Endianness, W: GammaWrite<E>> Codec<E, W> for GammaCodec {
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_gamma(value)?)
    }

    #[inline(always)]
    fn decode<R: GammaRead<E>>(reader: &mut R, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_gamma()?)
    }
}

impl GammaCodec {
    /// Encodes a value using gamma coding.
    #[inline(always)]
    pub fn encode<W: GammaWrite<E>, E: Endianness>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, ())
    }
    /// Decodes a value using gamma coding.
    #[inline(always)]
    pub fn decode<E: Endianness, R>(reader: &mut R) -> Result<u64, Box<dyn Error>>
    where
        R: BitRead<E> + GammaRead<E>,
    {
        Ok(reader.read_gamma()?)
    }
}

/// DeltaCodec: no extra runtime parameter.
///
/// Uses the delta code for encoding and decoding.
pub struct DeltaCodec;

impl<E: Endianness, W: DeltaWrite<E>> Codec<E, W> for DeltaCodec {
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_delta(value)?)
    }

    #[inline(always)]
    fn decode<R: DeltaRead<E>>(reader: &mut R, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_delta()?)
    }
}

impl DeltaCodec {
    /// Encodes a value using delta coding.
    #[inline(always)]
    pub fn encode<E: Endianness, W: DeltaWrite<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, ())
    }
    /// Decodes a value using delta coding.
    #[inline(always)]
    pub fn decode<E: Endianness, R>(reader: &mut R) -> Result<u64, Box<dyn Error>>
    where
        R: BitRead<E> + DeltaRead<E>,
    {
        Ok(reader.read_delta()?)
    }
}

/// Exp‑Golomb Codec: requires a runtime parameter (e.g. k).
///
/// This codec supports the Exp‑Golomb coding scheme which is parameterized by `k`.
pub struct ExpGolombCodec;

impl<E: Endianness, W: ExpGolombWrite<E>> Codec<E, W> for ExpGolombCodec {
    type Params = usize;

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, k: usize) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_exp_golomb(value, k)?)
    }

    #[inline(always)]
    fn decode<R: ExpGolombRead<E>>(reader: &mut R, k: usize) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_exp_golomb(k)?)
    }
}

impl ExpGolombCodec {
    /// Encodes a value using Exp‑Golomb coding with the specified parameter `k`.
    #[inline(always)]
    pub fn encode<E: Endianness, W: ExpGolombWrite<E>>(
        writer: &mut W,
        value: u64,
        k: usize,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, k)
    }

    /// Decodes a value using Exp‑Golomb coding with the specified parameter `k`.
    #[inline(always)]
    pub fn decode<E: Endianness, R>(reader: &mut R, k: usize) -> Result<u64, Box<dyn Error>>
    where
        R: BitRead<E> + ExpGolombRead<E>,
    {
        Ok(reader.read_exp_golomb(k)?)
    }
}

/// ZetaCodec: uses runtime parameter (k) with non‑parametric ζ functions.
///
/// The parameter is given as a `u64`.
pub struct ZetaCodec;

impl<E: Endianness, W: ZetaWrite<E>> Codec<E, W> for ZetaCodec {
    type Params = u64;

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, k: u64) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_zeta(value, k)?)
    }

    #[inline(always)]
    fn decode<R: ZetaRead<E>>(reader: &mut R, k: u64) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_zeta(k)?)
    }
}

impl ZetaCodec {
    /// Encodes a value using Zeta coding with parameter `k`.
    #[inline(always)]
    pub fn encode<E: Endianness, W: ZetaWrite<E>>(
        writer: &mut W,
        value: u64,
        k: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, k)
    }

    /// Decodes a value using Zeta coding with parameter `k`.
    #[inline(always)]
    pub fn decode<E: Endianness, R>(reader: &mut R, k: u64) -> Result<u64, Box<dyn Error>>
    where
        R: BitRead<E> + ZetaRead<E>,
    {
        Ok(reader.read_zeta(k)?)
    }
}

/// RiceCodec: uses the Rice functions with a runtime parameter (log2_b).
///
/// The parameter represents the logarithm base‑2 of the encoding base.
pub struct RiceCodec;

impl<E: Endianness, W: RiceWrite<E>> Codec<E, W> for RiceCodec {
    type Params = usize;

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, log2_b: usize) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_rice(value, log2_b)?)
    }

    #[inline(always)]
    fn decode<R: RiceRead<E>>(reader: &mut R, log2_b: usize) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_rice(log2_b)?)
    }
}

impl RiceCodec {
    /// Encodes a value using Rice coding with the specified `log2_b` parameter.
    #[inline(always)]
    pub fn encode<E: Endianness, W: RiceWrite<E>>(
        writer: &mut W,
        value: u64,
        log2_b: usize,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, log2_b)
    }

    /// Decodes a value using Rice coding with the specified `log2_b` parameter.
    #[inline(always)]
    pub fn decode<E: Endianness, R>(reader: &mut R, log2_b: usize) -> Result<u64, Box<dyn Error>>
    where
        R: BitRead<E> + RiceRead<E>,
    {
        Ok(reader.read_rice(log2_b)?)
    }
}

/// ParamZetaCodec: uses a compile‑time flag for ζ functions.
///
/// The compile‑time flag `USE_TABLE` determines whether a lookup table is used.
pub struct ParamZetaCodec<const USE_TABLE: bool>;

impl<E: Endianness, W: ZetaWriteParam<E>, const USE_TABLE: bool> Codec<E, W>
    for ParamZetaCodec<USE_TABLE>
{
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_zeta3_param::<USE_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode<R: ZetaReadParam<E>>(reader: &mut R, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_zeta3_param::<USE_TABLE>()?)
    }
}

impl<const USE_TABLE: bool> ParamZetaCodec<USE_TABLE> {
    /// Encodes a value using the parameterized Zeta codec.
    #[inline(always)]
    pub fn encode<E: Endianness, W: ZetaWriteParam<E>, R: ZetaReadParam<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, ())
    }
    /// Decodes a value using the parameterized Zeta codec.
    #[inline(always)]
    pub fn decode<E: Endianness, W: ZetaWriteParam<E>, R: ZetaReadParam<E>>(
        reader: &mut R,
    ) -> Result<u64, Box<dyn Error>>
    where
        R: ZetaReadParam<E>,
    {
        Ok(reader.read_zeta3_param::<USE_TABLE>()?)
    }
}

/// ParamDeltaCodec: uses compile‑time booleans for table usage.
///
/// The parameters `USE_DELTA_TABLE` and `USE_GAMMA_TABLE` are compile‑time flags.
pub struct ParamDeltaCodec<const USE_DELTA_TABLE: bool, const USE_GAMMA_TABLE: bool>;

impl<
        E: Endianness,
        const USE_DELTA_TABLE: bool,
        const USE_GAMMA_TABLE: bool,
        W: DeltaWriteParam<E>,
    > Codec<E, W> for ParamDeltaCodec<USE_DELTA_TABLE, USE_GAMMA_TABLE>
{
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode<R: DeltaReadParam<E>>(reader: &mut R, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>()?)
    }
}

impl<const USE_DELTA_TABLE: bool, const USE_GAMMA_TABLE: bool>
    ParamDeltaCodec<USE_DELTA_TABLE, USE_GAMMA_TABLE>
{
    /// Encodes a value using the parameterized Delta codec.
    #[inline(always)]
    pub fn encode<E: Endianness, W: DeltaWriteParam<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, ())
    }
    /// Decodes a value using the parameterized Delta codec.
    #[inline(always)]
    pub fn decode<E: Endianness, R: DeltaReadParam<E>>(
        reader: &mut R,
    ) -> Result<u64, Box<dyn Error>>
    where
        R: DeltaReadParam<E>,
    {
        Ok(reader.read_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>()?)
    }
}

/// ParamGammaCodec: uses a compile‑time flag for table usage in gamma coding.
pub struct ParamGammaCodec<const USE_TABLE: bool>;

impl<E: Endianness, W: GammaWriteParam<E>, const USE_TABLE: bool> Codec<E, W>
    for ParamGammaCodec<USE_TABLE>
{
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_gamma_param::<USE_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode<R: GammaReadParam<E>>(reader: &mut R, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_gamma_param::<USE_TABLE>()?)
    }
}

impl<const USE_TABLE: bool> ParamGammaCodec<USE_TABLE> {
    /// Encodes a value using the parameterized Gamma codec.
    #[inline(always)]
    pub fn encode<E: Endianness, W: GammaWriteParam<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, ())
    }
    /// Decodes a value using the parameterized Gamma codec.
    #[inline(always)]
    pub fn decode<E: Endianness, R: GammaReadParam<E>>(
        reader: &mut R,
    ) -> Result<u64, Box<dyn Error>>
    where
        R: GammaReadParam<E>,
    {
        Ok(reader.read_gamma_param::<USE_TABLE>()?)
    }
}

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
    /// use compressed_intvec::ExpGolombCodec;
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
    /// use compressed_intvec::GammaCodec;
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
    /// use compressed_intvec::GammaCodec;
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

impl<'a, C> Iterator for BEIntVecIter<'a, C>
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

impl<'a, C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>>> LEIntVec<C>
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
    /// use compressed_intvec::ExpGolombCodec;
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
    /// use compressed_intvec::GammaCodec;
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
    /// use compressed_intvec::GammaCodec;
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

impl<'a, C> Iterator for LEIntVecIter<'a, C>
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
