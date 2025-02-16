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
/// - `R`: A reader capable of reading bits/words in the specified codec.
///
/// # Associated Types
///
/// - `Params`: The type of extra parameters needed for the codec. For many codecs this is
///   `()`, but some require additional runtime parameters.
pub trait Codec<E: Endianness, W, R> {
    /// The type of parameters for encoding/decoding.
    type Params;

    /// Encodes `value` into the stream represented by `writer`, using the provided parameters.
    ///
    /// Returns the number of bits written.
    fn encode(writer: &mut W, value: u64, params: Self::Params) -> Result<usize, Box<dyn Error>>;

    /// Decodes a value from the stream represented by `reader`, using the provided parameters.
    ///
    /// Returns the decoded `u64` value.
    fn decode(reader: &mut R, params: Self::Params) -> Result<u64, Box<dyn Error>>;
}

/// GammaCodec: no extra runtime parameter.
///
/// Uses the gamma code for encoding and decoding.
pub struct GammaCodec;

impl<E: Endianness, W: GammaWrite<E>, R: GammaRead<E>> Codec<E, W, R> for GammaCodec {
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_gamma(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut R, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_gamma()?)
    }
}

impl GammaCodec {
    /// Encodes a value using gamma coding.
    #[inline(always)]
    pub fn encode<W: GammaWrite<E>, R: GammaRead<E>, E: Endianness>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::encode(writer, value, ())
    }
    /// Decodes a value using gamma coding.
    #[inline(always)]
    pub fn decode<W: GammaWrite<E>, R: GammaRead<E>, E: Endianness>(
        reader: &mut R,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::decode(reader, ())
    }
}

/// DeltaCodec: no extra runtime parameter.
///
/// Uses the delta code for encoding and decoding.
pub struct DeltaCodec;

impl<E: Endianness, W: DeltaWrite<E>, R: DeltaRead<E>> Codec<E, W, R> for DeltaCodec {
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_delta(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut R, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_delta()?)
    }
}

impl DeltaCodec {
    /// Encodes a value using delta coding.
    #[inline(always)]
    pub fn encode<E: Endianness, W: DeltaWrite<E>, R: DeltaRead<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::encode(writer, value, ())
    }
    /// Decodes a value using delta coding.
    #[inline(always)]
    pub fn decode<E: Endianness, W: DeltaWrite<E>, R: DeltaRead<E>>(
        reader: &mut R,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::decode(reader, ())
    }
}

/// Exp‑Golomb Codec: requires a runtime parameter (e.g. k).
///
/// This codec supports the Exp‑Golomb coding scheme which is parameterized by `k`.
pub struct ExpGolombCodec;

impl<E: Endianness, W: ExpGolombWrite<E>, R: ExpGolombRead<E>> Codec<E, W, R> for ExpGolombCodec {
    type Params = usize;

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, k: usize) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_exp_golomb(value, k)?)
    }

    #[inline(always)]
    fn decode(reader: &mut R, k: usize) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_exp_golomb(k)?)
    }
}

impl ExpGolombCodec {
    /// Encodes a value using Exp‑Golomb coding with the specified parameter `k`.
    #[inline(always)]
    pub fn encode<E: Endianness, W: ExpGolombWrite<E>, R: ExpGolombRead<E>>(
        writer: &mut W,
        value: u64,
        k: usize,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::encode(writer, value, k)
    }
    /// Decodes a value using Exp‑Golomb coding with the specified parameter `k`.
    #[inline(always)]
    pub fn decode<E: Endianness, W: ExpGolombWrite<E>, R: ExpGolombRead<E>>(
        reader: &mut R,
        k: usize,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::decode(reader, k)
    }
}

/// ZetaCodec: uses runtime parameter (k) with non‑parametric ζ functions.
///
/// The parameter is given as a `u64`.
pub struct ZetaCodec;

impl<E: Endianness, W: ZetaWrite<E>, R: ZetaRead<E>> Codec<E, W, R> for ZetaCodec {
    type Params = u64;

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, k: u64) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_zeta(value, k)?)
    }

    #[inline(always)]
    fn decode(reader: &mut R, k: u64) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_zeta(k)?)
    }
}

impl ZetaCodec {
    /// Encodes a value using Zeta coding with parameter `k`.
    #[inline(always)]
    pub fn encode<E: Endianness, W: ZetaWrite<E>, R: ZetaRead<E>>(
        writer: &mut W,
        value: u64,
        k: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::encode(writer, value, k)
    }

    /// Decodes a value using Zeta coding with parameter `k`.
    #[inline(always)]
    pub fn decode<E: Endianness, W: ZetaWrite<E>, R: ZetaRead<E>>(
        reader: &mut R,
        k: u64,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::decode(reader, k)
    }
}

/// RiceCodec: uses the Rice functions with a runtime parameter (log2_b).
///
/// The parameter represents the logarithm base‑2 of the encoding base.
pub struct RiceCodec;

impl<E: Endianness, W: RiceWrite<E>, R: RiceRead<E>> Codec<E, W, R> for RiceCodec {
    type Params = usize;

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, log2_b: usize) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_rice(value, log2_b)?)
    }

    #[inline(always)]
    fn decode(reader: &mut R, log2_b: usize) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_rice(log2_b)?)
    }
}

impl RiceCodec {
    /// Encodes a value using Rice coding with the specified `log2_b` parameter.
    #[inline(always)]
    pub fn encode<E: Endianness, W: RiceWrite<E>, R: RiceRead<E>>(
        writer: &mut W,
        value: u64,
        log2_b: usize,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::encode(writer, value, log2_b)
    }

    /// Decodes a value using Rice coding with the specified `log2_b` parameter.
    #[inline(always)]
    pub fn decode<E: Endianness, W: RiceWrite<E>, R: RiceRead<E>>(
        reader: &mut R,
        log2_b: usize,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::decode(reader, log2_b)
    }
}

/// ParamZetaCodec: uses a compile‑time flag for ζ functions.
///
/// The compile‑time flag `USE_TABLE` determines whether a lookup table is used.
pub struct ParamZetaCodec<const USE_TABLE: bool>;

impl<E: Endianness, W: ZetaWriteParam<E>, R: ZetaReadParam<E>, const USE_TABLE: bool> Codec<E, W, R>
    for ParamZetaCodec<USE_TABLE>
{
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_zeta3_param::<USE_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut R, _params: ()) -> Result<u64, Box<dyn Error>> {
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
        <Self as Codec<E, W, R>>::encode(writer, value, ())
    }
    /// Decodes a value using the parameterized Zeta codec.
    #[inline(always)]
    pub fn decode<E: Endianness, W: ZetaWriteParam<E>, R: ZetaReadParam<E>>(
        reader: &mut R,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::decode(reader, ())
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
        R: DeltaReadParam<E>,
    > Codec<E, W, R> for ParamDeltaCodec<USE_DELTA_TABLE, USE_GAMMA_TABLE>
{
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut R, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>()?)
    }
}

impl<const USE_DELTA_TABLE: bool, const USE_GAMMA_TABLE: bool>
    ParamDeltaCodec<USE_DELTA_TABLE, USE_GAMMA_TABLE>
{
    /// Encodes a value using the parameterized Delta codec.
    #[inline(always)]
    pub fn encode<E: Endianness, W: DeltaWriteParam<E>, R: DeltaReadParam<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::encode(writer, value, ())
    }
    /// Decodes a value using the parameterized Delta codec.
    #[inline(always)]
    pub fn decode<E: Endianness, W: DeltaWriteParam<E>, R: DeltaReadParam<E>>(
        reader: &mut R,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::decode(reader, ())
    }
}

/// ParamGammaCodec: uses a compile‑time flag for table usage in gamma coding.
pub struct ParamGammaCodec<const USE_TABLE: bool>;

impl<E: Endianness, W: GammaWriteParam<E>, R: GammaReadParam<E>, const USE_TABLE: bool>
    Codec<E, W, R> for ParamGammaCodec<USE_TABLE>
{
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_gamma_param::<USE_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut R, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_gamma_param::<USE_TABLE>()?)
    }
}

impl<const USE_TABLE: bool> ParamGammaCodec<USE_TABLE> {
    /// Encodes a value using the parameterized Gamma codec.
    #[inline(always)]
    pub fn encode<E: Endianness, W: GammaWriteParam<E>, R: GammaReadParam<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::encode(writer, value, ())
    }
    /// Decodes a value using the parameterized Gamma codec.
    #[inline(always)]
    pub fn decode<E: Endianness, W: GammaWriteParam<E>, R: GammaReadParam<E>>(
        reader: &mut R,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W, R>>::decode(reader, ())
    }
}

/// Type aliases to simplify inner type signatures.
pub type MyBitWrite<E> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;
pub type MyBitRead<E> = BufBitReader<E, MemWordReader<u64, Vec<u64>>>;

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
/// let intvec = BEIntVec::<GammaCodec>::from(input, 2).unwrap();
/// let value = intvec.get(3);
/// assert_eq!(value, Some(4));
/// assert_eq!(intvec.len(), 5);
/// ```
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct IntVec<E: Endianness, C: Codec<E, MyBitWrite<E>, MyBitRead<E>>> {
    pub data: Vec<u64>,
    pub samples: Vec<usize>,
    pub codec: PhantomData<C>,
    pub k: usize,
    pub len: usize,
    pub codec_param: C::Params,
    pub endian: PhantomData<E>,
}

/// Big-endian variant of `IntVec`.
pub type BEIntVec<C> = IntVec<BE, C>;

impl<C> BEIntVec<C>
where
    C: Codec<BE, MyBitWrite<BE>, MyBitRead<BE>>,
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
        let mut writer = MyBitWrite::<BE>::new(word_writer);
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
    /// let input = vec![1, 5, 3, 1991, 42];
    /// let intvec = BEIntVec::<GammaCodec>::from(input, 2).unwrap();
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
        let mut reader = MyBitRead::<BE>::new(MemWordReader::new(self.data.clone()));
        reader.set_bit_pos(start_bit as u64).ok()?;

        let mut value = 0;
        let start_index = sample_index * self.k;
        for _ in start_index..=index {
            value = C::decode(&mut reader, self.codec_param).ok()?;
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
    /// let intvec = BEIntVec::<GammaCodec>::from(input, 2).unwrap();
    /// let values = intvec.into_vec();
    /// assert_eq!(values, input);
    /// ```
    ///
    pub fn into_vec(self) -> Vec<u64> {
        let word_reader = MemWordReader::new(self.data);
        let mut reader = MyBitRead::<BE>::new(word_reader);
        let mut values = Vec::with_capacity(self.len);

        for _ in 0..self.len {
            values.push(C::decode(&mut reader, self.codec_param).unwrap());
        }

        values
    }

    pub fn iter(&self) -> IntVecIterBE<C> {
        IntVecIterBE {
            intvec: self,
            index: 0,
        }
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
impl<C: Codec<BE, MyBitWrite<BE>, MyBitRead<BE>, Params = ()>> BEIntVec<C> {
    /// Creates a new `BEIntVec` from input values, with no extra codec parameters.
    pub fn from(input: Vec<u64>, k: usize) -> Result<Self, Box<dyn Error>> {
        Self::from_with_param(input, k, ())
    }
}

/// Iterator for `BEIntVec`.
pub struct IntVecIterBE<'a, C>
where
    C: Codec<BE, MyBitWrite<BE>, MyBitRead<BE>>,
{
    intvec: &'a BEIntVec<C>,
    index: usize,
}

impl<'a, C> Iterator for IntVecIterBE<'a, C>
where
    C: Codec<BE, MyBitWrite<BE>, MyBitRead<BE>>,
    C::Params: Copy,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.intvec.len {
            return None;
        }

        let value = self.intvec.get(self.index);
        self.index += 1;
        value
    }
}

/// Little-endian variant of `IntVec`.
pub type LEIntVec<C> = IntVec<LE, C>;

impl<C> LEIntVec<C>
where
    C: Codec<LE, MyBitWrite<LE>, MyBitRead<LE>>,
    C::Params: Copy,
{
    /// Creates a new `LEIntVec` from a vector of unsigned 64-bit integers.
    ///
    /// Values are encoded with the specified codec parameter.
    ///
    /// # Arguments
    ///
    /// - `input`: The values to compress.
    /// - `k`: The sampling rate.
    /// - `codec_param`: Parameters for the codec.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::LEIntVec;
    /// use compressed_intvec::ZetaCodec;
    ///
    /// let input = vec![1, 5, 3, 1991, 42];
    ///
    /// let intvec = LEIntVec::<ZetaCodec>::from_with_param(input, 2, 3).unwrap();
    /// let value = intvec.get(3);
    /// assert_eq!(value, Some(1991));
    /// ```
    pub fn from_with_param(
        input: Vec<u64>,
        k: usize,
        codec_param: C::Params,
    ) -> Result<Self, Box<dyn Error>> {
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = MyBitWrite::<LE>::new(word_writer);
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
    /// use compressed_intvec::ZetaCodec;
    ///
    /// let input = vec![1, 5, 3, 1991, 42];
    /// let intvec = LEIntVec::<ZetaCodec>::from_with_param(input, 2, 3).unwrap();
    /// let value = intvec.get(3);
    /// assert_eq!(value, Some(1991));
    /// ```
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }

        let sample_index = index / self.k;
        let start_bit = self.samples[sample_index];
        let mut reader = MyBitRead::<LE>::new(MemWordReader::new(self.data.clone()));
        reader.set_bit_pos(start_bit as u64).ok()?;

        let mut value = 0;
        let start_index = sample_index * self.k;
        for _ in start_index..=index {
            value = C::decode(&mut reader, self.codec_param).ok()?;
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
    /// use compressed_intvec::ZetaCodec;
    ///
    /// let input = vec![43, 12, 5, 1991, 42];
    /// let intvec = LEIntVec::<ZetaCodec>::from_with_param(input, 2, 3).unwrap();
    ///
    /// let values = intvec.into_vec();
    /// assert_eq!(values, input);
    /// ```
    pub fn into_vec(self) -> Vec<u64> {
        let word_reader = MemWordReader::new(self.data);
        let mut reader = MyBitRead::<LE>::new(word_reader);
        let mut values = Vec::with_capacity(self.len);

        for _ in 0..self.len {
            values.push(C::decode(&mut reader, self.codec_param).unwrap());
        }

        values
    }

    pub fn iter(&self) -> IntVecIterLE<C> {
        IntVecIterLE {
            intvec: self,
            index: 0,
        }
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
impl<C: Codec<LE, MyBitWrite<LE>, MyBitRead<LE>, Params = ()>> LEIntVec<C> {
    /// Creates a new `LEIntVec` from input values, without extra codec parameters.
    pub fn from(input: Vec<u64>, k: usize) -> Result<Self, Box<dyn Error>> {
        Self::from_with_param(input, k, ())
    }
}

/// Iterator for `LEIntVec`.
pub struct IntVecIterLE<'a, C>
where
    C: Codec<LE, MyBitWrite<LE>, MyBitRead<LE>>,
{
    intvec: &'a LEIntVec<C>,
    index: usize,
}

impl<'a, C> Iterator for IntVecIterLE<'a, C>
where
    C: Codec<LE, MyBitWrite<LE>, MyBitRead<LE>>,
    C::Params: Copy,
{
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.intvec.len {
            return None;
        }

        let value = self.intvec.get(self.index);
        self.index += 1;
        value
    }
}
