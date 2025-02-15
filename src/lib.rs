use std::{error::Error, marker::PhantomData};

use dsi_bitstream::prelude::*;
use mem_dbg::{MemDbg, MemSize};

/// Type aliases for better readability, generic over the endianness type `E`.
type MyBitWriter<E: Endianness> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;
type MyBitReader<'a, E> = BufBitReader<E, MemWordReader<u64, &'a Vec<u64>>>;

/// Trait for encoding and decoding values using a variable-length code.
/// The trait is generic over an endianness type `E`.
pub trait Codec<E: Endianness, W> {
    /// The type of parameters for encoding/decoding.
    type Params;

    /// Encodes `value` into the stream represented by `writer`, using the provided parameters.
    fn encode(writer: &mut W, value: u64, params: Self::Params) -> Result<usize, Box<dyn Error>>;

    /// Decodes a value from the stream represented by `reader`, using the provided parameters.
    fn decode(reader: &mut W, params: Self::Params) -> Result<u64, Box<dyn Error>>;
}

/// GammaCodec: no extra runtime parameter.
pub struct GammaCodec;

impl<E: Endianness, W: GammaWrite<E> + GammaRead<E>> Codec<E, W> for GammaCodec {
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_gamma(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut W, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_gamma()?)
    }
}

impl GammaCodec {
    #[inline(always)]
    pub fn encode<W: GammaWrite<E> + GammaRead<E>, E: Endianness>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, ())
    }
    #[inline(always)]
    pub fn decode<W: GammaWrite<E> + GammaRead<E>, E: Endianness>(
        reader: &mut W,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W>>::decode(reader, ())
    }
}

/// DeltaCodec: no extra runtime parameter.
pub struct DeltaCodec;

impl<E: Endianness, W: DeltaWrite<E> + DeltaRead<E>> Codec<E, W> for DeltaCodec {
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_delta(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut W, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_delta()?)
    }
}

impl DeltaCodec {
    #[inline(always)]
    pub fn encode<E: Endianness, W: DeltaWrite<E> + DeltaRead<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, ())
    }
    #[inline(always)]
    pub fn decode<E: Endianness, W: DeltaWrite<E> + DeltaRead<E>>(
        reader: &mut W,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W>>::decode(reader, ())
    }
}

/// Exp‑Golomb Codec: requires a runtime parameter (e.g. k).
pub struct ExpGolombCodec;

impl<E: Endianness, W: ExpGolombWrite<E> + ExpGolombRead<E>> Codec<E, W> for ExpGolombCodec {
    type Params = usize;

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, k: usize) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_exp_golomb(value, k)?)
    }

    #[inline(always)]
    fn decode(reader: &mut W, k: usize) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_exp_golomb(k)?)
    }
}

impl ExpGolombCodec {
    #[inline(always)]
    pub fn encode<E: Endianness, W: ExpGolombWrite<E> + ExpGolombRead<E>>(
        writer: &mut W,
        value: u64,
        k: usize,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, k)
    }
    #[inline(always)]
    pub fn decode<E: Endianness, W: ExpGolombWrite<E> + ExpGolombRead<E>>(
        reader: &mut W,
        k: usize,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W>>::decode(reader, k)
    }
}

/// ZetaCodec: uses runtime parameter (k) with non‑parametric ζ functions.
pub struct ZetaCodec;

impl<E: Endianness, W: ZetaWrite<E> + ZetaRead<E>> Codec<E, W> for ZetaCodec {
    type Params = u64;

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, k: u64) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_zeta(value, k)?)
    }

    #[inline(always)]
    fn decode(reader: &mut W, k: u64) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_zeta(k)?)
    }
}

impl ZetaCodec {
    #[inline(always)]
    pub fn encode<E: Endianness, W: ZetaWrite<E> + ZetaRead<E>>(
        writer: &mut W,
        value: u64,
        k: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, k)
    }

    #[inline(always)]
    pub fn decode<E: Endianness, W: ZetaWrite<E> + ZetaRead<E>>(
        reader: &mut W,
        k: u64,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W>>::decode(reader, k)
    }
}

/// RiceCodec: uses the Rice functions with a runtime parameter (log2_b).
pub struct RiceCodec;

impl<E: Endianness, W: RiceWrite<E> + RiceRead<E>> Codec<E, W> for RiceCodec {
    type Params = usize;

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, log2_b: usize) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_rice(value, log2_b)?)
    }

    #[inline(always)]
    fn decode(reader: &mut W, log2_b: usize) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_rice(log2_b)?)
    }
}

impl RiceCodec {
    #[inline(always)]
    pub fn encode<E: Endianness, W: RiceWrite<E> + RiceRead<E>>(
        writer: &mut W,
        value: u64,
        log2_b: usize,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, log2_b)
    }

    #[inline(always)]
    pub fn decode<E: Endianness, W: RiceWrite<E> + RiceRead<E>>(
        reader: &mut W,
        log2_b: usize,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W>>::decode(reader, log2_b)
    }
}

/// ParamZetaCodec: uses a compile‑time flag for ζ functions.
pub struct ParamZetaCodec<const USE_TABLE: bool>;

impl<E: Endianness, W: ZetaWriteParam<E> + ZetaReadParam<E>, const USE_TABLE: bool> Codec<E, W>
    for ParamZetaCodec<USE_TABLE>
{
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_zeta3_param::<USE_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut W, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_zeta3_param::<USE_TABLE>()?)
    }
}

impl<const USE_TABLE: bool> ParamZetaCodec<USE_TABLE> {
    #[inline(always)]
    pub fn encode<E: Endianness, W: ZetaWriteParam<E> + ZetaReadParam<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, ())
    }
    #[inline(always)]
    pub fn decode<E: Endianness, W: ZetaWriteParam<E> + ZetaReadParam<E>>(
        reader: &mut W,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W>>::decode(reader, ())
    }
}

/// ParamDeltaCodec: uses compile‑time booleans for table usage.
pub struct ParamDeltaCodec<const USE_DELTA_TABLE: bool, const USE_GAMMA_TABLE: bool>;

impl<
        E: Endianness,
        const USE_DELTA_TABLE: bool,
        const USE_GAMMA_TABLE: bool,
        W: DeltaWriteParam<E> + DeltaReadParam<E>,
    > Codec<E, W> for ParamDeltaCodec<USE_DELTA_TABLE, USE_GAMMA_TABLE>
{
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut W, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>()?)
    }
}

impl<const USE_DELTA_TABLE: bool, const USE_GAMMA_TABLE: bool>
    ParamDeltaCodec<USE_DELTA_TABLE, USE_GAMMA_TABLE>
{
    #[inline(always)]
    pub fn encode<E: Endianness, W: DeltaWriteParam<E> + DeltaReadParam<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, ())
    }
    #[inline(always)]
    pub fn decode<E: Endianness, W: DeltaWriteParam<E> + DeltaReadParam<E>>(
        reader: &mut W,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W>>::decode(reader, ())
    }
}

/// ParamGammaCodec: uses a compile‑time flag for table usage in gamma coding.
pub struct ParamGammaCodec<const USE_TABLE: bool>;

impl<E: Endianness, W: GammaWriteParam<E> + GammaReadParam<E>, const USE_TABLE: bool> Codec<E, W>
    for ParamGammaCodec<USE_TABLE>
{
    type Params = ();

    #[inline(always)]
    fn encode(writer: &mut W, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_gamma_param::<USE_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut W, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_gamma_param::<USE_TABLE>()?)
    }
}

impl<const USE_TABLE: bool> ParamGammaCodec<USE_TABLE> {
    #[inline(always)]
    pub fn encode<E: Endianness, W: GammaWriteParam<E> + GammaReadParam<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        <Self as Codec<E, W>>::encode(writer, value, ())
    }
    #[inline(always)]
    pub fn decode<E: Endianness, W: GammaWriteParam<E> + GammaReadParam<E>>(
        reader: &mut W,
    ) -> Result<u64, Box<dyn Error>> {
        <Self as Codec<E, W>>::decode(reader, ())
    }
}

/// =========================
/// Compressed Integer Vector (IntVec)
/// =========================

#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct IntVec<E: Endianness, C: Codec<E, C>> {
    /// Compressed data (64-bit words)
    data: Vec<u64>,
    /// Sampled bit positions for quick random access
    samples: Vec<usize>,
    /// Marker for the codec used
    codec: PhantomData<C>,
    /// Sampling interval (used to create the samples)
    k: usize,
    /// Number of elements in the original vector
    len: usize,
    /// Codec parameter used during encoding/decoding
    codec_param: C::Params,
    /// Marker for the chosen endianness
    endian: PhantomData<E>,
}

impl<E: Endianness, C: Codec<E, C>> IntVec<E, C>
where
    C::Params: Copy,
{
    pub fn from_with_param(
        input: Vec<u64>,
        k: usize,
        codec_param: C::Params,
    ) -> Result<Self, Box<dyn Error>> {
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = MyBitWriter::<E>::new(word_writer);
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

    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }
        let sample_index = index / self.k;
        let start_bit = self.samples[sample_index];
        let word_reader = MemWordReader::new(&self.data);
        let mut reader = MyBitReader::<E>::new(word_reader);
        reader.set_bit_pos(start_bit as u64).ok()?;
        let mut value = 0;
        let start_index = sample_index * self.k;
        for _ in start_index..=index {
            value = C::decode(&mut reader, self.codec_param).ok()?;
        }
        Some(value)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn fast_iter(&self) -> IntVecIter<E, C> {
        IntVecIter::new(self)
    }
}

/// For codecs with no extra runtime parameter (`C::Params = ()`), provide a convenience constructor.
impl<E: Endianness, C: Codec<E, Params = ()>> IntVec<E, C> {
    pub fn from(input: Vec<u64>, k: usize) -> Result<Self, Box<dyn Error>> {
        Self::from_with_param(input, k, ())
    }
}

/// Iterator over an `IntVec`.
pub struct IntVecIter<'a, E: Endianness, C: Codec<E, C>> {
    reader: MyBitReader<'a, E>,
    remaining: usize,
    codec: PhantomData<C>,
    codec_param: C::Params,
}

impl<'a, E: Endianness, C: Codec<E, C>> IntVecIter<'a, E, C>
where
    C::Params: Copy,
{
    pub fn new(vec: &'a IntVec<E, C>) -> Self {
        let word_reader = MemWordReader::new(&vec.data);
        let reader = MyBitReader::<E>::new(word_reader);
        IntVecIter {
            reader,
            remaining: vec.len,
            codec: PhantomData,
            codec_param: vec.codec_param,
        }
    }
}

impl<'a, E: Endianness, C: Codec<E, C>> Iterator for IntVecIter<'a, E, C>
where
    C::Params: Copy,
{
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            let value = C::decode(&mut self.reader, self.codec_param).ok()?;
            self.remaining -= 1;
            Some(value)
        }
    }
}
