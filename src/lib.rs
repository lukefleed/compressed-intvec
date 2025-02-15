use std::{error::Error, marker::PhantomData};

use dsi_bitstream::prelude::*;
use mem_dbg::{MemDbg, MemSize};

// Type aliases for better readability.
type MyBitWriter = BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>;
type MyBitReader<'a> = BufBitReader<BE, MemWordReader<u64, &'a Vec<u64>>>;

/// Trait for encoding and decoding values using a variable-length code.
pub trait Codec {
    /// The type of parameters for encoding/decoding.
    type Params;

    /// Encodes `value` into the stream represented by `writer`, using the provided parameters.
    fn encode(
        writer: &mut MyBitWriter,
        value: u64,
        params: Self::Params,
    ) -> Result<usize, Box<dyn Error>>;

    /// Decodes a value from the stream represented by `reader`, using the provided parameters.
    fn decode(reader: &mut MyBitReader, params: Self::Params) -> Result<u64, Box<dyn Error>>;
}

/// Implementation of the Gamma Codec (without compile‑time parameters).
pub struct GammaCodec;

impl Codec for GammaCodec {
    type Params = (); // No parameter needed.

    #[inline(always)]
    fn encode(writer: &mut MyBitWriter, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_gamma(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut MyBitReader, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_gamma()?)
    }
}

impl GammaCodec {
    #[inline(always)]
    pub fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>> {
        <Self as Codec>::encode(writer, value, ())
    }
    #[inline(always)]
    pub fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>> {
        <Self as Codec>::decode(reader, ())
    }
}

/// Implementation of the Delta Codec (without compile‑time parameters).
pub struct DeltaCodec;

impl Codec for DeltaCodec {
    type Params = (); // No parameter needed.

    #[inline(always)]
    fn encode(writer: &mut MyBitWriter, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_delta(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut MyBitReader, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_delta()?)
    }
}

impl DeltaCodec {
    #[inline(always)]
    pub fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>> {
        <Self as Codec>::encode(writer, value, ())
    }
    #[inline(always)]
    pub fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>> {
        <Self as Codec>::decode(reader, ())
    }
}

/// Implementation of the Exp‑Golomb Codec.
pub struct ExpGolombCodec;

impl Codec for ExpGolombCodec {
    type Params = usize; // This codec requires a runtime parameter (typically the k value).

    #[inline(always)]
    fn encode(writer: &mut MyBitWriter, value: u64, k: usize) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_exp_golomb(value, k)?)
    }

    #[inline(always)]
    fn decode(reader: &mut MyBitReader, k: usize) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_exp_golomb(k)?)
    }
}

impl ExpGolombCodec {
    #[inline(always)]
    pub fn encode(writer: &mut MyBitWriter, value: u64, k: usize) -> Result<usize, Box<dyn Error>> {
        <Self as Codec>::encode(writer, value, k)
    }
    #[inline(always)]
    pub fn decode(reader: &mut MyBitReader, k: usize) -> Result<u64, Box<dyn Error>> {
        <Self as Codec>::decode(reader, k)
    }
}

/// Implementation of a delta codec with compile‑time parameters.
/// This codec uses the more general delta functions:
/// `write_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>(n)` and
/// `read_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>()`.
pub struct ParamDeltaCodec<const USE_DELTA_TABLE: bool, const USE_GAMMA_TABLE: bool>;

impl<const USE_DELTA_TABLE: bool, const USE_GAMMA_TABLE: bool> Codec
    for ParamDeltaCodec<USE_DELTA_TABLE, USE_GAMMA_TABLE>
{
    type Params = (); // No runtime parameter.

    #[inline(always)]
    fn encode(writer: &mut MyBitWriter, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut MyBitReader, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>()?)
    }
}

impl<const USE_DELTA_TABLE: bool, const USE_GAMMA_TABLE: bool>
    ParamDeltaCodec<USE_DELTA_TABLE, USE_GAMMA_TABLE>
{
    #[inline(always)]
    pub fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>> {
        <Self as Codec>::encode(writer, value, ())
    }
    #[inline(always)]
    pub fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>> {
        <Self as Codec>::decode(reader, ())
    }
}

/// Implementation of a gamma codec with a compile‑time parameter.
/// This codec uses the parametric gamma functions:
/// `write_gamma_param::<USE_TABLE>(n)` and `read_gamma_param::<USE_TABLE>()`.
pub struct ParamGammaCodec<const USE_TABLE: bool>;

impl<const USE_TABLE: bool> Codec for ParamGammaCodec<USE_TABLE> {
    type Params = (); // No runtime parameter.

    #[inline(always)]
    fn encode(writer: &mut MyBitWriter, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_gamma_param::<USE_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut MyBitReader, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_gamma_param::<USE_TABLE>()?)
    }
}

impl<const USE_TABLE: bool> ParamGammaCodec<USE_TABLE> {
    #[inline(always)]
    pub fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>> {
        <Self as Codec>::encode(writer, value, ())
    }
    #[inline(always)]
    pub fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>> {
        <Self as Codec>::decode(reader, ())
    }
}

/// Implementation of the Zeta Codec using runtime parameter.
/// This codec uses the non‑parametric ζ functions:
///   - For writing: `write_zeta(n, k)`
///   - For reading: `read_zeta_param(k)`
pub struct ZetaCodec;

impl Codec for ZetaCodec {
    // Here the runtime parameter (of type u64) represents the "k" value.
    type Params = u64;

    #[inline(always)]
    fn encode(writer: &mut MyBitWriter, value: u64, k: u64) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_zeta(value, k)?)
    }

    #[inline(always)]
    fn decode(reader: &mut MyBitReader, k: u64) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_zeta_param(k)?)
    }
}

impl ZetaCodec {
    #[inline(always)]
    pub fn encode(writer: &mut MyBitWriter, value: u64, k: u64) -> Result<usize, Box<dyn Error>> {
        <Self as Codec>::encode(writer, value, k)
    }
    #[inline(always)]
    pub fn decode(reader: &mut MyBitReader, k: u64) -> Result<u64, Box<dyn Error>> {
        <Self as Codec>::decode(reader, k)
    }
}

/// Implementation of a Zeta codec with a compile‑time parameter.
/// This codec uses the parametric ζ functions:
///   - For writing: `write_zeta3_param::<USE_TABLE>(n)`
///   - For reading: `read_zeta3_param::<USE_TABLE>()`
pub struct ParamZetaCodec<const USE_TABLE: bool>;

impl<const USE_TABLE: bool> Codec for ParamZetaCodec<USE_TABLE> {
    type Params = (); // No runtime parameter.

    #[inline(always)]
    fn encode(writer: &mut MyBitWriter, value: u64, _params: ()) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_zeta3_param::<USE_TABLE>(value)?)
    }

    #[inline(always)]
    fn decode(reader: &mut MyBitReader, _params: ()) -> Result<u64, Box<dyn Error>> {
        Ok(reader.read_zeta3_param::<USE_TABLE>()?)
    }
}

impl<const USE_TABLE: bool> ParamZetaCodec<USE_TABLE> {
    #[inline(always)]
    pub fn encode(writer: &mut MyBitWriter, value: u64) -> Result<usize, Box<dyn Error>> {
        <Self as Codec>::encode(writer, value, ())
    }
    #[inline(always)]
    pub fn decode(reader: &mut MyBitReader) -> Result<u64, Box<dyn Error>> {
        <Self as Codec>::decode(reader, ())
    }
}

/// Implementation of the Rice Codec.
/// This codec uses the Rice code functions:
///   - For writing: `write_rice(n, log2_b)`
///   - For reading: `read_rice(log2_b)`
pub struct RiceCodec;

impl Codec for RiceCodec {
    // The Rice codec requires a runtime parameter (log2_b), so we set Params to usize.
    type Params = usize;

    #[inline(always)]
    fn encode(
        writer: &mut MyBitWriter,
        value: u64,
        log2_b: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        // Calls the provided method from the RiceWrite trait.
        Ok(writer.write_rice(value, log2_b)?)
    }

    #[inline(always)]
    fn decode(reader: &mut MyBitReader, log2_b: usize) -> Result<u64, Box<dyn std::error::Error>> {
        // Calls the provided method from the RiceRead trait.
        Ok(reader.read_rice(log2_b)?)
    }
}

impl RiceCodec {
    /// Convenience method to encode a value using Rice coding with the specified log2_b.
    #[inline(always)]
    pub fn encode(
        writer: &mut MyBitWriter,
        value: u64,
        log2_b: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        <Self as Codec>::encode(writer, value, log2_b)
    }

    /// Convenience method to decode a Rice-encoded value using the specified log2_b.
    #[inline(always)]
    pub fn decode(
        reader: &mut MyBitReader,
        log2_b: usize,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        <Self as Codec>::decode(reader, log2_b)
    }
}

#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct IntVec<C: Codec> {
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
}

impl<C: Codec> IntVec<C>
where
    C::Params: Copy,
{
    /// Constructs a new `IntVec` from an input vector, a sampling interval `k`
    /// and a codec parameter.
    ///
    /// # Errors
    ///
    /// Returns an error if there is a problem writing or flushing the stream.
    pub fn from_with_param(
        input: Vec<u64>,
        k: usize,
        codec_param: C::Params,
    ) -> Result<Self, Box<dyn Error>> {
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = MyBitWriter::new(word_writer);
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
        })
    }

    /// Returns the value at the `index` position in the original vector.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len {
            return None;
        }
        let sample_index = index / self.k;
        let start_bit = self.samples[sample_index];
        let word_reader = MemWordReader::new(&self.data);
        let mut reader = MyBitReader::new(word_reader);
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

    pub fn fast_iter(&self) -> IntVecIter<C> {
        IntVecIter::new(self)
    }
}

/// For codecs with no extra runtime parameter (i.e. `C::Params = ()`),
/// provide a convenience constructor.
impl<C: Codec<Params = ()>> IntVec<C> {
    pub fn from(input: Vec<u64>, k: usize) -> Result<Self, Box<dyn Error>> {
        Self::from_with_param(input, k, ())
    }
}

/// Iterator over an `IntVec`.
pub struct IntVecIter<'a, C: Codec> {
    reader: MyBitReader<'a>,
    remaining: usize,
    codec: PhantomData<C>,
    codec_param: C::Params,
}

impl<'a, C: Codec> IntVecIter<'a, C>
where
    C::Params: Copy,
{
    pub fn new(vec: &'a IntVec<C>) -> Self {
        let word_reader = MemWordReader::new(&vec.data);
        let reader = MyBitReader::new(word_reader);
        IntVecIter {
            reader,
            remaining: vec.len,
            codec: PhantomData,
            codec_param: vec.codec_param,
        }
    }
}

impl<'a, C: Codec> Iterator for IntVecIter<'a, C>
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
