use std::error::Error;

use dsi_bitstream::prelude::*;
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
        R: GammaRead<E>,
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
        Ok(writer.write_delta(value)?)
    }
    /// Decodes a value using delta coding.
    #[inline(always)]
    pub fn decode<E: Endianness, R>(reader: &mut R) -> Result<u64, Box<dyn Error>>
    where
        R: DeltaRead<E>,
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
        Ok(writer.write_exp_golomb(value, k)?)
    }

    /// Decodes a value using Exp‑Golomb coding with the specified parameter `k`.
    #[inline(always)]
    pub fn decode<E: Endianness, R>(reader: &mut R, k: usize) -> Result<u64, Box<dyn Error>>
    where
        R: ExpGolombRead<E>,
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
        R: ZetaRead<E>,
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
        Ok(writer.write_rice(value, log2_b)?)
    }

    /// Decodes a value using Rice coding with the specified `log2_b` parameter.
    #[inline(always)]
    pub fn decode<E: Endianness, R>(reader: &mut R, log2_b: usize) -> Result<u64, Box<dyn Error>>
    where
        R: RiceRead<E>,
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
    pub fn encode<E: Endianness, W: ZetaWriteParam<E>>(
        writer: &mut W,
        value: u64,
    ) -> Result<usize, Box<dyn Error>> {
        Ok(writer.write_zeta3_param::<USE_TABLE>(value)?)
    }
    /// Decodes a value using the parameterized Zeta codec.
    #[inline(always)]
    pub fn decode<E: Endianness, R>(reader: &mut R) -> Result<u64, Box<dyn Error>>
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
        Ok(writer.write_delta_param::<USE_DELTA_TABLE, USE_GAMMA_TABLE>(value)?)
    }
    /// Decodes a value using the parameterized Delta codec.
    #[inline(always)]
    pub fn decode<E: Endianness, R>(reader: &mut R) -> Result<u64, Box<dyn Error>>
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
        Ok(writer.write_gamma_param::<USE_TABLE>(value)?)
    }
    /// Decodes a value using the parameterized Gamma codec.
    #[inline(always)]
    pub fn decode<E: Endianness, R>(reader: &mut R) -> Result<u64, Box<dyn Error>>
    where
        R: GammaReadParam<E>,
    {
        Ok(reader.read_gamma_param::<USE_TABLE>()?)
    }
}
