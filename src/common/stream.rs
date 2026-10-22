//! Writing and reading a stream of codewords over `u64` words, in little-endian bit order.
//!
//! [`Encoder`] and [`Decoder`] use dsi-bitstream's codes: every code with a function-pointer
//! implementation there is dispatched once, when they are created; the others match on the code
//! for every word.
//!
//! Every declared element in parts or serialized data is first read by [`CheckedDecoder`],
//! this crate's own decoder, which rejects the codewords on which dsi-bitstream's decoders
//! would overflow. Only then does [`Decoder`] read those elements. Trailing storage may remain
//! unread.

use super::{Codec, ValidCodec};
use dsi_bitstream::dispatch::{ConstCode, code_consts};
use dsi_bitstream::prelude::{
    BitRead, BitSeek, BufBitReader, BufBitWriter, Codes, FuncCodeReader, FuncCodeWriter, LE,
    MemWordReader, MemWordWriterVec, MinimalBinaryRead, StaticCodeRead, StaticCodeWrite, WordError,
};

/// The stream's bit reader, which returns an error at the end of the stored words.
type BitReader<'a> = BufBitReader<LE, MemWordReader<ReadWord, &'a [ReadWord]>>;

/// The word that the bit reader loads: `u32` on little-endian targets, where dsi-bitstream's
/// buffered reader decodes faster from `u32` words (into a `u64` buffer) than from `u64` words.
#[cfg(target_endian = "little")]
type ReadWord = u32;
#[cfg(target_endian = "big")]
type ReadWord = u64;

/// Returns the words of a stream as the bit reader loads them.
#[cfg(target_endian = "little")]
fn read_words(words: &[u64]) -> &[u32] {
    // SAFETY: `u32` has no invalid values and an alignment that divides that of `u64`, and the
    // `2 * words.len()` words cover exactly the bytes of `words`, borrowed for as long. On a
    // little-endian target the two halves of a `u64` word are, in memory order, its low and its
    // high 32 bits, so the bit order of the stream is unchanged.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u32>(), 2 * words.len()) }
}

/// Returns the words of a stream as the bit reader loads them.
#[cfg(target_endian = "big")]
fn read_words(words: &[u64]) -> &[u64] {
    words
}

/// The bit writer of a stream.
type BitWriter = BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>;

/// Reads the codewords of one code.
enum CodeReader<'a> {
    // The VByte codes are dispatched statically: their decoders are short enough that the call
    // through a function pointer would dominate them.
    VByteLe,
    VByteBe,
    Func(FuncCodeReader<LE, BitReader<'a>>),
    Match(Codes),
}

/// Writes the codewords of one code.
enum CodeWriter {
    Func(FuncCodeWriter<LE, BitWriter>),
    Match(Codes),
}

/// Decodes the words of a validated stream, starting at bit 0, with dsi-bitstream's decoders.
///
/// Reads only codewords emitted by [`Encoder`] or accepted by [`CheckedDecoder`].
/// dsi-bitstream's decoders trust their input and panic on some corrupt codewords in builds
/// with debug assertions or overflow checks.
pub(crate) struct Decoder<'a> {
    bits: BitReader<'a>,
    code: CodeReader<'a>,
}

impl<'a> Decoder<'a> {
    /// Creates a decoder of `words`, positioned at bit 0.
    #[inline]
    pub(crate) fn new(words: &'a [u64], codec: ValidCodec) -> Self {
        let codes = codec.to_codes();
        let code = match codes {
            Codes::VByteLe => CodeReader::VByteLe,
            Codes::VByteBe => CodeReader::VByteBe,
            codes => FuncCodeReader::new(codes).map_or(CodeReader::Match(codes), CodeReader::Func),
        };
        Self {
            bits: BufBitReader::new(MemWordReader::new(read_words(words))),
            code,
        }
    }

    /// Decodes the next word of a validated stream.
    ///
    /// # Panics
    ///
    /// If the stream ends first, which validation rules out.
    #[inline]
    pub(crate) fn read(&mut self) -> u64 {
        match &self.code {
            CodeReader::VByteLe => ConstCode::<{ code_consts::VBYTE_LE }>.read(&mut self.bits),
            CodeReader::VByteBe => ConstCode::<{ code_consts::VBYTE_BE }>.read(&mut self.bits),
            CodeReader::Func(func) => func.read(&mut self.bits),
            CodeReader::Match(codes) => codes.read(&mut self.bits),
        }
        .expect("a validated stream holds a codeword for every element")
    }

    /// Moves to a bit position of a validated stream.
    ///
    /// # Panics
    ///
    /// If the position lies past the end of the words, which validation rules out.
    #[inline]
    pub(crate) fn seek(&mut self, bit: u64) {
        self.bits
            .set_bit_pos(bit)
            .expect("a validated position lies within the words");
    }

    /// Returns the position of the next bit to decode.
    #[inline]
    pub(crate) fn bit_pos(&mut self) -> u64 {
        self.bits
            .bit_pos()
            .expect("a memory word reader always knows its position")
    }
}

/// A codeword that [`CheckedDecoder`] cannot decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodewordError {
    /// The words end before the codeword does.
    Truncated,
    /// The codeword encodes no word: its value does not fit in a `u64` (for the codes that
    /// encode `n + 1`, `n + 1` does not).
    Invalid,
}

impl From<WordError> for CodewordError {
    /// The memory word reader fails only at the end of the words.
    fn from(_: WordError) -> Self {
        Self::Truncated
    }
}

/// Decodes external codewords with checked lengths and arithmetic.
///
/// Follows each code's bit-level steps using primitive bit reads, returning
/// [`CodewordError::Invalid`] for operands outside the representable range. Every accepted
/// codeword matches the codeword that [`Encoder`] writes for its word. dsi-bitstream's
/// decoders, with tables enabled or disabled, recover the same word and stop at the same bit.
/// These properties establish the preconditions of [`Decoder`].
pub(crate) struct CheckedDecoder<'a> {
    bits: BitReader<'a>,
    codec: ValidCodec,
}

impl<'a> CheckedDecoder<'a> {
    /// Creates a decoder of `words`, positioned at bit 0.
    pub(crate) fn new(words: &'a [u64], codec: ValidCodec) -> Self {
        Self {
            bits: BufBitReader::new(MemWordReader::new(read_words(words))),
            codec,
        }
    }

    /// Decodes the next word, or returns why the codeword at the position cannot be decoded.
    #[inline]
    pub(crate) fn try_read(&mut self) -> Result<u64, CodewordError> {
        let bits = &mut self.bits;
        // The parameters are in range (`ValidCodec`), which the divisions and shifts below need.
        match self.codec.get() {
            Codec::Unary => Ok(bits.read_unary()?),
            Codec::Gamma => read_gamma(bits),
            Codec::Delta => {
                let len = read_gamma(bits)?;
                read_tail(bits, len)
            }
            Codec::Omega => read_omega(bits),
            Codec::VByteLe => read_vbyte_le(bits),
            Codec::VByteBe => read_vbyte_be(bits),
            Codec::Rice { log2_b } => read_rice(bits, log2_b),
            Codec::Golomb { b } => {
                let quotient = bits.read_unary()?;
                let remainder = bits.read_minimal_binary(b)?;
                quotient
                    .checked_mul(b)
                    .and_then(|high| high.checked_add(remainder))
                    .ok_or(CodewordError::Invalid)
            }
            Codec::ExpGolomb { k } => {
                let high = read_gamma(bits)?;
                read_low_bits(bits, high, k)
            }
            Codec::Zeta { k } => {
                // The codeword of `n` holds `h = floor(log2(n + 1) / k)` in unary, then
                // `n + 1 - 2^(h * k)` in minimal binary below `2^((h + 1) * k) - 2^(h * k)`.
                let h = bits.read_unary()?;
                if h > u64::from(63 / k) {
                    return Err(CodewordError::Invalid);
                }
                let low = 1_u64 << (h * u64::from(k));
                // With `(h + 1) * k >= 64` the bound wraps to `2^64 - low`, as in dsi-bitstream,
                // which is the number of words from `low - 1` to `u64::MAX - 1`.
                let rest = bits.read_minimal_binary((low << k).wrapping_sub(low))?;
                Ok(low + rest - 1)
            }
            Codec::Pi { k } => {
                let len = read_rice(bits, k)?;
                read_tail(bits, len)
            }
        }
    }

    /// Moves to bit `bit`, or returns an error if it lies past the end of the words.
    pub(crate) fn try_seek(&mut self, bit: u64) -> Result<(), WordError> {
        self.bits.set_bit_pos(bit)
    }

    /// Returns the position of the next bit to decode.
    #[inline]
    pub(crate) fn bit_pos(&mut self) -> u64 {
        self.bits
            .bit_pos()
            .expect("a memory word reader always knows its position")
    }
}

/// Reads the `len` bits below the implicit leading one of `n + 1`, and returns `n`: the tail of
/// the gamma, delta and pi codes.
#[inline]
fn read_tail(bits: &mut BitReader<'_>, len: u64) -> Result<u64, CodewordError> {
    if len >= 64 {
        return Err(CodewordError::Invalid);
    }
    // Lossless: `len < 64`.
    Ok(bits.read_bits(len as usize)? + (1 << len) - 1)
}

#[inline]
fn read_gamma(bits: &mut BitReader<'_>) -> Result<u64, CodewordError> {
    let len = bits.read_unary()?;
    read_tail(bits, len)
}

/// Reads `k` more bits below `high`: the low part of the Rice and exponential Golomb codes.
#[inline]
fn read_low_bits(bits: &mut BitReader<'_>, high: u64, k: u32) -> Result<u64, CodewordError> {
    if high > u64::MAX >> k {
        return Err(CodewordError::Invalid);
    }
    // Lossless: `k < 64`.
    Ok((high << k) | bits.read_bits(k as usize)?)
}

#[inline]
fn read_rice(bits: &mut BitReader<'_>, log2_b: u32) -> Result<u64, CodewordError> {
    let quotient = bits.read_unary()?;
    read_low_bits(bits, quotient, log2_b)
}

/// The omega code of `n` writes the binary representation of `n + 1` after that of its length
/// minus one, recursively down to 1, each starting with a one bit, and ends with a zero bit.
#[inline]
fn read_omega(bits: &mut BitReader<'_>) -> Result<u64, CodewordError> {
    let mut n = 1_u64;
    while bits.read_bits(1)? == 1 {
        // The group below the one bit holds `n` more bits.
        if n >= 64 {
            return Err(CodewordError::Invalid);
        }
        // Lossless: `n < 64`.
        n = bits.read_bits(n as usize)? | (1 << n);
    }
    Ok(n - 1)
}

/// The VByte code, least significant group first: each byte holds seven bits of the word and a
/// continuation bit, and each continuation adds `2^(7 * bytes read)`, so every word has one
/// codeword. The value is kept in 128 bits, so that a codeword of up to ten bytes whose value
/// exceeds `u64::MAX` is detected; a codeword of eleven bytes or more always exceeds it.
#[inline]
fn read_vbyte_le(bits: &mut BitReader<'_>) -> Result<u64, CodewordError> {
    let mut value = 0_u128;
    for byte_index in 0..10 {
        let shift = 7 * byte_index;
        let byte = bits.read_bits(8)?;
        value += u128::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return u64::try_from(value).map_err(|_| CodewordError::Invalid);
        }
        value += 1 << (shift + 7);
    }
    Err(CodewordError::Invalid)
}

/// The VByte code, most significant group first; see [`read_vbyte_le`].
#[inline]
fn read_vbyte_be(bits: &mut BitReader<'_>) -> Result<u64, CodewordError> {
    let mut value = 0_u128;
    for _ in 0..10 {
        let byte = bits.read_bits(8)?;
        value = (value << 7) | u128::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            return u64::try_from(value).map_err(|_| CodewordError::Invalid);
        }
        value += 1;
    }
    Err(CodewordError::Invalid)
}

/// The word `u64::MAX` cannot be encoded with the code of an [`Encoder`].
pub(crate) struct Unencodable;

/// Encodes words into a stream.
pub(crate) struct Encoder {
    bits: BitWriter,
    code: CodeWriter,
    encodes_max: bool,
    bit_len: u64,
}

impl Encoder {
    /// Creates an encoder of an empty stream.
    pub(crate) fn new(codec: ValidCodec) -> Self {
        let codes = codec.to_codes();
        let code = match FuncCodeWriter::new(codes) {
            Ok(func) => CodeWriter::Func(func),
            Err(_) => CodeWriter::Match(codes),
        };
        Self {
            bits: BufBitWriter::new(MemWordWriterVec::new(Vec::new())),
            code,
            encodes_max: codec.get().encodes_max(),
            bit_len: 0,
        }
    }

    /// Appends the codeword of `word`, or returns an error if the code cannot encode it.
    #[inline]
    pub(crate) fn write(&mut self, word: u64) -> Result<(), Unencodable> {
        if word == u64::MAX && !self.encodes_max {
            return Err(Unencodable);
        }
        let Ok(bits) = match &self.code {
            CodeWriter::Func(func) => func.write(&mut self.bits, word),
            CodeWriter::Match(codes) => codes.write(&mut self.bits, word),
        };
        // Lossless: usize is at most 64 bits wide on every supported target.
        self.bit_len += bits as u64;
        Ok(())
    }

    /// Returns the number of bits written so far, which is the position of the next codeword.
    pub(crate) fn bit_len(&self) -> u64 {
        self.bit_len
    }

    /// Returns the words of the stream. The bits past the last codeword are zero.
    pub(crate) fn finish(self) -> Vec<u64> {
        let Ok(words) = self.bits.into_inner();
        let mut words = words.into_inner();
        words.shrink_to_fit();
        words
    }
}
