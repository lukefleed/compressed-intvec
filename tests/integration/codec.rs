//! The codes that `VarVec` and `SeqVec` share: the parameters they accept, the words they can
//! encode, the automatic choice, and what the decoders accept from a corrupt stream.

use crate::{Rng, nz};
use compressed_intvec::{
    Codec, CodecError, FixedVec, SeqVec, VarVec, VarVecParts, VarVecPartsError,
};

/// Every code, with its parameter at the ends of its range, at the end of the candidates of the
/// automatic choice, and just past it.
pub fn codecs() -> Vec<Codec> {
    let mut codecs = vec![
        Codec::Unary,
        Codec::Gamma,
        Codec::Delta,
        Codec::Omega,
        Codec::VByteLe,
        Codec::VByteBe,
    ];
    for k in [0, 1, 2, 9, 10, 11, 63] {
        codecs.extend([
            Codec::Rice { log2_b: k },
            Codec::ExpGolomb { k },
            Codec::Pi { k },
        ]);
        if k > 0 {
            codecs.push(Codec::Zeta { k });
        }
    }
    for b in [1, 2, 3, 10, 11, (1 << 40) + 1, u64::MAX] {
        codecs.push(Codec::Golomb { b });
    }
    codecs
}

/// Whether `codec` can encode the word `u64::MAX`, as `Codec` documents.
pub fn encodes_max(codec: Codec) -> bool {
    !matches!(
        codec,
        Codec::Gamma
            | Codec::Delta
            | Codec::Omega
            | Codec::Zeta { .. }
            | Codec::Pi { .. }
            | Codec::ExpGolomb { k: 0 }
    )
}

/// The largest word that the tests encode with `codec`. A code with a unary part takes as many
/// bits as its quotient, so its words keep the quotient below `2^12`.
pub fn max_word(codec: Codec) -> u64 {
    match codec {
        Codec::Unary => 1 << 12,
        Codec::Rice { log2_b } => u64::MAX >> 52_u32.saturating_sub(log2_b),
        Codec::Golomb { b } => b.saturating_mul(1 << 12),
        codec if encodes_max(codec) => u64::MAX,
        _ => u64::MAX - 1,
    }
}

/// Returns `n` words up to `max_word(codec)`: zeros, the largest word, small words and words of
/// every magnitude.
pub fn words(rng: &mut Rng, codec: Codec, n: usize) -> Vec<u64> {
    let max = max_word(codec);
    let small = 1 << rng.below(8);
    (0..n)
        .map(|_| match rng.below(8) {
            0 => 0,
            1 => max,
            2..=4 => rng.next_u64() % small,
            _ => match max.checked_add(1) {
                Some(end) => rng.word() % end,
                None => rng.word(),
            },
        })
        .collect()
}

/// Returns bit `k` of a stream: bit `k % 8` of byte `k / 8` of the words in memory, as
/// `VarVecParts` and `SeqVecParts` document.
fn stream_bit(words: &[u64], k: u64) -> bool {
    let byte = words[(k / 64) as usize].to_ne_bytes()[(k % 64 / 8) as usize];
    (byte >> (k % 8)) & 1 == 1
}

/// Returns the vector of one sequence holding `words`, if `codec` can encode them.
fn single_sequence(codec: Codec, words: &[u64]) -> Option<SeqVec<u64>> {
    SeqVec::builder().codec(codec).build(&[words]).ok()
}

/// The number of bits that `codec` takes for `words`: the end offset of their sequence.
fn bits(codec: Codec, words: &[u64]) -> Option<u64> {
    single_sequence(codec, words)?.into_parts().offsets.get(1)
}

/// Checks that the stream `data`, from bit `start`, holds the codewords that the builder writes
/// for `words`. Every word has one codeword, so a stream that the decoders accept, even a
/// corrupt one, must hold exactly the codewords of the words they return.
#[track_caller]
pub fn assert_canonical(codec: Codec, words: &[u64], data: &[u64], start: u64) {
    let parts = single_sequence(codec, words)
        .expect("decoded words can be encoded")
        .into_parts();
    let len = parts.offsets.get(1).unwrap();
    for k in 0..len {
        assert_eq!(
            stream_bit(&parts.data, k),
            stream_bit(data, start + k),
            "{codec:?} decoded {words:?}, whose bit {k} differs"
        );
    }
}

/// The automatic choice takes no more bits than any of the candidates that `Codec` lists, for
/// both builders, up to the one bit that the choice may misjudge per word `u64::MAX`.
#[test]
#[cfg_attr(miri, ignore = "too slow under Miri, and safe code only")]
fn automatic_choice_takes_the_fewest_bits() {
    let mut candidates = vec![
        Codec::Unary,
        Codec::Gamma,
        Codec::Delta,
        Codec::Omega,
        Codec::VByteLe,
    ];
    candidates.extend((1..=10).map(|k| Codec::Zeta { k }));
    candidates.extend((1..=10).map(|b| Codec::Golomb { b }));
    candidates.extend((0..=9).map(|k| Codec::ExpGolomb { k }));
    candidates.extend((0..=9).map(|log2_b| Codec::Rice { log2_b }));
    candidates.extend((2..=10).map(|k| Codec::Pi { k }));

    let mut rng = Rng::new(1);
    for round in 0..300 {
        let n = 1 + rng.below(300);
        let words: Vec<u64> = match round % 6 {
            0 => vec![0; n],
            1 => (0..n).map(|_| rng.next_u64() % 3).collect(),
            2 => (0..n).map(|_| rng.word() >> 40).collect(),
            3 => (0..n).map(|_| rng.word()).collect(),
            4 => (0..n).map(|_| 1 << rng.below(64)).collect(),
            _ => (0..n).map(|_| [0, 1, u64::MAX][rng.below(3)]).collect(),
        };
        let largest = words.iter().copied().max().unwrap();
        // A code whose unary part reaches `2^17` bits for one word takes more bits than VByte,
        // at most 80 bits per word, for all of them, so it cannot be the best, and encoding it
        // would take too much memory.
        let best = candidates
            .iter()
            .filter(|&&codec| {
                let quotient = match codec {
                    Codec::Unary => largest,
                    Codec::Rice { log2_b } => largest >> log2_b,
                    Codec::Golomb { b } => largest / b,
                    _ => 0,
                };
                quotient < 1 << 17
            })
            .filter_map(|&codec| bits(codec, &words))
            .min()
            .unwrap();
        let slack = words.iter().filter(|&&w| w == u64::MAX).count() as u64;
        let seq = SeqVec::from_slices(&[&words]);
        let var = VarVec::from_slice(&words);
        for chosen in [seq.codec(), var.codec()] {
            let taken = bits(chosen, &words).unwrap();
            assert!(
                taken <= best + slack,
                "{chosen:?} takes {taken} bits, the best {best}"
            );
        }
        assert_eq!(var, words[..]);
        assert_eq!(seq, [&words][..]);
    }
}

#[test]
fn chosen_codes_with_a_parameter_out_of_range_fail() {
    let empty: [&[u64]; 0] = [];
    for codec in [
        Codec::Rice { log2_b: 64 },
        Codec::ExpGolomb { k: 64 },
        Codec::Pi { k: 64 },
        Codec::Zeta { k: 0 },
        Codec::Zeta { k: 64 },
        Codec::Golomb { b: 0 },
        Codec::Rice { log2_b: u32::MAX },
    ] {
        let error = CodecError::InvalidParameter(codec);
        let var = VarVec::<u64>::builder().codec(codec);
        assert_eq!(var.build(&[]).unwrap_err(), error);
        assert_eq!(var.build_from_iter([1]).unwrap_err(), error);
        let seq = SeqVec::<u64>::builder().codec(codec);
        assert_eq!(seq.build(&empty).unwrap_err(), error);
        assert_eq!(
            seq.store_lengths().build_from_iter([[1]]).unwrap_err(),
            error
        );
    }
}

/// The codes that `Codec` lists cannot encode the word `u64::MAX`, which stores `u64::MAX` and
/// `i64::MIN`; the others can, and the automatic choice always picks one of those for such data.
#[test]
fn codes_that_cannot_encode_the_largest_word() {
    for codec in codecs() {
        // A code with a unary part would take up to 2^64 bits for the largest word.
        if encodes_max(codec) && max_word(codec) < u64::MAX {
            continue;
        }
        let unsigned = VarVec::builder().codec(codec).build(&[1, u64::MAX]);
        let signed = VarVec::builder().codec(codec).build(&[0, i64::MIN]);
        let sequences = SeqVec::builder().codec(codec).build(&[[1, u64::MAX]]);
        if !encodes_max(codec) {
            let error = CodecError::Unencodable {
                codec,
                value: u64::MAX,
            };
            assert_eq!(unsigned.unwrap_err(), error);
            assert_eq!(sequences.unwrap_err(), error);
            assert_eq!(
                signed.unwrap_err(),
                CodecError::Unencodable {
                    codec,
                    value: i64::MIN
                }
            );
        } else {
            assert_eq!(unsigned.unwrap(), [1, u64::MAX][..]);
            assert_eq!(signed.unwrap(), [0, i64::MIN][..]);
            assert_eq!(sequences.unwrap(), [[1, u64::MAX]][..]);
        }
    }
    let values = [i64::MIN, -1, 0, i64::MAX];
    assert_eq!(VarVec::from_slice(&values), values[..]);
    assert_eq!(SeqVec::from_slices(&[values]), [values][..]);
    assert_eq!(VarVec::from_slice(&[u64::MAX]), [u64::MAX][..]);
}

/// Flips every bit of the codeword of each word next to a power of two, where the lengths that
/// the decoders compute change, and of the bits after it. Each stream that `from_parts`
/// accepts as one or two words reads back the same from every access path, and holds the
/// codewords of the words it decodes to.
#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn decoders_agree_on_every_flipped_bit_of_a_codeword() {
    let (mut accepted, mut rejected) = (0, 0);
    for codec in codecs() {
        let boundaries = (0..64).flat_map(|j| [(1_u64 << j) - 1, 1 << j]);
        let max = max_word(codec);
        for word in boundaries
            .chain([u64::MAX - 1, u64::MAX])
            .filter(|&w| w <= max)
        {
            let len = bits(codec, &[word]).unwrap();
            let mut data = single_sequence(codec, &[word]).unwrap().into_parts().data;
            data.extend([u64::MAX, 0, u64::MAX]);
            for bit in 0..len + 8 {
                let mut flipped = data.clone();
                flipped[(bit / 64) as usize] ^= 1 << (bit % 64);
                for count in 1..=2 {
                    let parts = VarVecParts {
                        data: flipped.clone(),
                        samples: FixedVec::from_slice(&[0]),
                        sampling_rate: nz(usize::MAX),
                        len: count,
                        codec,
                    };
                    let Ok(vec) = VarVec::<u64>::from_parts(parts) else {
                        rejected += 1;
                        break;
                    };
                    accepted += 1;
                    let decoded: Vec<u64> = vec.iter().collect();
                    assert!((0..count).all(|i| vec.get(i) == Some(decoded[i])));
                    assert_canonical(codec, &decoded, &flipped, 0);
                }
            }
        }
    }
    assert!(accepted > 0 && rejected > 0, "{accepted} {rejected}");
}

/// Returns the bits of the codeword that the builder writes for `word`, in stream order.
fn codeword(codec: Codec, word: u64) -> Vec<bool> {
    let data = single_sequence(codec, &[word]).unwrap().into_parts().data;
    (0..bits(codec, &[word]).unwrap())
        .map(|k| stream_bit(&data, k))
        .collect()
}

/// Returns `n` bits of `value`, least significant first, as the stream holds a field.
fn field(value: u64, n: u32) -> Vec<bool> {
    (0..n).map(|i| (value >> i) & 1 == 1).collect()
}

/// Returns the bits of `bytes`, each written as an 8-bit field.
fn byte_bits(bytes: &[u8]) -> Vec<bool> {
    bytes.iter().flat_map(|&b| field(b.into(), 8)).collect()
}

/// Returns the stream words of `bits`, in the memory layout that `VarVecParts` documents.
fn stream(bits: &[bool]) -> Vec<u64> {
    let mut words = vec![0_u64; bits.len().div_ceil(64)];
    for (k, &bit) in bits.iter().enumerate() {
        words[k / 64] |= u64::from(bit) << (k % 64);
    }
    words.into_iter().map(u64::to_le).collect()
}

/// The VByte codeword of `n`, least significant group first, from the bijective base-128 digits
/// that `Codec::VByteLe` documents, computed in 128 bits so that it reaches `u64::MAX + 1`.
fn vbyte_le(mut n: u128) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let low = (n & 0x7F) as u8;
        n >>= 7;
        if n == 0 {
            bytes.push(low);
            return bytes;
        }
        bytes.push(low | 0x80);
        n -= 1;
    }
}

/// The VByte codeword of `n`, most significant group first.
fn vbyte_be(mut n: u128) -> Vec<u8> {
    let mut bytes = vec![(n & 0x7F) as u8];
    n >>= 7;
    while n > 0 {
        n -= 1;
        bytes.push((n & 0x7F) as u8 | 0x80);
        n >>= 7;
    }
    bytes.reverse();
    bytes
}

/// The largest word that `codec` encodes, and codewords next to it that encode no word: for
/// every branch of the checked decoder that rejects a codeword, the codeword just past the
/// largest one that it accepts. Unary, `Rice { log2_b: 0 }` and `Golomb { b: 1 }` have no such
/// codeword: their unary quotient cannot exceed `u64::MAX` in any stream.
fn limits() -> Vec<(Codec, u64, Vec<Vec<bool>>)> {
    let zeros = |n: u64| vec![false; n as usize];
    let cat = |parts: &[Vec<bool>]| parts.concat();
    let one = vec![true];
    let max = u64::MAX;
    // A length of 64 below the leading one of `n + 1`, so that `n + 1 >= 2^64`.
    let gamma_past = cat(&[zeros(64), one.clone()]);
    let mut limits = vec![
        (Codec::Gamma, max - 1, vec![gamma_past.clone()]),
        (Codec::ExpGolomb { k: 0 }, max - 1, vec![gamma_past]),
        (
            Codec::Delta,
            max - 1,
            vec![cat(&[codeword(Codec::Gamma, 64), zeros(64)])],
        ),
        // The terminating zero of the codeword of `u64::MAX - 1` becomes the one that starts a
        // group after the group of 64 bits. Then the shortest group that announces 64 more
        // bits: the groups of 2, 6 and 64, each a one and the bits below it, and a one.
        (Codec::Omega, max - 1, {
            let mut bits = codeword(Codec::Omega, max - 1);
            *bits.last_mut().unwrap() = true;
            let group = |value, n| cat(&[one.clone(), field(value, n)]);
            let groups = cat(&[group(0, 1), group(0b10, 2), group(0, 6), one.clone()]);
            vec![bits, groups]
        }),
    ];
    for (codec, encode) in [
        (Codec::VByteLe, vbyte_le as fn(u128) -> Vec<u8>),
        (Codec::VByteBe, vbyte_be),
    ] {
        assert_eq!(
            byte_bits(&encode(max.into())),
            codeword(codec, max),
            "{codec:?}"
        );
        let past = encode(u128::from(max) + 1);
        assert_eq!(past.len(), 10, "{codec:?}");
        // Ten bytes that continue: every codeword of eleven bytes or more is too large.
        let eleven = cat(&[byte_bits(&[0x80; 10]), byte_bits(&[0])]);
        limits.push((codec, max, vec![byte_bits(&past), eleven]));
    }
    for k in [1, 2, 3, 10, 32, 63] {
        // The quotient `(u64::MAX >> k) + 1` does not fit `64 - k` bits.
        limits.push((
            Codec::ExpGolomb { k },
            max,
            vec![cat(&[
                codeword(Codec::Gamma, (max >> k) + 1),
                zeros(k.into()),
            ])],
        ));
        // `h = 63 / k + 1` puts the value at `2^(h * k) >= 2^64` at least.
        limits.push((
            Codec::Zeta { k },
            max - 1,
            vec![cat(&[zeros(u64::from(63 / k + 1)), one.clone()])],
        ));
    }
    for k in [0, 1, 2, 63] {
        limits.push((
            Codec::Pi { k },
            max - 1,
            vec![cat(&[codeword(Codec::Rice { log2_b: k }, 64), zeros(64)])],
        ));
    }
    for log2_b in [52, 62, 63] {
        let codec = Codec::Rice { log2_b };
        limits.push((
            codec,
            max,
            vec![cat(&[zeros((max >> log2_b) + 1), codeword(codec, 0)])],
        ));
    }
    for b in [(1 << 52) + 1, 1 << 63, (1 << 63) + 1, max - 1, max] {
        let codec = Codec::Golomb { b };
        let (quotient, remainder) = (max / b, max % b);
        // One more in the quotient overflows the product, one more in the remainder the sum.
        let mut past = vec![cat(&[zeros(quotient + 1), codeword(codec, 0)])];
        if remainder + 1 < b {
            past.push(cat(&[zeros(quotient), codeword(codec, remainder + 1)]));
        }
        limits.push((codec, max, past));
    }
    limits
}

/// For every code with a codeword that encodes no word, `from_parts` accepts the codeword of the
/// largest word and rejects each codeword just past it as `InvalidCodeword`, never as a
/// truncated stream.
#[test]
fn checked_decoder_accepts_the_largest_codeword_and_rejects_the_next() {
    let parts = |data| VarVecParts {
        data,
        samples: FixedVec::from_slice(&[0]),
        sampling_rate: nz(1),
        len: 1,
        codec: Codec::Unary,
    };
    for (codec, largest, past) in limits() {
        let accepted = VarVec::<u64>::from_parts(VarVecParts {
            codec,
            ..parts(stream(&codeword(codec, largest)))
        });
        assert_eq!(accepted.unwrap(), [largest][..], "{codec:?}");
        for (case, bits) in past.iter().enumerate() {
            // Enough bits follow that every rejection is for the value, not the end of the stream.
            let data = stream(&[&bits[..], &[false; 128]].concat());
            assert_eq!(
                VarVec::<u64>::from_parts(VarVecParts {
                    codec,
                    ..parts(data)
                })
                .unwrap_err(),
                VarVecPartsError::InvalidCodeword { index: 0 },
                "{codec:?}, case {case}"
            );
        }
    }
}
