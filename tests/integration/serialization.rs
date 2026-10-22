//! Serialization, with the `serde` feature: round trips through two formats, the element types
//! that a payload loads as, payloads that must not load, and samples of format version 1.
//!
//! Every format is canonical for a given element type: a payload that loads serializes back to
//! the same bytes.

use crate::codec::{codecs, words};
use crate::{AliasedFlag, Case, Rng, hash, nz, value};
use compressed_intvec::{AtomicFixedVec, Codec, FixedVec, SeqVec, SeqVecParts, VarVec};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::fmt::Debug;

/// Checks that `v` loads back from both formats, and that what loads serializes to the same
/// bytes.
#[track_caller]
fn round_trip<V: Serialize + DeserializeOwned + PartialEq + Debug>(v: &V) {
    let bytes = postcard::to_allocvec(v).unwrap();
    let loaded = postcard::from_bytes::<V>(&bytes).unwrap();
    assert_eq!(&loaded, v);
    assert_eq!(postcard::to_allocvec(&loaded).unwrap(), bytes);
    let text = serde_json::to_string(v).unwrap();
    let loaded = serde_json::from_str::<V>(&text).unwrap();
    assert_eq!(&loaded, v);
    assert_eq!(serde_json::to_string(&loaded).unwrap(), text);
}

fn to_json(v: &impl Serialize) -> Value {
    serde_json::to_value(v).unwrap()
}

#[test]
fn round_trips() {
    let mut rng = Rng::new(1);
    for bit_width in 0..=64 {
        for len in [0, 1, 100] {
            let mut vec = FixedVec::new(bit_width).unwrap();
            vec.extend((0..len).map(|_| value::<u64>(&mut rng, bit_width)));
            round_trip(&vec);
            let atomic = AtomicFixedVec::from(vec.clone());
            assert_eq!(to_json(&atomic), to_json(&vec));
            let loaded: AtomicFixedVec<u64> = serde_json::from_value(to_json(&vec)).unwrap();
            assert_eq!(FixedVec::from(loaded), vec);
            let (words, len, bit_width) = vec.clone().into_parts();
            let view = FixedVec::<u64, &[u64]>::from_parts(&words, len, bit_width).unwrap();
            assert_eq!(to_json(&view), to_json(&vec));
        }
    }
    round_trip(&FixedVec::from_slice(&[-3i16, 7, i16::MIN]));
    round_trip(&FixedVec::from_slice(&[usize::MAX, 0]));
    for codec in codecs() {
        let n = rng.below(100);
        let values = words(&mut rng, codec, n);
        for rate in [1, 32] {
            round_trip(
                &VarVec::builder()
                    .codec(codec)
                    .sampling_rate(nz(rate))
                    .build(&values)
                    .unwrap(),
            );
        }
        let sequences: Vec<&[u64]> = values.chunks(1 + rng.below(5)).collect();
        round_trip(&SeqVec::builder().codec(codec).build(&sequences).unwrap());
        round_trip(
            &SeqVec::builder()
                .codec(codec)
                .store_lengths()
                .build(&sequences)
                .unwrap(),
        );
    }
    round_trip(&VarVec::from_slice(&[i64::MIN, 0, i64::MAX]));
    round_trip(&SeqVec::from_slices(&[vec![-1i8, 1], vec![]]));
}

/// Changing only unused bits and padding words leaves serialization unchanged.
#[test]
fn unused_bits_do_not_change_serialization() {
    for (values, bit_width) in [(&[1u32, 2, 3][..], 2), (&[5, 6], 32), (&[], 7)] {
        let mut clean = FixedVec::new(bit_width).unwrap();
        clean.extend_from_slice(values);
        let (mut words, len, _) = clean.clone().into_parts();
        let used = values.len() * bit_width as usize;
        for k in used..64 * words.len() {
            words[k / 64] |= 1 << (k % 64);
        }
        let dirty = FixedVec::<u32>::from_parts(words, len, bit_width).unwrap();
        assert_eq!(dirty, clean);
        assert_eq!(
            postcard::to_allocvec(&dirty).unwrap(),
            postcard::to_allocvec(&clean).unwrap()
        );
    }
    let mut popped = FixedVec::from_slice(&[7u32, 7, 7]);
    popped.pop();
    assert_eq!(to_json(&popped), to_json(&FixedVec::from_slice(&[7u32, 7])));

    // The stream of a `VarVec` or a `SeqVec` is written up to its last codeword, and a `SeqVec`
    // stream from its first offset, whatever the parts hold around them.
    let clean = VarVec::builder()
        .codec(Codec::Gamma)
        .build(&[3u32, 1, 4])
        .unwrap();
    let mut parts = clean.clone().into_parts();
    // Stream words hold the bytes of the stream in memory order, hence `to_le`.
    parts.data[0] |= u64::to_le(u64::MAX << 20);
    parts.data.extend([u64::MAX, 1]);
    let dirty = VarVec::<u32>::from_parts(parts).unwrap();
    assert_eq!(to_json(&dirty), to_json(&clean));
    round_trip(&dirty);
    let clean = SeqVec::<u32>::from_parts(SeqVecParts {
        data: vec![u64::to_le(0b1010)],
        offsets: FixedVec::from_slice(&[1, 2, 4]),
        lengths: None,
        codec: Codec::Unary,
    })
    .unwrap();
    let mut parts = clean.clone().into_parts();
    parts.data = vec![u64::to_le(0b1011 | u64::MAX << 4), u64::MAX];
    let dirty = SeqVec::<u32>::from_parts(parts).unwrap();
    assert_eq!(dirty, [[0u32], [1]][..]);
    assert_eq!(to_json(&dirty), to_json(&clean));
    assert_eq!(to_json(&dirty)["V1"]["data"], json!([0b1010]));
    round_trip(&dirty);
}

#[test]
fn custom_element_aliases_preserve_encoded_words() {
    let canonical = FixedVec::<AliasedFlag>::from_parts(vec![1, 0], 1, 2).unwrap();
    let alias = FixedVec::<AliasedFlag>::from_parts(vec![3, 0], 1, 2).unwrap();
    assert_eq!(canonical, alias);
    assert_eq!(hash(&canonical), hash(&alias));
    assert_eq!(to_json(&canonical)["V1"]["words"], json!([1]));
    assert_eq!(to_json(&alias)["V1"]["words"], json!([3]));
    assert_ne!(
        postcard::to_allocvec(&canonical).unwrap(),
        postcard::to_allocvec(&alias).unwrap()
    );
    round_trip(&canonical);
    round_trip(&alias);
}

/// A payload loads as any type of the same signedness that holds its contents.
#[test]
fn payloads_load_as_types_that_hold_them() {
    #[track_caller]
    fn loads<V: DeserializeOwned>(payload: &Value) -> bool {
        serde_json::from_value::<V>(payload.clone()).is_ok()
    }

    // Bit width 10.
    let fixed = to_json(&FixedVec::from_slice(&[1000u32, 3]));
    let wider: FixedVec<u64> = serde_json::from_value(fixed.clone()).unwrap();
    assert_eq!(wider, [1000, 3][..]);
    assert!(loads::<FixedVec<u16>>(&fixed));
    assert!(!loads::<FixedVec<u8>>(&fixed));
    assert!(!loads::<FixedVec<i32>>(&fixed));
    // The zig-zag words 9 and 14, of bit width 4.
    let signed = to_json(&FixedVec::from_slice(&[-5i16, 7]));
    let narrower: FixedVec<i8> = serde_json::from_value(signed.clone()).unwrap();
    assert_eq!(narrower, [-5, 7][..]);
    assert!(loads::<AtomicFixedVec<i64>>(&signed));
    assert!(!loads::<FixedVec<u16>>(&signed));

    let fits = to_json(&VarVec::from_slice(&[1u64, 255]));
    let narrower: VarVec<u8> = serde_json::from_value(fits.clone()).unwrap();
    assert_eq!(narrower, [1, 255][..]);
    assert!(!loads::<VarVec<u8>>(&to_json(&VarVec::from_slice(&[
        1u64, 256
    ]))));
    assert!(!loads::<VarVec<i64>>(&fits));
    assert!(loads::<VarVec<i8>>(&to_json(&VarVec::from_slice(&[
        -128i64, 127
    ]))));
    assert!(!loads::<VarVec<i8>>(&to_json(&VarVec::from_slice(&[
        -129i64
    ]))));

    let sequences = to_json(&SeqVec::from_slices(&[vec![1u64, 255], vec![]]));
    let narrower: SeqVec<u8> = serde_json::from_value(sequences.clone()).unwrap();
    assert_eq!(narrower, [vec![1u8, 255], vec![]][..]);
    assert!(!loads::<SeqVec<u8>>(&to_json(&SeqVec::from_slices(&[[
        256u64
    ]]))));
    assert!(!loads::<SeqVec<i64>>(&sequences));
}

/// Each change makes the payload describe no vector, or another format version, and the error
/// names the rule that it breaks. The unchanged payloads load and serialize back to the same
/// text.
#[test]
fn malformed_payloads_do_not_load() {
    #[track_caller]
    fn rejects<V: Serialize + DeserializeOwned>(payload: &Value, edits: &[(&str, Case<Value>)]) {
        let loaded: V = serde_json::from_value(payload.clone()).unwrap();
        assert_eq!(&to_json(&loaded), payload);
        for (message, (name, edit)) in edits {
            let mut payload = payload.clone();
            edit(&mut payload);
            let error = serde_json::from_value::<V>(payload).err().expect(name);
            assert!(error.to_string().contains(message), "{name}: {error}");
        }
    }

    rejects::<FixedVec<u32>>(
        &to_json(&FixedVec::from_slice(&[1000u32, 3, 5])),
        &[
            (
                "unknown variant `V2`",
                ("version", |p| *p = json!({"V2": p["V1"]})),
            ),
            (
                "unknown variant `V2`",
                ("version first", |p| *p = json!({"V2": 0})),
            ),
            (
                "unknown field",
                ("field", |p| p["V1"]["element_bits"] = json!(32)),
            ),
            (
                "are signed",
                ("signedness", |p| p["V1"]["element_signed"] = json!(true)),
            ),
            (
                "bit width",
                ("bit width", |p| p["V1"]["bit_width"] = json!(33)),
            ),
            (
                "0 data words",
                ("one word less", |p| p["V1"]["words"] = json!([])),
            ),
            (
                "2 data words",
                ("one word more", |p| p["V1"]["words"] = json!([0, 0])),
            ),
            ("expected 157", ("length", |p| p["V1"]["len"] = json!(1000))),
            (
                "overflow",
                ("length overflow", |p| p["V1"]["len"] = json!(usize::MAX)),
            ),
            (
                "past the last element",
                ("unused bits", |p| {
                    p["V1"]["words"][0] = json!(p["V1"]["words"][0].as_u64().unwrap() | 1 << 30)
                }),
            ),
        ],
    );
    rejects::<VarVec<u32>>(
        &to_json(
            &VarVec::builder()
                .codec(Codec::Gamma)
                .build(&[3u32, 1, 4, 1, 5])
                .unwrap(),
        ),
        &[
            (
                "unknown variant `V2`",
                ("version", |p| *p = json!({"V2": p["V1"]})),
            ),
            (
                "unknown field",
                ("samples", |p| p["V1"]["samples"] = json!(null)),
            ),
            (
                "are signed",
                ("signedness", |p| p["V1"]["element_signed"] = json!(true)),
            ),
            (
                "sampling rate",
                ("sampling rate", |p| p["V1"]["sampling_rate"] = json!(0)),
            ),
            (
                "before element 5",
                ("length", |p| p["V1"]["len"] = json!(6)),
            ),
            (
                "out of range",
                ("code parameter", |p| {
                    p["V1"]["codec"] = json!({"Golomb": 0})
                }),
            ),
            (
                "Huffman",
                ("unknown code", |p| p["V1"]["codec"] = json!("Huffman")),
            ),
            (
                "before element 0",
                ("stream", |p| p["V1"]["data"] = json!([])),
            ),
            (
                "2 words",
                ("trailing word", |p| {
                    p["V1"]["data"].as_array_mut().unwrap().push(json!(0))
                }),
            ),
            (
                "after the last codeword",
                ("trailing bits", |p| {
                    p["V1"]["data"][0] = json!(p["V1"]["data"][0].as_u64().unwrap() | 1 << 63)
                }),
            ),
        ],
    );
    rejects::<SeqVec<u32>>(
        &to_json(
            &SeqVec::builder()
                .codec(Codec::Gamma)
                .store_lengths()
                .build(&[vec![1u32, 2], vec![], vec![3]])
                .unwrap(),
        ),
        &[
            (
                "unknown variant `V2`",
                ("version", |p| *p = json!({"V2": p["V1"]})),
            ),
            (
                "unknown field",
                ("lengths", |p| p["V1"]["lengths"] = json!(null)),
            ),
            (
                "are signed",
                ("signedness", |p| p["V1"]["element_signed"] = json!(true)),
            ),
            (
                "out of range",
                ("code parameter", |p| p["V1"]["codec"] = json!({"Zeta": 0})),
            ),
            (
                "offsets are empty",
                ("no offsets", |p| {
                    p["V1"]["offsets"] = to_json(&FixedVec::<u64>::from_slice(&[]))
                }),
            ),
            (
                "sequence 0 ends",
                ("offsets", |p| {
                    p["V1"]["offsets"] = to_json(&FixedVec::from_slice(&[0u64, 5, 6, 11]))
                }),
            ),
            (
                "unknown variant `V2`",
                ("offsets version", |p| {
                    p["V1"]["offsets"] = json!({"V2": p["V1"]["offsets"]["V1"]})
                }),
            ),
            ("ends before", ("stream", |p| p["V1"]["data"] = json!([0]))),
            (
                "2 words",
                ("trailing word", |p| {
                    p["V1"]["data"].as_array_mut().unwrap().push(json!(0))
                }),
            ),
            (
                "after the last codeword",
                ("trailing bits", |p| {
                    p["V1"]["data"][0] = json!(p["V1"]["data"][0].as_u64().unwrap() | 1 << 63)
                }),
            ),
        ],
    );
    // The sequences [0] and [1] in the unary code, from bit 1.
    let shifted = SeqVec::<u32>::from_parts(SeqVecParts {
        data: vec![u64::to_le(0b1010)],
        offsets: FixedVec::from_slice(&[1, 2, 4]),
        lengths: None,
        codec: Codec::Unary,
    })
    .unwrap();
    rejects::<SeqVec<u32>>(
        &to_json(&shifted),
        &[(
            "before the first codeword",
            ("leading bits", |p| p["V1"]["data"][0] = json!(0b1011)),
        )],
    );

    // Postcard writes the version as the first byte.
    let bytes = postcard::to_allocvec(&FixedVec::from_slice(&[1u32])).unwrap();
    assert_eq!(bytes[0], 0);
    let mut later = bytes.clone();
    later[0] = 1;
    assert!(postcard::from_bytes::<FixedVec<u32>>(&later).is_err());
    assert_eq!(
        postcard::from_bytes::<FixedVec<u32>>(&bytes).unwrap(),
        [1][..]
    );
}

/// Returns the bytes of a hexadecimal text.
fn from_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

/// Payloads of format version 1, as JSON text and as postcard bytes in hexadecimal, derived by
/// hand from the documented layout and codes. Each loads as the vector it describes, and both
/// that vector and the one loaded serialize back to the same text and bytes.
///
/// These are the samples of the 1.0.0 release. Once released, they are never edited or deleted:
/// a format change gets a new version with its own samples, and readers keep loading every
/// earlier version.
///
/// In postcard, a variant is its index in declaration order and an integer a LEB128 varint:
/// the version tag `V1` is `00`, `false` is `00` and `true` is `01`, and each code is its index
/// in `Unary, Gamma, Delta, Omega, VByteLe, VByteBe, Rice, Golomb, ExpGolomb, Zeta, Pi`
/// followed by its parameter. A sequence of words is its length, then the words.
#[test]
fn samples_of_format_version_1() {
    #[track_caller]
    fn sample<V: Serialize + DeserializeOwned + PartialEq + Debug>(
        expected: &V,
        text: &str,
        hex: &str,
    ) {
        let loaded: V = serde_json::from_str(text).unwrap();
        assert_eq!(&loaded, expected);
        assert_eq!(serde_json::to_string(&loaded).unwrap(), text);
        assert_eq!(serde_json::to_string(expected).unwrap(), text);
        let bytes = from_hex(hex);
        let loaded: V = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(&loaded, expected);
        assert_eq!(postcard::to_allocvec(&loaded).unwrap(), bytes);
        assert_eq!(postcard::to_allocvec(expected).unwrap(), bytes);
    }

    // 10, 20 and 30 in 5 bits each: 10 | 20 << 5 | 30 << 10 = 31370, the varint 8A F5 01.
    let fixed = FixedVec::from_slice(&[10u32, 20, 30]);
    let text = r#"{"V1":{"element_signed":false,"bit_width":5,"len":3,"words":[31370]}}"#;
    let hex = "00000503018af501";
    sample(&fixed, text, hex);
    let atomic = AtomicFixedVec::from(fixed);
    assert_eq!(serde_json::to_string(&atomic).unwrap(), text);
    assert_eq!(postcard::to_allocvec(&atomic).unwrap(), from_hex(hex));
    // The zig-zag words 1, 0 and 2 in 2 bits each: 1 | 2 << 4.
    sample(
        &FixedVec::from_slice(&[-1i8, 0, 1]),
        r#"{"V1":{"element_signed":true,"bit_width":2,"len":3,"words":[33]}}"#,
        "000102030121",
    );
    // Gamma of n: floor(log2(n + 1)) zeros, a one, then the bits of n + 1 below its leading one,
    // least significant first. 0..=4 give 1, 010, 011, 00100, 00110 from bit 0: the bits 0, 2,
    // 5, 6, 9, 14 and 15, which end at bit 16, in the first word.
    sample(
        &VarVec::builder()
            .codec(Codec::Gamma)
            .build(&[0u32, 1, 2, 3, 4])
            .unwrap(),
        r#"{"V1":{"element_signed":false,"codec":"Gamma","sampling_rate":32,"len":5,"data":[49765]}}"#,
        "000001200501e58403",
    );
    // Rice with log2_b = 1 of n: n >> 1 zeros, a one, then the low bit of n. 1, 2 and 3 give
    // 11, 010, 011: the bits 0, 1, 3, 6 and 7. The offsets 0, 5, 5 and 8 in 4 bits each, as a
    // serialized `FixedVec<u64>`, and the lengths, which the reader counts.
    sample(
        &SeqVec::builder()
            .codec(Codec::Rice { log2_b: 1 })
            .store_lengths()
            .build(&[vec![1u32, 2], vec![], vec![3]])
            .unwrap(),
        r#"{"V1":{"element_signed":false,"codec":{"Rice":1},"data":[203],"offsets":{"V1":{"element_signed":false,"bit_width":4,"len":4,"words":[34128]}},"store_lengths":true}}"#,
        "0000060101cb010000040401d08a0201",
    );
    // One sample for each remaining code, written by 1.0.0, so that a change to the bit layout
    // of any code fails here. Unary of n is n zeros and a one, so 0..=4 set the bits 0, 2, 5, 9
    // and 14. The byte codes write 0..=4 in one byte each and 1000 = 104 + 128 * (6 + 1) in
    // two, E8 06 for VByteLe and 86 68 for VByteBe. The Golomb codeword of 1000 starts with its
    // quotient 333 in unary, whose zeros fill four words.
    let small = [0u32, 1, 2, 3, 4];
    let wide = [0u32, 1, 2, 3, 4, 1000];
    for (codec, values, text, hex) in [
        (
            Codec::Unary,
            &small[..],
            r#"{"V1":{"element_signed":false,"codec":"Unary","sampling_rate":32,"len":5,"data":[16933]}}"#,
            "000000200501a58401",
        ),
        (
            Codec::Delta,
            &wide[..],
            r#"{"V1":{"element_signed":false,"codec":"Delta","sampling_rate":32,"len":6,"data":[32837438789]}}"#,
            "000002200601c59a8eaa7a",
        ),
        (
            Codec::Omega,
            &wide[..],
            r#"{"V1":{"element_signed":false,"codec":"Omega","sampling_rate":32,"len":6,"data":[32857760434]}}"#,
            "000003200601b2c5e6b37a",
        ),
        (
            Codec::VByteLe,
            &wide[..],
            r#"{"V1":{"element_signed":false,"codec":"VByteLe","sampling_rate":32,"len":6,"data":[1943953788240128]}}"#,
            "00000420060180828898c080ba03",
        ),
        (
            Codec::VByteBe,
            &wide[..],
            r#"{"V1":{"element_signed":false,"codec":"VByteBe","sampling_rate":32,"len":6,"data":[29420749366362368]}}"#,
            "00000520060180828898c0c0a134",
        ),
        (
            Codec::Golomb { b: 3 },
            &wide[..],
            r#"{"V1":{"element_signed":false,"codec":{"Golomb":3},"sampling_rate":32,"len":6,"data":[13037,0,0,0,0,805306368]}}"#,
            "00000703200606ed65000000008080808003",
        ),
        (
            Codec::ExpGolomb { k: 1 },
            &wide[..],
            r#"{"V1":{"element_signed":false,"codec":{"ExpGolomb":1},"sampling_rate":32,"len":6,"data":[8237640237]}}"#,
            "00000801200601add481d81e",
        ),
        (
            Codec::Zeta { k: 3 },
            &wide[..],
            r#"{"V1":{"element_signed":false,"codec":{"Zeta":3},"sampling_rate":32,"len":6,"data":[4106661273]}}"#,
            "0000090320060199db9aa60f",
        ),
        (
            Codec::Pi { k: 2 },
            &wide[..],
            r#"{"V1":{"element_signed":false,"codec":{"Pi":2},"sampling_rate":32,"len":6,"data":[32842263961]}}"#,
            "00000a0220060199dbb4ac7a",
        ),
    ] {
        sample(
            &VarVec::builder().codec(codec).build(values).unwrap(),
            text,
            hex,
        );
    }
}
