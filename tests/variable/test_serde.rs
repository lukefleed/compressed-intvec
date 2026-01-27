// tests/variable/test_serde.rs

//! Integration tests for Serde functionality.

#![cfg(feature = "serde")]
use crate::common::helpers::{generate_random_signed_vec, generate_random_vec};
use compressed_intvec::prelude::*;
use dsi_bitstream::prelude::{BE, LE};

/// A helper macro to generate round-trip serialization tests for `VarVec<u64, E>`.
macro_rules! test_intvec_serde_roundtrip {
    ($test_name:ident, $endianness:ty, $input:expr, $k:expr, $codec_spec:expr) => {
        #[test]
        fn $test_name() {
            let input_data = &$input;
            // FIX: Use the new generic signature VarVec<T, E> by specifying u64.
            let original_intvec = VarVec::<u64, $endianness>::builder()
                .k($k)
                .codec($codec_spec)
                .build(input_data)
                .unwrap();

            // 1. Test with bincode (binary format)
            let encoded_bincode = bincode::serialize(&original_intvec).unwrap();
            // FIX: Specify the full generic type for deserialization.
            let decoded_bincode: VarVec<u64, $endianness> =
                bincode::deserialize(&encoded_bincode).unwrap();

            assert_eq!(
                original_intvec.iter().collect::<Vec<_>>(),
                decoded_bincode.iter().collect::<Vec<_>>(),
                "Bincode round-trip failed: vectors do not match"
            );
            assert_eq!(
                original_intvec.encoding(),
                decoded_bincode.encoding(),
                "Bincode round-trip failed: encodings do not match"
            );

            // 2. Test with serde_json (text format)
            let encoded_json = serde_json::to_string(&original_intvec).unwrap();
            let decoded_json: VarVec<u64, $endianness> =
                serde_json::from_str(&encoded_json).unwrap();

            assert_eq!(
                original_intvec.iter().collect::<Vec<_>>(),
                decoded_json.iter().collect::<Vec<_>>(),
                "JSON round-trip failed: vectors do not match"
            );
            assert_eq!(
                original_intvec.encoding(),
                decoded_json.encoding(),
                "JSON round-trip failed: encodings do not match"
            );
        }
    };
}

/// A helper macro to generate round-trip serialization tests for `VarVec<i64, E>`.
macro_rules! test_sintvec_serde_roundtrip {
    ($test_name:ident, $endianness:ty, $input:expr, $k:expr, $codec_spec:expr) => {
        #[test]
        fn $test_name() {
            let input_data = &$input;
            // FIX: Use the generic VarVec<i64, E> instead of the old SVarVec<E>.
            let original_sintvec = VarVec::<i64, $endianness>::builder()
                .k($k)
                .codec($codec_spec)
                .build(input_data)
                .unwrap();

            // 1. Test with bincode
            let encoded_bincode = bincode::serialize(&original_sintvec).unwrap();
            let decoded_bincode: VarVec<i64, $endianness> =
                bincode::deserialize(&encoded_bincode).unwrap();

            assert_eq!(
                original_sintvec.iter().collect::<Vec<_>>(),
                decoded_bincode.iter().collect::<Vec<_>>(),
                "Bincode round-trip failed: SVarVec vectors do not match"
            );
            assert_eq!(
                original_sintvec.encoding(),
                decoded_bincode.encoding(),
                "Bincode round-trip failed: SVarVec encodings do not match"
            );

            // 2. Test with serde_json
            let encoded_json = serde_json::to_string(&original_sintvec).unwrap();
            let decoded_json: VarVec<i64, $endianness> =
                serde_json::from_str(&encoded_json).unwrap();

            assert_eq!(
                original_sintvec.iter().collect::<Vec<_>>(),
                decoded_json.iter().collect::<Vec<_>>(),
                "JSON round-trip failed: SVarVec vectors do not match"
            );
            assert_eq!(
                original_sintvec.encoding(),
                decoded_json.encoding(),
                "JSON round-trip failed: SVarVec encodings do not match"
            );
        }
    };
}

// --- VarVec Test Suite ---
test_intvec_serde_roundtrip!(test_intvec_empty_le, LE, Vec::<u64>::new(), 32, Codec::Auto);
test_intvec_serde_roundtrip!(test_intvec_empty_be, BE, Vec::<u64>::new(), 32, Codec::Auto);
test_intvec_serde_roundtrip!(
    test_intvec_uniform_small_auto_le,
    LE,
    generate_random_vec(1000, 100),
    32,
    Codec::Auto
);
test_intvec_serde_roundtrip!(
    test_intvec_uniform_large_auto_be,
    BE,
    generate_random_vec(1000, 1_000_000),
    32,
    Codec::Auto
);
test_intvec_serde_roundtrip!(
    test_intvec_gamma_explicit_le,
    LE,
    generate_random_vec(500, 2000),
    16,
    Codec::Gamma
);
test_intvec_serde_roundtrip!(
    test_intvec_vbyte_le,
    LE,
    generate_random_vec(500, 5000),
    32,
    Codec::VByteLe
);
test_intvec_serde_roundtrip!(
    test_intvec_omega_be,
    BE,
    generate_random_vec(500, 5000),
    32,
    Codec::Omega
);
test_intvec_serde_roundtrip!(
    test_intvec_golomb_le,
    LE,
    generate_random_vec(500, 5000),
    32,
    Codec::Golomb { b: Some(10) }
);

// --- SVarVec Test Suite ---
test_sintvec_serde_roundtrip!(
    test_sintvec_empty_le,
    LE,
    Vec::<i64>::new(),
    16,
    // FIX: Auto codec is now supported for signed integers as well.
    Codec::Auto
);
test_sintvec_serde_roundtrip!(
    test_sintvec_mixed_values_be,
    BE,
    generate_random_signed_vec(1000, 1000),
    32,
    Codec::Delta
);
test_sintvec_serde_roundtrip!(
    test_sintvec_vbyte_be,
    BE,
    generate_random_signed_vec(1000, 10_000),
    32,
    Codec::VByteBe
);
