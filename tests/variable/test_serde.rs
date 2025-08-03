//! Integration tests for Serde functionality.

#[cfg(all(test, feature = "serde"))]
mod test_serde {
    use crate::common::helpers::{generate_random_signed_vec, generate_random_vec};
    use compressed_intvec::prelude::*;
    use dsi_bitstream::prelude::{BE, LE};

    /// A helper macro to generate round-trip serialization tests for `IntVec`.
    macro_rules! test_intvec_serde_roundtrip {
        ($test_name:ident, $endianness:ty, $input:expr, $k:expr, $codec_spec:expr) => {
            #[test]
            fn $test_name() {
                let input_data = &$input;
                let original_intvec = IntVec::<$endianness>::builder(input_data)
                    .k($k)
                    .codec($codec_spec)
                    .build()
                    .unwrap();

                // 1. Test with bincode (binary format)
                let encoded_bincode = bincode::serialize(&original_intvec).unwrap();
                let decoded_bincode: IntVec<$endianness> =
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
                let decoded_json: IntVec<$endianness> =
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

    /// A helper macro to generate round-trip serialization tests for `SIntVec`.
    macro_rules! test_sintvec_serde_roundtrip {
        ($test_name:ident, $endianness:ty, $input:expr, $k:expr, $codec_spec:expr) => {
            #[test]
            fn $test_name() {
                let input_data = &$input;
                let original_sintvec = SIntVec::<$endianness>::builder(input_data)
                    .k($k)
                    .codec($codec_spec)
                    .build()
                    .unwrap();

                // 1. Test with bincode
                let encoded_bincode = bincode::serialize(&original_sintvec).unwrap();
                let decoded_bincode: SIntVec<$endianness> =
                    bincode::deserialize(&encoded_bincode).unwrap();

                assert_eq!(
                    original_sintvec.iter().collect::<Vec<_>>(),
                    decoded_bincode.iter().collect::<Vec<_>>(),
                    "Bincode round-trip failed: SIntVec vectors do not match"
                );
                assert_eq!(
                    original_sintvec.encoding(),
                    decoded_bincode.encoding(),
                    "Bincode round-trip failed: SIntVec encodings do not match"
                );

                // 2. Test with serde_json
                let encoded_json = serde_json::to_string(&original_sintvec).unwrap();
                let decoded_json: SIntVec<$endianness> =
                    serde_json::from_str(&encoded_json).unwrap();

                assert_eq!(
                    original_sintvec.iter().collect::<Vec<_>>(),
                    decoded_json.iter().collect::<Vec<_>>(),
                    "JSON round-trip failed: SIntVec vectors do not match"
                );
                assert_eq!(
                    original_sintvec.encoding(),
                    decoded_json.encoding(),
                    "JSON round-trip failed: SIntVec encodings do not match"
                );
            }
        };
    }

    // --- IntVec Test Suite ---
    test_intvec_serde_roundtrip!(
        test_intvec_empty_le,
        LE,
        Vec::<u64>::new(),
        32,
        VariableCodecSpec::Auto
    );
    test_intvec_serde_roundtrip!(
        test_intvec_empty_be,
        BE,
        Vec::<u64>::new(),
        32,
        VariableCodecSpec::Auto
    );
    test_intvec_serde_roundtrip!(
        test_intvec_uniform_small_auto_le,
        LE,
        generate_random_vec(1000, 100),
        32,
        VariableCodecSpec::Auto
    );
    test_intvec_serde_roundtrip!(
        test_intvec_uniform_large_auto_be,
        BE,
        generate_random_vec(1000, 1_000_000),
        32,
        VariableCodecSpec::Auto
    );
    test_intvec_serde_roundtrip!(
        test_intvec_gamma_explicit_le,
        LE,
        generate_random_vec(500, 2000),
        16,
        VariableCodecSpec::Gamma
    );
    test_intvec_serde_roundtrip!(
        test_intvec_vbyte_le,
        LE,
        generate_random_vec(500, 5000),
        32,
        VariableCodecSpec::VByteLe
    );
    test_intvec_serde_roundtrip!(
        test_intvec_omega_be,
        BE,
        generate_random_vec(500, 5000),
        32,
        VariableCodecSpec::Omega
    );
    test_intvec_serde_roundtrip!(
        test_intvec_golomb_le,
        LE,
        generate_random_vec(500, 5000),
        32,
        VariableCodecSpec::Golomb { b: Some(10) }
    );

    // --- SIntVec Test Suite ---
    test_sintvec_serde_roundtrip!(
        test_sintvec_empty_le,
        LE,
        Vec::<i64>::new(),
        16,
        VariableCodecSpec::Gamma // SIntVec requires a specified codec
    );
    test_sintvec_serde_roundtrip!(
        test_sintvec_mixed_values_be,
        BE,
        generate_random_signed_vec(1000, 1000),
        32,
        VariableCodecSpec::Delta
    );
    test_sintvec_serde_roundtrip!(
        test_sintvec_vbyte_be,
        BE,
        generate_random_signed_vec(1000, 10_000),
        32,
        VariableCodecSpec::VByteBe
    );
}
