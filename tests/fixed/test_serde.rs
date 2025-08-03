//! Integration tests for Serde functionality.

#[cfg(all(test, feature = "serde"))]
mod test_serde {
    use crate::common::helpers::{generate_random_signed_vec, generate_random_vec};
    use compressed_intvec::prelude::*;
    use dsi_bitstream::prelude::{BE, LE};

    /// A helper macro to generate round-trip serialization tests for `FixedVec`.
    macro_rules! test_fixedvec_serde_roundtrip {
        ($test_name:ident, $endianness:ty, $input:expr, $bit_width:expr) => {
            #[test]
            fn $test_name() {
                let input_data = &$input;
                let original_vec = FixedVec::<$endianness>::builder(input_data)
                    .bit_width($bit_width)
                    .build()
                    .unwrap();

                // 1. Test with bincode (binary format)
                let encoded_bincode = bincode::serialize(&original_vec).unwrap();
                let decoded_bincode: FixedVec<$endianness> =
                    bincode::deserialize(&encoded_bincode).unwrap();

                assert_eq!(
                    original_vec, decoded_bincode,
                    "Bincode round-trip failed: vectors do not match"
                );

                // 2. Test with serde_json (text format)
                let encoded_json = serde_json::to_string(&original_vec).unwrap();
                let decoded_json: FixedVec<$endianness> =
                    serde_json::from_str(&encoded_json).unwrap();

                assert_eq!(
                    original_vec, decoded_json,
                    "JSON round-trip failed: vectors do not match"
                );
            }
        };
    }

    /// A helper macro to generate round-trip serialization tests for `SFixedVec`.
    macro_rules! test_sfixedvec_serde_roundtrip {
        ($test_name:ident, $endianness:ty, $input:expr, $bit_width:expr) => {
            #[test]
            fn $test_name() {
                let input_data = &$input;
                let original_vec = SFixedVec::<$endianness>::builder(input_data)
                    .bit_width($bit_width)
                    .build()
                    .unwrap();

                // 1. Test with bincode
                let encoded_bincode = bincode::serialize(&original_vec).unwrap();
                let decoded_bincode: SFixedVec<$endianness> =
                    bincode::deserialize(&encoded_bincode).unwrap();

                assert_eq!(
                    original_vec, decoded_bincode,
                    "Bincode round-trip failed: vectors do not match"
                );

                // 2. Test with serde_json
                let encoded_json = serde_json::to_string(&original_vec).unwrap();
                let decoded_json: SFixedVec<$endianness> =
                    serde_json::from_str(&encoded_json).unwrap();

                assert_eq!(
                    original_vec, decoded_json,
                    "JSON round-trip failed: vectors do not match"
                );
            }
        };
    }

    // --- FixedVec Test Suite ---
    test_fixedvec_serde_roundtrip!(
        test_fixedvec_empty_le,
        LE,
        Vec::<u64>::new(),
        BitWidth::Explicit(8)
    );
    test_fixedvec_serde_roundtrip!(
        test_fixedvec_uniform_small_auto_le,
        LE,
        generate_random_vec(1000, 100),
        BitWidth::Minimal
    );
    test_fixedvec_serde_roundtrip!(
        test_fixedvec_uniform_large_explicit_be,
        BE,
        generate_random_vec(1000, 1_000_000),
        BitWidth::Explicit(20) // 1_000_000 fits in 20 bits
    );
    test_fixedvec_serde_roundtrip!(
        test_fixedvec_full_64_bits_le,
        LE,
        vec![u64::MAX],
        BitWidth::Explicit(64)
    );

    // --- SFixedVec Test Suite ---
    test_sfixedvec_serde_roundtrip!(
        test_sfixedvec_empty_le,
        LE,
        Vec::<i64>::new(),
        BitWidth::Explicit(8)
    );
    test_sfixedvec_serde_roundtrip!(
        test_sfixedvec_mixed_auto_be,
        BE,
        generate_random_signed_vec(1000, 1000),
        BitWidth::Minimal
    );
    test_sfixedvec_serde_roundtrip!(
        test_sfixedvec_mixed_explicit_le,
        LE,
        vec![-128, 0, 127], // Zigzag of -128 is 255. Fits in 8 bits.
        BitWidth::Explicit(8)
    );
}
