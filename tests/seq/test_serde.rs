//! Integration tests for Serde functionality in [`SeqVec`].

#![cfg(feature = "serde")]

use compressed_intvec::seq::SeqVec;
use dsi_bitstream::prelude::{BE, LE};

/// Macro to generate round-trip serialization tests for `SeqVec<T, E>`.
macro_rules! test_seqvec_serde_roundtrip {
    ($test_name:ident, $type:ty, $endianness:ty, $input:expr) => {
        #[test]
        fn $test_name() {
            let sequences = $input;

            // Build the original vector
            let original_vec: SeqVec<$type, $endianness> =
                SeqVec::from_slices(&sequences).expect("Failed to build SeqVec");

            // 1. Test with bincode (binary format)
            let encoded_bincode =
                bincode::serialize(&original_vec).expect("Bincode serialization failed");
            let decoded_bincode: SeqVec<$type, $endianness> =
                bincode::deserialize(&encoded_bincode).expect("Bincode deserialization failed");

            // Verify content equality
            let original_collected: Vec<Vec<$type>> =
                original_vec.iter().map(|s| s.collect()).collect();
            let decoded_collected: Vec<Vec<$type>> =
                decoded_bincode.iter().map(|s| s.collect()).collect();

            assert_eq!(
                original_collected, decoded_collected,
                "Bincode round-trip content mismatch"
            );

            // Verify structural equality (encoding, length, etc)
            assert_eq!(
                original_vec.len(),
                decoded_bincode.len(),
                "Bincode round-trip length mismatch"
            );
            assert_eq!(
                original_vec.encoding(),
                decoded_bincode.encoding(),
                "Bincode round-trip encoding mismatch"
            );

            // 2. Test with serde_json (text format)
            let encoded_json =
                serde_json::to_string(&original_vec).expect("JSON serialization failed");
            let decoded_json: SeqVec<$type, $endianness> =
                serde_json::from_str(&encoded_json).expect("JSON deserialization failed");

            let decoded_json_collected: Vec<Vec<$type>> =
                decoded_json.iter().map(|s| s.collect()).collect();

            assert_eq!(
                original_collected, decoded_json_collected,
                "JSON round-trip content mismatch"
            );
            assert_eq!(
                original_vec.encoding(),
                decoded_json.encoding(),
                "JSON round-trip encoding mismatch"
            );
        }
    };
}

// --- Test Data Generators ---

fn sequences_u32() -> Vec<Vec<u32>> {
    vec![vec![1, 2, 3], vec![], vec![100, 200, 300, 400], vec![5]]
}

fn sequences_i64() -> Vec<Vec<i64>> {
    vec![vec![-1, 2, -3], vec![0], vec![-100, -200, 300, 400]]
}

fn sequences_u16() -> Vec<Vec<u16>> {
    vec![vec![10, 20], vec![100, 200, 300], vec![]]
}

fn sequences_i32() -> Vec<Vec<i32>> {
    vec![vec![-50, 50], vec![-1, -2, -3, 4, 5]]
}

// --- Macro Invocations for LE ---

test_seqvec_serde_roundtrip!(test_serde_u32_le, u32, LE, sequences_u32());
test_seqvec_serde_roundtrip!(test_serde_i64_le, i64, LE, sequences_i64());
test_seqvec_serde_roundtrip!(test_serde_u16_le, u16, LE, sequences_u16());
test_seqvec_serde_roundtrip!(test_serde_i32_le, i32, LE, sequences_i32());

// --- Macro Invocations for BE ---

test_seqvec_serde_roundtrip!(test_serde_u32_be, u32, BE, sequences_u32());
test_seqvec_serde_roundtrip!(test_serde_i64_be, i64, BE, sequences_i64());
test_seqvec_serde_roundtrip!(test_serde_u16_be, u16, BE, sequences_u16());
test_seqvec_serde_roundtrip!(test_serde_i32_be, i32, BE, sequences_i32());
