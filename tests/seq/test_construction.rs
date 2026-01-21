//! Integration tests for [`SeqVec`] construction patterns and builders.
//!
//! This test suite validates:
//! - [`SeqVecBuilder`]: Construction with custom codec specification
//! - [`SeqVecFromIterBuilder`]: Single-pass construction from iterators
//! - Error handling for invalid parameters
//! - Codec selection and validation

use compressed_intvec::seq::{SeqVec, SeqVecError, VariableCodecSpec};
use dsi_bitstream::prelude::LE;

// --- SeqVecBuilder Tests ---

#[test]
fn test_builder_with_auto_codec() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];

    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Auto)
        .build(&sequences);

    assert!(vec.is_ok());
}

#[test]
fn test_builder_with_gamma_codec() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];

    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences);

    assert!(vec.is_ok());
    let vec = vec.unwrap();
    assert_eq!(vec.num_sequences(), 3);
    assert_eq!(vec.decode_vec(0), Some(vec![1, 2, 3]));
}

#[test]
fn test_builder_with_delta_codec() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];

    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences);

    assert!(vec.is_ok());
    let vec = vec.unwrap();
    assert_eq!(vec.decode_vec(1), Some(vec![10, 20]));
}

#[test]
fn test_builder_with_zeta_codec() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20]];

    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Zeta { k: Some(3) })
        .build(&sequences);

    assert!(vec.is_ok());
    let vec = vec.unwrap();
    assert_eq!(vec.decode_vec(0), Some(vec![1, 2, 3]));
    assert_eq!(vec.decode_vec(1), Some(vec![10, 20]));
}

#[test]
fn test_builder_with_byte_codec() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200]];

    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::VByteLe)
        .build(&sequences);

    assert!(vec.is_ok());
    let vec = vec.unwrap();
    assert_eq!(vec.decode_vec(0), Some(vec![1, 2, 3]));
    assert_eq!(vec.decode_vec(2), Some(vec![100, 200]));
}

#[test]
fn test_builder_empty_sequences() {
    let sequences: Vec<Vec<u32>> = vec![vec![], vec![], vec![]];

    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences);

    assert!(vec.is_ok());
    let vec = vec.unwrap();
    assert_eq!(vec.num_sequences(), 3);
    for i in 0..3 {
        assert_eq!(vec.decode_vec(i), Some(vec![]));
    }
}

#[test]
fn test_builder_mixed_empty_non_empty() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2], vec![], vec![3, 4, 5], vec![], vec![6]];

    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences);

    assert!(vec.is_ok());
    let vec = vec.unwrap();
    assert_eq!(vec.num_sequences(), 5);
    assert_eq!(vec.decode_vec(0), Some(vec![1, 2]));
    assert_eq!(vec.decode_vec(1), Some(vec![]));
    assert_eq!(vec.decode_vec(2), Some(vec![3, 4, 5]));
    assert_eq!(vec.decode_vec(3), Some(vec![]));
    assert_eq!(vec.decode_vec(4), Some(vec![6]));
}

#[test]
fn test_builder_with_different_types() {
    // Test with u8
    {
        let sequences: Vec<Vec<u8>> = vec![vec![1, 2], vec![255]];
        let vec = SeqVec::builder()
            .codec(VariableCodecSpec::Gamma)
            .build(&sequences);
        assert!(vec.is_ok());
    }

    // Test with u64
    {
        let sequences: Vec<Vec<u64>> = vec![vec![1, 2, 3], vec![1000000000000]];
        let vec = SeqVec::builder()
            .codec(VariableCodecSpec::Gamma)
            .build(&sequences);
        assert!(vec.is_ok());
    }

    // Test with i32
    {
        let sequences: Vec<Vec<i32>> = vec![vec![-1, 2, -3], vec![100, -200]];
        let vec = SeqVec::builder()
            .codec(VariableCodecSpec::Gamma)
            .build(&sequences);
        assert!(vec.is_ok());
    }

    // Test with i64
    {
        let sequences: Vec<Vec<i64>> = vec![vec![-1, 2, -3], vec![100, -200]];
        let vec = SeqVec::builder()
            .codec(VariableCodecSpec::Gamma)
            .build(&sequences);
        assert!(vec.is_ok());
    }
}

#[test]
fn test_builder_store_lengths() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2], vec![], vec![3, 4, 5]];

    let vec_with_lengths = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .store_lengths(true)
        .build(&sequences)
        .unwrap();

    assert_eq!(vec_with_lengths.sequence_len(0), Some(2));
    assert_eq!(vec_with_lengths.sequence_len(1), Some(0));
    assert_eq!(vec_with_lengths.sequence_len(2), Some(3));
    assert_eq!(vec_with_lengths.sequence_len(3), None);

    let vec_without_lengths = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .store_lengths(false)
        .build(&sequences)
        .unwrap();

    assert_eq!(vec_without_lengths.sequence_len(0), None);
    assert_eq!(vec_without_lengths.sequence_len(2), None);
}

// --- SeqVecFromIterBuilder Tests ---

#[test]
fn test_from_iter_builder_gamma() {
    let sequences = vec![vec![1, 2, 3], vec![10, 20], vec![100]];

    let vec = SeqVec::from_iter_builder(sequences.iter().cloned())
        .codec(VariableCodecSpec::Gamma)
        .build();

    assert!(vec.is_ok());
    let vec = vec.unwrap();
    assert_eq!(vec.num_sequences(), 3);
    assert_eq!(vec.decode_vec(0), Some(vec![1, 2, 3]));
    assert_eq!(vec.decode_vec(1), Some(vec![10, 20]));
    assert_eq!(vec.decode_vec(2), Some(vec![100]));
}

#[test]
fn test_from_iter_builder_delta() {
    let sequences = vec![vec![1, 2, 3], vec![10, 20]];

    let vec = SeqVec::from_iter_builder(sequences.iter().cloned())
        .codec(VariableCodecSpec::Delta)
        .build();

    assert!(vec.is_ok());
    let vec = vec.unwrap();
    assert_eq!(vec.decode_vec(0), Some(vec![1, 2, 3]));
    assert_eq!(vec.decode_vec(1), Some(vec![10, 20]));
}

#[test]
fn test_from_iter_builder_with_empty_sequences() {
    let sequences = vec![vec![1, 2], vec![], vec![3, 4, 5]];

    let vec = SeqVec::from_iter_builder(sequences.iter().cloned())
        .codec(VariableCodecSpec::Gamma)
        .build();

    assert!(vec.is_ok());
    let vec = vec.unwrap();
    assert_eq!(vec.num_sequences(), 3);
    assert_eq!(vec.decode_vec(0), Some(vec![1, 2]));
    assert_eq!(vec.decode_vec(1), Some(vec![]));
    assert_eq!(vec.decode_vec(2), Some(vec![3, 4, 5]));
}

#[test]
fn test_from_iter_builder_rejects_auto_codec() {
    let sequences = vec![vec![1, 2, 3], vec![10, 20]];

    let result = SeqVec::from_iter_builder(sequences.iter().cloned())
        .codec(VariableCodecSpec::Auto)
        .build();

    assert!(matches!(result, Err(SeqVecError::InvalidParameters(_))));
}

#[test]
fn test_from_iter_builder_single_sequence() {
    let sequences = vec![vec![1, 2, 3, 4, 5]];

    let vec = SeqVec::from_iter_builder(sequences.iter().cloned())
        .codec(VariableCodecSpec::Gamma)
        .build();

    assert!(vec.is_ok());
    let vec = vec.unwrap();
    assert_eq!(vec.num_sequences(), 1);
    assert_eq!(vec.decode_vec(0), Some(vec![1, 2, 3, 4, 5]));
}

#[test]
fn test_from_iter_builder_empty() {
    let sequences: Vec<Vec<u32>> = vec![];

    let vec = SeqVec::from_iter_builder(sequences.iter().cloned())
        .codec(VariableCodecSpec::Gamma)
        .build();

    assert!(vec.is_ok());
    let vec = vec.unwrap();
    assert_eq!(vec.num_sequences(), 0);
    assert!(vec.is_empty());
}

// --- Error Handling Tests ---

#[test]
fn test_builder_invalid_auto_on_non_iterator() {
    // Auto codec on builder with slices is OK
    let sequences = vec![vec![1, 2, 3]];
    let result = SeqVec::builder()
        .codec(VariableCodecSpec::Auto)
        .build(&sequences);
    assert!(result.is_ok());
}

#[test]
fn test_from_iter_builder_codec_validation() {
    // Auto codec should be rejected
    let sequences = vec![vec![1, 2, 3]];
    let result = SeqVec::from_iter_builder(sequences.iter().cloned())
        .codec(VariableCodecSpec::Auto)
        .build();
    assert!(matches!(result, Err(SeqVecError::InvalidParameters(_))));

    // Non-Auto codecs should work
    let sequences = vec![vec![1, 2, 3]];
    let result = SeqVec::from_iter_builder(sequences.iter().cloned())
        .codec(VariableCodecSpec::Gamma)
        .build();
    assert!(result.is_ok());
}

// --- Construction Consistency Tests ---

#[test]
fn test_from_slices_and_builder_consistency() {
    let sequences = vec![vec![1, 2, 3], vec![10, 20], vec![100]];

    // Using from_slices (auto codec)
    let vec1 = SeqVec::from_slices(&sequences).unwrap();

    // Using builder with auto codec
    let vec2 = SeqVec::builder()
        .codec(VariableCodecSpec::Auto)
        .build(&sequences)
        .unwrap();

    // Should decode to same data
    assert_eq!(vec1.num_sequences(), vec2.num_sequences());
    for i in 0..sequences.len() {
        assert_eq!(vec1.decode_vec(i), vec2.decode_vec(i));
    }
}

#[test]
fn test_builder_determinism() {
    let sequences = vec![vec![1, 2, 3], vec![10, 20]];

    let vec1 = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences)
        .unwrap();

    let vec2 = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences)
        .unwrap();

    // Same codec should produce same encoding for same data
    for i in 0..sequences.len() {
        assert_eq!(vec1.decode_vec(i), vec2.decode_vec(i));
    }
}

#[test]
fn test_builder_large_sequences() {
    let large_seq: Vec<u32> = (0..10000).collect();
    let sequences = vec![large_seq.clone()];

    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences)
        .unwrap();

    assert_eq!(vec.decode_vec(0), Some(large_seq));
}

#[test]
fn test_from_iter_builder_large_sequences() {
    let large_seq: Vec<u32> = (0..10000).collect();
    let sequences = vec![large_seq.clone()];

    let vec = SeqVec::from_iter_builder(sequences.iter().cloned())
        .codec(VariableCodecSpec::Gamma)
        .build()
        .unwrap();

    assert_eq!(vec.decode_vec(0), Some(large_seq));
}

#[test]
fn test_build_with_auto_codec() {
    let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
    let vec: SeqVec<u32, LE, Vec<u64>> = SeqVec::builder()
        .codec(VariableCodecSpec::Auto)
        .build(sequences)
        .expect("Failed to build SeqVec with Auto codec");

    assert_eq!(vec.num_sequences(), 3);
}

#[test]
fn test_build_with_explicit_codec() {
    let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
    let vec: SeqVec<u32, LE, Vec<u64>> = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(sequences)
        .expect("Failed to build with Gamma codec");

    assert_eq!(vec.num_sequences(), 3);
}

#[test]
fn test_build_with_stored_lengths() {
    let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20]];
    let vec: SeqVec<u32, LE, Vec<u64>> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .store_lengths(true)
        .build(sequences)
        .expect("Failed to build with stored lengths");

    assert_eq!(vec.num_sequences(), 2);
}

#[test]
fn test_build_empty_sequences() {
    let sequences: &[&[u32]] = &[];
    let vec: SeqVec<u32, LE, Vec<u64>> = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(sequences)
        .expect("Failed to build empty SeqVec");

    assert_eq!(vec.num_sequences(), 0);
}

#[test]
fn test_from_iter_builder_with_gamma() {
    let sequences_iter = (0..5).map(|i| vec![i as u32; 3]);
    let vec: SeqVec<u32, LE, Vec<u64>> = SeqVec::from_iter_builder(sequences_iter)
        .codec(VariableCodecSpec::Gamma)
        .build()
        .expect("Failed to build from iterator");

    assert_eq!(vec.num_sequences(), 5);
}

#[test]
fn test_from_iter_builder_rejects_auto_codec() {
    let sequences_iter = (0..3).map(|i| vec![i as u32; 2]);
    let result: Result<SeqVec<u32, LE, Vec<u64>>, _> = SeqVec::from_iter_builder(sequences_iter)
        .codec(VariableCodecSpec::Auto)
        .build();

    assert!(
        result.is_err(),
        "Should reject Auto codec in iterator builder"
    );
}

#[test]
fn test_multiple_codecs() {
    let sequences: &[&[u32]] = &[&[5, 10, 15], &[20, 25]];

    // Test Gamma codec
    let vec_gamma: SeqVec<u32, LE, Vec<u64>> = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(sequences)
        .expect("Gamma codec failed");
    assert_eq!(vec_gamma.num_sequences(), 2);

    // Test Delta codec
    let vec_delta: SeqVec<u32, LE, Vec<u64>> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(sequences)
        .expect("Delta codec failed");
    assert_eq!(vec_delta.num_sequences(), 2);
}
