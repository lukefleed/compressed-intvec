//! Comprehensive integration tests for [`SeqVecReader`] functionality.
//!
//! This test suite validates:
//! - Basic random access via [`SeqVecReader::get`]
//! - Bounds checking and out-of-bounds behavior
//! - Convenience methods: `get_vec()`, `get_into()`
//! - Empty sequence handling
//! - Reader reusability across multiple accesses
//! - Different codec configurations
//! - Type-parameterized testing across integer types

use compressed_intvec::seq::{LESeqVec, SeqVec, VariableCodecSpec};
use dsi_bitstream::prelude::{BE, LE};
use dsi_bitstream::traits::Endianness;
use std::fmt::Debug;

/// Helper function to run basic reader access tests for a type.
fn run_reader_basic_tests_for_type<T, E>(sequences: &[Vec<T>], type_name: &str)
where
    T: compressed_intvec::variable::traits::Storable
        + Debug
        + PartialEq
        + Copy
        + Send
        + Sync
        + 'static,
    for<'a> compressed_intvec::seq::iter::SeqVecBitReader<'a, E>: dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
        + dsi_bitstream::prelude::CodesRead<E>
        + dsi_bitstream::prelude::BitSeek<Error = core::convert::Infallible>,
    E: Endianness + Debug,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<u64, Vec<u64>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = core::convert::Infallible>
            + dsi_bitstream::prelude::CodesWrite<E>,
{
    let context = |op: &str| {
        format!(
            "<{}> on {} in <{}>",
            type_name,
            op,
            std::any::type_name::<E>()
        )
    };

    let vec: SeqVec<T, E> = SeqVec::from_slices(&sequences)
        .unwrap_or_else(|e| panic!("Build failed: {} - {}", context("from_slices"), e));

    let reader = vec.reader();

    // Test random access in non-sequential order
    if sequences.len() >= 3 {
        let seq2: Vec<T> = reader.get(2).unwrap().collect();
        assert_eq!(
            &seq2,
            &sequences[2],
            "Reader get(2) mismatch {}",
            context("reader.get(2)")
        );

        let seq0: Vec<T> = reader.get(0).unwrap().collect();
        assert_eq!(
            &seq0,
            &sequences[0],
            "Reader get(0) mismatch {}",
            context("reader.get(0)")
        );

        let seq1: Vec<T> = reader.get(1).unwrap().collect();
        assert_eq!(
            &seq1,
            &sequences[1],
            "Reader get(1) mismatch {}",
            context("reader.get(1)")
        );
    }

    // Test sequential access
    for (i, expected_seq) in sequences.iter().enumerate() {
        let retrieved: Vec<T> = reader
            .get(i)
            .unwrap_or_else(|| panic!("Reader get({}) returned None {}", i, context("get")))
            .collect();
        assert_eq!(
            &retrieved,
            expected_seq,
            "Reader get({}) mismatch {}",
            i,
            context("sequential access")
        );
    }
}

#[test]
fn test_reader_basic_access_u32_le() {
    let sequences = vec![vec![1u32, 2, 3], vec![10, 20], vec![100, 200, 300, 400]];
    run_reader_basic_tests_for_type::<u32, LE>(&sequences, "u32");
}

#[test]
fn test_reader_basic_access_u64_be() {
    let sequences = vec![vec![1u64, 2, 3], vec![10, 20], vec![100, 200, 300, 400]];
    run_reader_basic_tests_for_type::<u64, BE>(&sequences, "u64");
}

/// Helper function to test out-of-bounds access for a type.
fn run_reader_bounds_tests_for_type<T, E>(sequences: &[Vec<T>], type_name: &str)
where
    T: compressed_intvec::variable::traits::Storable
        + Debug
        + PartialEq
        + Copy
        + Send
        + Sync
        + 'static,
    for<'a> compressed_intvec::seq::iter::SeqVecBitReader<'a, E>: dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
        + dsi_bitstream::prelude::CodesRead<E>
        + dsi_bitstream::prelude::BitSeek<Error = core::convert::Infallible>,
    E: Endianness + Debug,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<u64, Vec<u64>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = core::convert::Infallible>
            + dsi_bitstream::prelude::CodesWrite<E>,
{
    let context = |op: &str| {
        format!(
            "<{}> on {} in <{}>",
            type_name,
            op,
            std::any::type_name::<E>()
        )
    };

    let vec: SeqVec<T, E> = SeqVec::from_slices(&sequences)
        .unwrap_or_else(|e| panic!("Build failed: {} - {}", context("from_slices"), e));

    let reader = vec.reader();

    // Test valid access for all sequences
    for i in 0..sequences.len() {
        assert!(
            reader.get(i).is_some(),
            "Reader get({}) should be Some {}",
            i,
            context("valid access")
        );
    }

    // Test out-of-bounds access
    assert!(
        reader.get(sequences.len()).is_none(),
        "Reader get({}) should be None {}",
        sequences.len(),
        context("at boundary")
    );

    assert!(
        reader.get(sequences.len() + 100).is_none(),
        "Reader get({}) should be None {}",
        sequences.len() + 100,
        context("far out of bounds")
    );
}

#[test]
fn test_reader_out_of_bounds_u32() {
    let sequences = vec![vec![1u32, 2], vec![3, 4]];
    run_reader_bounds_tests_for_type::<u32, LE>(&sequences, "u32");
}

#[test]
fn test_reader_out_of_bounds_i64() {
    let sequences = vec![vec![-1i64, 0], vec![1, 2]];
    run_reader_bounds_tests_for_type::<i64, LE>(&sequences, "i64");
}

/// Helper function to test empty sequence handling for a type.
fn run_reader_empty_sequences_tests<T, E>(type_name: &str)
where
    T: compressed_intvec::variable::traits::Storable
        + Debug
        + PartialEq
        + Copy
        + Send
        + Sync
        + 'static
        + From<u8>,
    for<'a> compressed_intvec::seq::iter::SeqVecBitReader<'a, E>: dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
        + dsi_bitstream::prelude::CodesRead<E>
        + dsi_bitstream::prelude::BitSeek<Error = core::convert::Infallible>,
    E: Endianness + Debug,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<u64, Vec<u64>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = core::convert::Infallible>
            + dsi_bitstream::prelude::CodesWrite<E>,
{
    let context = |op: &str| {
        format!(
            "<{}> on {} in <{}>",
            type_name,
            op,
            std::any::type_name::<E>()
        )
    };

    let sequences = vec![
        vec![],
        vec![T::from(1), T::from(2)],
        vec![],
        vec![T::from(3)],
    ];

    let vec: SeqVec<T, E> = SeqVec::from_slices(&sequences)
        .unwrap_or_else(|e| panic!("Build failed: {} - {}", context("from_slices"), e));

    let reader = vec.reader();

    // Test empty sequence at index 0
    let seq0: Vec<T> = reader.get(0).unwrap().collect();
    assert!(
        seq0.is_empty(),
        "Reader get(0) should be empty {}",
        context("empty at 0")
    );

    // Test non-empty sequence
    let seq1: Vec<T> = reader.get(1).unwrap().collect();
    assert_eq!(
        &seq1,
        &sequences[1],
        "Reader get(1) mismatch {}",
        context("non-empty")
    );

    // Test empty sequence at index 2
    let seq2: Vec<T> = reader.get(2).unwrap().collect();
    assert!(
        seq2.is_empty(),
        "Reader get(2) should be empty {}",
        context("empty at 2")
    );

    // Test non-empty sequence
    let seq3: Vec<T> = reader.get(3).unwrap().collect();
    assert_eq!(
        &seq3,
        &sequences[3],
        "Reader get(3) mismatch {}",
        context("non-empty")
    );
}

#[test]
fn test_reader_empty_sequences_u32() {
    run_reader_empty_sequences_tests::<u32, LE>("u32");
}

#[test]
fn test_reader_empty_sequences_u16() {
    run_reader_empty_sequences_tests::<u16, BE>("u16");
}

#[test]
fn test_reader_get_vec_u64() {
    let sequences = vec![vec![10u64, 20, 30], vec![100, 200]];
    let vec: LESeqVec<u64> = SeqVec::from_slices(&sequences).unwrap();

    let reader = vec.reader();

    assert_eq!(
        reader.get_vec(0),
        Some(vec![10, 20, 30]),
        "Reader get_vec(0) mismatch"
    );
    assert_eq!(
        reader.get_vec(1),
        Some(vec![100, 200]),
        "Reader get_vec(1) mismatch"
    );
    assert_eq!(reader.get_vec(2), None, "Reader get_vec(2) should be None");
}

#[test]
fn test_reader_get_into_u32() {
    let sequences = vec![vec![1u32, 2, 3], vec![10, 20, 30, 40]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    let reader = vec.reader();
    let mut buffer = Vec::new();

    // First sequence
    let count = reader
        .get_into(0, &mut buffer)
        .expect("Reader get_into(0) should succeed");
    assert_eq!(count, 3, "Reader get_into(0) count mismatch");
    assert_eq!(buffer, vec![1, 2, 3], "Reader get_into(0) buffer mismatch");

    // Buffer is reused (cleared internally)
    let count = reader
        .get_into(1, &mut buffer)
        .expect("Reader get_into(1) should succeed");
    assert_eq!(count, 4, "Reader get_into(1) count mismatch");
    assert_eq!(
        buffer,
        vec![10, 20, 30, 40],
        "Reader get_into(1) buffer mismatch"
    );

    // Out of bounds
    assert!(
        reader.get_into(2, &mut buffer).is_none(),
        "Reader get_into(2) should be None"
    );
}

#[test]
fn test_reader_with_gamma_codec() {
    let sequences = vec![vec![1u32, 2, 3], vec![100, 200, 300]];

    let vec: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences)
        .expect("Build with Gamma codec should succeed");

    let reader = vec.reader();
    assert_eq!(
        reader.get_vec(0),
        Some(vec![1, 2, 3]),
        "Gamma codec: get_vec(0) mismatch"
    );
    assert_eq!(
        reader.get_vec(1),
        Some(vec![100, 200, 300]),
        "Gamma codec: get_vec(1) mismatch"
    );
}

#[test]
fn test_reader_with_delta_codec() {
    let sequences = vec![vec![1u32, 2, 3], vec![100, 200, 300]];

    let vec: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences)
        .expect("Build with Delta codec should succeed");

    let reader = vec.reader();
    assert_eq!(
        reader.get_vec(0),
        Some(vec![1, 2, 3]),
        "Delta codec: get_vec(0) mismatch"
    );
    assert_eq!(
        reader.get_vec(1),
        Some(vec![100, 200, 300]),
        "Delta codec: get_vec(1) mismatch"
    );
}

#[test]
fn test_reader_with_zeta_codec() {
    let sequences = vec![vec![1u64, 2, 3], vec![100, 200, 300]];

    let vec: LESeqVec<u64> = SeqVec::builder()
        .codec(VariableCodecSpec::Zeta { k: Some(3) })
        .build(&sequences)
        .expect("Build with Zeta codec should succeed");

    let reader = vec.reader();
    assert_eq!(
        reader.get_vec(0),
        Some(vec![1, 2, 3]),
        "Zeta codec: get_vec(0) mismatch"
    );
    assert_eq!(
        reader.get_vec(1),
        Some(vec![100, 200, 300]),
        "Zeta codec: get_vec(1) mismatch"
    );
}

#[test]
fn test_reader_signed_i32() {
    let sequences = vec![vec![-1i32, 0, 1], vec![-100, -50, 0, 50, 100]];
    let vec: LESeqVec<i32> =
        SeqVec::from_slices(&sequences).expect("Build with signed integers should succeed");

    let reader = vec.reader();

    let seq0: Vec<i32> = reader.get(0).expect("get(0) should succeed").collect();
    assert_eq!(seq0, vec![-1, 0, 1], "Signed i32: seq0 mismatch");

    let seq1: Vec<i32> = reader.get(1).expect("get(1) should succeed").collect();
    assert_eq!(
        seq1,
        vec![-100, -50, 0, 50, 100],
        "Signed i32: seq1 mismatch"
    );
}

#[test]
fn test_reader_signed_i64() {
    let sequences = vec![vec![-1000i64, -1, 0, 1, 1000]];
    let vec: LESeqVec<i64> =
        SeqVec::from_slices(&sequences).expect("Build with signed i64 should succeed");

    let reader = vec.reader();
    let seq: Vec<i64> = reader.get(0).expect("get(0) should succeed").collect();
    assert_eq!(seq, vec![-1000, -1, 0, 1, 1000], "Signed i64: seq mismatch");
}

#[test]
fn test_reader_reusability_same_sequence() {
    let sequences = vec![vec![1u32], vec![2], vec![3], vec![4], vec![5]];
    let vec: LESeqVec<u32> =
        SeqVec::from_slices(&sequences).expect("Build for reusability test should succeed");

    let reader = vec.reader();

    // Access the same sequence multiple times
    for iteration in 0..3 {
        let seq: Vec<u32> = reader
            .get(2)
            .unwrap_or_else(|| panic!("Iteration {}: get(2) should succeed", iteration))
            .collect();
        assert_eq!(
            seq,
            vec![3],
            "Iteration {}: reusability same sequence mismatch",
            iteration
        );
    }
}

#[test]
fn test_reader_reusability_different_sequences() {
    let sequences = vec![vec![1u32], vec![2], vec![3], vec![4], vec![5]];
    let vec: LESeqVec<u32> =
        SeqVec::from_slices(&sequences).expect("Build for reusability test should succeed");

    let reader = vec.reader();

    // Access different sequences
    for i in 0..5 {
        let seq: Vec<u32> = reader
            .get(i)
            .unwrap_or_else(|| panic!("get({}) should succeed", i))
            .collect();
        assert_eq!(
            seq,
            vec![(i + 1) as u32],
            "Reusability: sequence {} mismatch",
            i
        );
    }
}

#[test]
fn test_reader_large_sequences() {
    let large_seq: Vec<u64> = (0..1000).collect();
    let sequences: Vec<Vec<u64>> = vec![large_seq[..500].to_vec(), large_seq[500..].to_vec()];

    let vec: LESeqVec<u64> =
        SeqVec::from_slices(&sequences).expect("Build with large sequences should succeed");
    let reader = vec.reader();

    let seq0: Vec<u64> = reader.get(0).expect("get(0) should succeed").collect();
    assert_eq!(seq0.len(), 500, "Large sequence 0: length mismatch");
    assert_eq!(seq0[0], 0, "Large sequence 0: first element mismatch");
    assert_eq!(seq0[499], 499, "Large sequence 0: last element mismatch");

    let seq1: Vec<u64> = reader.get(1).expect("get(1) should succeed").collect();
    assert_eq!(seq1.len(), 500, "Large sequence 1: length mismatch");
    assert_eq!(seq1[0], 500, "Large sequence 1: first element mismatch");
    assert_eq!(seq1[499], 999, "Large sequence 1: last element mismatch");
}

#[test]
fn test_reader_get_unchecked_u32() {
    let sequences = vec![vec![10u32, 20], vec![30, 40, 50]];
    let vec: LESeqVec<u32> =
        SeqVec::from_slices(&sequences).expect("Build for get_unchecked test should succeed");

    let reader = vec.reader();

    // Safe usage within bounds
    let seq0: Vec<u32> = unsafe { reader.get_unchecked(0) }.collect();
    assert_eq!(seq0, vec![10, 20], "get_unchecked(0) mismatch");

    let seq1: Vec<u32> = unsafe { reader.get_unchecked(1) }.collect();
    assert_eq!(seq1, vec![30, 40, 50], "get_unchecked(1) mismatch");
}

#[test]
fn test_reader_single_sequence() {
    let sequences = vec![vec![1u32, 2, 3, 4, 5]];
    let vec: LESeqVec<u32> =
        SeqVec::from_slices(&sequences).expect("Build with single sequence should succeed");

    let reader = vec.reader();

    assert_eq!(
        reader.get_vec(0),
        Some(vec![1, 2, 3, 4, 5]),
        "Single sequence: get_vec(0) mismatch"
    );
    assert!(
        reader.get(1).is_none(),
        "Single sequence: get(1) should be None"
    );
}

#[test]
fn test_reader_empty_seqvec() {
    let sequences: Vec<Vec<u32>> = vec![];
    let vec: LESeqVec<u32> =
        SeqVec::from_slices(&sequences).expect("Build with empty sequences should succeed");

    let reader = vec.reader();

    assert!(
        reader.get(0).is_none(),
        "Empty SeqVec: get(0) should be None"
    );
    assert!(
        reader.get_vec(0).is_none(),
        "Empty SeqVec: get_vec(0) should be None"
    );
}

#[test]
fn test_reader_multiple_empty_sequences() {
    let sequences = vec![vec![] as Vec<u32>, vec![], vec![]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences)
        .expect("Build with multiple empty sequences should succeed");

    let reader = vec.reader();

    for i in 0..3 {
        let seq: Vec<u32> = reader
            .get(i)
            .unwrap_or_else(|| panic!("get({}) should succeed", i))
            .collect();
        assert!(
            seq.is_empty(),
            "Multiple empty sequences: seq {} should be empty",
            i
        );
    }
}
