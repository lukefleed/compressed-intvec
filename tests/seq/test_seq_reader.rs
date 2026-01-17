//! Comprehensive integration tests for [`SeqVecSeqReader`] functionality.
//!
//! This test suite validates:
//! - Basic sequential access via [`SeqVecSeqReader::decode_into`]
//! - API uniformity: `decode_vec()` and `decode_into()` return consistent results
//! - Bounds checking and out-of-bounds behavior
//! - Empty sequence handling
//! - Buffer reuse in `decode_into()`
//! - State tracking for sequential access patterns
//! - Correctness compared to [`SeqVecReader`]
//! - Different codec configurations
//! - Type-parameterized testing across integer types

use compressed_intvec::seq::{LESeqVec, SeqVec, VariableCodecSpec};
use dsi_bitstream::prelude::{BE, LE};
use dsi_bitstream::traits::Endianness;
use std::fmt::Debug;

/// Helper function to run basic seq_reader access tests for a type.
fn run_seq_reader_basic_tests_for_type<T, E>(sequences: &[Vec<T>], type_name: &str)
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
    for<'a> compressed_intvec::common::codec_reader::IntVecBitReader<'a, E>:
        dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
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

    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    // Test sequential access with decode_into - primary use case
    for (i, expected_seq) in sequences.iter().enumerate() {
        let count = seq_reader.decode_into(i, &mut buffer).unwrap_or_else(|| {
            panic!(
                "seq_reader.decode_into({}) returned None {}",
                i,
                context("decode_into")
            )
        });

        assert_eq!(
            count,
            expected_seq.len(),
            "seq_reader.decode_into({}) count mismatch {}",
            i,
            context("count")
        );

        assert_eq!(
            &buffer,
            expected_seq,
            "seq_reader.decode_into({}) content mismatch {}",
            i,
            context("content")
        );
    }

    // Test that SeqVec::get() returns consistent results with decode_into()
    for (i, expected_seq) in sequences.iter().enumerate() {
        let retrieved: Vec<T> = vec
            .get(i)
            .unwrap_or_else(|| panic!("vec.get({}) returned None {}", i, context("get")))
            .collect();

        assert_eq!(
            &retrieved,
            expected_seq,
            "vec.get({}) mismatch {}",
            i,
            context("get consistency")
        );
    }

    // Test that decode_vec() returns consistent results
    let mut seq_reader3 = vec.seq_reader();
    for (i, expected_seq) in sequences.iter().enumerate() {
        let retrieved = seq_reader3.decode_vec(i).unwrap_or_else(|| {
            panic!(
                "seq_reader.decode_vec({}) returned None {}",
                i,
                context("decode_vec")
            )
        });

        assert_eq!(
            &retrieved,
            expected_seq,
            "seq_reader.decode_vec({}) mismatch {}",
            i,
            context("decode_vec consistency")
        );
    }
}

#[test]
fn test_seq_reader_basic_access_u32_le() {
    let sequences = vec![vec![1u32, 2, 3], vec![10, 20], vec![100, 200, 300, 400]];
    run_seq_reader_basic_tests_for_type::<u32, LE>(&sequences, "u32");
}

#[test]
fn test_seq_reader_basic_access_u64_be() {
    let sequences = vec![vec![1u64, 2, 3], vec![10, 20], vec![100, 200, 300, 400]];
    run_seq_reader_basic_tests_for_type::<u64, BE>(&sequences, "u64");
}

#[test]
fn test_seq_reader_basic_access_i32_le() {
    let sequences = vec![vec![-10i32, 0, 10], vec![100, -100], vec![1, 2, 3, 4, 5]];
    run_seq_reader_basic_tests_for_type::<i32, LE>(&sequences, "i32");
}

#[test]
fn test_seq_reader_empty_sequences() {
    let sequences = vec![
        vec![1u32, 2, 3],
        vec![], // Empty sequence
        vec![10, 20],
        vec![], // Another empty sequence
        vec![100],
    ];

    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();
    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    // Test sequential access including empty sequences
    for (i, expected_seq) in sequences.iter().enumerate() {
        let count = seq_reader.decode_into(i, &mut buffer).unwrap();
        assert_eq!(
            count,
            expected_seq.len(),
            "Empty sequence handling failed at index {}",
            i
        );
        assert_eq!(&buffer, expected_seq, "Content mismatch at index {}", i);
    }
}

#[test]
fn test_seq_reader_out_of_bounds() {
    let sequences = vec![vec![1u32, 2, 3], vec![10, 20], vec![100]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();
    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    // Access within bounds
    assert!(seq_reader.decode_into(0, &mut buffer).is_some());
    assert!(seq_reader.decode_into(1, &mut buffer).is_some());
    assert!(seq_reader.decode_into(2, &mut buffer).is_some());

    // Access out of bounds
    assert!(seq_reader.decode_into(3, &mut buffer).is_none());
    assert!(seq_reader.decode_into(100, &mut buffer).is_none());

    // SeqVec::get() should also return None
    assert!(vec.get(3).is_none());
    assert!(vec.get(100).is_none());

    // decode_vec() should also return None
    assert!(seq_reader.decode_vec(3).is_none());
    assert!(seq_reader.decode_vec(100).is_none());
}

#[test]
fn test_seq_reader_buffer_reuse() {
    let sequences = vec![vec![1u32, 2, 3], vec![10, 20, 30, 40, 50], vec![100]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();
    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    // First access allocates buffer
    seq_reader.decode_into(0, &mut buffer).unwrap();
    assert_eq!(buffer, vec![1, 2, 3]);
    let capacity_after_first = buffer.capacity();

    // Second access clears and reuses buffer
    seq_reader.decode_into(1, &mut buffer).unwrap();
    assert_eq!(buffer, vec![10, 20, 30, 40, 50]);
    // Capacity should be >= previous capacity (may grow but not shrink)
    assert!(buffer.capacity() >= capacity_after_first);

    // Third access with smaller sequence
    seq_reader.decode_into(2, &mut buffer).unwrap();
    assert_eq!(buffer, vec![100]);
    // Buffer should have preserved its capacity
    assert!(buffer.capacity() >= capacity_after_first);
}

#[test]
fn test_seq_reader_random_access_pattern() {
    let sequences = vec![
        vec![1u32, 2, 3],
        vec![10, 20],
        vec![100, 200, 300],
        vec![1000],
        vec![10000, 20000],
    ];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();
    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    // Access in non-sequential order
    let access_order = [2, 4, 0, 3, 1];
    for &index in &access_order {
        seq_reader.decode_into(index, &mut buffer).unwrap();
        assert_eq!(
            &buffer, &sequences[index],
            "Random access failed at index {}",
            index
        );
    }
}

#[test]
fn test_seq_reader_consistency_with_seqvec_reader() {
    // Compare results between SeqVecSeqReader and SeqVecReader
    let sequences = vec![
        vec![1u32, 2, 3, 4, 5],
        vec![10, 20, 30],
        vec![],
        vec![100],
        vec![1000, 2000, 3000, 4000],
    ];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    let mut reader = vec.reader(); // Stateless reader
    let mut seq_reader = vec.seq_reader(); // Stateful reader
    let mut buffer = Vec::new();

    // Compare all sequences
    for i in 0..sequences.len() {
        // Get from stateless reader
        let from_reader: Vec<u32> = reader.decode_vec(i).unwrap();

        // Get from stateful reader via decode_into
        seq_reader.decode_into(i, &mut buffer).unwrap();

        assert_eq!(
            &buffer, &from_reader,
            "Inconsistency between SeqVecReader and SeqVecSeqReader at index {}",
            i
        );

        assert_eq!(
            &buffer, &sequences[i],
            "Inconsistency with original data at index {}",
            i
        );
    }
}

#[test]
fn test_seq_reader_with_delta_codec() {
    let sequences = vec![
        vec![100u64, 101, 102, 103, 104], // Sequential, good for Delta
        vec![200, 202, 204, 206],         // Small deltas
        vec![1000, 1001],                 // Small deltas
    ];

    let vec: SeqVec<u64, LE> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences)
        .unwrap();

    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    for (i, expected_seq) in sequences.iter().enumerate() {
        seq_reader.decode_into(i, &mut buffer).unwrap();
        assert_eq!(
            &buffer, expected_seq,
            "Delta codec: content mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_seq_reader_with_gamma_codec() {
    let sequences = vec![
        vec![1u32, 2, 3, 4, 5], // Small values, good for Gamma
        vec![10, 11, 12],
        vec![100],
    ];

    let vec: SeqVec<u32, LE> = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences)
        .unwrap();

    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    for (i, expected_seq) in sequences.iter().enumerate() {
        seq_reader.decode_into(i, &mut buffer).unwrap();
        assert_eq!(
            &buffer, expected_seq,
            "Gamma codec: content mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_seq_reader_single_sequence() {
    let sequences = vec![vec![1u32, 2, 3, 4, 5]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();
    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    seq_reader.decode_into(0, &mut buffer).unwrap();
    assert_eq!(buffer, vec![1, 2, 3, 4, 5]);

    // Out of bounds
    assert!(seq_reader.decode_into(1, &mut buffer).is_none());
}

#[test]
fn test_seq_reader_all_empty_sequences() {
    let sequences = vec![vec![], vec![], vec![]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();
    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    for i in 0..3 {
        let count = seq_reader.decode_into(i, &mut buffer).unwrap();
        assert_eq!(count, 0, "Empty sequence should have count 0");
        assert!(buffer.is_empty(), "Buffer should be empty");
    }
}

#[test]
fn test_seq_reader_forward_backward_access() {
    let sequences = vec![vec![1u32, 2], vec![3, 4], vec![5, 6], vec![7, 8]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();
    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    // Forward access
    seq_reader.decode_into(0, &mut buffer).unwrap();
    assert_eq!(buffer, vec![1, 2]);

    seq_reader.decode_into(1, &mut buffer).unwrap();
    assert_eq!(buffer, vec![3, 4]);

    // Backward access (should still work correctly)
    seq_reader.decode_into(0, &mut buffer).unwrap();
    assert_eq!(buffer, vec![1, 2]);

    // Jump forward
    seq_reader.decode_into(3, &mut buffer).unwrap();
    assert_eq!(buffer, vec![7, 8]);

    // Jump backward
    seq_reader.decode_into(2, &mut buffer).unwrap();
    assert_eq!(buffer, vec![5, 6]);
}

#[test]
fn test_seq_reader_large_values() {
    let sequences = vec![
        vec![u64::MAX - 2, u64::MAX - 1, u64::MAX],
        vec![u64::MAX / 2, u64::MAX / 2 + 1],
        vec![1, 2, u64::MAX],
    ];

    let vec: SeqVec<u64, LE> = SeqVec::from_slices(&sequences).unwrap();
    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    for (i, expected_seq) in sequences.iter().enumerate() {
        seq_reader.decode_into(i, &mut buffer).unwrap();
        assert_eq!(
            &buffer, expected_seq,
            "Large values: content mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_seq_reader_many_sequences() {
    // Create many small sequences to test scalability
    let sequences: Vec<Vec<u32>> = (0..1000).map(|i| vec![i, i + 1, i + 2]).collect();

    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();
    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    // Test every 100th sequence
    for i in (0..1000).step_by(100) {
        seq_reader.decode_into(i, &mut buffer).unwrap();
        assert_eq!(buffer, vec![i as u32, i as u32 + 1, i as u32 + 2]);
    }

    // Test last sequence
    seq_reader.decode_into(999, &mut buffer).unwrap();
    assert_eq!(buffer, vec![999, 1000, 1001]);
}

#[test]
fn test_seq_reader_api_uniformity() {
    // Verify that SeqVec::get(), decode_vec(), and decode_into() return the same data
    let sequences = vec![vec![1u32, 2, 3], vec![], vec![10, 20, 30, 40]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();
    let mut seq_reader = vec.seq_reader();
    let mut buffer = Vec::new();

    for i in 0..sequences.len() {
        // SeqVec::get() returns iterator
        let from_get: Vec<u32> = vec.get(i).unwrap().collect();

        // decode_vec() returns Vec
        let from_get_vec = seq_reader.decode_vec(i).unwrap();

        // decode_into() fills buffer
        seq_reader.decode_into(i, &mut buffer).unwrap();

        assert_eq!(
            from_get, from_get_vec,
            "get() vs decode_vec() mismatch at {}",
            i
        );
        assert_eq!(from_get, buffer, "get() vs decode_into() mismatch at {}", i);
        assert_eq!(
            from_get_vec, buffer,
            "decode_vec() vs decode_into() mismatch at {}",
            i
        );
        assert_eq!(
            buffer, sequences[i],
            "All methods vs original mismatch at {}",
            i
        );
    }
}
