//! Comprehensive integration tests for core [`SeqVec`] functionality.
//!
//! This test suite validates:
//! - Construction via [`SeqVec::from_slices`]
//! - Query methods: `num_sequences()`, `is_empty()`, `encoding()`, bit offset queries
//! - Access methods: `get()`, `get_vec()`, `get_into()`
//! - Type-parameterized testing across all integer types and endianness

use compressed_intvec::seq::{LESeqVec, SeqVec};
use dsi_bitstream::prelude::{BE, LE};
use dsi_bitstream::traits::Endianness;
use num_traits::AsPrimitive;
use std::fmt::Debug;

// Import helper functions from the common module.
#[path = "../common/mod.rs"]
mod common;
use common::helpers::{generate_random_signed_vec, generate_random_vec};

/// Central test function called by the macro for each type combination.
fn run_test_for_type<T, E>(sequences: &[Vec<T>], type_name: &str)
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

    // Build the SeqVec
    let vec: SeqVec<T, E> = SeqVec::from_slices(&sequences)
        .unwrap_or_else(|e| panic!("Build failed: {} - {}", context("from_slices"), e));

    // Test num_sequences()
    assert_eq!(
        vec.num_sequences(),
        sequences.len(),
        "num_sequences mismatch {}",
        context("num_sequences()")
    );

    // Test is_empty()
    let is_empty_expected = sequences.is_empty();
    assert_eq!(
        vec.is_empty(),
        is_empty_expected,
        "is_empty mismatch {}",
        context("is_empty()")
    );

    // Test encoding is set
    let _encoding = vec.encoding();
    // Just verify it doesn't panic

    // Test as_limbs returns data
    let limbs = vec.as_limbs();
    assert!(
        !limbs.is_empty() || sequences.is_empty(),
        "as_limbs should not be empty"
    );

    // Test total_bits
    let total_bits = vec.total_bits();
    assert!(
        total_bits > 0 || sequences.is_empty(),
        "total_bits should be > 0 or empty"
    );

    // Test bit_offsets_ref
    let offsets_ref = vec.bit_offsets_ref();
    assert_eq!(
        offsets_ref.len(),
        sequences.len() + 1,
        "bit_offsets should have N+1 entries {}",
        context("bit_offsets_ref()")
    );

    // Test sequence_start_bit for each sequence
    for i in 0..sequences.len() {
        let start_bit = vec.sequence_start_bit(i);
        assert!(
            start_bit.is_some(),
            "sequence_start_bit({}) should be Some {}",
            i,
            context("sequence_start_bit()")
        );
    }

    // Test sequence_start_bit out of bounds
    assert_eq!(
        vec.sequence_start_bit(sequences.len()),
        None,
        "sequence_start_bit out of bounds {}",
        context("sequence_start_bit()")
    );

    // Test unsafe sequence_start_bit_unchecked
    for i in 0..sequences.len() {
        let start_safe = vec.sequence_start_bit(i).unwrap();
        let start_unsafe = unsafe { vec.sequence_start_bit_unchecked(i) };
        assert_eq!(
            start_safe,
            start_unsafe,
            "sequence_start_bit_unchecked({}) mismatch {}",
            i,
            context("sequence_start_bit_unchecked()")
        );
    }

    // Test unsafe sequence_end_bit_unchecked
    for i in 0..sequences.len() {
        let end_bit = unsafe { vec.sequence_end_bit_unchecked(i) };
        let next_start = if i + 1 < sequences.len() {
            vec.sequence_start_bit(i + 1).unwrap()
        } else {
            vec.total_bits()
        };
        assert_eq!(
            end_bit,
            next_start,
            "sequence_end_bit_unchecked({}) should equal next sequence start {}",
            i,
            context("sequence_end_bit_unchecked()")
        );
    }

    // Test get() for each sequence
    for i in 0..sequences.len() {
        let seq_iter = vec.get(i);
        assert!(
            seq_iter.is_some(),
            "get({}) should return Some {}",
            i,
            context("get()")
        );

        let collected: Vec<T> = seq_iter.unwrap().collect();
        assert_eq!(
            &collected,
            &sequences[i],
            "get({}) content mismatch {}",
            i,
            context("get()")
        );
    }

    // Test get() out of bounds
    assert_eq!(
        vec.get(sequences.len()),
        None,
        "get(out_of_bounds) should be None {}",
        context("get()")
    );

    // Test unsafe get_unchecked()
    for i in 0..sequences.len() {
        let collected_safe: Vec<T> = vec.get(i).unwrap().collect();
        let collected_unsafe: Vec<T> = unsafe { vec.get_unchecked(i) }.collect();
        assert_eq!(
            collected_safe,
            collected_unsafe,
            "get_unchecked({}) mismatch {}",
            i,
            context("get_unchecked()")
        );
    }

    // Test get_vec()
    for i in 0..sequences.len() {
        let vec_result = vec.get_vec(i);
        assert_eq!(
            vec_result,
            Some(sequences[i].clone()),
            "get_vec({}) mismatch {}",
            i,
            context("get_vec()")
        );
    }

    // Test get_vec() out of bounds
    assert_eq!(
        vec.get_vec(sequences.len()),
        None,
        "get_vec(out_of_bounds) should be None {}",
        context("get_vec()")
    );

    // Test get_into() with buffer reuse
    if !sequences.is_empty() {
        let mut buf = Vec::new();

        for i in 0..sequences.len() {
            let len_result = vec.get_into(i, &mut buf);
            assert_eq!(
                len_result,
                Some(sequences[i].len()),
                "get_into({}) returned incorrect length {}",
                i,
                context("get_into()")
            );
            assert_eq!(
                &buf,
                &sequences[i],
                "get_into({}) content mismatch {}",
                i,
                context("get_into()")
            );
        }
    }

    // Test get_into() out of bounds
    let mut buf = Vec::new();
    assert_eq!(
        vec.get_into(sequences.len(), &mut buf),
        None,
        "get_into(out_of_bounds) should be None {}",
        context("get_into()")
    );
}

// --- Macro for Type-Parameterized Testing ---

/// Macro to test all combinations of integer types and endianness for SeqVec.
macro_rules! test_all_types {
    ($test_name:ident, $E:ty) => {
        #[test]
        fn $test_name() {
            // Unsigned types
            {
                let data: Vec<Vec<u8>> =
                    vec![vec![1, 2, 3], vec![10, 20], vec![], vec![100, 200, 300]];
                run_test_for_type::<u8, $E>(&data, stringify!(u8));
            }

            {
                let data: Vec<Vec<u16>> =
                    vec![vec![1000, 2000, 3000], vec![100, 200], vec![], vec![65535]];
                run_test_for_type::<u16, $E>(&data, stringify!(u16));
            }

            {
                let data: Vec<Vec<u32>> = vec![
                    vec![1, 2, 3],
                    vec![100, 200, 300, 400],
                    vec![],
                    vec![1000000],
                ];
                run_test_for_type::<u32, $E>(&data, stringify!(u32));
            }

            {
                let data: Vec<Vec<u64>> = vec![
                    vec![1, 2, 3],
                    vec![100, 200, 300],
                    vec![],
                    vec![9999999999999],
                ];
                run_test_for_type::<u64, $E>(&data, stringify!(u64));
            }

            // Signed types
            {
                let data: Vec<Vec<i8>> =
                    vec![vec![-1, 2, -3], vec![10, -20], vec![], vec![-100, 127]];
                run_test_for_type::<i8, $E>(&data, stringify!(i8));
            }

            {
                let data: Vec<Vec<i16>> =
                    vec![vec![-1000, 2000, -3000], vec![-100], vec![], vec![32767]];
                run_test_for_type::<i16, $E>(&data, stringify!(i16));
            }

            {
                let data: Vec<Vec<i32>> = vec![
                    vec![-1, 2, -3],
                    vec![-100, 200, -300],
                    vec![],
                    vec![2000000000],
                ];
                run_test_for_type::<i32, $E>(&data, stringify!(i32));
            }

            {
                let data: Vec<Vec<i64>> = vec![
                    vec![-1, 2, -3],
                    vec![-100, 200, -300],
                    vec![],
                    vec![-9999999999999],
                ];
                run_test_for_type::<i64, $E>(&data, stringify!(i64));
            }
        }
    };
}

// Generate tests for both endianness
test_all_types!(test_le_all_types, LE);
test_all_types!(test_be_all_types, BE);

// --- Edge Case Tests ---

#[test]
fn test_empty_seqvec() {
    let empty_sequences: Vec<Vec<u32>> = vec![];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&empty_sequences).unwrap();

    assert_eq!(vec.num_sequences(), 0);
    assert!(vec.is_empty());
    assert_eq!(vec.get(0), None);
    assert_eq!(vec.get_vec(0), None);

    let collected: Vec<_> = vec.iter().collect();
    assert!(collected.is_empty());
}

#[test]
fn test_single_sequence() {
    let sequences: &[&[u32]] = &[&[42]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    assert_eq!(vec.num_sequences(), 1);
    assert!(!vec.is_empty());
    assert_eq!(vec.get_vec(0), Some(vec![42]));
}

#[test]
fn test_all_empty_sequences() {
    let sequences: Vec<Vec<u32>> = vec![vec![], vec![], vec![]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    assert_eq!(vec.num_sequences(), 3);

    for i in 0..3 {
        let seq = vec.get_vec(i).unwrap();
        assert!(seq.is_empty(), "Sequence {} should be empty", i);
    }
}

#[test]
fn test_mixed_empty_non_empty() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2], vec![], vec![3, 4, 5], vec![], vec![6]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    assert_eq!(vec.num_sequences(), 5);
    assert_eq!(vec.get_vec(0), Some(vec![1, 2]));
    assert_eq!(vec.get_vec(1), Some(vec![]));
    assert_eq!(vec.get_vec(2), Some(vec![3, 4, 5]));
    assert_eq!(vec.get_vec(3), Some(vec![]));
    assert_eq!(vec.get_vec(4), Some(vec![6]));
}

#[test]
fn test_single_element_sequences() {
    let sequences: Vec<Vec<u32>> = vec![vec![1], vec![2], vec![3], vec![4]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    assert_eq!(vec.num_sequences(), 4);

    for i in 0..4 {
        let seq = vec.get_vec(i).unwrap();
        assert_eq!(seq.len(), 1);
        assert_eq!(seq[0], (i + 1) as u32);
    }
}

#[test]
fn test_large_sequences() {
    let seq1: Vec<u32> = (0..1000).collect();
    let seq2: Vec<u32> = (1000..2000).collect();
    let sequences: Vec<Vec<u32>> = vec![seq1.clone(), seq2.clone()];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    assert_eq!(vec.get_vec(0), Some(seq1));
    assert_eq!(vec.get_vec(1), Some(seq2));
}

#[test]
fn test_identical_values() {
    let sequences: Vec<Vec<u32>> = vec![vec![42; 100], vec![99; 50], vec![1; 200]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    assert_eq!(vec.get_vec(0), Some(vec![42; 100]));
    assert_eq!(vec.get_vec(1), Some(vec![99; 50]));
    assert_eq!(vec.get_vec(2), Some(vec![1; 200]));
}

#[test]
fn test_all_zeros() {
    let sequences: Vec<Vec<u32>> = vec![vec![0; 50], vec![0; 100]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    assert_eq!(vec.get_vec(0), Some(vec![0; 50]));
    assert_eq!(vec.get_vec(1), Some(vec![0; 100]));
}

#[test]
fn test_random_data() {
    let seq1 = generate_random_vec::<u32>(50);
    let seq2 = generate_random_vec::<u32>(100);
    let seq3 = generate_random_vec::<u32>(30);
    let sequences: Vec<Vec<u32>> = vec![seq1.clone(), seq2.clone(), seq3.clone()];

    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    assert_eq!(vec.get_vec(0), Some(seq1));
    assert_eq!(vec.get_vec(1), Some(seq2));
    assert_eq!(vec.get_vec(2), Some(seq3));
}

#[test]
fn test_signed_random_data() {
    let seq1 = generate_random_signed_vec::<i32>(50);
    let seq2 = generate_random_signed_vec::<i32>(100);
    let sequences: Vec<Vec<i32>> = vec![seq1.clone(), seq2.clone()];

    let vec: LESeqVec<i32> = SeqVec::from_slices(&sequences).unwrap();

    assert_eq!(vec.get_vec(0), Some(seq1));
    assert_eq!(vec.get_vec(1), Some(seq2));
}
