//! Comprehensive integration tests for iterator functionality in [`SeqVec`].
//!
//! This test suite validates:
//! - [`SeqIter`]: Forward iteration, FusedIterator behavior, sequence termination
//! - [`SeqVecIter`]: Forward/backward iteration, ExactSizeIterator, DoubleEndedIterator
//! - Empty sequences, single-element sequences, mixed iteration patterns

use compressed_intvec::seq::{LESeqVec, SEqVecIter, SeqVec};
use dsi_bitstream::prelude::{BE, LE};
use dsi_bitstream::traits::Endianness;
use std::fmt::Debug;

/// Helper function to run comprehensive iterator tests for a type.
fn run_iterator_tests_for_type<T, E>(sequences: &[Vec<T>], type_name: &str)
where
    T: compressed_intvec::variable::traits::Storable
        + Debug
        + PartialEq
        + Copy
        + Send
        + Sync
        + 'static,
    for<'a> compressed_intvec::seq::iter::SeqVecBitReader<'a, E>:
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

    // --- Test Case 1: Non-empty SeqVecIter ---
    if !sequences.is_empty() {
        // 1.1. Forward iteration with SeqVecIter
        let all_sequences: Vec<Vec<T>> = vec.iter().map(|seq| seq.collect()).collect();
        assert_eq!(
            &all_sequences, sequences,
            "SeqVecIter forward iteration mismatch {}",
            context("iter()")
        );

        // 1.2. Reverse iteration with DoubleEndedIterator
        if sequences.len() > 1 {
            let reversed_sequences: Vec<Vec<T>> =
                vec.iter().rev().map(|seq| seq.collect()).collect();
            let mut expected_reversed = sequences.clone();
            expected_reversed.reverse();
            assert_eq!(
                &reversed_sequences, &expected_reversed,
                "SeqVecIter reverse iteration mismatch {}",
                context("iter().rev()")
            );
        }

        // 1.3. Mixed forward and backward iteration
        if sequences.len() > 2 {
            let mut mixed_iter = vec.iter();
            let first = mixed_iter.next().map(|seq| seq.collect::<Vec<_>>());
            let last = mixed_iter.next_back().map(|seq| seq.collect::<Vec<_>>());

            assert_eq!(
                first,
                Some(sequences[0].clone()),
                "Mixed iteration: first element mismatch {}",
                context("iter().next()")
            );
            assert_eq!(
                last,
                Some(sequences[sequences.len() - 1].clone()),
                "Mixed iteration: last element mismatch {}",
                context("iter().next_back()")
            );
        }

        // 1.4. ExactSizeIterator
        assert_eq!(
            vec.iter().len(),
            sequences.len(),
            "ExactSizeIterator.len() mismatch {}",
            context("iter().len()")
        );

        // 1.5. FusedIterator behavior - after None, always None
        let mut iter = vec.iter();
        while iter.next().is_some() {}
        assert_eq!(
            iter.next(),
            None,
            "FusedIterator should continue returning None {}",
            context("iter() exhaustion")
        );
    }

    // --- Test Case 2: Individual SeqIter behavior ---
    for (seq_idx, expected_seq) in sequences.iter().enumerate() {
        if let Some(seq_iter) = vec.get(seq_idx) {
            // 2.1. Collect all elements
            let collected: Vec<T> = seq_iter.collect();
            assert_eq!(
                &collected, expected_seq,
                "SeqIter[{}] collection mismatch {}",
                seq_idx,
                context("get().collect()")
            );

            // 2.2. Iterator termination at sequence boundary
            if let Some(seq_iter) = vec.get(seq_idx) {
                let mut count = 0;
                for _ in seq_iter {
                    count += 1;
                }
                assert_eq!(
                    count, expected_seq.len(),
                    "SeqIter[{}] count mismatch {}",
                    seq_idx,
                    context("get().count()")
                );
            }

            // 2.3. FusedIterator - after exhaustion, always None
            if let Some(seq_iter) = vec.get(seq_idx) {
                let mut iter = seq_iter;
                while iter.next().is_some() {}
                assert_eq!(
                    iter.next(),
                    None,
                    "SeqIter[{}] should return None after exhaustion {}",
                    seq_idx,
                    context("get() exhaustion")
                );
            }
        }
    }

    // --- Test Case 3: Empty sequences ---
    for (seq_idx, expected_seq) in sequences.iter().enumerate() {
        if expected_seq.is_empty() {
            if let Some(seq_iter) = vec.get(seq_idx) {
                assert_eq!(
                    seq_iter.next(),
                    None,
                    "Empty SeqIter[{}] should return None immediately {}",
                    seq_idx,
                    context("get().next()")
                );
            }
        }
    }

    // --- Test Case 4: SeqVecIter exhaustion ---
    if !sequences.is_empty() {
        let mut iter = vec.iter();
        for _ in 0..sequences.len() {
            assert!(iter.next().is_some());
        }
        // After exhaustion, should always return None
        assert_eq!(iter.next(), None, "SeqVecIter exhaustion mismatch");
        assert_eq!(iter.next(), None, "SeqVecIter should fuse after exhaustion");
    }
}

// --- Macro for Type-Parameterized Iterator Testing ---

macro_rules! test_iterator_all_types {
    ($test_name:ident, $E:ty) => {
        #[test]
        fn $test_name() {
            // Test with u32
            {
                let sequences: Vec<Vec<u32>> = vec![
                    vec![1, 2, 3],
                    vec![10, 20],
                    vec![],
                    vec![100, 200, 300, 400, 500],
                ];
                run_iterator_tests_for_type::<u32, $E>(&sequences, stringify!(u32));
            }

            // Test with i32
            {
                let sequences: Vec<Vec<i32>> =
                    vec![vec![-1, 2, -3], vec![10, -20], vec![], vec![-100, 200]];
                run_iterator_tests_for_type::<i32, $E>(&sequences, stringify!(i32));
            }

            // Test with u64
            {
                let sequences: Vec<Vec<u64>> = vec![vec![1, 2, 3], vec![100, 200, 300]];
                run_iterator_tests_for_type::<u64, $E>(&sequences, stringify!(u64));
            }
        }
    };
}

test_iterator_all_types!(test_iterator_le, LE);
test_iterator_all_types!(test_iterator_be, BE);

// --- Specific Iterator Edge Cases ---

#[test]
fn test_seq_iter_single_element() {
    let sequences: Vec<Vec<u32>> = vec![vec![42]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    let seq_iter = vec.get(0).unwrap();
    assert_eq!(seq_iter.next(), Some(42));
    let mut iter_after = vec.get(0).unwrap();
    while iter_after.next().is_some() {}
    assert_eq!(iter_after.next(), None, "Should fuse after exhaustion");
}

#[test]
fn test_seqvec_iter_single_sequence() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4, 5]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    let mut iter = vec.iter();
    assert_eq!(iter.len(), 1);
    let collected: Vec<u32> = iter.next().unwrap().collect();
    assert_eq!(collected, vec![1, 2, 3, 4, 5]);
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_seqvec_iter_double_ended() {
    let sequences: Vec<Vec<u32>> =
        vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![9, 10]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    let mut iter = vec.iter();

    // Collect from both ends
    let from_front = iter.next().map(|s| s.collect::<Vec<_>>());
    let from_back = iter.next_back().map(|s| s.collect::<Vec<_>>());

    assert_eq!(from_front, Some(vec![1, 2]));
    assert_eq!(from_back, Some(vec![9, 10]));
    assert_eq!(iter.len(), 3); // 3 sequences left in the middle
}

#[test]
fn test_exact_size_iterator() {
    let sequences: Vec<Vec<u32>> = vec![
        vec![1],
        vec![2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9, 10],
    ];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    let mut iter = vec.iter();
    assert_eq!(iter.len(), 4);
    assert_eq!(iter.size_hint(), (4, Some(4)));

    iter.next();
    assert_eq!(iter.len(), 3);

    iter.next_back();
    assert_eq!(iter.len(), 2);

    iter.next();
    iter.next();
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_seqvec_iter_exhaustion_patterns() {
    let sequences: Vec<Vec<u32>> = vec![vec![1], vec![2], vec![3], vec![4]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    // Pattern 1: Exhaust from front
    {
        let mut iter = vec.iter();
        for _ in 0..4 {
            assert!(iter.next().is_some());
        }
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None); // Fused
    }

    // Pattern 2: Exhaust from back
    {
        let mut iter = vec.iter();
        for _ in 0..4 {
            assert!(iter.next_back().is_some());
        }
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next_back(), None); // Fused
    }

    // Pattern 3: Alternate front/back until exhaustion
    {
        let mut iter = vec.iter();
        assert!(iter.next().is_some());
        assert!(iter.next_back().is_some());
        assert!(iter.next().is_some());
        assert!(iter.next_back().is_some());
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }
}

#[test]
fn test_all_empty_sequences_iteration() {
    let sequences: Vec<Vec<u32>> = vec![vec![], vec![], vec![]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    let collected: Vec<Vec<u32>> = vec.iter().map(|seq| seq.collect()).collect();
    assert_eq!(collected.len(), 3);
    for seq in collected {
        assert!(seq.is_empty());
    }
}

#[test]
fn test_mixed_empty_non_empty_iteration() {
    let sequences: Vec<Vec<u32>> =
        vec![vec![1, 2], vec![], vec![3], vec![], vec![4, 5, 6]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    let collected: Vec<Vec<u32>> = vec.iter().map(|seq| seq.collect()).collect();
    assert_eq!(collected.len(), 5);
    assert_eq!(collected[0], vec![1, 2]);
    assert!(collected[1].is_empty());
    assert_eq!(collected[2], vec![3]);
    assert!(collected[3].is_empty());
    assert_eq!(collected[4], vec![4, 5, 6]);
}

#[test]
fn test_seq_iter_termination_accuracy() {
    // Verify that SeqIter terminates exactly at sequence boundaries
    let seq1 = vec![1, 2, 3];
    let seq2 = vec![10, 20, 30, 40];
    let sequences: Vec<Vec<u32>> = vec![seq1.clone(), seq2.clone()];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    // Get first sequence
    let collected1: Vec<u32> = vec.get(0).unwrap().collect();
    assert_eq!(collected1, seq1, "First sequence must terminate correctly");

    // Get second sequence
    let collected2: Vec<u32> = vec.get(1).unwrap().collect();
    assert_eq!(collected2, seq2, "Second sequence must not include first");
}
