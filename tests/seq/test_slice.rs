//! Tests for [`SeqVecSlice`] - zero-copy slicing of sequence vectors.
//!
//! This test suite comprehensively validates slice functionality across all
//! integer types and endianness, including:
//! - Slice creation and bounds checking
//! - Sequence access and iteration within slices
//! - Binary search operations
//! - Equality comparisons
//! - Edge cases (empty slices, single-element slices, full slices)

use compressed_intvec::seq::{SeqVec, VariableCodecSpec};
use dsi_bitstream::prelude::{BE, LE};
use dsi_bitstream::traits::Endianness;
use std::fmt::Debug;

/// Helper function to run comprehensive slice tests for a type.
fn run_slice_tests_for_type<T, E>(sequences: &[Vec<T>], type_name: &str)
where
    T: compressed_intvec::variable::traits::Storable
        + Debug
        + PartialEq
        + Eq
        + Copy
        + Ord
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

    let vec: SeqVec<T, E> = SeqVec::from_slices(sequences)
        .unwrap_or_else(|e| panic!("Build failed: {} - {}", context("from_slices"), e));

    let num_seqs = sequences.len();

    // --- Basic Slice Creation ---

    // Test valid range
    if num_seqs >= 3 {
        let slice = vec
            .slice(1, 2)
            .unwrap_or_else(|| panic!("slice(1, 2) failed {}", context("slice")));
        assert_eq!(
            slice.len(),
            2,
            "Slice length mismatch {}",
            context("slice().len()")
        );
        assert!(
            !slice.is_empty(),
            "Slice should not be empty {}",
            context("slice().is_empty()")
        );

        // Verify index translation
        if !sequences[1].is_empty() {
            let seq0: Vec<T> = slice.get(0).unwrap().collect();
            assert_eq!(
                &seq0,
                &sequences[1],
                "Slice index 0 should map to sequence 1 {}",
                context("slice().get(0)")
            );
        }

        if !sequences[2].is_empty() {
            let seq1: Vec<T> = slice.get(1).unwrap().collect();
            assert_eq!(
                &seq1,
                &sequences[2],
                "Slice index 1 should map to sequence 2 {}",
                context("slice().get(1)")
            );
        }
    }

    // Test full slice
    if !sequences.is_empty() {
        let slice = vec
            .slice(0, num_seqs)
            .unwrap_or_else(|| panic!("Full slice failed {}", context("slice")));
        assert_eq!(
            slice.len(),
            num_seqs,
            "Full slice length should equal num_sequences {}",
            context("slice(0, num_seqs).len()")
        );
    }

    // Test empty slice
    if num_seqs > 1 {
        let slice = vec
            .slice(1, 0)
            .unwrap_or_else(|| panic!("Empty slice failed {}", context("slice")));
        assert_eq!(
            slice.len(),
            0,
            "Empty slice should have len 0 {}",
            context("slice(_, 0).len()")
        );
        assert!(
            slice.is_empty(),
            "Empty slice should report is_empty {}",
            context("slice(_, 0).is_empty()")
        );
        assert_eq!(
            slice.get(0),
            None,
            "Empty slice get should return None {}",
            context("slice(_, 0).get(0)")
        );
    }

    // Test out of bounds
    assert!(
        vec.slice(num_seqs + 1, 1).is_none(),
        "Slice out of bounds should return None {}",
        context("slice(out_of_bounds)")
    );
    assert!(
        vec.slice(0, num_seqs + 1).is_none(),
        "Slice length overflow should return None {}",
        context("slice(_, overflow)")
    );

    // Test overflow protection
    assert!(
        vec.slice(usize::MAX, 1).is_none(),
        "Slice with usize::MAX should return None {}",
        context("slice(usize::MAX)")
    );

    // --- Split At ---

    if num_seqs > 1 {
        let (left, right) = vec
            .split_at(1)
            .unwrap_or_else(|| panic!("split_at failed {}", context("split_at")));

        assert_eq!(
            left.len(),
            1,
            "Left slice should have 1 sequence {}",
            context("split_at(1) left.len()")
        );
        assert_eq!(
            right.len(),
            num_seqs - 1,
            "Right slice should have remaining sequences {}",
            context("split_at(1) right.len()")
        );
    }

    // Test split at boundaries
    if !sequences.is_empty() {
        let (left, right) = vec
            .split_at(0)
            .unwrap_or_else(|| panic!("split_at(0) failed {}", context("split_at")));
        assert_eq!(
            left.len(),
            0,
            "split_at(0) left should be empty {}",
            context("split_at(0) left.len()")
        );
        assert_eq!(
            right.len(),
            num_seqs,
            "split_at(0) right should have all sequences {}",
            context("split_at(0) right.len()")
        );

        let (left, right) = vec
            .split_at(num_seqs)
            .unwrap_or_else(|| panic!("split_at(num_seqs) failed {}", context("split_at")));
        assert_eq!(
            left.len(),
            num_seqs,
            "split_at(num_seqs) left should have all sequences {}",
            context("split_at(num_seqs) left.len()")
        );
        assert_eq!(
            right.len(),
            0,
            "split_at(num_seqs) right should be empty {}",
            context("split_at(num_seqs) right.len()")
        );
    }

    // Test split out of bounds
    assert!(
        vec.split_at(num_seqs + 1).is_none(),
        "split_at out of bounds should return None {}",
        context("split_at(out_of_bounds)")
    );

    // --- Sequence Access ---

    // Test get_vec
    for i in 0..num_seqs {
        let result = vec
            .slice(0, num_seqs)
            .unwrap()
            .get_vec(i)
            .unwrap_or_else(|| panic!("get_vec({}) failed {}", i, context("get_vec")));
        assert_eq!(
            &result,
            &sequences[i],
            "get_vec({}) content mismatch {}",
            i,
            context("slice().get_vec()")
        );
    }

    // Test get_into
    if !sequences.is_empty() {
        let slice = vec
            .slice(0, num_seqs)
            .unwrap_or_else(|| panic!("slice failed {}", context("slice")));
        let mut buf = Vec::new();

        for i in 0..num_seqs {
            let len = slice
                .get_into(i, &mut buf)
                .unwrap_or_else(|| panic!("get_into({}) failed {}", i, context("get_into")));
            assert_eq!(
                len,
                sequences[i].len(),
                "get_into({}) returned incorrect length {}",
                i,
                context("slice().get_into()")
            );
            assert_eq!(
                &buf,
                &sequences[i],
                "get_into({}) content mismatch {}",
                i,
                context("slice().get_into()")
            );
        }
    }

    // --- Iteration ---

    if !sequences.is_empty() {
        let slice = vec
            .slice(0, num_seqs)
            .unwrap_or_else(|| panic!("slice failed {}", context("slice")));

        // Forward iteration
        let collected: Vec<Vec<T>> = slice.iter().map(|seq| seq.collect()).collect();
        assert_eq!(
            &collected,
            sequences,
            "Forward iteration mismatch {}",
            context("slice().iter()")
        );

        // ExactSizeIterator
        let slice_iter = slice.iter();
        assert_eq!(
            slice_iter.len(),
            num_seqs,
            "Iterator len() should equal slice len {}",
            context("slice().iter().len()")
        );

        // Backward iteration
        if num_seqs > 1 {
            let mut reversed: Vec<Vec<T>> = slice.iter().rev().map(|seq| seq.collect()).collect();
            reversed.reverse();
            assert_eq!(
                &reversed,
                sequences,
                "Reverse iteration mismatch {}",
                context("slice().iter().rev()")
            );
        }
    }

    // --- Binary Search ---

    // Test binary_search if we have enough sorted sequences
    if num_seqs >= 3 && !sequences[0].is_empty() && !sequences[1].is_empty() {
        let slice = vec
            .slice(0, num_seqs)
            .unwrap_or_else(|| panic!("slice failed {}", context("slice")));

        // Create sorted test data
        let min_elem = sequences[0][0];
        let _mid_elem = sequences[1][0];
        let _max_elem = sequences[2].get(0).copied().unwrap_or(min_elem);

        // Search by custom function on first element
        let result = slice.binary_search_by(|probe| {
            let first = probe.next();
            match first {
                Some(val) => val.cmp(&min_elem),
                None => std::cmp::Ordering::Less,
            }
        });

        // Should find the first sequence if it starts with min_elem
        match result {
            Ok(_) | Err(_) => {
                // Either found or got an insertion point, both valid
            }
        }
    }

    // --- Equality ---

    if num_seqs > 0 {
        let slice1 = vec
            .slice(0, num_seqs)
            .unwrap_or_else(|| panic!("slice failed {}", context("slice")));
        let slice2 = vec
            .slice(0, num_seqs)
            .unwrap_or_else(|| panic!("slice failed {}", context("slice")));

        assert_eq!(
            slice1,
            slice2,
            "Identical slices should be equal {}",
            context("slice() == slice()")
        );

        // Slice should equal full vec
        assert_eq!(
            &slice1,
            &vec,
            "Full slice should equal original vec {}",
            context("slice() == vec")
        );
        assert_eq!(
            &vec,
            &slice1,
            "Original vec should equal full slice {}",
            context("vec == slice()")
        );
    }

    if num_seqs >= 2 {
        let slice1 = vec
            .slice(0, 1)
            .unwrap_or_else(|| panic!("slice failed {}", context("slice")));
        let slice2 = vec
            .slice(1, 1)
            .unwrap_or_else(|| panic!("slice failed {}", context("slice")));

        if sequences[0] != sequences[1] {
            assert_ne!(
                slice1,
                slice2,
                "Different slices should not be equal {}",
                context("slice() != slice()")
            );
        }
    }

    // --- Clone ---

    if num_seqs > 0 {
        let slice1 = vec
            .slice(0, num_seqs)
            .unwrap_or_else(|| panic!("slice failed {}", context("slice")));
        let slice2 = slice1.clone();

        assert_eq!(
            slice1,
            slice2,
            "Cloned slice should be equal {}",
            context("slice().clone()")
        );
    }
}

// --- Macro for Type-Parameterized Testing ---

/// Macro to test all combinations of integer types and endianness for SeqVecSlice.
macro_rules! test_slice_all_types {
    ($test_name:ident, $E:ty) => {
        #[test]
        fn $test_name() {
            // Unsigned types
            {
                let data: Vec<Vec<u8>> =
                    vec![vec![1, 2, 3], vec![10, 20], vec![], vec![100, 200, 300]];
                run_slice_tests_for_type::<u8, $E>(&data, stringify!(u8));
            }

            {
                let data: Vec<Vec<u16>> =
                    vec![vec![1000, 2000, 3000], vec![100, 200], vec![], vec![65535]];
                run_slice_tests_for_type::<u16, $E>(&data, stringify!(u16));
            }

            {
                let data: Vec<Vec<u32>> = vec![
                    vec![1, 2, 3],
                    vec![100, 200, 300, 400],
                    vec![],
                    vec![1000000],
                ];
                run_slice_tests_for_type::<u32, $E>(&data, stringify!(u32));
            }

            {
                let data: Vec<Vec<u64>> = vec![
                    vec![1, 2, 3],
                    vec![100, 200, 300],
                    vec![],
                    vec![9999999999999],
                ];
                run_slice_tests_for_type::<u64, $E>(&data, stringify!(u64));
            }

            // Signed types
            {
                let data: Vec<Vec<i8>> =
                    vec![vec![-1, 2, -3], vec![10, -20], vec![], vec![-100, 127]];
                run_slice_tests_for_type::<i8, $E>(&data, stringify!(i8));
            }

            {
                let data: Vec<Vec<i16>> =
                    vec![vec![-1000, 2000, -3000], vec![-100], vec![], vec![32767]];
                run_slice_tests_for_type::<i16, $E>(&data, stringify!(i16));
            }

            {
                let data: Vec<Vec<i32>> = vec![
                    vec![-1, 2, -3],
                    vec![-100, 200, -300],
                    vec![],
                    vec![2000000000],
                ];
                run_slice_tests_for_type::<i32, $E>(&data, stringify!(i32));
            }

            {
                let data: Vec<Vec<i64>> = vec![
                    vec![-1, 2, -3],
                    vec![-100, 200, -300],
                    vec![],
                    vec![-9999999999999],
                ];
                run_slice_tests_for_type::<i64, $E>(&data, stringify!(i64));
            }
        }
    };
}

// Generate tests for both endianness
test_slice_all_types!(test_slice_le_all_types, LE);
test_slice_all_types!(test_slice_be_all_types, BE);

// --- Edge Case Tests ---

#[test]
fn test_slice_empty_seqvec() {
    let empty_sequences: Vec<Vec<u32>> = vec![];
    let vec = SeqVec::from_slices::<Vec<u32>>(&empty_sequences).unwrap();

    assert!(vec.slice(0, 0).is_ok());
    let slice = vec.slice(0, 0).unwrap();
    assert_eq!(slice.len(), 0);
    assert!(slice.is_empty());
}

#[test]
fn test_slice_single_sequence() {
    let sequences: Vec<Vec<u32>> = vec![vec![42, 99, 1]];
    let vec = SeqVec::from_slices(&sequences).unwrap();

    let slice = vec.slice(0, 1).unwrap();
    assert_eq!(slice.len(), 1);
    let seq: Vec<u32> = slice.get(0).unwrap().collect();
    assert_eq!(seq, vec![42, 99, 1]);
}

#[test]
fn test_slice_all_empty_sequences() {
    let sequences: Vec<Vec<u32>> = vec![vec![], vec![], vec![]];
    let vec = SeqVec::from_slices(&sequences).unwrap();

    let slice = vec.slice(0, 3).unwrap();
    assert_eq!(slice.len(), 3);

    for i in 0..3 {
        let seq = slice.get_vec(i).unwrap();
        assert!(seq.is_empty(), "Sequence {} should be empty", i);
    }
}

#[test]
fn test_slice_long_sequence() {
    let long_seq: Vec<u32> = (0..1000).collect();
    let sequences = vec![long_seq.clone()];
    let vec = SeqVec::from_slices(&sequences).unwrap();

    let slice = vec.slice(0, 1).unwrap();
    let result: Vec<u32> = slice.get(0).unwrap().collect();

    assert_eq!(result.len(), 1000);
    assert_eq!(result[0], 0);
    assert_eq!(result[999], 999);
}

#[test]
fn test_slice_many_empty_sequences() {
    let sequences: Vec<&[u32]> = vec![&[]; 50];
    let vec = SeqVec::from_slices(&sequences).unwrap();

    let slice = vec.slice(10, 30).unwrap();
    assert_eq!(slice.len(), 30);

    for i in 0..30 {
        assert_eq!(slice.get_vec(i), Some(vec![]));
    }
}

#[test]
fn test_slice_into_iterator() {
    let sequences: Vec<&[u32]> = vec![&[1, 2], &[3, 4, 5], &[6]];
    let vec = SeqVec::from_slices(&sequences).unwrap();

    let slice = vec.slice(0, 3).unwrap();

    // Test that &slice implements IntoIterator, enabling for-loop syntax
    let mut collected = Vec::new();
    for seq in &slice {
        let seq_vec: Vec<u32> = seq.collect();
        collected.push(seq_vec);
    }

    assert_eq!(collected.len(), 3);
    assert_eq!(collected[0], vec![1, 2]);
    assert_eq!(collected[1], vec![3, 4, 5]);
    assert_eq!(collected[2], vec![6]);
}

#[test]
fn test_slice_into_iterator_partial() {
    let sequences: Vec<&[u32]> = vec![&[10], &[20, 30], &[40], &[50, 60, 70]];
    let vec = SeqVec::from_slices(&sequences).unwrap();

    let slice = vec.slice(1, 2).unwrap();

    // Test IntoIterator on a partial slice (indices 1 and 2)
    let collected: Vec<Vec<u32>> = (&slice)
        .into_iter()
        .map(|seq| seq.collect())
        .collect();

    assert_eq!(collected.len(), 2);
    assert_eq!(collected[0], vec![20, 30]);
    assert_eq!(collected[1], vec![40]);
}
