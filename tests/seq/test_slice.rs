//! Tests for [`SeqVecSlice`] - zero-copy slicing of sequence vectors.
//!
//! This module comprehensively tests the slicing functionality, including:
//! - Slice creation and bounds checking
//! - Sequence access and iteration within slices
//! - Binary search operations
//! - Equality comparisons
//! - Edge cases (empty slices, single-element slices, full slices)

use compressed_intvec::seq::{SeqVec, VariableCodecSpec, LESeqVec};

// --- Basic Slice Creation ---

#[test]
fn test_slice_valid_range() {
    let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5], &[6], &[7, 8, 9]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(1, 2).unwrap();
    assert_eq!(slice.len(), 2);
    assert!(!slice.is_empty());

    let seq0: Vec<u32> = slice.get(0).unwrap().collect();
    assert_eq!(seq0, vec![3, 4, 5]);

    let seq1: Vec<u32> = slice.get(1).unwrap().collect();
    assert_eq!(seq1, vec![6]);
}

#[test]
fn test_slice_full() {
    let sequences: &[&[u32]] = &[&[1], &[2], &[3]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 3).unwrap();
    assert_eq!(slice.len(), 3);

    for i in 0..3 {
        let expected = vec![(i + 1) as u32];
        assert_eq!(slice.get_vec(i), Some(expected));
    }
}

#[test]
fn test_slice_empty() {
    let sequences: &[&[u32]] = &[&[1, 2], &[3], &[4, 5, 6]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(1, 0).unwrap();
    assert_eq!(slice.len(), 0);
    assert!(slice.is_empty());
    assert_eq!(slice.get(0), None);
}

#[test]
fn test_slice_single_sequence() {
    let sequences: &[&[u32]] = &[&[10, 20, 30], &[40, 50], &[60]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(1, 1).unwrap();
    assert_eq!(slice.len(), 1);
    assert_eq!(slice.get_vec(0), Some(vec![40, 50]));
}

#[test]
fn test_slice_out_of_bounds() {
    let sequences: &[&[u32]] = &[&[1], &[2], &[3]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    assert!(vec.slice(3, 1).is_none()); // Start beyond end
    assert!(vec.slice(2, 2).is_none()); // Length extends past end
    assert!(vec.slice(0, 4).is_none()); // Length too large
}

#[test]
fn test_slice_overflow_protection() {
    let sequences: &[&[u32]] = &[&[1], &[2]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    // saturating_add should prevent overflow
    assert!(vec.slice(usize::MAX, 1).is_none());
    assert!(vec.slice(1, usize::MAX).is_none());
}

// --- Split At ---

#[test]
fn test_split_at_middle() {
    let sequences: &[&[u32]] = &[&[1], &[2, 3], &[4], &[5, 6, 7]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let (left, right) = vec.split_at(2).unwrap();

    assert_eq!(left.len(), 2);
    assert_eq!(right.len(), 2);

    assert_eq!(left.get_vec(0), Some(vec![1]));
    assert_eq!(left.get_vec(1), Some(vec![2, 3]));

    assert_eq!(right.get_vec(0), Some(vec![4]));
    assert_eq!(right.get_vec(1), Some(vec![5, 6, 7]));
}

#[test]
fn test_split_at_boundaries() {
    let sequences: &[&[u32]] = &[&[1], &[2], &[3]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    // Split at start
    let (left, right) = vec.split_at(0).unwrap();
    assert_eq!(left.len(), 0);
    assert_eq!(right.len(), 3);

    // Split at end
    let (left, right) = vec.split_at(3).unwrap();
    assert_eq!(left.len(), 3);
    assert_eq!(right.len(), 0);
}

#[test]
fn test_split_at_out_of_bounds() {
    let sequences: &[&[u32]] = &[&[1], &[2]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    assert!(vec.split_at(3).is_none());
    assert!(vec.split_at(100).is_none());
}

// --- Sequence Access ---

#[test]
fn test_slice_get_index_translation() {
    let sequences: &[&[u32]] = &[&[10], &[20], &[30], &[40], &[50]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(2, 2).unwrap(); // Sequences 2 and 3 (30 and 40)

    // Index 0 of slice is sequence 2 of parent
    assert_eq!(slice.get_vec(0), Some(vec![30]));
    // Index 1 of slice is sequence 3 of parent
    assert_eq!(slice.get_vec(1), Some(vec![40]));
    // Index 2 is out of bounds for the slice
    assert_eq!(slice.get_vec(2), None);
}

#[test]
fn test_slice_get_into() {
    let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5], &[6]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 3).unwrap();
    let mut buf = Vec::new();

    assert_eq!(slice.get_into(0, &mut buf), Some(2));
    assert_eq!(buf, vec![1, 2]);

    // Buffer should be reused (cleared)
    assert_eq!(slice.get_into(1, &mut buf), Some(3));
    assert_eq!(buf, vec![3, 4, 5]);

    assert_eq!(slice.get_into(2, &mut buf), Some(1));
    assert_eq!(buf, vec![6]);

    // Out of bounds
    assert_eq!(slice.get_into(3, &mut buf), None);
}

#[test]
fn test_slice_with_empty_sequences() {
    let sequences: &[&[u32]] = &[&[1, 2], &[], &[3], &[], &[4, 5]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(1, 3).unwrap(); // Empty, [3], Empty

    assert_eq!(slice.len(), 3);

    assert_eq!(slice.get_vec(0), Some(vec![])); // Empty sequence
    assert_eq!(slice.get_vec(1), Some(vec![3]));
    assert_eq!(slice.get_vec(2), Some(vec![])); // Empty sequence
}

// --- Iteration ---

#[test]
fn test_slice_iter_forward() {
    let sequences: &[&[u32]] = &[&[1], &[2, 3], &[4, 5, 6], &[7]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(1, 2).unwrap();

    let collected: Vec<Vec<u32>> = slice.iter()
        .map(|seq| seq.collect())
        .collect();

    assert_eq!(collected, vec![vec![2, 3], vec![4, 5, 6]]);
}

#[test]
fn test_slice_iter_backward() {
    let sequences: &[&[u32]] = &[&[1], &[2], &[3], &[4]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(1, 2).unwrap();

    let collected: Vec<Vec<u32>> = slice.iter()
        .rev()
        .map(|seq| seq.collect())
        .collect();

    assert_eq!(collected, vec![vec![3], vec![2]]);
}

#[test]
fn test_slice_iter_exact_size() {
    let sequences: &[&[u32]] = &[&[1], &[2], &[3], &[4], &[5]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(1, 3).unwrap();
    let mut iter = slice.iter();

    assert_eq!(iter.len(), 3);
    iter.next();
    assert_eq!(iter.len(), 2);
    iter.next();
    assert_eq!(iter.len(), 1);
    iter.next();
    assert_eq!(iter.len(), 0);
}

#[test]
fn test_slice_iter_size_hint() {
    let sequences: &[&[u32]] = &[&[1], &[2], &[3]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 3).unwrap();
    let iter = slice.iter();

    let (lower, upper) = iter.size_hint();
    assert_eq!(lower, 3);
    assert_eq!(upper, Some(3));
}

#[test]
fn test_slice_iter_fused() {
    let sequences: &[&[u32]] = &[&[1]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 1).unwrap();
    let mut iter = slice.iter();

    assert!(iter.next().is_some());
    assert!(iter.next().is_none());
    assert!(iter.next().is_none()); // Still None after exhaustion
}

#[test]
fn test_slice_iter_empty_slice() {
    let sequences: &[&[u32]] = &[&[1], &[2]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(1, 0).unwrap();
    let mut iter = slice.iter();

    assert_eq!(iter.len(), 0);
    assert!(iter.next().is_none());
}

// --- Binary Search ---

#[test]
fn test_binary_search_found() {
    let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5], &[6, 7], &[8, 9, 10]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 4).unwrap();

    assert_eq!(slice.binary_search(&[1, 2]), Ok(0));
    assert_eq!(slice.binary_search(&[3, 4, 5]), Ok(1));
    assert_eq!(slice.binary_search(&[6, 7]), Ok(2));
    assert_eq!(slice.binary_search(&[8, 9, 10]), Ok(3));
}

#[test]
fn test_binary_search_not_found() {
    let sequences: &[&[u32]] = &[&[1], &[3, 4], &[6, 7, 8]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 3).unwrap();

    assert_eq!(slice.binary_search(&[2]), Err(1)); // Would insert between [1] and [3,4]
    assert_eq!(slice.binary_search(&[5]), Err(2)); // Would insert between [3,4] and [6,7,8]
    assert_eq!(slice.binary_search(&[9]), Err(3)); // Would insert at end
}

#[test]
fn test_binary_search_by_length() {
    let sequences: &[&[u32]] = &[&[], &[1], &[2, 3], &[4, 5, 6]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 4).unwrap();

    let result = slice.binary_search_by(|probe| {
        let len = probe.count();
        len.cmp(&2)
    });

    assert_eq!(result, Ok(2)); // Sequence [2, 3] has length 2
}

#[test]
fn test_binary_search_by_key_first_element() {
    let sequences: &[&[u32]] = &[&[1, 10], &[2, 20], &[3, 30], &[4, 40]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 4).unwrap();

    let result = slice.binary_search_by_key(&Some(3), |probe| probe.next());
    assert_eq!(result, Ok(2)); // Sequence starting with 3
}

#[test]
fn test_binary_search_empty_slice() {
    let sequences: &[&[u32]] = &[&[1], &[2]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(1, 0).unwrap();

    assert_eq!(slice.binary_search(&[5]), Err(0)); // Would insert at position 0
}

#[test]
fn test_binary_search_early_exit() {
    // Sequences differ in first element
    let sequences: &[&[u32]] = &[&[1, 100, 100], &[2, 200, 200], &[3, 300, 300]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 3).unwrap();

    assert_eq!(slice.binary_search(&[2, 200, 200]), Ok(1));
    assert_eq!(slice.binary_search(&[2, 999]), Err(2)); // Early exit on second element
}

#[test]
fn test_binary_search_length_mismatch() {
    let sequences: &[&[u32]] = &[&[1], &[1, 2], &[1, 2, 3]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 3).unwrap();

    // Shorter sequence compares as Less
    assert_eq!(slice.binary_search(&[1, 2]), Ok(1));
    assert_eq!(slice.binary_search(&[1, 2, 3, 4]), Err(3)); // Longer sequence compares as Greater
}

// --- Equality ---

#[test]
fn test_slice_eq_slice() {
    let sequences1: &[&[u32]] = &[&[1, 2], &[3], &[4, 5, 6]];
    let vec1: LESeqVec<u32> = SeqVec::from_slices(sequences1).unwrap();

    let sequences2: &[&[u32]] = &[&[99], &[1, 2], &[3], &[4, 5, 6], &[100]];
    let vec2: LESeqVec<u32> = SeqVec::from_slices(sequences2).unwrap();

    let slice1 = vec1.slice(0, 3).unwrap();
    let slice2 = vec2.slice(1, 3).unwrap();

    assert_eq!(slice1, slice2);
}

#[test]
fn test_slice_neq_different_lengths() {
    let sequences: &[&[u32]] = &[&[1], &[2], &[3]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice1 = vec.slice(0, 2).unwrap();
    let slice2 = vec.slice(0, 3).unwrap();

    assert_ne!(slice1, slice2);
}

#[test]
fn test_slice_neq_different_content() {
    let sequences: &[&[u32]] = &[&[1, 2], &[3, 4], &[5, 6]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice1 = vec.slice(0, 2).unwrap();
    let slice2 = vec.slice(1, 2).unwrap();

    assert_ne!(slice1, slice2);
}

#[test]
fn test_slice_eq_vec() {
    let sequences: &[&[u32]] = &[&[1, 2], &[3], &[4, 5]];
    let vec1: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    let vec2: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec1.slice(0, 3).unwrap();

    assert_eq!(slice, vec2);
    assert_eq!(vec2, slice); // Symmetric
}

#[test]
fn test_slice_neq_vec_different_length() {
    let sequences: &[&[u32]] = &[&[1], &[2], &[3]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 2).unwrap();

    assert_ne!(slice, vec);
    assert_ne!(vec, slice);
}

#[test]
fn test_slice_eq_ref_vec() {
    let sequences: &[&[u32]] = &[&[1], &[2]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice = vec.slice(0, 2).unwrap();

    assert_eq!(slice, &vec);
}

// --- Multiple Codecs ---

#[test]
fn test_slice_with_gamma_codec() {
    let sequences: &[&[u32]] = &[&[1, 2, 3], &[4, 5], &[6, 7, 8, 9]];
    let vec: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(sequences)
        .unwrap();

    let slice = vec.slice(1, 2).unwrap();
    assert_eq!(slice.get_vec(0), Some(vec![4, 5]));
    assert_eq!(slice.get_vec(1), Some(vec![6, 7, 8, 9]));
}

#[test]
fn test_slice_with_delta_codec() {
    let sequences: &[&[u64]] = &[&[10, 20, 30], &[100, 200], &[1000]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(sequences)
        .unwrap();

    let slice = vec.slice(0, 2).unwrap();
    assert_eq!(slice.len(), 2);

    let seq0: Vec<u64> = slice.get(0).unwrap().collect();
    assert_eq!(seq0, vec![10, 20, 30]);
}

#[test]
fn test_slice_with_zeta_codec() {
    let sequences: &[&[u32]] = &[&[5, 10, 15], &[20, 25], &[30]];
    let vec: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Zeta { k: Some(3) })
        .build(sequences)
        .unwrap();

    let slice = vec.slice(1, 2).unwrap();

    let all: Vec<Vec<u32>> = slice.iter()
        .map(|seq| seq.collect())
        .collect();

    assert_eq!(all, vec![vec![20, 25], vec![30]]);
}

// --- Edge Cases ---

#[test]
fn test_slice_of_single_long_sequence() {
    let long_seq: Vec<u32> = (0..1000).collect();
    let sequences = vec![long_seq.as_slice()];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    let slice = vec.slice(0, 1).unwrap();
    let result: Vec<u32> = slice.get(0).unwrap().collect();

    assert_eq!(result.len(), 1000);
    assert_eq!(result[0], 0);
    assert_eq!(result[999], 999);
}

#[test]
fn test_slice_many_empty_sequences() {
    let sequences: Vec<&[u32]> = vec![&[]; 100];
    let vec: LESeqVec<u32> = SeqVec::from_slices(&sequences).unwrap();

    let slice = vec.slice(25, 50).unwrap();
    assert_eq!(slice.len(), 50);

    for i in 0..50 {
        assert_eq!(slice.get_vec(i), Some(vec![]));
    }
}

#[test]
fn test_slice_nested_slicing_simulation() {
    let sequences: &[&[u32]] = &[&[1], &[2], &[3], &[4], &[5], &[6]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice1 = vec.slice(1, 4).unwrap(); // [2, 3, 4, 5]
    let slice2 = vec.slice(2, 2).unwrap(); // [3, 4] - overlaps with slice1

    // Both should access the same underlying data
    assert_eq!(slice1.get_vec(1), Some(vec![3]));
    assert_eq!(slice2.get_vec(0), Some(vec![3]));
}

#[test]
fn test_slice_clone() {
    let sequences: &[&[u32]] = &[&[1, 2], &[3, 4]];
    let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

    let slice1 = vec.slice(0, 2).unwrap();
    let slice2 = slice1.clone();

    assert_eq!(slice1, slice2);
    assert_eq!(slice1.get_vec(0), slice2.get_vec(0));
}
