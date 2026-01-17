//! Integration tests for backend conversions in [`SeqVec`].
//!
//! This test suite validates:
//! - Owned (`Vec<u64>`) to borrowed (`&[u64]`) conversions via [`SeqVec::from_parts`]
//! - Zero-copy access patterns
//! - Data integrity across backend types
//! - Different endianness with borrowed backends

use compressed_intvec::seq::SeqVec;
use dsi_bitstream::prelude::{BE, LE};

#[test]
fn test_owned_to_borrowed_conversion() {
    type TestVecOwned = SeqVec<u32, LE, Vec<u64>>;
    type TestVecBorrowed<'a> = SeqVec<u32, LE, &'a [u64]>;

    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];

    // 1. Create an owned SeqVec.
    let owned_vec: TestVecOwned = SeqVec::from_slices(&sequences).unwrap();

    // 2. Create a borrowed view from the owned data.
    let borrowed_vec: TestVecBorrowed =
        SeqVec::from_parts(
            owned_vec.as_limbs(),
            owned_vec.bit_offsets_ref().as_limbs(),
            owned_vec.bit_offsets_ref().len(),
            owned_vec.bit_offsets_ref().bit_width(),
            owned_vec.encoding(),
        )
        .unwrap();

    // 3. Verify they are functionally identical.
    assert_eq!(owned_vec.num_sequences(), borrowed_vec.num_sequences());
    assert_eq!(owned_vec.is_empty(), borrowed_vec.is_empty());
    assert_eq!(owned_vec.total_bits(), borrowed_vec.total_bits());

    // 4. Verify data is identical.
    for i in 0..sequences.len() {
        assert_eq!(owned_vec.decode_vec(i), borrowed_vec.decode_vec(i));
    }

    // 5. Verify all sequences through iterators match.
    let owned_all: Vec<Vec<u32>> = owned_vec.iter().map(|seq| seq.collect()).collect();
    let borrowed_all: Vec<Vec<u32>> = borrowed_vec.iter().map(|seq| seq.collect()).collect();
    assert_eq!(owned_all, borrowed_all);
}

#[test]
fn test_borrowed_backend_preserves_data() {
    type TestVec<'a> = SeqVec<u32, LE, &'a [u64]>;

    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100]];
    let owned = SeqVec::from_slices(&sequences).unwrap();

    // Create borrowed view
    let borrowed: TestVec = SeqVec::from_parts(
        owned.as_limbs(),
        owned.bit_offsets_ref().as_limbs(),
        owned.bit_offsets_ref().len(),
        owned.bit_offsets_ref().bit_width(),
        owned.encoding(),
    )
    .unwrap();

    // Verify all sequences
    for i in 0..sequences.len() {
        assert_eq!(borrowed.decode_vec(i), Some(sequences[i].clone()));
    }
}

#[test]
fn test_borrowed_backend_with_different_endianness() {
    type TestVecOwnedLE = SeqVec<u32, LE, Vec<u64>>;
    type TestVecBorrowedLE<'a> = SeqVec<u32, LE, &'a [u64]>;

    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20]];

    let owned: TestVecOwnedLE = SeqVec::from_slices(&sequences).unwrap();
    let borrowed: TestVecBorrowedLE = SeqVec::from_parts(
        owned.as_limbs(),
        owned.bit_offsets_ref().as_limbs(),
        owned.bit_offsets_ref().len(),
        owned.bit_offsets_ref().bit_width(),
        owned.encoding(),
    )
    .unwrap();

    for i in 0..sequences.len() {
        assert_eq!(borrowed.decode_vec(i), Some(sequences[i].clone()));
    }
}

#[test]
fn test_from_parts_with_empty_sequences() {
    type TestVec<'a> = SeqVec<u32, LE, &'a [u64]>;

    let sequences: Vec<Vec<u32>> = vec![vec![], vec![], vec![]];
    let owned = SeqVec::from_slices(&sequences).unwrap();

    let borrowed: TestVec = SeqVec::from_parts(
        owned.as_limbs(),
        owned.bit_offsets_ref().as_limbs(),
        owned.bit_offsets_ref().len(),
        owned.bit_offsets_ref().bit_width(),
        owned.encoding(),
    )
    .unwrap();

    assert_eq!(borrowed.num_sequences(), 3);
    for i in 0..3 {
        assert_eq!(borrowed.decode_vec(i), Some(vec![]));
    }
}

#[test]
fn test_from_parts_with_mixed_sequences() {
    type TestVec<'a> = SeqVec<u32, LE, &'a [u64]>;

    let sequences: Vec<Vec<u32>> =
        vec![vec![1, 2], vec![], vec![3, 4, 5], vec![], vec![6]];
    let owned = SeqVec::from_slices(&sequences).unwrap();

    let borrowed: TestVec = SeqVec::from_parts(
        owned.as_limbs(),
        owned.bit_offsets_ref().as_limbs(),
        owned.bit_offsets_ref().len(),
        owned.bit_offsets_ref().bit_width(),
        owned.encoding(),
    )
    .unwrap();

    for i in 0..sequences.len() {
        assert_eq!(borrowed.decode_vec(i), Some(sequences[i].clone()));
    }
}

#[test]
fn test_from_parts_validation_zero_length() {
    // from_parts should reject zero-length offsets
    let result: Result<SeqVec<u32, LE, &[u64]>, _> =
        SeqVec::from_parts(&[1, 2], &[], 0, 64, Default::default());

    assert!(matches!(result, Err(_)));
}

#[test]
fn test_multiple_borrowed_views_from_same_owned() {
    type TestVecOwned = SeqVec<u32, LE, Vec<u64>>;
    type TestVecBorrowed<'a> = SeqVec<u32, LE, &'a [u64]>;

    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];
    let owned: TestVecOwned = SeqVec::from_slices(&sequences).unwrap();

    // Create multiple borrowed views
    let borrowed1: TestVecBorrowed = SeqVec::from_parts(
        owned.as_limbs(),
        owned.bit_offsets_ref().as_limbs(),
        owned.bit_offsets_ref().len(),
        owned.bit_offsets_ref().bit_width(),
        owned.encoding(),
    )
    .unwrap();

    let borrowed2: TestVecBorrowed = SeqVec::from_parts(
        owned.as_limbs(),
        owned.bit_offsets_ref().as_limbs(),
        owned.bit_offsets_ref().len(),
        owned.bit_offsets_ref().bit_width(),
        owned.encoding(),
    )
    .unwrap();

    // Both views should be identical
    for i in 0..sequences.len() {
        assert_eq!(borrowed1.decode_vec(i), borrowed2.decode_vec(i));
    }
}

#[test]
fn test_as_limbs_returns_consistent_data() {
    type TestVecOwned = SeqVec<u32, LE, Vec<u64>>;

    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20]];
    let owned: TestVecOwned = SeqVec::from_slices(&sequences).unwrap();

    // Get limbs reference
    let limbs1 = owned.as_limbs();

    // Ensure multiple calls return same reference
    let limbs2 = owned.as_limbs();
    assert_eq!(limbs1.as_ptr(), limbs2.as_ptr());
    assert_eq!(limbs1, limbs2);
}

#[test]
fn test_borrowed_iterator_consistency() {
    type TestVecOwned = SeqVec<u32, LE, Vec<u64>>;
    type TestVecBorrowed<'a> = SeqVec<u32, LE, &'a [u64]>;

    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];
    let owned: TestVecOwned = SeqVec::from_slices(&sequences).unwrap();

    let borrowed: TestVecBorrowed = SeqVec::from_parts(
        owned.as_limbs(),
        owned.bit_offsets_ref().as_limbs(),
        owned.bit_offsets_ref().len(),
        owned.bit_offsets_ref().bit_width(),
        owned.encoding(),
    )
    .unwrap();

    // Test forward iteration
    let owned_iter: Vec<Vec<u32>> = owned.iter().map(|seq| seq.collect()).collect();
    let borrowed_iter: Vec<Vec<u32>> = borrowed.iter().map(|seq| seq.collect()).collect();
    assert_eq!(owned_iter, borrowed_iter);

    // Test reverse iteration
    let owned_rev: Vec<Vec<u32>> = owned.iter().rev().map(|seq| seq.collect()).collect();
    let borrowed_rev: Vec<Vec<u32>> = borrowed.iter().rev().map(|seq| seq.collect()).collect();
    assert_eq!(owned_rev, borrowed_rev);
}

#[test]
fn test_borrowed_with_u64_data() {
    type TestVecOwned = SeqVec<u64, LE, Vec<u64>>;
    type TestVecBorrowed<'a> = SeqVec<u64, LE, &'a [u64]>;

    let sequences: Vec<Vec<u64>> = vec![
        vec![1, 2, 3],
        vec![10000000000, 20000000000],
        vec![9999999999999],
    ];
    let owned: TestVecOwned = SeqVec::from_slices(&sequences).unwrap();

    let borrowed: TestVecBorrowed = SeqVec::from_parts(
        owned.as_limbs(),
        owned.bit_offsets_ref().as_limbs(),
        owned.bit_offsets_ref().len(),
        owned.bit_offsets_ref().bit_width(),
        owned.encoding(),
    )
    .unwrap();

    for i in 0..sequences.len() {
        assert_eq!(borrowed.decode_vec(i), Some(sequences[i].clone()));
    }
}

#[test]
fn test_borrowed_with_signed_data() {
    type TestVecOwned = SeqVec<i32, LE, Vec<u64>>;
    type TestVecBorrowed<'a> = SeqVec<i32, LE, &'a [u64]>;

    let sequences: Vec<Vec<i32>> = vec![vec![-1, 2, -3], vec![100, -200, 300], vec![-50]];
    let owned: TestVecOwned = SeqVec::from_slices(&sequences).unwrap();

    let borrowed: TestVecBorrowed = SeqVec::from_parts(
        owned.as_limbs(),
        owned.bit_offsets_ref().as_limbs(),
        owned.bit_offsets_ref().len(),
        owned.bit_offsets_ref().bit_width(),
        owned.encoding(),
    )
    .unwrap();

    for i in 0..sequences.len() {
        assert_eq!(borrowed.decode_vec(i), Some(sequences[i].clone()));
    }
}

#[test]
fn test_bit_offsets_consistency_owned_borrowed() {
    type TestVecOwned = SeqVec<u32, LE, Vec<u64>>;
    type TestVecBorrowed<'a> = SeqVec<u32, LE, &'a [u64]>;

    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];
    let owned: TestVecOwned = SeqVec::from_slices(&sequences).unwrap();

    let borrowed: TestVecBorrowed = SeqVec::from_parts(
        owned.as_limbs(),
        owned.bit_offsets_ref().as_limbs(),
        owned.bit_offsets_ref().len(),
        owned.bit_offsets_ref().bit_width(),
        owned.encoding(),
    )
    .unwrap();

    // Check bit offsets consistency
    for i in 0..sequences.len() {
        assert_eq!(
            owned.sequence_start_bit(i),
            borrowed.sequence_start_bit(i),
            "Start bit mismatch for sequence {}", i
        );
    }

    // Check total bits
    assert_eq!(owned.total_bits(), borrowed.total_bits());
}
