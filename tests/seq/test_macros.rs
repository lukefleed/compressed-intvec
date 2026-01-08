//! Integration tests for the `seq_vec!` and `seq_vec_signed!` macros.

use compressed_intvec::{seq_vec, seq_vec_signed};
use compressed_intvec::seq::{LESeqVec, LESEqVec, BESEqVec, BESeqVec};

// --- Tests for seq_vec! macro (unsigned) ---

#[test]
fn test_seq_vec_macro_basic() {
    // Explicit type to guide the macro
    let v: LESeqVec<u32> = seq_vec![
        [1, 2, 3],
        [4, 5],
        []
    ];

    assert_eq!(v.len(), 3, "seq_vec! length mismatch");
    assert_eq!(
        v.get(0).unwrap().collect::<Vec<_>>(),
        vec![1, 2, 3],
        "seq_vec! first sequence"
    );
    assert_eq!(
        v.get(1).unwrap().collect::<Vec<_>>(),
        vec![4, 5],
        "seq_vec! second sequence"
    );
    assert!(
        v.get(2).unwrap().collect::<Vec<_>>().is_empty(),
        "seq_vec! third sequence should be empty"
    );
}

#[test]
fn test_seq_vec_macro_empty() {
    let v: LESeqVec<u32> = seq_vec![];
    assert!(v.is_empty(), "seq_vec![] should be empty");
}

#[test]
fn test_seq_vec_macro_single_sequence() {
    let v: LESeqVec<u64> = seq_vec![[100, 200, 300]];
    assert_eq!(v.len(), 1, "Single sequence length");
    assert_eq!(
        v.get(0).unwrap().collect::<Vec<_>>(),
        vec![100, 200, 300],
        "Single sequence content"
    );
}

#[test]
fn test_seq_vec_macro_large_values() {
    let v: BESeqVec<u32> = seq_vec![
        [u32::MAX - 1, u32::MAX],
        [1000000, 2000000]
    ];
    assert_eq!(v.len(), 2, "Large values length");
    assert_eq!(
        v.get(0).unwrap().collect::<Vec<_>>(),
        vec![u32::MAX - 1, u32::MAX],
        "Large values first sequence"
    );
}

#[test]
fn test_seq_vec_macro_with_trailing_comma() {
    let v: LESeqVec<u32> = seq_vec![
        [1, 2],
        [3, 4],
    ];
    assert_eq!(v.len(), 2, "Trailing comma should not affect length");
    assert_eq!(
        v.get(0).unwrap().collect::<Vec<_>>(),
        vec![1, 2],
        "First sequence with trailing comma"
    );
}

// --- Tests for seq_vec_signed! macro (signed) ---

#[test]
fn test_seq_vec_signed_macro_basic() {
    let v: LESEqVec<i32> = seq_vec_signed![
        [-1, -2],
        [10, 20]
    ];

    assert_eq!(v.len(), 2, "seq_vec_signed! length");
    assert_eq!(
        v.get(0).unwrap().collect::<Vec<_>>(),
        vec![-1, -2],
        "seq_vec_signed! first sequence"
    );
    assert_eq!(
        v.get(1).unwrap().collect::<Vec<_>>(),
        vec![10, 20],
        "seq_vec_signed! second sequence"
    );
}

#[test]
fn test_seq_vec_signed_macro_empty() {
    let v: LESEqVec<i64> = seq_vec_signed![];
    assert!(v.is_empty(), "seq_vec_signed![] should be empty");
}

#[test]
fn test_seq_vec_signed_macro_mixed_values() {
    let v: BESEqVec<i16> = seq_vec_signed![
        [-100, -50, 0],
        [50, 100],
        [-1, -2, -3, -4]
    ];

    assert_eq!(v.len(), 3, "Mixed values length");
    assert_eq!(
        v.get(0).unwrap().collect::<Vec<_>>(),
        vec![-100, -50, 0],
        "Mixed values first sequence"
    );
    assert_eq!(
        v.get(1).unwrap().collect::<Vec<_>>(),
        vec![50, 100],
        "Mixed values second sequence"
    );
    assert_eq!(
        v.get(2).unwrap().collect::<Vec<_>>(),
        vec![-4, -3, -2, -1],
        "Mixed values third sequence"
    );
}

#[test]
fn test_seq_vec_signed_macro_single_sequence() {
    let v: LESEqVec<i32> = seq_vec_signed![[-42, -21, 0, 21, 42]];
    assert_eq!(v.len(), 1, "Single signed sequence length");
    assert_eq!(
        v.get(0).unwrap().collect::<Vec<_>>(),
        vec![-42, -21, 0, 21, 42],
        "Single signed sequence content"
    );
}

#[test]
fn test_seq_vec_signed_macro_extreme_values() {
    let v: BESEqVec<i64> = seq_vec_signed![
        [i64::MIN, i64::MIN + 1],
        [i64::MAX - 1, i64::MAX]
    ];

    assert_eq!(v.len(), 2, "Extreme values length");
    let first = v.get(0).unwrap().collect::<Vec<_>>();
    assert!(first[0] < first[1], "First extreme sequence ordering");
}

#[test]
fn test_seq_vec_signed_macro_with_trailing_comma() {
    let v: LESEqVec<i16> = seq_vec_signed![
        [-10, -5],
        [5, 10],
    ];
    assert_eq!(v.len(), 2, "Trailing comma should not affect length");
}

// --- Cross-endianness tests to ensure macros work with both LE and BE ---

#[test]
fn test_seq_vec_macro_le_vs_be_consistency() {
    let sequences = vec![vec![1u32, 2, 3], vec![4, 5]];

    let vec_le: LESeqVec<u32> = LESeqVec::from_slices(&sequences).unwrap();
    let vec_be: BESeqVec<u32> = BESeqVec::from_slices(&sequences).unwrap();

    // Both should contain the same data
    let le_collected: Vec<Vec<u32>> = vec_le.iter().map(|s| s.collect()).collect();
    let be_collected: Vec<Vec<u32>> = vec_be.iter().map(|s| s.collect()).collect();

    assert_eq!(le_collected, be_collected, "LE and BE should have same content");
    assert_eq!(le_collected, sequences, "Content should match original");
}
