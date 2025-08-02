//! Integration tests for `FixedVecSlice` and `SFixedVecSlice`.

use crate::common::helpers::{generate_random_signed_vec, generate_random_vec};
use compressed_intvec::prelude::*;

#[test]
fn test_fixedvec_slice_creation_and_access() {
    let data = generate_random_vec(100, 1000);
    let fixed_vec = LEFixedVec::builder(&data)
        .bit_width(BitWidth::Explicit(10))
        .build()
        .unwrap();

    // Valid slice
    let slice = fixed_vec.slice(10, 20).unwrap();
    assert_eq!(slice.len(), 20);
    assert!(!slice.is_empty());
    assert_eq!(slice.get(0), Some(data[10]));
    assert_eq!(slice.get(19), Some(data[29]));
    assert_eq!(slice.get(20), None);

    // Full slice
    let full_slice = fixed_vec.slice(0, 100).unwrap();
    assert_eq!(full_slice, fixed_vec);
    assert_eq!(full_slice, &data[..]);

    // Empty slice
    let empty_slice = fixed_vec.slice(10, 0).unwrap();
    assert_eq!(empty_slice.len(), 0);
    assert!(empty_slice.is_empty());
    assert_eq!(empty_slice.get(0), None);

    // Invalid slice requests
    assert!(fixed_vec.slice(90, 20).is_none()); // a + b > len
    assert!(fixed_vec.slice(101, 0).is_none()); // a > len
}

#[test]
fn test_sfixedvec_slice_creation_and_access() {
    let data = generate_random_signed_vec(100, 1000);
    let s_fixed_vec = LESFixedVec::builder(&data)
        .bit_width(BitWidth::Explicit(11))
        .build()
        .unwrap();

    // Valid slice
    let slice = s_fixed_vec.slice(10, 20).unwrap();
    assert_eq!(slice.len(), 20);
    assert_eq!(slice.get(0), Some(data[10]));
    assert_eq!(slice.get(19), Some(data[29]));
    assert_eq!(slice.get(20), None);

    // Full slice
    let full_slice = s_fixed_vec.slice(0, 100).unwrap();
    assert_eq!(full_slice, s_fixed_vec);
    assert_eq!(full_slice, &data[..]);
}

#[test]
fn test_fixedvec_split_at() {
    let data = generate_random_vec(100, 1000);
    let fixed_vec = BEFixedVec::builder(&data).build().unwrap();

    // Valid split
    let (left, right) = fixed_vec.split_at(30).unwrap();
    assert_eq!(left.len(), 30);
    assert_eq!(right.len(), 70);
    assert_eq!(left, &data[0..30]);
    assert_eq!(right, &data[30..100]);
    assert_eq!(left.get(0), Some(data[0]));
    assert_eq!(right.get(0), Some(data[30]));

    // Split at start
    let (left, right) = fixed_vec.split_at(0).unwrap();
    assert!(left.is_empty());
    assert_eq!(right.len(), 100);
    assert_eq!(right, fixed_vec);

    // Split at end
    let (left, right) = fixed_vec.split_at(100).unwrap();
    assert_eq!(left.len(), 100);
    assert!(right.is_empty());
    assert_eq!(left, fixed_vec);

    // Invalid split
    assert!(fixed_vec.split_at(101).is_none());
}

#[test]
fn test_sfixedvec_split_at() {
    let data = generate_random_signed_vec(100, 1000);
    let s_fixed_vec = BESFixedVec::builder(&data).build().unwrap();

    // Valid split
    let (left, right) = s_fixed_vec.split_at(30).unwrap();
    assert_eq!(left.len(), 30);
    assert_eq!(right.len(), 70);
    assert_eq!(left, &data[0..30]);
    assert_eq!(right, &data[30..100]);
}

#[test]
fn test_slice_iterators() {
    let data = generate_random_vec(100, 1000);
    let fixed_vec = LEFixedVec::builder(&data).build().unwrap();
    let s_fixed_vec = LESFixedVec::builder(&generate_random_signed_vec(100, 1000))
        .build()
        .unwrap();

    // Unsigned
    let slice = fixed_vec.slice(20, 50).unwrap();
    let collected: Vec<u64> = slice.iter().collect();
    assert_eq!(collected.len(), 50);
    assert_eq!(collected, &data[20..70]);

    // Signed
    let s_slice = s_fixed_vec.slice(20, 50).unwrap();
    let s_collected: Vec<i64> = s_slice.iter().collect();
    assert_eq!(s_collected.len(), 50);
    assert_eq!(s_slice.get(0), s_fixed_vec.get(20));
}

#[test]
fn test_into_iterator_implementations() {
    // Test for FixedVec by reference
    let data_u64 = generate_random_vec(50, 100);
    let fixed_vec_u64 = LEFixedVec::builder(&data_u64).build().unwrap();
    let mut collected_u64 = Vec::new();
    for value in &fixed_vec_u64 {
        collected_u64.push(value);
    }
    assert_eq!(collected_u64, data_u64);

    // Test for FixedVec by value
    let collected_u64_owned: Vec<u64> = fixed_vec_u64.into_iter().collect();
    assert_eq!(collected_u64_owned, data_u64);

    // Test for SFixedVec by reference
    let data_i64 = generate_random_signed_vec(50, 100);
    let s_fixed_vec_i64 = LESFixedVec::builder(&data_i64).build().unwrap();
    let mut collected_i64 = Vec::new();
    for value in &s_fixed_vec_i64 {
        collected_i64.push(value);
    }
    assert_eq!(collected_i64, data_i64);

    // Test for FixedVecSlice
    let slice = s_fixed_vec_i64.slice(10, 20).unwrap();
    let mut collected_slice = Vec::new();
    for value in slice {
        collected_slice.push(value);
    }
    assert_eq!(collected_slice, &data_i64[10..30]);
}