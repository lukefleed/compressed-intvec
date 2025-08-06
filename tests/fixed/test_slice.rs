//! Integration tests for `FixedVecSlice` and `SFixedVecSlice`.

use crate::common::helpers::{generate_random_signed_vec, generate_random_vec};
use compressed_intvec::prelude::*;

#[test]
fn test_fixedvec_slice_creation_and_access() {
    let data: Vec<u64> = generate_random_vec(100, 1000);
    // Use the new primary alias UFixedVec for testing.
    let fixed_vec: UFixedVec<u64> = FixedVec::builder()
        .bit_width(BitWidth::Explicit(10))
        .build(&data)
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
    let data: Vec<i64> = generate_random_signed_vec(100, 1000);
    // Use the new primary alias SFixedVec for testing.
    let s_fixed_vec: SFixedVec<i64> = FixedVec::builder()
        .bit_width(BitWidth::Explicit(11))
        .build(&data)
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
    let data: Vec<u64> = generate_random_vec(100, 1000);
    // Use the BE alias to test non-default endianness.
    let fixed_vec: BEFixedVec = FixedVec::builder().build(&data).unwrap();

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
    let data: Vec<i64> = generate_random_signed_vec(100, 1000);
    // Use the BE alias to test non-default endianness.
    let s_fixed_vec: BESFixedVec = FixedVec::builder().build(&data).unwrap();

    // Valid split
    let (left, right) = s_fixed_vec.split_at(30).unwrap();
    assert_eq!(left.len(), 30);
    assert_eq!(right.len(), 70);
    assert_eq!(left, &data[0..30]);
    assert_eq!(right, &data[30..100]);
}

#[test]
fn test_slice_iterators() {
    let data_u: Vec<u32> = generate_random_vec(100, 1000)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let data_s: Vec<i32> = generate_random_signed_vec(100, 1000)
        .into_iter()
        .map(|x| x as i32)
        .collect();

    let fixed_vec: UFixedVec<u32> = FixedVec::builder().build(&data_u).unwrap();
    let s_fixed_vec: SFixedVec<i32> = FixedVec::builder().build(&data_s).unwrap();

    // Unsigned
    let slice = fixed_vec.slice(20, 50).unwrap();
    let collected: Vec<u32> = slice.iter().collect();
    assert_eq!(collected.len(), 50);
    assert_eq!(collected, &data_u[20..70]);

    // Signed
    let s_slice = s_fixed_vec.slice(20, 50).unwrap();
    let s_collected: Vec<i32> = s_slice.iter().collect();
    assert_eq!(s_collected.len(), 50);
    assert_eq!(s_slice.get(0), s_fixed_vec.get(20));
}

#[test]
fn test_into_iterator_implementations() {
    // Test for FixedVec by reference
    let data_u64: Vec<u64> = generate_random_vec(50, 100);
    let fixed_vec_u64: UFixedVec<u64> = FixedVec::builder().build(&data_u64).unwrap();
    let mut collected_u64 = Vec::new();
    for value in &fixed_vec_u64 {
        collected_u64.push(value);
    }
    assert_eq!(collected_u64, data_u64);

    // Test for FixedVec by value
    let collected_u64_owned: Vec<u64> = fixed_vec_u64.into_iter().collect();
    assert_eq!(collected_u64_owned, data_u64);

    // Test for SFixedVec by reference
    let data_i64: Vec<i64> = generate_random_signed_vec(50, 100);
    let s_fixed_vec_i64: SFixedVec<i64> = FixedVec::builder().build(&data_i64).unwrap();
    let mut collected_i64 = Vec::new();
    for value in &s_fixed_vec_i64 {
        collected_i64.push(value);
    }
    assert_eq!(collected_i64, data_i64);

    // Test for FixedVecSlice
    let slice = s_fixed_vec_i64.slice(10, 20).unwrap();
    let mut collected_slice = Vec::new();
    // The new API requires calling .iter() explicitly on a slice.
    for value in slice.iter() {
        collected_slice.push(value);
    }
    assert_eq!(collected_slice, &data_i64[10..30]);
}

#[test]
fn test_partial_eq_vec_and_slice() {
    let data: Vec<u64> = (0..100).collect();
    let vec: UFixedVec<u64> = FixedVec::builder().build(&data).unwrap();
    let slice_all = vec.slice(0, 100).unwrap();
    let slice_part = vec.slice(10, 20).unwrap();

    // Test FixedVec == FixedVecSlice
    assert_eq!(vec, slice_all);
    assert_ne!(vec, slice_part);

    // Test FixedVecSlice == FixedVec (already covered, but good for completeness)
    let part_vec: UFixedVec<u64> = FixedVec::builder().build(&data[10..30]).unwrap();
    assert_eq!(slice_part, part_vec);
    assert_ne!(slice_all, part_vec);
}