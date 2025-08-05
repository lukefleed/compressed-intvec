//! Integration tests for `IntVecSlice` and `SIntVecSlice`.

use crate::common::helpers::{generate_random_signed_vec, generate_random_vec};
use compressed_intvec::prelude::*;

#[test]
fn test_intvec_slice_creation_and_access() {
    let data = generate_random_vec(100, 1000);
    let intvec = LEIntVec::builder(&data)
        .codec(VariableCodecSpec::Delta)
        .k(8)
        .build()
        .unwrap();

    // Valid slice
    let slice = intvec.slice(10, 20).unwrap();
    assert_eq!(slice.len(), 20);
    assert!(!slice.is_empty());
    assert_eq!(slice.get(0), Some(data[10]));
    assert_eq!(slice.get(19), Some(data[29]));
    assert_eq!(slice.get(20), None);

    // Full slice
    let full_slice = intvec.slice(0, 100).unwrap();
    assert_eq!(full_slice.iter().collect::<Vec<_>>(), intvec.iter().collect::<Vec<_>>());

    // Empty slice
    let empty_slice = intvec.slice(10, 0).unwrap();
    assert_eq!(empty_slice.len(), 0);
    assert!(empty_slice.is_empty());
    assert_eq!(empty_slice.get(0), None);

    // Invalid slice requests
    assert!(intvec.slice(90, 20).is_none()); // a + b > len
    assert!(intvec.slice(101, 0).is_none()); // a > len
}

#[test]
fn test_sintvec_slice_creation_and_access() {
    let data = generate_random_signed_vec(100, 1000);
    let sintvec = LESIntVec::builder(&data)
        .codec(VariableCodecSpec::Gamma)
        .k(8)
        .build()
        .unwrap();

    // Valid slice
    let slice = sintvec.slice(10, 20).unwrap();
    assert_eq!(slice.len(), 20);
    assert_eq!(slice.get(0), Some(data[10]));
    assert_eq!(slice.get(19), Some(data[29]));
    assert_eq!(slice.get(20), None);

    // Full slice
    let full_slice = sintvec.slice(0, 100).unwrap();
    assert_eq!(full_slice.iter().collect::<Vec<_>>(), sintvec.iter().collect::<Vec<_>>());
}

#[test]
fn test_intvec_split_at() {
    let data = generate_random_vec(100, 1000);
    let intvec = BEIntVec::builder(&data).build().unwrap();

    // Valid split
    let (left, right) = intvec.split_at(30).unwrap();
    assert_eq!(left.len(), 30);
    assert_eq!(right.len(), 70);
    assert_eq!(left.iter().collect::<Vec<_>>(), &data[0..30]);
    assert_eq!(right.iter().collect::<Vec<_>>(), &data[30..100]);
    assert_eq!(left.get(0), Some(data[0]));
    assert_eq!(right.get(0), Some(data[30]));

    // Split at start
    let (left, right) = intvec.split_at(0).unwrap();
    assert!(left.is_empty());
    assert_eq!(right.len(), 100);

    // Split at end
    let (left, right) = intvec.split_at(100).unwrap();
    assert_eq!(left.len(), 100);
    assert!(right.is_empty());

    // Invalid split
    assert!(intvec.split_at(101).is_none());
}

#[test]
fn test_sintvec_split_at() {
    let data = generate_random_signed_vec(100, 1000);
    let sintvec = BESIntVec::builder(&data).build().unwrap();

    // Valid split
    let (left, right) = sintvec.split_at(30).unwrap();
    assert_eq!(left.len(), 30);
    assert_eq!(right.len(), 70);
    assert_eq!(left.iter().collect::<Vec<_>>(), &data[0..30]);
    assert_eq!(right.iter().collect::<Vec<_>>(), &data[30..100]);
}

#[test]
fn test_slice_binary_search() {
    let data: Vec<u64> = (0..100).map(|x| x * 10).collect();
    let vec = LEIntVec::builder(&data).build().unwrap();

    // Slice the middle part: [200, 210, ..., 690]
    let slice = vec.slice(20, 50).unwrap();
    assert_eq!(slice.len(), 50);

    // Found in slice
    assert_eq!(slice.binary_search(400), Ok(20)); // data[40] is at index 20 of slice
    assert_eq!(slice.binary_search(200), Ok(0));
    assert_eq!(slice.binary_search(690), Ok(49));

    // Not found in slice
    assert_eq!(slice.binary_search(100), Err(0)); // Before start
    assert_eq!(slice.binary_search(405), Err(21)); // In the middle
    assert_eq!(slice.binary_search(800), Err(50)); // After end
}

#[test]
fn test_sintvec_slice_binary_search() {
    let data: Vec<i64> = (-50..50).map(|x| x * 10).collect();
    let vec = LESIntVec::builder(&data).build().unwrap();

    // Slice the middle part: [-300, -290, ..., 190]
    let slice = vec.slice(20, 50).unwrap();
    assert_eq!(slice.len(), 50);

    // Found in slice
    assert_eq!(slice.binary_search(0), Ok(30)); // data[50] is at index 30 of slice
    assert_eq!(slice.binary_search(-300), Ok(0));
    assert_eq!(slice.binary_search(190), Ok(49));

    // Not found in slice
    assert_eq!(slice.binary_search(-400), Err(0));
    assert_eq!(slice.binary_search(5), Err(31));
    assert_eq!(slice.binary_search(200), Err(50));
}