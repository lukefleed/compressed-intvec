//! Integration tests for binary search functionality.

use compressed_intvec::prelude::*;

#[test]
fn test_fixedvec_binary_search() {
    let data: Vec<u64> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    let vec = LEFixedVec::builder(&data).build().unwrap();

    // Found cases
    assert_eq!(vec.binary_search(30), Ok(2));
    assert_eq!(vec.binary_search(10), Ok(0));
    assert_eq!(vec.binary_search(100), Ok(9));

    // Not found cases
    assert_eq!(vec.binary_search(0), Err(0));
    assert_eq!(vec.binary_search(35), Err(3));
    assert_eq!(vec.binary_search(101), Err(10));
    assert_eq!(vec.binary_search(u64::MAX), Err(10));

    // Empty vec
    let empty_vec = LEFixedVec::builder(&Vec::<u64>::new()).build().unwrap();
    assert_eq!(empty_vec.binary_search(10), Err(0));

    // Duplicates
    let dup_data: Vec<u64> = vec![10, 20, 20, 20, 30];
    let dup_vec = LEFixedVec::builder(&dup_data).build().unwrap();
    let res = dup_vec.binary_search(20);
    assert!(res.is_ok());
    assert!((1..=3).contains(&res.unwrap())); // Can be any of the matching indices
}

#[test]
fn test_sfixedvec_binary_search() {
    let data: Vec<i64> = vec![-30, -20, -10, 0, 10, 20, 30, 40, 50];
    let vec = LESFixedVec::builder(&data).build().unwrap();

    // Found cases
    assert_eq!(vec.binary_search(-10), Ok(2));
    assert_eq!(vec.binary_search(-30), Ok(0));
    assert_eq!(vec.binary_search(50), Ok(8));

    // Not found cases
    assert_eq!(vec.binary_search(-100), Err(0));
    assert_eq!(vec.binary_search(5), Err(4));
    assert_eq!(vec.binary_search(100), Err(9));

    // Duplicates
    let dup_data: Vec<i64> = vec![-10, 0, 0, 0, 10];
    let dup_vec = LESFixedVec::builder(&dup_data).build().unwrap();
    let res = dup_vec.binary_search(0);
    assert!(res.is_ok());
    assert!((1..=3).contains(&res.unwrap()));
}

#[test]
fn test_slice_binary_search() {
    let data: Vec<u64> = (0..100).map(|x| x * 10).collect();
    let vec = LEFixedVec::builder(&data).build().unwrap();

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
fn test_sfixed_slice_binary_search() {
    let data: Vec<i64> = (-50..50).map(|x| x * 10).collect();
    let vec = LESFixedVec::builder(&data).build().unwrap();

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

#[test]
fn test_binary_search_by_key() {
    let data: Vec<u64> = vec![1, 2, 5, 10, 21];
    let vec = LEFixedVec::builder(&data).build().unwrap();

    // Search for a key that is the element squared
    // Find x where x*x = 25
    assert_eq!(vec.binary_search_by_key(&25, |x| x * x), Ok(2));
    // Find x where x*x = 100
    assert_eq!(vec.binary_search_by_key(&100, |x| x * x), Ok(3));
    // Find x where x*x = 9 (not found).
    // The keys are [1, 4, 25, 100, 441]. The insertion point for 9 is at index 2.
    assert_eq!(vec.binary_search_by_key(&9, |x| x * x), Err(2));
}