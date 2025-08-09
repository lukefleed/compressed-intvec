//! Integration tests for binary search functionality.

use compressed_intvec::{fixed::{FixedVec, SFixedVec, UFixedVec}, fixed_vec};
use dsi_bitstream::prelude::LE;

#[test]
fn test_fixedvec_binary_search() {
    type TestVec = UFixedVec<u32>;
    let data: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    let vec: TestVec = FixedVec::builder().build(&data).unwrap();

    // Found cases
    assert_eq!(vec.binary_search(&30), Ok(2));
    assert_eq!(vec.binary_search(&10), Ok(0));
    assert_eq!(vec.binary_search(&100), Ok(9));

    // Not found cases
    assert_eq!(vec.binary_search(&0), Err(0));
    assert_eq!(vec.binary_search(&35), Err(3));
    assert_eq!(vec.binary_search(&101), Err(10));
    assert_eq!(vec.binary_search(&u32::MAX), Err(10));

    // Empty vec
    let empty_vec: TestVec = FixedVec::builder().build(&[]).unwrap();
    assert_eq!(empty_vec.binary_search(&10), Err(0));

    // Duplicates
    let dup_data: Vec<u32> = vec![10, 20, 20, 20, 30];
    let dup_vec: TestVec = FixedVec::builder().build(&dup_data).unwrap();
    let res = dup_vec.binary_search(&20);
    assert!(res.is_ok());
    assert!((1..=3).contains(&res.unwrap())); // Can be any of the matching indices
}

#[test]
fn test_sfixedvec_binary_search() {
    type TestVec = SFixedVec<i16>;
    let data: Vec<i16> = vec![-30, -20, -10, 0, 10, 20, 30, 40, 50];
    let vec: TestVec = FixedVec::builder().build(&data).unwrap();

    // Found cases
    assert_eq!(vec.binary_search(&-10), Ok(2));
    assert_eq!(vec.binary_search(&-30), Ok(0));
    assert_eq!(vec.binary_search(&50), Ok(8));

    // Not found cases
    assert_eq!(vec.binary_search(&-100), Err(0));
    assert_eq!(vec.binary_search(&5), Err(4));
    assert_eq!(vec.binary_search(&100), Err(9));

    // Duplicates
    let dup_data: Vec<i16> = vec![-10, 0, 0, 0, 10];
    let dup_vec: TestVec = FixedVec::builder().build(&dup_data).unwrap();
    let res = dup_vec.binary_search(&0);
    assert!(res.is_ok());
    assert!((1..=3).contains(&res.unwrap()));
}

#[test]
fn test_view_binary_search() {
    type TestVec = UFixedVec<u64>;
    let data: Vec<u64> = (0..100).map(|x| x * 10).collect();
    let vec: TestVec = FixedVec::builder().build(&data).unwrap();

    // Slice the middle part: [200, 210, ..., 690]
    let view = vec.slice(20, 50).unwrap();
    assert_eq!(view.len(), 50);

    // Found in view
    assert_eq!(view.binary_search(&400), Ok(20)); // data[40] is at index 20 of view
    assert_eq!(view.binary_search(&200), Ok(0));
    assert_eq!(view.binary_search(&690), Ok(49));

    // Not found in view
    assert_eq!(view.binary_search(&100), Err(0)); // Before start
    assert_eq!(view.binary_search(&405), Err(21)); // In the middle
    assert_eq!(view.binary_search(&800), Err(50)); // After end
}

#[test]
fn test_binary_search_by() {
    let vec: UFixedVec<u32> = fixed_vec![1, 2, 5, 10, 21];

    assert_eq!(vec.binary_search_by(|p| p.cmp(&10)), Ok(3));
    assert_eq!(vec.binary_search_by(|p| p.cmp(&9)), Err(3));
    assert_eq!(
        vec.binary_search_by(|p| (p * 2).cmp(&10)),
        Ok(2) /* 5*2=10 */
    );
}

#[test]
fn test_binary_search_by_key() {
    type TestVec = FixedVec<u16, u32, LE>; // Use a non-default word size
    let data: Vec<u16> = vec![1, 2, 5, 10, 21];
    let vec: TestVec = FixedVec::builder().build(&data).unwrap();

    // Search for a key that is the element squared
    // Find x where x*x = 25
    assert_eq!(vec.binary_search_by_key(&25, |x| (x as u32).pow(2)), Ok(2));
    // Find x where x*x = 100
    assert_eq!(vec.binary_search_by_key(&100, |x| (x as u32).pow(2)), Ok(3));

    // Find x where x*x = 9 (not found).
    // The keys are [1, 4, 25, 100, 441]. The insertion point for 9 is at index 2.
    assert_eq!(vec.binary_search_by_key(&9, |x| (x as u32).pow(2)), Err(2));
}

#[test]
fn test_partition_point() {
    let vec: UFixedVec<u32> = fixed_vec![0, 1, 1, 2, 3, 5, 8, 13];

    // Partition point is the index of the first element that is false
    assert_eq!(vec.partition_point(|&x| x < 5), 5); // Index of 5
    assert_eq!(vec.partition_point(|&x| x <= 1), 3); // Index of 2
    assert_eq!(vec.partition_point(|&x| x > 100), 0); // All are false
    assert_eq!(vec.partition_point(|&x| x < 20), 8); // All are true

    let empty_vec: UFixedVec<u32> = fixed_vec![];
    assert_eq!(empty_vec.partition_point(|&x| x < 5), 0);
}