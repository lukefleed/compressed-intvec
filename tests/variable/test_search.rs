//! Integration tests for binary search functionality in `IntVec`.

use compressed_intvec::prelude::*;

#[test]
fn test_intvec_binary_search_simple() {
    let data: Vec<u64> = (0..100).map(|x| x * 10).collect();
    let vec = LEIntVec::builder(&data)
        .codec(VariableCodecSpec::Delta)
        .k(8)
        .build()
        .unwrap();

    // Found cases
    assert_eq!(vec.binary_search(300), Ok(30));
    assert_eq!(vec.binary_search(0), Ok(0));
    assert_eq!(vec.binary_search(990), Ok(99));
    assert_eq!(vec.binary_search(500), Ok(50));

    // Not found cases
    assert_eq!(vec.binary_search(1), Err(1)); // Before first element (but after 0)
    assert_eq!(vec.binary_search(305), Err(31)); // Between two elements
    assert_eq!(vec.binary_search(1000), Err(100)); // After last element
    assert_eq!(vec.binary_search(u64::MAX), Err(100)); // Large value after last element
}

#[test]
fn test_intvec_binary_search_with_duplicates() {
    let data: Vec<u64> = vec![10, 20, 20, 20, 20, 30, 40, 40, 50];
    let vec = LEIntVec::builder(&data)
        .codec(VariableCodecSpec::Gamma)
        .k(4)
        .build()
        .unwrap();

    // Search for a unique value
    assert_eq!(vec.binary_search(10), Ok(0));
    assert_eq!(vec.binary_search(30), Ok(5));

    // Search for a duplicate value
    let res_20 = vec.binary_search(20);
    assert!(res_20.is_ok(), "Expected to find value 20");
    let index_20 = res_20.unwrap();
    assert!(
        (1..=4).contains(&index_20),
        "Expected index for 20 to be between 1 and 4, but got {}",
        index_20
    );

    let res_40 = vec.binary_search(40);
    assert!(res_40.is_ok(), "Expected to find value 40");
    let index_40 = res_40.unwrap();
    assert!(
        (6..=7).contains(&index_40),
        "Expected index for 40 to be 6 or 7, but got {}",
        index_40
    );

    // Search for non-existent values
    assert_eq!(vec.binary_search(15), Err(1));
    assert_eq!(vec.binary_search(25), Err(5));
    assert_eq!(vec.binary_search(45), Err(8));
}

#[test]
fn test_intvec_binary_search_empty_and_single() {
    // Empty vector
    let empty_data: Vec<u64> = Vec::new();
    let empty_vec = LEIntVec::builder(&empty_data).build().unwrap();
    assert_eq!(empty_vec.binary_search(10), Err(0));

    // Single element vector, found
    let single_data = vec![42];
    let single_vec = LEIntVec::builder(&single_data).build().unwrap();
    assert_eq!(single_vec.binary_search(42), Ok(0));

    // Single element vector, not found
    assert_eq!(single_vec.binary_search(10), Err(0));
    assert_eq!(single_vec.binary_search(100), Err(1));
}

#[test]
fn test_intvec_binary_search_by_key() {
    // A vector of even numbers
    let data: Vec<u64> = vec![2, 4, 6, 8, 10, 12, 14];
    let vec = LEIntVec::builder(&data)
        .codec(VariableCodecSpec::Zeta { k: Some(3) })
        .k(3)
        .build()
        .unwrap();

    // Search for a key `k` such that its value in the vector `v` is `k*2`.
    // In other words, we are searching for `k` in `[1, 2, 3, 4, 5, 6, 7]`.
    let f = |v: u64| v / 2;

    // Found cases
    assert_eq!(vec.binary_search_by_key(&5, f), Ok(4)); // 5*2 = 10 is at index 4
    assert_eq!(vec.binary_search_by_key(&1, f), Ok(0));
    assert_eq!(vec.binary_search_by_key(&7, f), Ok(6));

    // Not found cases
    // Search for key 3.5. We can do this by searching for integer key 7 where the
    // key extraction function is `v / 2 * 2`, effectively searching for an odd number.
    // The keys are [2, 4, 6, 8, 10, 12, 14]. The insertion point for 7 is at index 3.
    assert_eq!(vec.binary_search_by_key(&7, |v| v), Err(3));

    // Search for key 8. Keys are [1, 2, 3, 4, 5, 6, 7]. Insertion point is after 7.
    assert_eq!(vec.binary_search_by_key(&8, f), Err(7));
}

#[test]
fn test_sintvec_binary_search_simple() {
    let data: Vec<i64> = (-50..50).map(|x| x * 2).collect(); // A sorted range of even numbers
    let vec = LESIntVec::builder(&data)
        .codec(VariableCodecSpec::Delta)
        .k(8)
        .build()
        .unwrap();

    // Found cases
    assert_eq!(vec.binary_search(-20), Ok(40)); // -20 is at index 40
    assert_eq!(vec.binary_search(-100), Ok(0));
    assert_eq!(vec.binary_search(98), Ok(99));
    assert_eq!(vec.binary_search(0), Ok(50));

    // Not found cases
    assert_eq!(vec.binary_search(-101), Err(0)); // Before first element
    assert_eq!(vec.binary_search(1), Err(51)); // Between 0 and 2
    assert_eq!(vec.binary_search(99), Err(100)); // After last element
    assert_eq!(vec.binary_search(i64::MAX), Err(100));
}

#[test]
fn test_sintvec_binary_search_with_duplicates() {
    let data: Vec<i64> = vec![-20, -10, -10, 0, 0, 0, 10];
    let vec = LESIntVec::builder(&data)
        .codec(VariableCodecSpec::Gamma)
        .build()
        .unwrap();

    // Search for a duplicate value
    let res_minus_10 = vec.binary_search(-10);
    assert!(res_minus_10.is_ok());
    let index_minus_10 = res_minus_10.unwrap();
    assert!(
        (1..=2).contains(&index_minus_10),
        "Expected index for -10 to be 1 or 2, but got {}",
        index_minus_10
    );

    let res_0 = vec.binary_search(0);
    assert!(res_0.is_ok());
    let index_0 = res_0.unwrap();
    assert!(
        (3..=5).contains(&index_0),
        "Expected index for 0 to be between 3 and 5, but got {}",
        index_0
    );

    // Search for non-existent values
    assert_eq!(vec.binary_search(-15), Err(1));
    assert_eq!(vec.binary_search(5), Err(6));
}

#[test]
fn test_sintvec_binary_search_by_key() {
    // A vector of numbers sorted by their absolute value
    let data: Vec<i64> = vec![-1, 2, -3, 4, -5, 10];
    let vec = LESIntVec::builder(&data)
        .codec(VariableCodecSpec::Zeta { k: Some(3) })
        .build()
        .unwrap();

    // Search for a key `k` such that `abs(v) == k`.
    let f = |v: i64| v.abs();

    // Found cases
    assert_eq!(vec.binary_search_by_key(&3, f), Ok(2));
    assert_eq!(vec.binary_search_by_key(&1, f), Ok(0));
    assert_eq!(vec.binary_search_by_key(&10, f), Ok(5));

    // Not found cases
    // Keys are [1, 2, 3, 4, 5, 10]. Insertion point for key 6 is at index 5.
    assert_eq!(vec.binary_search_by_key(&6, f), Err(5));
    assert_eq!(vec.binary_search_by_key(&0, f), Err(0));
}
