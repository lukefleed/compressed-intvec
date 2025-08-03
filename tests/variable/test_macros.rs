//! Integration tests for the `int_vec!` macro.

use compressed_intvec::{int_vec, prelude::*};

#[test]
fn test_int_vec_macro_empty() {
    let v: LEIntVec = int_vec![];
    assert!(v.is_empty());
    assert_eq!(v.len(), 0);
    // The default codec for an empty vec is Gamma.
    assert_eq!(v.encoding(), dsi_bitstream::prelude::Codes::Gamma);
}

#[test]
fn test_int_vec_macro_from_list() {
    let data = vec![100u64, 200, 300, 400, 500];
    let v = int_vec![100, 200, 300, 400, 500];

    assert_eq!(v.len(), 5);
    assert_eq!(v.get(0), Some(100));
    assert_eq!(v.get(4), Some(500));
    assert_eq!(v.get(5), None);

    // Check full content
    assert_eq!(v.into_vec(), data);
}

#[test]
fn test_int_vec_macro_from_list_with_trailing_comma() {
    let v = int_vec![1, 2, 3,];
    assert_eq!(v.len(), 3);
    assert_eq!(v.get(2), Some(3));
}

#[test]
fn test_int_vec_macro_from_repeated_element() {
    let v = int_vec![42u64; 100];
    assert_eq!(v.len(), 100);

    for i in 0..100 {
        assert_eq!(v.get(i), Some(42), "Element at index {} is incorrect", i);
    }
    assert_eq!(v.get(100), None);
}
