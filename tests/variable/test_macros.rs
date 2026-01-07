//! Integration tests for the `int_vec!` and `sint_vec!` macros and other convenience APIs.

use compressed_intvec::prelude::LESIntVec;
use compressed_intvec::variable::LEIntVec;
use compressed_intvec::int_vec;
use compressed_intvec::sint_vec;


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
    let data: [u64; 5] = [100u64, 200, 300, 400, 500];
    // Add an explicit type annotation to help the compiler resolve trait bounds.
    let v: LEIntVec = int_vec![100, 200, 300, 400, 500];

    assert_eq!(v.len(), 5);
    assert_eq!(v.get(0), Some(100));
    assert_eq!(v.get(4), Some(500));
    assert_eq!(v.get(5), None);

    // Check full content
    assert_eq!(v, &data[..]);
}

#[test]
fn test_int_vec_macro_from_list_with_trailing_comma() {
    // Add an explicit type annotation.
    let v: LEIntVec = int_vec![1, 2, 3,];
    assert_eq!(v.len(), 3);
    assert_eq!(v.get(2), Some(3));
}

#[test]
fn test_int_vec_macro_from_repeated_element() {
    // Add an explicit type annotation.
    let v: LEIntVec = int_vec![42u64; 100];
    assert_eq!(v.len(), 100);

    for i in 0..100 {
        assert_eq!(v.get(i), Some(42), "Element at index {} is incorrect", i);
    }
    assert_eq!(v.get(100), None);
}

#[test]
fn test_sint_vec_macro_empty() {
    let v: LESIntVec = sint_vec![];
    assert!(v.is_empty());
    assert_eq!(v.len(), 0);
    // The builder for an empty vec falls back to Gamma.
    assert_eq!(v.encoding(), dsi_bitstream::prelude::Codes::Gamma);
}

#[test]
fn test_sint_vec_macro_from_list() {
    let data: [i64; 5] = [-100i64, 0, 200, -300, 500];
    // Add an explicit type annotation.
    let v: LESIntVec = sint_vec![-100, 0, 200, -300, 500];

    assert_eq!(v.len(), 5);
    assert_eq!(v.get(0), Some(-100));
    assert_eq!(v.get(4), Some(500));
    assert_eq!(v.get(5), None);

    // Check full content
    assert_eq!(v, &data[..]);
}

#[test]
fn test_sint_vec_macro_from_repeated_element() {
    // Add an explicit type annotation.
    let v: LESIntVec = sint_vec![-42; 100];
    assert_eq!(v.len(), 100);

    for i in 0..100 {
        assert_eq!(v.get(i), Some(-42), "Element at index {} is incorrect", i);
    }
    assert_eq!(v.get(100), None);
}

#[test]
fn test_from_slice_method() {
    // Test IntVec::from_slice
    let data_u64: &[u64] = &[10, 20, 30, 1000];
    let vec_u64 = LEIntVec::from_slice(data_u64).unwrap();
    assert_eq!(vec_u64.get_sampling_rate(), 16);
    assert_eq!(vec_u64, data_u64);

    // Test SIntVec::from_slice
    let data_i64: &[i64] = &[-10, 20, -300];
    let vec_i64 = LESIntVec::from_slice(data_i64).unwrap();
    assert_eq!(vec_i64.get_sampling_rate(), 16);
    assert_eq!(vec_i64, data_i64);
}
