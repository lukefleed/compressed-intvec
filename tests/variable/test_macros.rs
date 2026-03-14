//! Integration tests for the `int_vec!` and `sint_vec!` macros and other convenience APIs.

use compressed_intvec::int_vec;
use compressed_intvec::prelude::LESVarVec;
use compressed_intvec::sint_vec;
use compressed_intvec::variable::LEVarVec;

#[test]
fn test_int_vec_macro_empty() {
    let v: LEVarVec = int_vec![];
    assert!(v.is_empty());
    assert_eq!(v.len(), 0);
    // The default codec for an empty vec is Gamma.
    assert_eq!(v.encoding(), dsi_bitstream::prelude::Codes::Gamma);
}

#[test]
fn test_int_vec_macro_from_list() {
    let v_u64: LEVarVec = int_vec![100, 200, 300, 400, 500];

    assert_eq!(v_u64.len(), 5);
    assert_eq!(v_u64.get(0), Some(100));
    assert_eq!(v_u64.get(4), Some(500));

    // Polymorphism: test with u32
    let v_u32 = int_vec![10u32, 20, 30];
    assert_eq!(v_u32.get(0), Some(10));
    assert_eq!(v_u32.get(2), Some(30));
}

#[test]
fn test_int_vec_macro_from_list_with_trailing_comma() {
    // Add an explicit type annotation.
    let v: LEVarVec = int_vec![1, 2, 3,];
    assert_eq!(v.len(), 3);
    assert_eq!(v.get(2), Some(3));
}

#[test]
fn test_int_vec_macro_from_repeated_element() {
    // Add an explicit type annotation.
    let v: LEVarVec = int_vec![42u64; 100];
    assert_eq!(v.len(), 100);

    for i in 0..100 {
        assert_eq!(v.get(i), Some(42), "Element at index {} is incorrect", i);
    }
    assert_eq!(v.get(100), None);
}

#[test]
fn test_sint_vec_macro_empty() {
    let v: LESVarVec = sint_vec![];
    assert!(v.is_empty());
    assert_eq!(v.len(), 0);
    // The builder for an empty vec falls back to Gamma.
    assert_eq!(v.encoding(), dsi_bitstream::prelude::Codes::Gamma);
}

#[test]
fn test_sint_vec_macro_from_list() {
    let data: [i64; 5] = [-100i64, 0, 200, -300, 500];
    // Add an explicit type annotation.
    let v: LESVarVec = sint_vec![-100, 0, 200, -300, 500];

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
    let v: LESVarVec = sint_vec![-42; 100];
    assert_eq!(v.len(), 100);

    for i in 0..100 {
        assert_eq!(v.get(i), Some(-42), "Element at index {} is incorrect", i);
    }
    assert_eq!(v.get(100), None);
}

#[test]
fn test_from_slice_method() {
    // Test VarVec::from_slice
    let data_u64: &[u64] = &[10, 20, 30, 1000];
    let vec_u64 = LEVarVec::from_slice(data_u64).unwrap();
    assert_eq!(vec_u64.sampling_rate(), 16);
    assert_eq!(vec_u64, data_u64);

    // Test SVarVec::from_slice
    let data_i64: &[i64] = &[-10, 20, -300];
    let vec_i64 = LESVarVec::from_slice(data_i64).unwrap();
    assert_eq!(vec_i64.sampling_rate(), 16);
    assert_eq!(vec_i64, data_i64);
}
