//! Integration tests for the `int_vec!` and `sint_vec!` macros.

use compressed_intvec::{int_vec, prelude::*, sint_vec};

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

#[test]
fn test_sint_vec_macro_empty() {
    let v: LESIntVec = sint_vec![];
    assert!(v.is_empty());
    assert_eq!(v.len(), 0);
    // The builder for an empty vec falls back to Gamma, even if Delta is requested.
    assert_eq!(v.encoding(), dsi_bitstream::prelude::Codes::Gamma);
}

#[test]
fn test_sint_vec_macro_from_list() {
    let data = vec![-100i64, 0, 200, -300, 500];
    let v = sint_vec![-100, 0, 200, -300, 500];

    assert_eq!(v.len(), 5);
    assert_eq!(v.get(0), Some(-100));
    assert_eq!(v.get(4), Some(500));
    assert_eq!(v.get(5), None);

    // Check full content
    let collected: Vec<i64> = v.iter().collect();
    assert_eq!(collected, data);
}

#[test]
fn test_sint_vec_macro_from_repeated_element() {
    let v = sint_vec![-42; 100];
    assert_eq!(v.len(), 100);

    for i in 0..100 {
        assert_eq!(v.get(i), Some(-42), "Element at index {} is incorrect", i);
    }
    assert_eq!(v.get(100), None);
}

#[test]
fn test_sint_vec_macro_default_parameters() {
    let data = vec![-1, 2, -3, 5, -8, 13, -21];
    let v = sint_vec![-1, 2, -3, 5, -8, 13, -21];

    // The macro should use the default k=32 and CodecSpec::Delta.
    assert_eq!(v.get_sampling_rate(), Some(32));
    assert_eq!(v.encoding(), dsi_bitstream::prelude::Codes::Delta);

    // Verify the content.
    let collected: Vec<i64> = v.iter().collect();
    assert_eq!(
        collected, data,
        "The content of the vector created by the macro is incorrect."
    );
}
