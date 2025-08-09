//! Integration tests for the `fixed_vec!` macro.

use compressed_intvec::fixed::{SFixedVec, UFixedVec};
use compressed_intvec::fixed_vec;

#[test]
fn test_fixed_vec_macro_empty() {
    // For the empty case, type annotation is still required as there are no
    // values from which to infer the type. This is expected and correct.
    let v: UFixedVec<u32> = fixed_vec![];
    assert!(v.is_empty());
    assert_eq!(v.len(), 0);
    assert_eq!(v.bit_width(), 1);
}

#[test]
fn test_fixed_vec_macro_from_list_unsigned() {
    // The type is now fully inferred by the macro!
    // We create an explicitly typed slice to compare against.
    let v = fixed_vec![100u32, 200, 300, 400, 500];
    let expected: &[u32] = &[100, 200, 300, 400, 500];

    assert_eq!(v.len(), 5);
    assert_eq!(v.get(0), Some(100));
    assert_eq!(v.get(4), Some(500));
    assert_eq!(v, expected);

    // Verify that the inferred type is indeed UFixedVec<u32>
    let _: UFixedVec<u32> = v;
}

#[test]
fn test_fixed_vec_macro_from_list_signed() {
    // The type is inferred to SFixedVec<i16>.
    let v = fixed_vec![-100i16, 0, 200, -300, 500];
    let expected: &[i16] = &[-100, 0, 200, -300, 500];

    assert_eq!(v.len(), 5);
    assert_eq!(v.get(0), Some(-100));
    assert_eq!(v, expected);

    // Verify that the inferred type is SFixedVec<i16>
    let _: SFixedVec<i16> = v;
}

#[test]
fn test_fixed_vec_macro_from_list_with_trailing_comma() {
    let v = fixed_vec![1u8, 2, 3,];
    assert_eq!(v.len(), 3);
    assert_eq!(v.get(2), Some(3));
    let _: UFixedVec<u8> = v;
}

#[test]
fn test_fixed_vec_macro_from_repeated_element() {
    // Unsigned
    let v_unsigned = fixed_vec![42u64; 100];
    assert_eq!(v_unsigned.len(), 100);
    for i in 0..100 {
        assert_eq!(v_unsigned.get(i), Some(42));
    }
    assert_eq!(v_unsigned.bit_width(), 6);
    let _: UFixedVec<u64> = v_unsigned;

    // Signed
    let v_signed = fixed_vec![-5isize; 50];
    assert_eq!(v_signed.len(), 50);
    for i in 0..50 {
        assert_eq!(v_signed.get(i), Some(-5));
    }
    assert_eq!(v_signed.bit_width(), 4);
    let _: SFixedVec<isize> = v_signed;
}
