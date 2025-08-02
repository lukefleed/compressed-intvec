//! Integration tests for API ergonomics, such as macros and convenience methods.

use compressed_intvec::{fixed_vec, prelude::*, s_fixed_vec};

#[test]
fn test_fixed_vec_macro() {
    // Empty
    let v_empty: LEFixedVec = fixed_vec![8];
    assert!(v_empty.is_empty());
    assert_eq!(v_empty.num_bits(), 8);

    // From list
    let v_list = fixed_vec![10; 100, 200, 300];
    assert_eq!(v_list.len(), 3);
    assert_eq!(v_list.num_bits(), 10);
    assert_eq!(v_list.get(1), Some(200));
    assert_eq!(v_list, &[100u64, 200, 300][..]);

    // From element and length
    let v_repeat = fixed_vec![5 => 13; 100];
    assert_eq!(v_repeat.len(), 100);
    assert_eq!(v_repeat.num_bits(), 5);
    for i in 0..100 {
        assert_eq!(v_repeat.get(i), Some(13));
    }
}

#[test]
fn test_s_fixed_vec_macro() {
    // Empty
    let v_empty: LESFixedVec = s_fixed_vec![8];
    assert!(v_empty.is_empty());
    assert_eq!(v_empty.num_bits(), 8);

    // From list
    let v_list = s_fixed_vec![12; -100, 200, -300];
    assert_eq!(v_list.len(), 3);
    assert_eq!(v_list.num_bits(), 12);
    assert_eq!(v_list.get(2), Some(-300));
    assert_eq!(v_list, &[-100i64, 200, -300][..]);

    // From element and length
    let v_repeat = s_fixed_vec![16 => -42; 100];
    assert_eq!(v_repeat.len(), 100);
    assert_eq!(v_repeat.num_bits(), 16);
    for i in 0..100 {
        assert_eq!(v_repeat.get(i), Some(-42));
    }
}

#[test]
fn test_from_slice_method() {
    let data_u32: &[u32] = &[10, 20, 30, 1000];
    let vec_u32 = LEFixedVec::from_slice(data_u32).unwrap();
    // With ByteAligned as default, 10 bits (for 1000) rounds up to 16.
    assert_eq!(vec_u32.num_bits(), 16);
    assert_eq!(vec_u32, data_u32);

    let data_i16: &[i16] = &[-10, 20, -300];
    let vec_i16 = LESFixedVec::from_slice(data_i16).unwrap();
    // ZigZag(-300) = 599, requires 10 bits, which rounds up to 16.
    assert_eq!(vec_i16.num_bits(), 16);
    assert_eq!(vec_i16, data_i16);
}