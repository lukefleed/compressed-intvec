//! Tests for architecture-dependent `Storable` implementations.
//!
//! This test module is only compiled when the `arch-dependent-storable`
//! feature flag is enabled. It verifies the correct behavior of `Storable`
//! for `usize` and `isize`.

#![cfg(feature = "arch-dependent-storable")]

use compressed_intvec::prelude::*;
use dsi_bitstream::prelude::LE;

#[test]
fn test_storable_for_usize() {
    let data: Vec<usize> = (0..100).collect();
    // Explicitly type the VarVec to use the new Storable impl.
    let vec = VarVec::<usize, LE>::from_slice(&data).unwrap();
    assert_eq!(vec.len(), 100);
    assert_eq!(vec.get(50), Some(50));
    assert_eq!(vec.iter().collect::<Vec<_>>(), data);
}

#[test]
fn test_storable_for_isize() {
    let data: Vec<isize> = (-50..50).collect();
    let vec = VarVec::<isize, LE>::from_slice(&data).unwrap();
    assert_eq!(vec.len(), 100);
    assert_eq!(vec.get(0), Some(-50));
    assert_eq!(vec.get(50), Some(0));
    assert_eq!(vec.iter().collect::<Vec<_>>(), data);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn test_large_usize_on_64bit() {
    // This value will not fit in a 32-bit usize.
    let large_value = u32::MAX as usize + 1;
    let data: Vec<usize> = vec![1, 2, large_value];
    let vec = VarVec::<usize, LE>::from_slice(&data).unwrap();
    assert_eq!(vec.get(2), Some(large_value));
}

#[test]
#[should_panic]
#[cfg(target_pointer_width = "32")]
fn test_panic_on_32bit_from_large_u64() {
    // This simulates reading a large value on a 32-bit system.
    let large_word = u32::MAX as u64 + 1;
    // The `from_word` implementation should panic here.
    let _ = <usize as VariableStorable>::from_word(large_word);
}
