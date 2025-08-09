//! Integration tests for capacity and modification methods in `FixedVec`.

use compressed_intvec::fixed::{
    traits::{Storable, Word},
    Error, FixedVec,
};
use dsi_bitstream::{
    prelude::{BE, LE},
    traits::Endianness,
};
use num_traits::{Bounded, ToPrimitive};
use std::fmt::Debug;

/// A helper function to run a comprehensive suite of modification tests.
fn run_modification_tests<T, W, E>()
where
    T: Storable<W> + Bounded + ToPrimitive + From<u8> + Ord + Debug + Copy + PartialEq,
    W: Word,
    E: Endianness + Debug,
    // Bound needed for FixedVec::new() and other builder methods.
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
{
    // --- Test `push` and `pop` ---
    let mut vec: FixedVec<T, W, E> = FixedVec::new(8).unwrap();
    for i in 0..10 {
        vec.push(T::from(i));
    }
    assert_eq!(vec.len(), 10);
    assert_eq!(vec.get(9), Some(T::from(9)));
    assert_eq!(vec.pop(), Some(T::from(9)));
    assert_eq!(vec.pop(), Some(T::from(8)));
    assert_eq!(vec.len(), 8);

    // --- Test `remove` ---
    // vec is now [0, 1, 2, 3, 4, 5, 6, 7]
    assert_eq!(vec.remove(2), T::from(2)); // remove '2'
    assert_eq!(vec.len(), 7);
    assert_eq!(vec.get(1), Some(T::from(1)));
    assert_eq!(vec.get(2), Some(T::from(3))); // '3' shifted left
    assert_eq!(vec.get(6), Some(T::from(7)));

    // --- Test `insert` ---
    // vec is now [0, 1, 3, 4, 5, 6, 7]
    vec.insert(0, T::from(99)); // insert at start
    assert_eq!(vec.len(), 8);
    assert_eq!(vec.get(0), Some(T::from(99)));
    assert_eq!(vec.get(1), Some(T::from(0)));

    vec.insert(8, T::from(88)); // insert at end
    assert_eq!(vec.len(), 9);
    assert_eq!(vec.get(7), Some(T::from(7)));
    assert_eq!(vec.get(8), Some(T::from(88)));

    // --- Test `clear` ---
    vec.clear();
    assert!(vec.is_empty());

    // --- Test `resize` ---
    vec.resize(5, T::from(42));
    assert_eq!(vec.len(), 5);
    let expected_resize: Vec<T> = vec![T::from(42); 5];
    assert_eq!(vec, &expected_resize[..]);
    vec.resize(2, T::from(0));
    assert_eq!(vec.len(), 2);
    assert_eq!(vec.get(1), Some(T::from(42)));
}

/// A macro to instantiate the modification test suite for different configurations.
macro_rules! test_modifiers {
    ($test_name:ident, $T:ty, $W:ty, $E:ty) => {
        #[test]
        fn $test_name() {
            run_modification_tests::<$T, $W, $E>();
        }
    };
}

// Instantiate tests for various interesting combinations.
test_modifiers!(modifiers_u32_usize_le, u32, usize, LE);
test_modifiers!(modifiers_u64_u64_be, u64, u64, BE);
test_modifiers!(modifiers_i16_u32_le, i16, u32, LE);
test_modifiers!(modifiers_u8_u16_be, u8, u16, BE);

#[test]
fn test_with_capacity() {
    // This test is specific and doesn't need to be in the macro.
    let vec_u: FixedVec<u32, usize, LE> = FixedVec::with_capacity(10, 1000).unwrap();
    assert_eq!(vec_u.len(), 0);
    assert!(vec_u.capacity() >= 1000);
}

#[test]
fn test_reserve() {
    let mut vec: FixedVec<u64, u64, LE> = FixedVec::with_capacity(20, 10).unwrap();
    assert_eq!(vec.len(), 0);
    assert!(vec.capacity() >= 10);

    // Reserve space for 100 additional elements.
    vec.reserve(100);

    // The capacity must be sufficient for at least len (0) + additional (100) elements.
    assert!(
        vec.capacity() >= 100,
        "Capacity after reserve should be >= 100, but is {}",
        vec.capacity()
    );

    // Add some elements and reserve more.
    for i in 0..50 {
        vec.push(i);
    }
    let current_len = vec.len(); // 50
    vec.reserve(100); // Reserve for 100 *additional* elements.
    assert!(
        vec.capacity() >= current_len + 100,
        "Capacity should be >= 150, but is {}",
        vec.capacity()
    );
}

#[test]
fn test_complex_unaligned_shifts() {
    // A specific, tricky case for remove and insert.
    // 11 bits on a 64-bit word forces unaligned access.
    let mut vec: FixedVec<u32, usize, LE> = FixedVec::with_capacity(11, 100).unwrap();
    for i in 0..50 {
        vec.push(i);
    }

    // Test remove
    let mut expected: Vec<u32> = (0..50).collect();
    let removed = vec.remove(25);
    expected.remove(25);
    assert_eq!(removed, 25);
    assert_eq!(vec, &expected[..]);

    // Test insert
    vec.insert(10, 999);
    expected.insert(10, 999);
    assert_eq!(vec, &expected[..]);
}

#[test]
#[should_panic]
fn test_insert_out_of_bounds() {
    let mut vec: FixedVec<u32, usize, LE> = (0..10u32).collect();
    vec.insert(11, 0);
}

#[test]
#[should_panic]
fn test_remove_out_of_bounds() {
    let mut vec: FixedVec<u32, usize, LE> = (0..10u32).collect();
    vec.remove(10);
}

#[test]
fn test_replace() {
    let mut vec: FixedVec<u32, usize, LE> = (0..10u32).collect();
    assert_eq!(vec.bit_width(), 4); // 0-9 fits in 4 bits

    // Replace in the middle
    let old_val = vec.replace(5, 15);
    assert_eq!(old_val, 5);
    assert_eq!(vec.get(5), Some(15));
    assert_eq!(vec.get(4), Some(4)); // Verify adjacent element
    assert_eq!(vec.get(6), Some(6)); // Verify adjacent element

    // Replace at the start
    let old_val_start = vec.replace(0, 11);
    assert_eq!(old_val_start, 0);
    assert_eq!(vec.get(0), Some(11));

    // Replace at the end
    let old_val_end = vec.replace(9, 12);
    assert_eq!(old_val_end, 9);
    assert_eq!(vec.get(9), Some(12));

    // Verify the final state of the vector
    let expected: Vec<u32> = vec![11, 1, 2, 3, 4, 15, 6, 7, 8, 12];
    assert_eq!(vec, &expected[..]);
}

#[test]
#[should_panic(expected = "replace: index out of bounds")]
fn test_replace_out_of_bounds() {
    let mut vec: FixedVec<u32, usize, LE> = (0..10u32).collect();
    vec.replace(10, 0); // This must panic.
}

#[test]
#[should_panic(expected = "Value 16 does not fit in the configured bit_width of 4")]
fn test_replace_value_too_large_panics() {
    let mut vec: FixedVec<u32, usize, LE> = (0..10u32).collect();
    vec.replace(5, 16); // 16 requires 5 bits, vec is configured for 4. Must panic.
}

#[test]
fn test_swap() {
    let mut vec: FixedVec<u32, usize, LE> = (0..10u32).collect();

    vec.swap(2, 8);
    assert_eq!(vec.get(2), Some(8));
    assert_eq!(vec.get(8), Some(2));

    // Test swapping adjacent elements
    vec.swap(0, 1);
    assert_eq!(vec.get(0), Some(1));
    assert_eq!(vec.get(1), Some(0));

    // Test swapping with self
    vec.swap(5, 5);
    assert_eq!(vec.get(5), Some(5));
}

#[test]
#[should_panic(expected = "swap: index a out of bounds")]
fn test_swap_out_of_bounds_a() {
    let mut vec: FixedVec<u32, usize, LE> = (0..10u32).collect();
    vec.swap(10, 0);
}

#[test]
fn test_swap_remove() {
    let mut vec: FixedVec<u32, usize, LE> = (0..10u32).collect();

    // Remove from the middle
    let removed = vec.swap_remove(3);
    assert_eq!(removed, 3);
    assert_eq!(vec.len(), 9);
    assert_eq!(vec.get(3), Some(9)); // Last element (9) moved here
    assert_eq!(vec, &[0, 1, 2, 9, 4, 5, 6, 7, 8][..]);

    // Remove the newly moved element
    let removed_again = vec.swap_remove(3);
    assert_eq!(removed_again, 9);
    assert_eq!(vec.len(), 8);
    assert_eq!(vec.get(3), Some(8)); // New last element (8) moved here
}

#[test]
#[should_panic(expected = "swap_remove: index out of bounds")]
fn test_swap_remove_out_of_bounds() {
    let mut vec: FixedVec<u32, usize, LE> = (0..10u32).collect();
    vec.swap_remove(10);
}

#[test]
fn test_try_api() {
    let mut vec: FixedVec<u32, usize, LE> = FixedVec::new(4).unwrap(); // bit_width=4, max_val=15

    // Test try_push
    assert!(vec.try_push(10).is_ok());
    assert!(vec.try_push(15).is_ok());
    let res_push = vec.try_push(16);
    assert!(matches!(res_push, Err(Error::ValueTooLarge { .. })));
    assert_eq!(vec.len(), 2); // Push should not have happened

    // Test try_set
    assert!(vec.try_set(0, 5).is_ok());
    assert_eq!(vec.get(0), Some(5));
    let res_set = vec.try_set(1, 20);
    assert!(matches!(res_set, Err(Error::ValueTooLarge { .. })));
    assert_eq!(vec.get(1), Some(15)); // Value should be unchanged
}

#[test]
fn test_extend_from_slice() {
    let mut vec: FixedVec<u32, usize, LE> = FixedVec::new(8).unwrap();
    vec.push(1);
    vec.push(2);

    let extension = [3, 4, 5, 6, 7, 8, 9, 10];
    vec.extend_from_slice(&extension);

    assert_eq!(vec.len(), 10);
    let expected: Vec<u32> = (1..=10).collect();
    assert_eq!(vec, &expected[..]);

    // Extend an empty vec
    let mut empty_vec: FixedVec<u32, usize, LE> = FixedVec::new(8).unwrap();
    empty_vec.extend_from_slice(&[10, 20, 30]);
    assert_eq!(empty_vec, &[10, 20, 30][..]);
}

#[test]
#[should_panic]
fn test_extend_from_slice_panic_on_overflow() {
    let mut vec: FixedVec<u32, usize, LE> = FixedVec::new(4).unwrap(); // max val 15
    vec.extend_from_slice(&[10, 11, 20, 12]); // 20 will cause a panic
}
