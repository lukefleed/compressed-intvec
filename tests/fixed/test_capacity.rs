//! Integration tests for capacity and modification methods in `FixedVec`.

use compressed_intvec::fixed::{
    traits::{Storable, Word},
    FixedVec,
};
use dsi_bitstream::{prelude::{BE, LE}, traits::Endianness};
use num_traits::{Bounded, ToPrimitive};
use std::fmt::Debug;

/// A helper function to run a comprehensive suite of modification tests.
fn run_modification_tests<T, W, E>()
where
    T: Storable<W>
        + Bounded
        + ToPrimitive
        + From<u8>
        + Ord
        + Debug
        + Copy
        + PartialEq,
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
    assert!(vec.capacity() >= 100, "Capacity after reserve should be >= 100, but is {}", vec.capacity());

    // Add some elements and reserve more.
    for i in 0..50 {
        vec.push(i);
    }
    let current_len = vec.len(); // 50
    vec.reserve(100); // Reserve for 100 *additional* elements.
    assert!(vec.capacity() >= current_len + 100, "Capacity should be >= 150, but is {}", vec.capacity());
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