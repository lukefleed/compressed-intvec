//! Comprehensive integration tests for iterator functionality, including
//! forward, reverse, and unchecked iterators.

use compressed_intvec::fixed::{
    traits::{Storable, Word},
    BitWidth, FixedVec, UFixedVec,
};
use dsi_bitstream::{
    prelude::{BE, LE},
    traits::Endianness,
};
use num_traits::{Bounded, ToPrimitive};
use std::fmt::Debug;

/// Helper function to run a comprehensive suite of iterator tests.
fn run_iterator_tests_for_type<T, W, E>()
where
    T: Storable<W> + Bounded + ToPrimitive + Ord + Debug + Copy + PartialEq + From<u8>,
    W: Word,
    E: Endianness + Debug,
    // Bound needed for FixedVec::new() and other builder methods.
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
{
    let context = format!(
        "<T={}, W={}, E={}>",
        std::any::type_name::<T>(),
        std::any::type_name::<W>(),
        std::any::type_name::<E>()
    );

    // --- Test Case 1: Non-empty vector ---
    let data: Vec<T> = (0..100).map(|i| T::from(i)).collect();
    let vec: FixedVec<T, W, E> = data.iter().copied().collect();
    assert_eq!(vec.len(), 100, "Context: {}", context);

    // 1.1. Forward safe iteration
    let collected_forward: Vec<T> = vec.iter().collect();
    assert_eq!(
        collected_forward, data,
        "Forward iter failed. Context: {}",
        context
    );

    // 1.2. Reverse safe iteration (`DoubleEndedIterator`)
    let mut expected_rev = data.clone();
    expected_rev.reverse();
    let collected_rev: Vec<T> = vec.iter().rev().collect();
    assert_eq!(
        collected_rev, expected_rev,
        "Reverse iter failed. Context: {}",
        context
    );

    // 1.3. Mixed forward and reverse safe iteration
    let mut mixed_iter = vec.iter();
    assert_eq!(mixed_iter.next(), Some(T::from(0)), "Context: {}", context);
    assert_eq!(
        mixed_iter.next_back(),
        Some(T::from(99)),
        "Context: {}",
        context
    );
    assert_eq!(mixed_iter.next(), Some(T::from(1)), "Context: {}", context);
    assert_eq!(
        mixed_iter.next_back(),
        Some(T::from(98)),
        "Context: {}",
        context
    );
    assert_eq!(mixed_iter.len(), 96, "Context: {}", context);

    // --- Test Case 2: Empty vector ---
    let empty_vec: FixedVec<T, W, E> = FixedVec::new(8).unwrap();

    assert!(
        empty_vec.iter().next().is_none(),
        "Empty iter().next() should be None. Context: {}",
        context
    );
    assert!(
        empty_vec.iter().next_back().is_none(),
        "Empty iter().next_back() should be None. Context: {}",
        context
    );

    // --- Test Case 3: Single element vector ---
    let single_data: Vec<T> = vec![T::from(42)];
    let single_vec: FixedVec<T, W, E> = single_data.iter().copied().collect();

    assert_eq!(
        single_vec.iter().collect::<Vec<_>>(),
        single_data,
        "Single element iter failed. Context: {}",
        context
    );
    assert_eq!(
        single_vec.iter().rev().collect::<Vec<_>>(),
        single_data,
        "Single element iter().rev() failed. Context: {}",
        context
    );

    let mut single_iter = single_vec.iter();
    assert_eq!(single_iter.next(), Some(T::from(42)), "Context: {}", context);
    assert_eq!(single_iter.next_back(), None, "Context: {}", context);

    let mut single_iter_rev = single_vec.iter();
    assert_eq!(
        single_iter_rev.next_back(),
        Some(T::from(42)),
        "Context: {}",
        context
    );
    assert_eq!(single_iter_rev.next(), None, "Context: {}", context);
}

/// A macro to instantiate the iterator test suite for different configurations.
macro_rules! test_iterators {
    ($test_name:ident, $T:ty, $W:ty, $E:ty) => {
        #[test]
        fn $test_name() {
            run_iterator_tests_for_type::<$T, $W, $E>();
        }
    };
}

// Instantiate tests for various interesting combinations of types.
test_iterators!(iterators_u32_usize_le, u32, usize, LE);
test_iterators!(iterators_u64_u64_be, u64, u64, BE);
test_iterators!(iterators_i16_u32_le, i16, u32, LE);
test_iterators!(iterators_u8_u16_be, u8, u16, BE);

#[test]
fn test_iter_mut() {
    // Max value will be 4 + 4*10 = 44, which needs 6 bits.
    let mut vec: UFixedVec<u32> = FixedVec::with_capacity(6, 5).unwrap();
    vec.extend(0..5);

    for (i, mut proxy) in vec.iter_mut().enumerate() {
        *proxy += i as u32 * 10;
    }

    let expected_data: Vec<u32> = vec![0, 11, 22, 33, 44];
    assert_eq!(vec, &expected_data[..]);
}

#[test]
fn test_windows() {
    let vec: UFixedVec<u32> = (1..=5).collect();
    let mut windows_iter = vec.windows(3);

    let expected1: Vec<u32> = (1..=3).collect();
    assert_eq!(windows_iter.next().unwrap(), &expected1[..]);

    let expected2: Vec<u32> = (2..=4).collect();
    assert_eq!(windows_iter.next().unwrap(), &expected2[..]);

    let expected3: Vec<u32> = (3..=5).collect();
    assert_eq!(windows_iter.next().unwrap(), &expected3[..]);

    assert!(windows_iter.next().is_none());

    // Test window size larger than vec
    assert!(vec.windows(6).next().is_none());
}

#[test]
fn test_unchecked_iterators() {
    let data: Vec<u32> = (0..100).collect();
    // Create with enough bit width for the modification test. 99*2=198 -> 8 bits
    let vec: UFixedVec<u32> = FixedVec::builder()
        .bit_width(BitWidth::Explicit(8))
        .build(&data)
        .unwrap();

    // Test iter_unchecked
    let mut collected = Vec::with_capacity(data.len());
    let mut iter = unsafe { vec.iter_unchecked() };
    for _ in 0..data.len() {
        collected.push(unsafe { iter.next_unchecked() });
    }
    assert_eq!(collected, data);

    // Test iter_mut_unchecked
    let mut mut_vec = vec;
    let len = mut_vec.len(); // Get the length *before* borrowing.
    let mut iter_mut = unsafe { mut_vec.iter_mut_unchecked() };
    for i in 0..len {
        let mut proxy = unsafe { iter_mut.next_unchecked() };
        *proxy = i as u32 * 2;
    }
    drop(iter_mut); // Drop the iterator to release the borrow

    for i in 0..len {
        assert_eq!(mut_vec.get(i), Some(i as u32 * 2));
    }
}