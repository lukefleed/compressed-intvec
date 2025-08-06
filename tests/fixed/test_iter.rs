//! Comprehensive integration tests for iterator functionality, including
//! forward, reverse, and unchecked iterators.

use compressed_intvec::fixed::{
    traits::{Storable, Word},
    FixedVec,
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
    assert_eq!(collected_forward, data, "Forward iter failed. Context: {}", context);

    // 1.2. Reverse safe iteration (`DoubleEndedIterator`)
    let mut expected_rev = data.clone();
    expected_rev.reverse();
    let collected_rev: Vec<T> = vec.iter().rev().collect();
    assert_eq!(collected_rev, expected_rev, "Reverse iter failed. Context: {}", context);

    // 1.3. Mixed forward and reverse safe iteration
    let mut mixed_iter = vec.iter();
    assert_eq!(mixed_iter.next(), Some(T::from(0)), "Context: {}", context);
    assert_eq!(mixed_iter.next_back(), Some(T::from(99)), "Context: {}", context);
    assert_eq!(mixed_iter.next(), Some(T::from(1)), "Context: {}", context);
    assert_eq!(mixed_iter.next_back(), Some(T::from(98)), "Context: {}", context);
    assert_eq!(mixed_iter.len(), 96, "Context: {}", context);

    // 1.4. Forward unchecked iteration
    let mut collected_unchecked_fwd = Vec::with_capacity(vec.len());
    let mut unchecked_iter_fwd = unsafe { vec.iter_unchecked() };
    for _ in 0..vec.len() {
        collected_unchecked_fwd.push(unsafe { unchecked_iter_fwd.next_unchecked() });
    }
    assert_eq!(collected_unchecked_fwd, data, "Forward unchecked iter failed. Context: {}", context);

    // 1.5. Reverse unchecked iteration
    let mut collected_unchecked_rev = Vec::with_capacity(vec.len());
    let mut unchecked_iter_rev = unsafe { vec.iter_rev_unchecked() };
    for _ in 0..vec.len() {
        collected_unchecked_rev.push(unsafe { unchecked_iter_rev.next_unchecked() });
    }
    assert_eq!(collected_unchecked_rev, expected_rev, "Reverse unchecked iter failed. Context: {}", context);

    // --- Test Case 2: Empty vector ---
    let empty_vec: FixedVec<T, W, E> = FixedVec::new(8).unwrap();

    assert!(empty_vec.iter().next().is_none(), "Empty iter().next() should be None. Context: {}", context);
    assert!(empty_vec.iter().next_back().is_none(), "Empty iter().next_back() should be None. Context: {}", context);
    let empty_unchecked_fwd = unsafe { empty_vec.iter_unchecked() };
    let empty_unchecked_rev = unsafe { empty_vec.iter_rev_unchecked() };
    // These should not be called, but we have the iterators.
    let _ = (empty_unchecked_fwd, empty_unchecked_rev);


    // --- Test Case 3: Single element vector ---
    let single_data: Vec<T> = vec![T::from(42)];
    let single_vec: FixedVec<T, W, E> = single_data.iter().copied().collect();

    assert_eq!(single_vec.iter().collect::<Vec<_>>(), single_data, "Single element iter failed. Context: {}", context);
    assert_eq!(single_vec.iter().rev().collect::<Vec<_>>(), single_data, "Single element iter().rev() failed. Context: {}", context);
    
    let mut single_iter = single_vec.iter();
    assert_eq!(single_iter.next(), Some(T::from(42)), "Context: {}", context);
    assert_eq!(single_iter.next_back(), None, "Context: {}", context);
    
    let mut single_iter_rev = single_vec.iter();
    assert_eq!(single_iter_rev.next_back(), Some(T::from(42)), "Context: {}", context);
    assert_eq!(single_iter_rev.next(), None, "Context: {}", context);

    let mut single_unchecked = unsafe { single_vec.iter_unchecked() };
    assert_eq!(unsafe { single_unchecked.next_unchecked() }, T::from(42), "Context: {}", context);

    let mut single_rev_unchecked = unsafe { single_vec.iter_rev_unchecked() };
    assert_eq!(unsafe { single_rev_unchecked.next_unchecked() }, T::from(42), "Context: {}", context);
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
#[should_panic]
fn test_unchecked_iterator_forward_panic_safety() {
    let vec: FixedVec<u32, usize, LE> = vec![1, 2].into_iter().collect();
    let mut iter = unsafe { vec.iter_unchecked() };
    let _ = unsafe { iter.next_unchecked() };
    let _ = unsafe { iter.next_unchecked() };
    // This call should go out of bounds. In debug mode, it should panic due to the assert.
    let _ = unsafe { iter.next_unchecked() };
}

#[test]
#[should_panic]
fn test_unchecked_iterator_reverse_panic_safety() {
    let vec: FixedVec<u32, usize, LE> = vec![1, 2].into_iter().collect();
    let mut iter = unsafe { vec.iter_rev_unchecked() };
    let _ = unsafe { iter.next_unchecked() };
    let _ = unsafe { iter.next_unchecked() };
    // This call should go out of bounds. In debug mode, it should panic due to the assert.
    let _ = unsafe { iter.next_unchecked() };
}