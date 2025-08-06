//! Integration tests for the `map_in_place` method.

use compressed_intvec::fixed::{
    traits::{Storable, Word},
    BitWidth, FixedVec, SFixedVec, UFixedVec,
};
use dsi_bitstream::prelude::{BE, LE};
use dsi_bitstream::traits::Endianness;
use std::fmt::Debug;

/// A generic helper to run a map_in_place test case.
fn run_map_test<T, W, E>(
    initial_data: &[T],
    bit_width: usize,
    mut f: impl FnMut(T) -> T,
    expected_data: &[T],
) where
    T: Storable<W> + Debug + PartialEq + Copy,
    W: Word,
    E: Endianness + Debug,
    for<'a> dsi_bitstream::impls::BufBitWriter<
        E,
        dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>,
    >: dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
{
    let mut vec: FixedVec<T, W, E> = FixedVec::builder()
        .bit_width(BitWidth::Explicit(bit_width))
        .build(initial_data)
        .unwrap();

    vec.map_in_place(&mut f);

    assert_eq!(
        vec,
        expected_data,
        "Vector content mismatch after map_in_place"
    );
}

#[test]
fn test_map_in_place_fast_path_exact_multiple() {
    // This tests the fast path where the number of elements is an exact
    // multiple of elements per word. bit_width=8, Word=u64 -> 8 elems/word.
    // We use 16 elements to test the main word-by-word loop thoroughly.
    let initial: Vec<u32> = (0..16).collect();
    let expected: Vec<u32> = initial.iter().map(|&x| x * 10).collect();
    run_map_test::<u32, u64, LE>(&initial, 8, |x| x * 10, &expected);
}

#[test]
fn test_map_in_place_fast_path_with_remainder() {
    // This is a critical test for the fast path. The number of elements (19)
    // is NOT a multiple of elements per word (8). This ensures both the main
    // word-by-word loop and the final element-by-element remainder loop are executed.
    let initial: Vec<u32> = (0..19).collect();
    let expected: Vec<u32> = initial.iter().map(|&x| x.wrapping_add(5)).collect();
    run_map_test::<u32, u64, BE>(&initial, 16, |x| x.wrapping_add(5), &expected);
}

#[test]
fn test_map_in_place_generic_path() {
    // bit_width=11 is not a power of two, forcing the generic path.
    let initial: Vec<u16> = (0..100).collect();
    let expected: Vec<u16> = initial.iter().map(|&x| x ^ 0b10101).collect();
    run_map_test::<u16, usize, LE>(&initial, 11, |x| x ^ 0b10101, &expected);
}

#[test]
fn test_map_in_place_signed_types() {
    // Test with signed integers to ensure Storable trait and ZigZag are handled.
    let initial: Vec<i16> = (-50..50).collect();

    // Test generic path for signed
    let expected_generic: Vec<i16> = initial.iter().map(|&x| x * -2).collect();
    run_map_test::<i16, u32, LE>(&initial, 13, |x| x * -2, &expected_generic);

    // Test fast path for signed
    let expected_fast: Vec<i16> = initial.iter().map(|&x| x + 1).collect();
    run_map_test::<i16, u64, BE>(&initial, 16, |x| x + 1, &expected_fast);
}

#[test]
fn test_map_in_place_stateful_closure() {
    // This test verifies that a stateful FnMut closure works correctly.
    let initial: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let expected: Vec<u32> = vec![1, 3, 6, 10, 15, 21, 28, 36, 45, 55]; // Cumulative sum

    let mut total = 0;
    let cumulative_sum = |x: u32| {
        total += x;
        total
    };

    run_map_test::<u32, usize, LE>(&initial, 10, cumulative_sum, &expected);
}

#[test]
#[should_panic(
    expected = "map_in_place: returned value 18 does not fit in the configured bit_width of 4"
)]
fn test_map_in_place_panic_on_overflow() {
    let mut vec: UFixedVec<u8> = FixedVec::builder()
        .bit_width(BitWidth::Explicit(4)) // Max value is 15
        .build(&[1, 2, 3])
        .unwrap();

    // This will process the first element (1), produce 18, which is > 15, and should panic.
    vec.map_in_place(|x| x + 17);
}

#[test]
fn test_map_in_place_edge_cases() {
    // Empty vector
    let mut empty_vec: UFixedVec<u32> = FixedVec::new(8).unwrap();
    empty_vec.map_in_place(|x| x + 1); // Should do nothing and not panic.
    assert!(empty_vec.is_empty());

    // Single element vector
    let mut single_vec: SFixedVec<i8> = FixedVec::builder()
        .bit_width(BitWidth::Explicit(8))
        .build(&[-10])
        .unwrap();
    single_vec.map_in_place(|x| x * 10);
    assert_eq!(single_vec.get(0), Some(-100));
}