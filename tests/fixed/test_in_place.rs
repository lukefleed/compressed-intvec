//! Integration tests for the `map_in_place` method and other in-place operations.

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
#[should_panic]
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

#[test]
fn test_split_at_mut() {
    let mut vec: UFixedVec<u32> = (10..=50).step_by(10).collect();
    {
        let (mut left, mut right) = vec.split_at_mut(3);

        assert_eq!(left.len(), 3);
        assert_eq!(right.len(), 2);
        assert_eq!(left.get(0), Some(10));
        assert_eq!(right.get(0), Some(40));

        *left.at_mut(0).unwrap() = 11;
        *right.at_mut(1).unwrap() = 55;
    }

    // After the block above the borrows have ended and `vec` can be re-borrowed.
    let expected_data: Vec<u32> = vec![11, 20, 30, 40, 55];
    let expected_vec: UFixedVec<u32> = expected_data.into_iter().collect();
    assert_eq!(vec, expected_vec);
}

#[test]
fn test_rotate() {
    let mut vec: UFixedVec<u32> = (0..5).collect();
    vec.rotate_left(2);
    let expected_left: Vec<u32> = vec![2, 3, 4, 0, 1];
    assert_eq!(vec, &expected_left[..]);

    let mut vec2: UFixedVec<u32> = (0..5).collect();
    vec2.rotate_right(2);
    let expected_right: Vec<u32> = vec![3, 4, 0, 1, 2];
    assert_eq!(vec2, &expected_right[..]);
}

#[test]
fn test_fill_and_fill_with() {
    // Test fill
    let mut vec: UFixedVec<u32> = FixedVec::with_capacity(8, 5).unwrap();
    vec.resize(5, 0); // Initialize with some value
    vec.fill(42);
    let expected_fill: Vec<u32> = vec![42; 5];
    assert_eq!(vec, &expected_fill[..]);

    // Test fill_with
    let mut counter = 0;
    vec.fill_with(|| {
        counter += 1;
        counter * 10
    });
    let expected_fill_with: Vec<u32> = vec![10, 20, 30, 40, 50];
    assert_eq!(vec, &expected_fill_with[..]);
}

#[test]
#[should_panic]
fn test_fill_with_panic() {
    let mut vec: UFixedVec<u32> = FixedVec::with_capacity(4, 5).unwrap();
    vec.resize(5, 0);
    vec.fill(16); // 16 does not fit in 4 bits
}

#[test]
fn test_copy_from_slice() {
    // max value for dest is 1099, which needs 11 bits.
    let bit_width = 11;
    let src_data: Vec<u32> = (0..100).collect();
    let dest_data: Vec<u32> = (0..100).map(|x| x + 1000).collect();

    let src: UFixedVec<u32> = FixedVec::builder()
        .bit_width(BitWidth::Explicit(bit_width))
        .build(&src_data)
        .unwrap();
    let mut dest: UFixedVec<u32> = FixedVec::builder()
        .bit_width(BitWidth::Explicit(bit_width))
        .build(&dest_data)
        .unwrap();

    // Non-overlapping
    dest.copy_from_slice(&src, 10..30, 50);
    for i in 0..20 {
        assert_eq!(dest.get(50 + i), Some(10 + i as u32));
    }
    // Verify that other elements are untouched
    assert_eq!(dest.get(49), Some(1049));
    assert_eq!(dest.get(70), Some(1070));

    // Overlapping (dest > src)
    let mut vec: UFixedVec<u32> = FixedVec::builder()
        .bit_width(BitWidth::Minimal)
        .build(&(0..10).collect::<Vec<u32>>())
        .unwrap();
    vec.copy_from_slice(&vec.clone(), 0..5, 2);
    let expected: Vec<u32> = vec![0, 1, 0, 1, 2, 3, 4, 7, 8, 9];
    assert_eq!(vec, &expected[..]);

    // Overlapping (src > dest)
    let mut vec2: UFixedVec<u32> = FixedVec::builder()
        .bit_width(BitWidth::Minimal)
        .build(&(0..10).collect::<Vec<u32>>())
        .unwrap();
    vec2.copy_from_slice(&vec2.clone(), 2..7, 0);
    let expected2: Vec<u32> = vec![2, 3, 4, 5, 6, 5, 6, 7, 8, 9];
    assert_eq!(vec2, &expected2[..]);
}
