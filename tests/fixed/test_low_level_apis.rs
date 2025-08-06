//! Integration tests for low-level, performance-oriented APIs.

use compressed_intvec::fixed::{
    traits::{Storable, Word},
    FixedVec,
};
use dsi_bitstream::{prelude::{BE, LE}, traits::Endianness};
use num_traits::{AsPrimitive, Bounded, ToPrimitive};
use std::fmt::Debug;

// --- Helper trait for test data generation ---
// This allows us to create different test data for signed and unsigned types.
trait TestData {
    fn get_test_data() -> Vec<Self> where Self: Sized;
    fn get_test_index_and_val() -> (usize, Self) where Self: Sized;
}

impl TestData for u8 { fn get_test_data() -> Vec<Self> { (0..100).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, 10) }}
impl TestData for u16 { fn get_test_data() -> Vec<Self> { (0..100).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, 10) }}
impl TestData for u32 { fn get_test_data() -> Vec<Self> { (0..100).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, 10) }}
impl TestData for u64 { fn get_test_data() -> Vec<Self> { (0..100).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, 10) }}

// For signed types, we use a range that is guaranteed to fit in a bit_width of 7.
impl TestData for i8 { fn get_test_data() -> Vec<Self> { (-50..50).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, -40) }}
impl TestData for i16 { fn get_test_data() -> Vec<Self> { (-50..50).map(|x| x as i16).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, -40) }}
impl TestData for i32 { fn get_test_data() -> Vec<Self> { (-50..50).map(|x| x).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, -40) }}
impl TestData for i64 { fn get_test_data() -> Vec<Self> { (-50..50).map(|x| x as i64).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, -40) }}


/// Helper function to run the as_mut_limbs test for a specific generic configuration.
fn run_as_mut_limbs_test<T, W, E>()
where
    T: Storable<W> + Bounded + ToPrimitive + Ord + Debug + Copy + PartialEq + TestData,
    W: Word,
    E: Endianness + Debug,
    u64: AsPrimitive<W>,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
{
    let bit_width = 7;
    let mut vec: FixedVec<T, W, E> = FixedVec::new(bit_width).unwrap();
    
    let test_data = T::get_test_data();
    for &val in &test_data {
        vec.push(val);
    }
    assert_eq!(vec.len(), 100);

    let (index_to_test, expected_original_val) = T::get_test_index_and_val();
    let original_val = vec.get(index_to_test).unwrap();
    assert_eq!(original_val, expected_original_val);
    
    let word_bits = <W as Word>::BITS;
    let bit_pos = index_to_test * bit_width;
    let word_idx = bit_pos / word_bits;
    let offset_in_word = bit_pos % word_bits;
    
    let corruption_pattern_u64 = 0b1010101010101010101010101010101010101010101010101010101010101010u64;
    let corruption_pattern: W = corruption_pattern_u64.as_();

    // --- 1. Corrupt the data ---
    {
        let limbs = vec.as_mut_limbs();
        let corruption_mask = corruption_pattern << offset_in_word;
        limbs[word_idx] ^= corruption_mask;

        if offset_in_word + bit_width > word_bits && word_idx + 1 < limbs.len() {
            let spill_mask = corruption_pattern >> (word_bits - offset_in_word);
            limbs[word_idx + 1] ^= spill_mask;
        }
    }

    // --- 2. Verify corruption ---
    let corrupted_val = vec.get(index_to_test).unwrap();
    assert_ne!(original_val, corrupted_val, "Value should have been corrupted");

    // --- 3. Restore the data ---
    {
        let limbs = vec.as_mut_limbs();
        let corruption_mask = corruption_pattern << offset_in_word;
        limbs[word_idx] ^= corruption_mask;

        if offset_in_word + bit_width > word_bits && word_idx + 1 < limbs.len() {
            let spill_mask = corruption_pattern >> (word_bits - offset_in_word);
            limbs[word_idx + 1] ^= spill_mask;
        }
    }

    // --- 4. Verify restoration ---
    let restored_val = vec.get(index_to_test).unwrap();
    assert_eq!(original_val, restored_val, "Value should be restored");
}

/// Macro to instantiate the `as_mut_limbs` test for different generic configurations.
macro_rules! test_as_mut_limbs {
    ($test_name:ident, $T:ty, $W:ty, $E:ty) => {
        #[test]
        fn $test_name() {
            run_as_mut_limbs_test::<$T, $W, $E>();
        }
    };
}

// Instantiate tests for various interesting combinations.
test_as_mut_limbs!(as_mut_limbs_u32_usize_le, u32, usize, LE);
test_as_mut_limbs!(as_mut_limbs_u64_u64_be, u64, u64, BE);
test_as_mut_limbs!(as_mut_limbs_i16_u32_le, i16, u32, LE);
test_as_mut_limbs!(as_mut_limbs_u8_u16_be, u8, u16, BE);