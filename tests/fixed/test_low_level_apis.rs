//! Integration tests for low-level, performance-oriented APIs.

use compressed_intvec::fixed::{
    traits::{Storable, Word},
    FixedVec,
};
use dsi_bitstream::{prelude::{BE, LE}, traits::Endianness};
use num_traits::{AsPrimitive, Bounded, ToPrimitive};
use std::fmt::Debug;

// --- Helper trait for test data generation ---
trait TestData {
    fn get_test_data() -> Vec<Self> where Self: Sized;
    fn get_test_index_and_val() -> (usize, Self) where Self: Sized;
}

impl TestData for u8 { fn get_test_data() -> Vec<Self> { (0..100).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, 10) }}
impl TestData for u16 { fn get_test_data() -> Vec<Self> { (0..100).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, 10) }}
impl TestData for u32 { fn get_test_data() -> Vec<Self> { (0..100).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, 10) }}
impl TestData for u64 { fn get_test_data() -> Vec<Self> { (0..100).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, 10) }}

impl TestData for i8 { fn get_test_data() -> Vec<Self> { (-50..50).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, -40) }}
impl TestData for i16 { fn get_test_data() -> Vec<Self> { (-50..50).map(|x| x as i16).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, -40) }}
impl TestData for i32 { fn get_test_data() -> Vec<Self> { (-50..50).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, -40) }}
impl TestData for i64 { fn get_test_data() -> Vec<Self> { (-50..50).map(|x| x as i64).collect() } fn get_test_index_and_val() -> (usize, Self) { (10, -40) }}


/// Helper function to run the as_mut_limbs test for a specific generic configuration.
fn run_as_mut_limbs_test<T, W, E>()
where
    T: Storable<W> + Bounded + ToPrimitive + Ord + Debug + Copy + PartialEq + TestData,
    W: Word,
    E: Endianness,
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

    {
        let limbs = vec.as_mut_limbs();
        let corruption_mask = corruption_pattern << offset_in_word;
        limbs[word_idx] ^= corruption_mask;
        if offset_in_word + bit_width > word_bits && word_idx + 1 < limbs.len() {
            let spill_mask = corruption_pattern >> (word_bits - offset_in_word);
            limbs[word_idx + 1] ^= spill_mask;
        }
    }

    let corrupted_val = vec.get(index_to_test).unwrap();
    assert_ne!(original_val, corrupted_val, "Value should have been corrupted");

    {
        let limbs = vec.as_mut_limbs();
        let corruption_mask = corruption_pattern << offset_in_word;
        limbs[word_idx] ^= corruption_mask;

        if offset_in_word + bit_width > word_bits && word_idx + 1 < limbs.len() {
            let spill_mask = corruption_pattern >> (word_bits - offset_in_word);
            limbs[word_idx + 1] ^= spill_mask;
        }
    }
    
    let restored_val = vec.get(index_to_test).unwrap();
    assert_eq!(original_val, restored_val, "Value should be restored");
}

/// Helper function to run the addr_of test for a specific generic configuration.
fn run_addr_of_test<T, W, E>()
where
    T: Storable<W> + Bounded + ToPrimitive + Ord + Debug + Copy + PartialEq + TestData,
    W: Word,
    E: Endianness,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
{
    let test_data = T::get_test_data();
    let vec: FixedVec<T, W, E> = test_data.into_iter().collect();

    let (index, _) = T::get_test_index_and_val();
    let bit_pos = index * vec.bit_width();
    let word_idx = bit_pos / <W as Word>::BITS;

    let ptr = vec.addr_of(index).unwrap();
    let expected_ptr = vec.as_limbs().as_ptr().wrapping_add(word_idx);
    assert_eq!(ptr, expected_ptr, "Pointer for index {} should point to word {}", index, word_idx);
    
    unsafe {
        assert_eq!(*ptr, vec.as_limbs()[word_idx]);
    }

    assert!(vec.addr_of(vec.len()).is_none());
}

/// Helper function to run the prefetch test for a specific generic configuration.
fn run_prefetch_test<T, W, E>()
where
    T: Storable<W> + Bounded + ToPrimitive + Ord + Debug + Copy + PartialEq + TestData,
    W: Word,
    E: Endianness,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
{
    let test_data = T::get_test_data();
    let vec: FixedVec<T, W, E> = test_data.into_iter().collect();

    if !vec.is_empty() {
        vec.prefetch(0);
        vec.prefetch(vec.len() / 2);
        vec.prefetch(vec.len() - 1);
    }
    
    vec.prefetch(vec.len());
    vec.prefetch(usize::MAX);

    let empty_vec: FixedVec<T,W,E> = FixedVec::new(8).unwrap();
    empty_vec.prefetch(0);
}

/// Helper function to test get_unaligned_unchecked against get_unchecked.
fn run_unaligned_access_test<T, W, E>()
where
    T: Storable<W> + Bounded + ToPrimitive + Ord + Debug + Copy + PartialEq + TestData,
    W: Word,
    E: Endianness,
    u64: AsPrimitive<W>,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
{
    for bit_width in [1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64] {
        if bit_width > <W as Word>::BITS { continue; }

        let mut vec: FixedVec<T, W, E> = FixedVec::new(bit_width).unwrap();
        
        let test_data = T::get_test_data();
        for &val in &test_data {
            // Use fully qualified syntax to disambiguate.
            let word_val = <T as Storable<W>>::into_word(val);
            if bit_width < <W as Word>::BITS && word_val >= (W::ONE << bit_width) {
                continue;
            }
            vec.push(val);
        }

        if vec.is_empty() { continue; }

        for i in 0..vec.len() {
            let val_normal = unsafe { vec.get_unchecked(i) };
            let val_unaligned = unsafe { vec.get_unaligned_unchecked(i) };
            assert_eq!(
                val_normal, val_unaligned,
                "Mismatch at index {} for bit_width {}. <T={}, W={}, E={}>",
                i, bit_width,
                std::any::type_name::<T>(), std::any::type_name::<W>(), std::any::type_name::<E>()
            );
        }
    }
}

macro_rules! test_low_level_apis {
    ($test_name:ident, $T:ty, $W:ty, $E:ty) => {
        #[test]
        fn $test_name() {
            run_as_mut_limbs_test::<$T, $W, $E>();
            run_addr_of_test::<$T, $W, $E>();
            run_prefetch_test::<$T, $W, $E>();
            run_unaligned_access_test::<$T, $W, $E>();
        }
    };
}

// Instantiate tests for various interesting combinations.
test_low_level_apis!(low_level_apis_u32_usize_le, u32, usize, LE);
test_low_level_apis!(low_level_apis_u64_u64_be, u64, u64, BE);
test_low_level_apis!(low_level_apis_i16_u32_le, i16, u32, LE);
test_low_level_apis!(low_level_apis_u8_u16_be, u8, u16, BE);