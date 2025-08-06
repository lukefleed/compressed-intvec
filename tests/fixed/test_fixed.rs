//! Comprehensive integration tests for the generic `FixedVec`.

use compressed_intvec::fixed::{
    traits::{Storable, Word},
    BitWidth, FixedVec, SFixedVec, UFixedVec,
};
use dsi_bitstream::{prelude::{BE, LE}, traits::Endianness};
use num_traits::{ToPrimitive};
use std::fmt::Debug;

// Import helper functions from the common module.
#[path = "../common/mod.rs"]
mod common;
use common::helpers::{generate_random_signed_vec, generate_random_vec};

/// Central test function called by the macro for each type combination.
fn run_test_for_type<T, W, E>(data: &[T], type_name: &str)
where
    T: Storable<W> + Debug + PartialEq + Default + Copy + ToPrimitive,
    W: Word,
    E: Endianness,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
{
    let context = |bw_strat: &str| format!("<{}> on {} in {}<{}>", type_name, bw_strat, std::any::type_name::<W>(), std::any::type_name::<E>());

    // We iterate over a set of strategies to test.
    // `Explicit(0)` is a placeholder that will be replaced by the calculated minimal bits.
    for strategy in [BitWidth::Minimal, BitWidth::PowerOfTwo, BitWidth::Explicit(0)] {
        let bit_width_strategy = if strategy == BitWidth::Explicit(0) {
            let max_val: W = data.iter().map(|&v| <T as Storable<W>>::into_word(v)).max().unwrap_or(W::ZERO);
            let min_bits = if max_val == W::ZERO { 1 } else { <W as Word>::BITS - max_val.leading_zeros() as usize };
            // Skip creating a vector with bit_width=0 for empty data, as it's invalid.
            if data.is_empty() && min_bits == 0 { continue; }
            BitWidth::Explicit(min_bits)
        } else {
            strategy
        };
        let strat_name = format!("{:?}", strategy);

        let vec = FixedVec::<T, W, E>::builder()
            .bit_width(bit_width_strategy)
            .build(data)
            .unwrap();

        assert_eq!(vec.len(), data.len(), "Length mismatch {}", context(&strat_name));
        assert_eq!(vec.is_empty(), data.is_empty(), "is_empty mismatch {}", context(&strat_name));

        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(vec.get(i), Some(expected), "get({}) mismatch {}", i, context(&strat_name));
            assert_eq!(unsafe { vec.get_unchecked(i) }, expected, "get_unchecked({}) mismatch {}", i, context(&strat_name));
        }
        assert_eq!(vec.get(data.len()), None, "get(out_of_bounds) should be None {}", context(&strat_name));

        assert_eq!(vec.iter().collect::<Vec<_>>(), data, "iter mismatch {}", context(&strat_name));
        assert_eq!(vec.clone().into_iter().collect::<Vec<_>>(), data, "into_iter mismatch {}", context(&strat_name));

        if data.len() > 1 {
            let mid = data.len() / 2;
            let (left, right) = vec.split_at(mid).unwrap();
            assert_eq!(left.len(), mid, "split_at left len mismatch {}", context(&strat_name));
            assert_eq!(right.len(), data.len() - mid, "split_at right len mismatch {}", context(&strat_name));
            assert_eq!(left.get(0), vec.get(0), "split_at left content mismatch {}", context(&strat_name));
            assert_eq!(right.get(0), vec.get(mid), "split_at right content mismatch {}", context(&strat_name));
        }

        let mut m_vec = vec.clone();
        if !data.is_empty() {
            let mid_idx = data.len() / 2;
            let new_val = data[mid_idx];
            m_vec.set(mid_idx, new_val);
            assert_eq!(m_vec.get(mid_idx), Some(new_val), "set/get mismatch {}", context(&strat_name));
        }

        m_vec.clear();
        assert!(m_vec.is_empty(), "clear failed {}", context(&strat_name));
        
        for &val in data {
            m_vec.push(val);
        }
        assert_eq!(m_vec.iter().collect::<Vec<_>>(), data, "push loop content mismatch {}", context(&strat_name));
        
        for i in (0..data.len()).rev() {
            assert_eq!(m_vec.pop(), Some(data[i]), "pop mismatch {}", context(&strat_name));
        }
        assert!(m_vec.is_empty(), "pop loop end state mismatch {}", context(&strat_name));
    }
}

/// A macro to orchestrate tests across all primitive integer types.
macro_rules! test_all_types {
    ($test_name:ident, $W:ty, $E:ty) => {
        #[test]
        fn $test_name() {
            // Unsigned types
            let u_data_8: Vec<u8> = generate_random_vec(100, 200).into_iter().map(|x| x as u8).collect();
            run_test_for_type::<u8, $W, $E>(&u_data_8, "u8");

            let u_data_16: Vec<u16> = generate_random_vec(100, 50_000).into_iter().map(|x| x as u16).collect();
            run_test_for_type::<u16, $W, $E>(&u_data_16, "u16");

            let u_data_32: Vec<u32> = generate_random_vec(100, 1_000_000_000).into_iter().map(|x| x as u32).collect();
            run_test_for_type::<u32, $W, $E>(&u_data_32, "u32");

            let u_data_64: Vec<u64> = generate_random_vec(100, 1_000_000_000_000);
            run_test_for_type::<u64, $W, $E>(&u_data_64, "u64");

            // Signed types
            let s_data_8: Vec<i8> = generate_random_signed_vec(100, 100).into_iter().map(|x| x as i8).collect();
            run_test_for_type::<i8, $W, $E>(&s_data_8, "i8");

            let s_data_16: Vec<i16> = generate_random_signed_vec(100, 30_000).into_iter().map(|x| x as i16).collect();
            run_test_for_type::<i16, $W, $E>(&s_data_16, "i16");

            let s_data_32: Vec<i32> = generate_random_signed_vec(100, 1_000_000_000).into_iter().map(|x| x as i32).collect();
            run_test_for_type::<i32, $W, $E>(&s_data_32, "i32");
            
            let s_data_64: Vec<i64> = generate_random_signed_vec(100, 1_000_000_000_000);
            run_test_for_type::<i64, $W, $E>(&s_data_64, "i64");
            
            // Empty data
            run_test_for_type::<u32, $W, $E>(&Vec::<u32>::new(), "empty");
        }
    };
}

// Instantiate the test suites for different Word and Endianness combinations.
test_all_types!(le_u64_word, u64, LE);
test_all_types!(be_u64_word, u64, BE);
test_all_types!(le_usize_word, usize, LE);
test_all_types!(be_usize_word, usize, BE);


#[test]
fn test_edge_cases_and_failures() {
    let data = vec![10u8, 255, 100];
    let res = FixedVec::<u8, u64, LE>::builder()
        .bit_width(BitWidth::Explicit(7))
        .build(&data);
    assert!(matches!(res, Err(compressed_intvec::fixed::Error::ValueTooLarge {..})));

    let res_zero_bits = FixedVec::<u8, u64, LE>::builder()
        .bit_width(BitWidth::Explicit(0))
        .build(&data);
    assert!(matches!(res_zero_bits, Err(compressed_intvec::fixed::Error::InvalidParameters(_))));
    
    let mut vec = FixedVec::<u16, u64, LE>::builder().bit_width(BitWidth::Explicit(8)).build(&[0, 0]).unwrap();
    let result = std::panic::catch_unwind(move || {
        vec.set(0, 300);
    });
    assert!(result.is_err(), "set() should panic on value too large");
}

#[test]
fn test_from_iterator() {
    // Unsigned
    let data_u32: Vec<u32> = (0..1000).collect();
    let vec_u32: UFixedVec<u32> = data_u32.iter().copied().collect();
    assert_eq!(vec_u32.len(), 1000);
    assert_eq!(vec_u32.get(123), Some(123));
    assert_eq!(vec_u32, &data_u32[..]);

    // Signed
    let data_i16: Vec<i16> = (-500..500).collect();
    let vec_i16: SFixedVec<i16> = data_i16.iter().copied().collect();
    assert_eq!(vec_i16.len(), 1000);
    assert_eq!(vec_i16.get(0), Some(-500));
    assert_eq!(vec_i16.get(500), Some(0));
    assert_eq!(vec_i16, &data_i16[..]);

    // Empty
    let data_empty: Vec<u64> = vec![];
    let vec_empty: UFixedVec<u64> = data_empty.iter().copied().collect();
    assert!(vec_empty.is_empty());
}

// In tests/fixed/test_fixed.rs

#[test]
fn test_default() {
    let vec_u: UFixedVec<u32> = FixedVec::default();
    assert!(vec_u.is_empty());
    assert_eq!(vec_u.len(), 0);
    assert_eq!(vec_u.bit_width(), 1);
    assert_eq!(vec_u.capacity(), 0);

    let mut vec_s: SFixedVec<i16> = FixedVec::default();
    assert!(vec_s.is_empty());
    vec_s.push(0);
    assert_eq!(vec_s.get(0), Some(0));
    // Pushing a value > 0 requires more than 1 bit for zigzag, so it should panic.
    let result = std::panic::catch_unwind(move || {
        vec_s.push(1);
    });
    assert!(result.is_err());
}