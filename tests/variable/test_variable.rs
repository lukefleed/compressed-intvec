//! # Comprehensive integration tests for the generic `IntVec`.
//!
//! This test suite is designed to validate the functionality of the new generic
//! `IntVec` across a wide range of types, configurations, and edge cases.
//! It is modeled after the test suite for `FixedVec` and uses a macro-based
//! approach to test multiple combinations of integer types and endianness.

use compressed_intvec::variable::codec::VariableCodecSpec;
use compressed_intvec::variable::traits::Storable;
use compressed_intvec::variable::{IntVec, IntVecError, LEIntVec, SIntVec, UIntVec};
use dsi_bitstream::prelude::{BE, LE};
use dsi_bitstream::traits::Endianness;
use num_traits::{AsPrimitive, PrimInt};
use std::fmt::Debug;

// Import helper functions from the common module.
#[path = "../common/mod.rs"]
mod common;
use common::helpers::{generate_random_signed_vec, generate_random_vec};

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Central test function called by the macro for each type combination.
fn run_test_for_type<T, E>(data: &[T], type_name: &str)
where
    T: Storable + Debug + PartialEq + PrimInt + AsPrimitive<u64> + 'static,
    for<'a> IntVec<T, E, &'a [u64]>: PartialEq<&'a [T]>,
    E: Endianness + Send + Sync,
{
    // A list of codecs to test for each configuration.
    // We select a representative subset to keep test times reasonable.
    let codecs_to_test = vec![
        VariableCodecSpec::Gamma,
        VariableCodecSpec::Delta,
        VariableCodecSpec::VByteLe,
        VariableCodecSpec::Auto,
    ];

    for codec_spec in codecs_to_test {
        let context = |op: &str| {
            format!(
                "<{}> on {} with codec {:?} in <{}>",
                type_name,
                op,
                codec_spec,
                std::any::type_name::<E>()
            )
        };

        // Build the IntVec
        let intvec = IntVec::<T, E>::builder(data)
            .k(32) // A reasonable default for testing
            .codec(codec_spec)
            .build()
            .unwrap_or_else(|e| panic!("Build failed: {}", context(&format!("{:?}", e))));

        // Basic property checks
        assert_eq!(intvec.len(), data.len(), "{}", context("len()"));
        assert_eq!(intvec.is_empty(), data.is_empty(), "{}", context("is_empty()"));

        // Test full decompression
        assert_eq!(
            &intvec.clone().into_iter().collect::<Vec<T>>(),
            data,
            "{}",
            context("into_iter")
        );
        assert_eq!(
            &intvec.iter().collect::<Vec<T>>(),
            data,
            "{}",
            context("iter")
        );

        // Test random access
        if !data.is_empty() {
            for i in (0..data.len()).step_by(10.max(data.len() / 10)) {
                assert_eq!(intvec.get(i), Some(data[i]), "{}", context("get()"));
                unsafe {
                    assert_eq!(
                        intvec.get_unchecked(i),
                        data[i],
                        "{}",
                        context("get_unchecked()")
                    );
                }
            }
        }

        // Parallel tests (if enabled)
        #[cfg(feature = "parallel")]
        {
            assert_eq!(
                &intvec.par_iter().collect::<Vec<T>>(),
                data,
                "{}",
                context("par_iter")
            );
        }

        // Test slicing
        if data.len() > 1 {
            let mid = data.len() / 2;
            let slice = intvec.slice(0, mid).unwrap();
            assert_eq!(slice.len(), mid, "{}", context("slice.len()"));
            assert_eq!(slice.get(0), Some(data[0]), "{}", context("slice.get()"));
        }
    }
}

/// A macro to orchestrate tests across all supported primitive integer types.
macro_rules! test_all_types {
    ($test_name:ident, $E:ty) => {
        #[test]
        fn $test_name() {
            // Unsigned types
            let u_data_8: Vec<u8> = generate_random_vec(100, 200).into_iter().map(|x| x as u8).collect();
            run_test_for_type::<u8, $E>(&u_data_8, "u8");

            let u_data_16: Vec<u16> = generate_random_vec(100, 50_000).into_iter().map(|x| x as u16).collect();
            run_test_for_type::<u16, $E>(&u_data_16, "u16");

            let u_data_32: Vec<u32> = generate_random_vec(100, 1_000_000_000).into_iter().map(|x| x as u32).collect();
            run_test_for_type::<u32, $E>(&u_data_32, "u32");

            let u_data_64: Vec<u64> = generate_random_vec(100, 1_000_000_000_000);
            run_test_for_type::<u64, $E>(&u_data_64, "u64");

            // Signed types
            let s_data_8: Vec<i8> = generate_random_signed_vec(100, 100).into_iter().map(|x| x as i8).collect();
            run_test_for_type::<i8, $E>(&s_data_8, "i8");

            let s_data_16: Vec<i16> = generate_random_signed_vec(100, 30_000).into_iter().map(|x| x as i16).collect();
            run_test_for_type::<i16, $E>(&s_data_16, "i16");

            let s_data_32: Vec<i32> = generate_random_signed_vec(100, 1_000_000_000).into_iter().map(|x| x as i32).collect();
            run_test_for_type::<i32, $E>(&s_data_32, "i32");

            let s_data_64: Vec<i64> = generate_random_signed_vec(100, 1_000_000_000_000);
            run_test_for_type::<i64, $E>(&s_data_64, "i64");

            // Edge cases
            run_test_for_type::<u64, $E>(&[], "empty u64");
            run_test_for_type::<i64, $E>(&[], "empty i64");
            run_test_for_type::<u64, $E>(&[0], "single zero");
            run_test_for_type::<i64, $E>(&[0], "single zero signed");
            run_test_for_type::<u64, $E>(&vec![0; 100], "all zeros");
        }
    };
}

// Instantiate the test suites for Little-Endian and Big-Endian.
test_all_types!(generic_tests_le, LE);
test_all_types!(generic_tests_be, BE);

#[test]
fn test_from_iter_builder_generic() {
    // Unsigned
    let data_u32: Vec<u32> = (0..100).collect();
    let intvec_u32 = UIntVec::<u32>::from_iter_builder(data_u32.clone())
        .codec(VariableCodecSpec::Gamma)
        .build()
        .unwrap();
    assert_eq!(intvec_u32.len(), data_u32.len());
    assert_eq!(intvec_u32.get(50), Some(50));

    // Signed
    let data_i16: Vec<i16> = (-50..50).collect();
    let intvec_i16 = SIntVec::<i16>::from_iter_builder(data_i16.clone())
        .codec(VariableCodecSpec::Delta)
        .build()
        .unwrap();
    assert_eq!(intvec_i16.len(), data_i16.len());
    assert_eq!(intvec_i16.get(0), Some(-50));
    assert_eq!(intvec_i16.get(50), Some(0));
}

#[test]
fn test_builder_rejects_auto_on_iter() {
    let data: Vec<i32> = vec![-10, 20, 100];
    let result = SIntVec::<i32>::from_iter_builder(data)
        .codec(VariableCodecSpec::Auto)
        .build();
    assert!(matches!(result, Err(IntVecError::InvalidParameters(_))));
}

// A simple test for binary search on a sorted vector.
#[test]
fn test_binary_search() {
    let data: Vec<u64> = (0..100).map(|x| x * 2).collect();
    let intvec = LEIntVec::builder(&data).build().unwrap();
    assert_eq!(intvec.binary_search(&10), Ok(5));
    assert_eq!(intvec.binary_search(&11), Err(6));
    assert_eq!(intvec.binary_search(&0), Ok(0));
    assert_eq!(intvec.binary_search(&198), Ok(99));
    assert_eq!(intvec.binary_search(&199), Err(100));
}
