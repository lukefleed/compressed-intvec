//! Integration tests for `SFixedVec`.

use compressed_intvec::prelude::*;
use dsi_bitstream::prelude::{BE, LE};
use rand::{rngs::StdRng, Rng, SeedableRng};

// Import helper functions from the common module.
#[path = "../common/mod.rs"]
mod common;
use common::helpers::generate_random_signed_vec;

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

macro_rules! test_sintvec_configuration {
    ($test_name:ident, $endianness:ty, $input:expr, $bit_width:expr) => {
        #[test]
        fn $test_name() {
            let input: &[i64] = &$input;
            let bit_width = $bit_width;

            // Build the SFixedVec
            let s_fixed_vec = SFixedVec::<$endianness>::builder(input)
                .bit_width(bit_width)
                .build()
                .unwrap();

            // Basic property checks
            assert_eq!(s_fixed_vec.len(), input.len());
            assert_eq!(s_fixed_vec.is_empty(), input.is_empty());
            if let BitWidth::Explicit(n) = bit_width {
                assert_eq!(s_fixed_vec.num_bits(), n);
            }

            // Test full decompression
            assert_eq!(
                &s_fixed_vec.clone().into_vec(),
                input,
                "into_vec from owned failed"
            );
            assert_eq!(
                &s_fixed_vec.iter().collect::<Vec<i64>>(),
                input,
                "iter failed"
            );

            // Test PartialEq implementations
            assert_eq!(
                s_fixed_vec,
                s_fixed_vec.clone(),
                "PartialEq with self failed"
            );
            // Correctly compare with a slice
            assert_eq!(s_fixed_vec, &input[..], "PartialEq with slice failed");

            // Test as_limbs
            let cloned_limbs = s_fixed_vec.as_limbs().to_vec();
            assert_eq!(s_fixed_vec.as_limbs(), cloned_limbs.as_slice());

            // Test a non-equal case
            if input.len() > 1 && input[0] != input[1] {
                let mut different_input = input.to_vec();
                different_input.swap(0, 1);
                let different_vec = SFixedVec::<$endianness>::builder(&different_input)
                    .bit_width(bit_width)
                    .build()
                    .unwrap();
                assert_ne!(
                    s_fixed_vec, different_vec,
                    "PartialEq with different vec should fail"
                );
            }

            if !input.is_empty() {
                let mut rng = StdRng::seed_from_u64(42);
                let num_indices = 100.min(input.len());
                let indices: Vec<usize> = (0..num_indices)
                    .map(|_| rng.random_range(0..input.len()))
                    .collect();
                let expected: Vec<i64> = indices.iter().map(|&i| input[i]).collect();

                // Test safe accessors
                for &idx in &indices {
                    assert_eq!(s_fixed_vec.get(idx), Some(input[idx]), "get failed");
                }
                assert_eq!(
                    s_fixed_vec.get_many(&indices).unwrap(),
                    expected,
                    "get_many failed"
                );

                // Test unsafe accessors
                unsafe {
                    for &idx in &indices {
                        assert_eq!(
                            s_fixed_vec.get_unchecked(idx),
                            input[idx],
                            "get_unchecked failed"
                        );
                    }
                    assert_eq!(
                        s_fixed_vec.get_many_unchecked(&indices),
                        expected,
                        "get_many_unchecked failed"
                    );
                }

                // Parallel tests
                #[cfg(feature = "parallel")]
                {
                    assert_eq!(
                        &s_fixed_vec.par_iter().collect::<Vec<i64>>(),
                        input,
                        "par_iter failed"
                    );
                    assert_eq!(
                        s_fixed_vec.par_get_many(&indices).unwrap(),
                        expected,
                        "par_get_many failed"
                    );
                    unsafe {
                        assert_eq!(
                            s_fixed_vec.par_get_many_unchecked(&indices),
                            expected,
                            "par_get_many_unchecked failed"
                        );
                    }
                }
            } else {
                // Special checks for empty vec
                assert!(s_fixed_vec.get(0).is_none());
                assert_eq!(s_fixed_vec.get_many(&[]).unwrap(), Vec::<i64>::new());
                unsafe {
                    assert_eq!(s_fixed_vec.get_many_unchecked(&[]), Vec::<i64>::new());
                }
            }
        }
    };
}

// TEST SUITE

// Empty Vector
test_sintvec_configuration!(
    test_empty_le,
    LE,
    generate_random_signed_vec(0, 0),
    BitWidth::Explicit(8)
);
test_sintvec_configuration!(
    test_empty_auto_bits_be,
    BE,
    generate_random_signed_vec(0, 0),
    BitWidth::Minimal
);

// Single Element Vector
test_sintvec_configuration!(
    test_single_element_le,
    LE,
    vec![-42i64],
    BitWidth::Explicit(7)
); // -42 -> 83, needs 7 bits
test_sintvec_configuration!(
    test_single_element_auto_bits_be,
    BE,
    vec![-500i64],
    BitWidth::Minimal
); // -500 -> 999, needs 10 bits

// Zeros Vector
test_sintvec_configuration!(test_zeros_le, LE, vec![0i64; 1000], BitWidth::Explicit(1));
test_sintvec_configuration!(
    test_zeros_auto_bits_be,
    BE,
    vec![0i64; 1000],
    BitWidth::Minimal
);

// Mixed positive and negative values
test_sintvec_configuration!(
    test_mixed_values_le,
    LE,
    generate_random_signed_vec(1000, 500), // Range [-499, 499] -> max zigzag is 998, needs 10 bits
    BitWidth::Explicit(10)
);
test_sintvec_configuration!(
    test_mixed_values_auto_bits_be,
    BE,
    generate_random_signed_vec(1000, 1_000_000),
    BitWidth::Minimal
);

#[test]
fn test_invalid_parameters() {
    // num_bits > 64
    let result = LESFixedVec::builder(&[-1i64, 2, -3])
        .bit_width(BitWidth::Explicit(65))
        .build();
    assert!(matches!(result, Err(FixedVecError::InvalidParameters(_))));

    // Value (after zigzag) too large for specified bits
    let input_large = vec![-10i64, 20, 128]; // zigzag(128) = 256, requires 9 bits
    let result_large = LESFixedVec::builder(&input_large)
        .bit_width(BitWidth::Explicit(8))
        .build();
    assert!(matches!(
        result_large,
        Err(FixedVecError::ValueTooLarge { .. })
    ));
}

#[test]
fn test_build_from_i32_slice() {
    let data_i32: Vec<i32> = (-500..500).collect();
    let s_fixed_vec = LESFixedVec::builder(&data_i32).build().unwrap();

    assert_eq!(s_fixed_vec.len(), data_i32.len());
    assert_eq!(s_fixed_vec.get(0), Some(-500));
    assert_eq!(s_fixed_vec.get(999), Some(499));

    let expected_i64: Vec<i64> = data_i32.iter().map(|&x| x as i64).collect();
    assert_eq!(s_fixed_vec, expected_i64.as_slice());
}

#[test]
fn test_edge_case_i64_min_max() {
    // i64::MIN ZigZag-encodes to u64::MAX.
    // i64::MAX ZigZag-encodes to u64::MAX - 1.
    let data = vec![i64::MIN, i64::MAX];
    let vec = LESFixedVec::builder(&data)
        .bit_width(BitWidth::Minimal)
        .build()
        .unwrap();

    // The presence of i64::MIN should force the bit width to 64.
    assert_eq!(vec.num_bits(), 64);
    assert_eq!(vec.get(0), Some(i64::MIN));
    assert_eq!(vec.get(1), Some(i64::MAX));
}
