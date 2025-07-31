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
    ($test_name:ident, $endianness:ty, $input:expr, $num_bits:expr) => {
        #[test]
        fn $test_name() {
            let input = &$input;
            let num_bits = $num_bits;

            // Build the SFixedVec
            let s_fixed_vec = SFixedVec::<$endianness>::builder(input)
                .num_bits(num_bits)
                .build()
                .unwrap();

            // Basic property checks
            assert_eq!(s_fixed_vec.len(), input.len());
            assert_eq!(s_fixed_vec.is_empty(), input.is_empty());
            if num_bits.is_some() {
                assert_eq!(s_fixed_vec.num_bits(), num_bits.unwrap());
            }

            // Test full decompression
            assert_eq!(
                &s_fixed_vec.clone().iter().collect::<Vec<_>>(),
                input,
                "iter failed"
            );

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
                        &s_fixed_vec.par_iter().collect::<Vec<_>>(),
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
                assert_eq!(s_fixed_vec.get_many(&[]).unwrap(), vec![]);
                unsafe {
                    assert_eq!(s_fixed_vec.get_many_unchecked(&[]), vec![]);
                }
            }
        }
    };
}

// TEST SUITE

// Empty Vector
test_sintvec_configuration!(test_empty_le, LE, generate_random_signed_vec(0, 0), Some(8));
test_sintvec_configuration!(
    test_empty_auto_bits_be,
    BE,
    generate_random_signed_vec(0, 0),
    None
);

// Single Element Vector
test_sintvec_configuration!(test_single_element_le, LE, vec![-42], Some(7)); // -42 -> 83, needs 7 bits
test_sintvec_configuration!(test_single_element_auto_bits_be, BE, vec![-500], None); // -500 -> 999, needs 10 bits

// Zeros Vector
test_sintvec_configuration!(test_zeros_le, LE, vec![0; 1000], Some(1));
test_sintvec_configuration!(test_zeros_auto_bits_be, BE, vec![0; 1000], None);

// Mixed positive and negative values
test_sintvec_configuration!(
    test_mixed_values_le,
    LE,
    generate_random_signed_vec(1000, 500), // Range [-499, 499] -> max zigzag is 998, needs 10 bits
    Some(10)
);
test_sintvec_configuration!(
    test_mixed_values_auto_bits_be,
    BE,
    generate_random_signed_vec(1000, 1_000_000),
    None
);

#[test]
fn test_invalid_parameters() {
    // num_bits > 64
    let result = LESFixedVec::builder(&[-1, 2, -3])
        .num_bits(Some(65))
        .build();
    assert!(matches!(result, Err(FixedVecError::InvalidParameters(_))));

    // Value (after zigzag) too large for specified bits
    let input_large = vec![-10, 20, 128]; // zigzag(128) = 256, requires 9 bits
    let result_large = LESFixedVec::builder(&input_large).num_bits(Some(8)).build();
    assert!(matches!(
        result_large,
        Err(FixedVecError::ValueTooLarge { .. })
    ));
}
