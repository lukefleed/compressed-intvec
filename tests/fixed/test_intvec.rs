//! Integration tests for `FixedVec`.

use compressed_intvec::prelude::*;
use dsi_bitstream::prelude::{BE, LE};
use rand::{rngs::StdRng, Rng, SeedableRng};

// Import helper functions from the common module.
#[path = "../common/mod.rs"]
mod common;
use common::helpers::generate_random_vec;

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

macro_rules! test_configuration {
    ($test_name:ident, $endianness:ty, $input:expr, $bit_width:expr) => {
        #[test]
        fn $test_name() {
            let input = &$input;
            let bit_width = $bit_width;

            // Build the FixedVec
            let fixed_vec: FixedVec<$endianness> = FixedVec::builder(input)
                .bit_width(bit_width)
                .build()
                .unwrap();

            // Basic property checks
            assert_eq!(fixed_vec.len(), input.len());
            assert_eq!(fixed_vec.is_empty(), input.is_empty());
            if let BitWidth::Explicit(n) = bit_width {
                assert_eq!(fixed_vec.num_bits(), n);
            }

            // Test full decompression
            assert_eq!(&fixed_vec.clone().into_vec(), input, "into_vec failed");
            assert_eq!(&fixed_vec.iter().collect::<Vec<_>>(), input, "iter failed");

            // Test PartialEq implementations
            assert_eq!(fixed_vec, fixed_vec.clone(), "PartialEq with self failed");
            // Correctly compare with a slice
            assert_eq!(fixed_vec, &input[..], "PartialEq with slice failed");

            // Test as_limbs
            let cloned_limbs = fixed_vec.limbs();
            assert_eq!(fixed_vec.as_limbs(), cloned_limbs.as_slice());

            // Test a non-equal case
            if input.len() > 1 && input[0] != input[1] {
                let mut different_input = input.clone();
                different_input.swap(0, 1);
                let different_vec = FixedVec::<$endianness>::builder(&different_input)
                    .bit_width(bit_width)
                    .build()
                    .unwrap();
                assert_ne!(
                    fixed_vec, different_vec,
                    "PartialEq with different vec should fail"
                );
            }

            if !input.is_empty() {
                let mut rng = StdRng::seed_from_u64(42);
                let num_indices = 100.min(input.len());
                let indices: Vec<usize> = (0..num_indices)
                    .map(|_| rng.random_range(0..input.len()))
                    .collect();
                let expected: Vec<u64> = indices.iter().map(|&i| input[i]).collect();

                // Test safe accessors
                for &idx in &indices {
                    assert_eq!(fixed_vec.get(idx), Some(input[idx]), "get failed");
                }
                assert_eq!(
                    fixed_vec.get_many(&indices).unwrap(),
                    expected,
                    "get_many failed"
                );

                // Test unsafe accessors
                unsafe {
                    for &idx in &indices {
                        assert_eq!(
                            fixed_vec.get_unchecked(idx),
                            input[idx],
                            "get_unchecked failed"
                        );
                    }
                    assert_eq!(
                        fixed_vec.get_many_unchecked(&indices),
                        expected,
                        "get_many_unchecked failed"
                    );
                }

                // Parallel tests
                #[cfg(feature = "parallel")]
                {
                    assert_eq!(
                        &fixed_vec.par_iter().collect::<Vec<_>>(),
                        input,
                        "par_iter failed"
                    );
                    assert_eq!(
                        fixed_vec.par_get_many(&indices).unwrap(),
                        expected,
                        "par_get_many failed"
                    );
                    unsafe {
                        assert_eq!(
                            fixed_vec.par_get_many_unchecked(&indices),
                            expected,
                            "par_get_many_unchecked failed"
                        );
                    }
                }
            } else {
                // Special checks for empty vec
                assert!(fixed_vec.get(0).is_none());
                assert_eq!(fixed_vec.get_many(&[]).unwrap(), Vec::<u64>::new());
                unsafe {
                    assert_eq!(fixed_vec.get_many_unchecked(&[]), Vec::<u64>::new());
                }
            }
        }
    };
}

// TEST SUITE

// Empty Vector
test_configuration!(
    test_empty_le,
    LE,
    generate_random_vec(0, 0),
    BitWidth::Explicit(8)
);
test_configuration!(
    test_empty_be,
    BE,
    generate_random_vec(0, 0),
    BitWidth::Minimal
);

// Single Element Vector
test_configuration!(
    test_single_element_le,
    LE,
    vec![42u64],
    BitWidth::Explicit(7)
);
test_configuration!(
    test_single_element_auto_bits_be,
    BE,
    vec![1000u64],
    BitWidth::Minimal
);

// Zeros Vector
test_configuration!(test_zeros_le, LE, vec![0u64; 1000], BitWidth::Explicit(1));
test_configuration!(
    test_zeros_auto_bits_be,
    BE,
    vec![0u64; 1000],
    BitWidth::Minimal
);

// Uniform Distributions
test_configuration!(
    test_uniform_small_le,
    LE,
    generate_random_vec(1000, 100),
    BitWidth::Explicit(7) // 100 requires 7 bits
);
test_configuration!(
    test_uniform_large_auto_bits_be,
    BE,
    generate_random_vec(1000, 1_000_000),
    BitWidth::Minimal
);

// Full 64-bit values
test_configuration!(
    test_full_64_bits_be,
    BE,
    vec![u64::MAX - 1, u64::MAX],
    BitWidth::Explicit(64)
);

#[test]
fn test_invalid_parameters() {
    // num_bits > 64
    let result = LEFixedVec::builder(&[1u64, 2, 3])
        .bit_width(BitWidth::Explicit(65))
        .build();
    assert!(matches!(result, Err(FixedVecError::InvalidParameters(_))));

    // Value too large for specified bits
    let input_large = vec![10u64, 20, 256]; // 256 requires 9 bits
    let result_large = LEFixedVec::builder(&input_large)
        .bit_width(BitWidth::Explicit(8))
        .build();
    assert!(matches!(
        result_large,
        Err(FixedVecError::ValueTooLarge { .. })
    ));
}

#[test]
fn test_out_of_bounds() {
    let input = vec![10u64, 20, 30];
    let fixed_vec = LEFixedVec::builder(&input).build().unwrap();
    assert!(matches!(
        fixed_vec.get_many(&[0, 1, 3]),
        Err(FixedVecError::IndexOutOfBounds(3))
    ));
}

#[test]
fn test_from_iter_builder() {
    let data: Vec<u64> = (0..1000).collect();

    // Success case
    let fixed_vec = LEFixedVec::from_iter_builder(data.clone(), 10)
        .build()
        .unwrap();
    assert_eq!(fixed_vec.len(), data.len());
    assert_eq!(fixed_vec.get(500), Some(500));
    assert_eq!(fixed_vec.clone().into_vec(), data);

    // Failure case: Value too large for specified bits
    let data_too_large = vec![10, 20, 256];
    let result = LEFixedVec::from_iter_builder(data_too_large, 8).build();
    assert!(matches!(result, Err(FixedVecError::ValueTooLarge { .. })));
}

#[test]
fn test_build_from_u32_slice() {
    let data_u32: Vec<u32> = (0..1000).map(|x| x * 2).collect();
    let fixed_vec = LEFixedVec::builder(&data_u32).build().unwrap();

    assert_eq!(fixed_vec.len(), data_u32.len());
    assert_eq!(fixed_vec.get(500), Some(1000));

    let expected_u64: Vec<u64> = data_u32.iter().map(|&x| x as u64).collect();
    assert_eq!(fixed_vec, expected_u64.as_slice());
}

#[test]
fn test_bit_width_power_of_two() {
    // 5 bits required -> rounds up to 8
    let data0 = vec![1u64, 2, 3, 4, 31]; // max needs 5 bits
    let vec0 = LEFixedVec::builder(&data0)
        .bit_width(BitWidth::PowerOfTwo)
        .build()
        .unwrap();
    assert_eq!(vec0.num_bits(), 8);
    assert_eq!(vec0, &data0[..]);

    // 8 bits required -> remains 8
    let data1 = vec![1u64, 2, 3, 4, 255]; // max needs 8 bits
    let vec1 = LEFixedVec::builder(&data1)
        .bit_width(BitWidth::PowerOfTwo)
        .build()
        .unwrap();
    assert_eq!(vec1.num_bits(), 8);
    assert_eq!(vec1, &data1[..]);

    // 9 bits required -> rounds up to 16
    let data2 = vec![1u64, 2, 3, 4, 511]; // max needs 9 bits
    let vec2 = LEFixedVec::builder(&data2)
        .bit_width(BitWidth::PowerOfTwo)
        .build()
        .unwrap();
    assert_eq!(vec2.num_bits(), 16);
    assert_eq!(vec2, &data2[..]);

    // 16 bits required -> remains 16
    let data3 = vec![1u64, 2, 3, 4, 65535]; // max needs 16 bits
    let vec3 = LEFixedVec::builder(&data3)
        .bit_width(BitWidth::PowerOfTwo)
        .build()
        .unwrap();
    assert_eq!(vec3.num_bits(), 16);
    assert_eq!(vec3, &data3[..]);

    // 17 bits required -> rounds up to 32
    let data4 = vec![1u64, 2, 3, 4, 131071]; // max needs 17 bits
    let vec4 = LEFixedVec::builder(&data4)
        .bit_width(BitWidth::PowerOfTwo)
        .build()
        .unwrap();
    assert_eq!(vec4.num_bits(), 32);
    assert_eq!(vec4, &data4[..]);

    // 32 bits required -> remains 32
    let data5 = vec![1u64, 2, 3, 4, 4294967295]; // max needs 32 bits
    let vec5 = LEFixedVec::builder(&data5)
        .bit_width(BitWidth::PowerOfTwo)
        .build()
        .unwrap();
    assert_eq!(vec5.num_bits(), 32);
    assert_eq!(vec5, &data5[..]);

    // 33 bits required -> rounds up to 64
    let data6 = vec![1u64, 2, 3, 4, 8589934591]; // max needs 33 bits
    let vec6 = LEFixedVec::builder(&data6)
        .bit_width(BitWidth::PowerOfTwo)
        .build()
        .unwrap();
    assert_eq!(vec6.num_bits(), 64);
    assert_eq!(vec6, &data6[..]);
}

#[test]
fn test_edge_case_zero_bits() {
    let result = LEFixedVec::builder(&[0u64])
        .bit_width(BitWidth::Explicit(0))
        .build();
    // Building with 0 bits should be an error if there is data.
    // If the vector is empty, it might be allowed, but `get` would be impossible.
    // Let's enforce it's invalid for non-empty slices.
    assert!(matches!(result, Err(FixedVecError::InvalidParameters(_))));

    // Test building an empty vector with 0 bits.
    let empty_vec = LEFixedVec::builder(&Vec::<u64>::new())
        .bit_width(BitWidth::Explicit(0))
        .build()
        .unwrap();
    assert_eq!(empty_vec.len(), 0);
    assert_eq!(empty_vec.num_bits(), 0);
}

#[test]
fn test_edge_case_u64_max() {
    let data = vec![u64::MAX];
    let vec = LEFixedVec::builder(&data)
        .bit_width(BitWidth::Minimal)
        .build()
        .unwrap();
    assert_eq!(vec.num_bits(), 64);
    assert_eq!(vec.get(0), Some(u64::MAX));
}
