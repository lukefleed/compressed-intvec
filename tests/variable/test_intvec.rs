//! Integration tests for `IntVec`.

use compressed_intvec::prelude::*;
use dsi_bitstream::prelude::{len_delta, len_gamma, BE, LE};
use rand::{rngs::StdRng, Rng, SeedableRng};

// Import helper functions from the common module.
#[path = "../common/mod.rs"]
mod common;
use common::helpers::{generate_random_vec, generate_with_distribution};

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

macro_rules! test_configuration {
    ($test_name:ident, $endianness:ty, $input:expr, $k:expr, $codec_spec:expr) => {
        #[test]
        fn $test_name() {
            let input: &[u64] = &$input;
            let k = $k;
            let codec_spec = $codec_spec;

            // Build the IntVec
            let intvec = IntVec::<$endianness>::builder(input)
                .k(k)
                .codec(codec_spec)
                .build()
                .unwrap();

            // Basic property checks
            assert_eq!(intvec.len(), input.len());
            assert_eq!(intvec.is_empty(), input.is_empty());
            assert_eq!(intvec.get_sampling_rate(), k);
            if input.is_empty() {
                assert_eq!(intvec.get_num_samples(), 0);
            } else {
                let expected_samples = (input.len() + k - 1) / k;
                assert_eq!(intvec.get_num_samples(), expected_samples);
            }

            // Test full decompression
            assert_eq!(&intvec.clone().into_vec(), input, "into_vec failed");
            assert_eq!(&intvec.iter().collect::<Vec<u64>>(), input, "iter failed");

            if !input.is_empty() {
                let mut rng = StdRng::seed_from_u64(42);
                let num_indices = 100.min(input.len());
                let indices: Vec<usize> = (0..num_indices)
                    .map(|_| rng.random_range(0..input.len()))
                    .collect();
                let expected: Vec<u64> = indices.iter().map(|&i| input[i]).collect();

                // Test safe accessors
                for &idx in &indices {
                    assert_eq!(intvec.get(idx), Some(input[idx]), "get failed");
                }
                assert_eq!(
                    intvec.get_many(&indices).unwrap(),
                    expected,
                    "get_many failed"
                );

                // Test unsafe accessors
                unsafe {
                    for &idx in &indices {
                        assert_eq!(
                            intvec.get_unchecked(idx),
                            input[idx],
                            "get_unchecked failed"
                        );
                    }
                    assert_eq!(
                        intvec.get_many_unchecked(&indices),
                        expected,
                        "get_many_unchecked failed"
                    );
                }

                // Parallel tests
                #[cfg(feature = "parallel")]
                {
                    assert_eq!(
                        &intvec.par_iter().collect::<Vec<u64>>(),
                        input,
                        "par_iter failed"
                    );
                    assert_eq!(
                        intvec.par_get_many(&indices).unwrap(),
                        expected,
                        "par_get_many failed"
                    );
                    assert_eq!(
                        unsafe { intvec.par_get_many_unchecked(&indices) },
                        expected,
                        "par_get_many_unchecked failed"
                    );
                }
            } else {
                // Special checks for empty vec
                assert!(intvec.get(0).is_none());
                assert_eq!(intvec.get_many(&[]).unwrap(), Vec::<u64>::new());
                unsafe {
                    assert_eq!(intvec.get_many_unchecked(&[]), Vec::<u64>::new());
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
    32,
    VariableCodecSpec::Auto
);
test_configuration!(
    test_empty_be,
    BE,
    generate_random_vec(0, 0),
    32,
    VariableCodecSpec::Auto
);

// Single Element Vector
test_configuration!(
    test_single_gamma_le,
    LE,
    vec![42u64],
    1,
    VariableCodecSpec::Gamma
);
test_configuration!(
    test_single_gamma_be,
    BE,
    vec![42u64],
    1,
    VariableCodecSpec::Gamma
);

// Zeros Vector
test_configuration!(
    test_zeros_auto_le,
    LE,
    vec![0u64; 1000],
    16,
    VariableCodecSpec::Auto
);
test_configuration!(
    test_zeros_unary_be,
    BE,
    vec![0u64; 1000],
    16,
    VariableCodecSpec::Unary
);

// Uniform Distributions
test_configuration!(
    test_uniform_small_auto_le,
    LE,
    generate_random_vec(1000, 100),
    32,
    VariableCodecSpec::Auto
);
test_configuration!(
    test_uniform_large_auto_be,
    BE,
    generate_random_vec(1000, 1_000_000),
    32,
    VariableCodecSpec::Auto
);

// Specific Code Distributions
test_configuration!(
    test_gamma_dist_le,
    LE,
    generate_with_distribution(1000, len_gamma),
    32,
    VariableCodecSpec::Gamma
);
test_configuration!(
    test_delta_dist_be,
    BE,
    generate_with_distribution(1000, len_delta),
    32,
    VariableCodecSpec::Delta
);

// Explicit Code Parameters
test_configuration!(
    test_explicit_zeta_le,
    LE,
    generate_random_vec(500, 1000),
    16,
    VariableCodecSpec::Zeta { k: Some(5) }
);
test_configuration!(
    test_explicit_rice_be,
    BE,
    generate_random_vec(500, 1000),
    16,
    VariableCodecSpec::Rice { log2_b: Some(4) }
);

// Test k that is NOT a power of two
test_configuration!(
    test_k_non_power_of_two,
    LE,
    generate_random_vec(1000, 1000),
    24, // Not a power of two
    VariableCodecSpec::Auto
);

// Edge case for sampling rate
test_configuration!(
    test_k_equals_len,
    LE,
    generate_random_vec(100, 100),
    100,
    VariableCodecSpec::Auto
);

#[test]
fn test_invalid_parameters() {
    let input = vec![1u64, 2, 3];
    // k=0 is invalid
    let result = IntVec::<LE>::builder(&input)
        .k(0)
        .codec(VariableCodecSpec::Gamma)
        .build();
    assert!(matches!(result, Err(IntVecError::InvalidParameters(_))));
}

#[test]
fn test_out_of_bounds() {
    let input = vec![10u64, 20, 30];
    let intvec = IntVec::<LE>::builder(&input).build().unwrap();
    assert!(matches!(
        intvec.get_many(&[0, 1, 3]),
        Err(IntVecError::IndexOutOfBounds(3))
    ));
}

#[test]
fn test_all_codecs_systematic() {
    let input = generate_random_vec(1000, 1000);
    let mut rng = StdRng::seed_from_u64(42);
    let test_indices: Vec<usize> = (0..50).map(|_| rng.random_range(0..input.len())).collect();

    let codecs_to_test = vec![
        VariableCodecSpec::Gamma,
        VariableCodecSpec::Delta,
        VariableCodecSpec::Unary,
        VariableCodecSpec::Omega,
        VariableCodecSpec::VByteLe,
        VariableCodecSpec::VByteBe,
        VariableCodecSpec::Rice { log2_b: Some(4) },
        VariableCodecSpec::Zeta { k: Some(3) },
        VariableCodecSpec::Golomb { b: Some(8) },
        VariableCodecSpec::Pi { k: Some(3) },
        VariableCodecSpec::ExpGolomb { k: Some(2) },
        VariableCodecSpec::Auto,
    ];

    for codec_spec in codecs_to_test {
        let intvec = IntVec::<LE>::builder(&input)
            .codec(codec_spec)
            .build()
            .unwrap_or_else(|e| panic!("Build failed for {:?}: {:?}", codec_spec, e));

        assert_eq!(
            &intvec.clone().into_vec(),
            &input,
            "into_vec failed for {:?}",
            codec_spec
        );

        let expected: Vec<u64> = test_indices.iter().map(|&i| input[i]).collect();
        assert_eq!(
            intvec.get_many(&test_indices).unwrap(),
            expected,
            "get_many failed for {:?}",
            codec_spec
        );
        unsafe {
            assert_eq!(
                intvec.get_many_unchecked(&test_indices),
                expected,
                "get_many_unchecked failed for {:?}",
                codec_spec
            );
        }
    }
}

#[test]
fn test_from_iter_builder() {
    let data: Vec<u64> = (0..1000).collect();

    // Success case
    let intvec_gamma = LEIntVec::from_iter_builder(data.clone())
        .codec(VariableCodecSpec::Gamma)
        .k(16)
        .build()
        .unwrap();
    assert_eq!(intvec_gamma.len(), data.len());
    assert_eq!(intvec_gamma.get(500), Some(500));
    assert_eq!(intvec_gamma.clone().into_vec(), data);

    // Failure cases: Automatic parameter selection
    let codecs_with_auto_params = vec![
        VariableCodecSpec::Auto,
        VariableCodecSpec::Rice { log2_b: None },
        VariableCodecSpec::Zeta { k: None },
    ];
    for codec in codecs_with_auto_params {
        let result = LEIntVec::from_iter_builder(data.clone().into_iter())
            .codec(codec)
            .build();
        assert!(
            matches!(result, Err(IntVecError::InvalidParameters(_))),
            "Expected failure for codec: {:?}",
            codec
        );
    }
}
