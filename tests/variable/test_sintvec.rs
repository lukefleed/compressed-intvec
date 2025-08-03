//! Integration tests for `SIntVec`.

use compressed_intvec::prelude::*;
use dsi_bitstream::{
    prelude::{ToInt, ToNat},
    traits::{BE, LE},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

// Import helper functions from the common module.
#[path = "../common/mod.rs"]
mod common;
use common::helpers::generate_random_signed_vec;

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

macro_rules! test_sintvec_configuration {
    ($test_name:ident, $endianness:ty, $input:expr, $k:expr, $codec_spec:expr) => {
        #[test]
        fn $test_name() {
            let input: &[i64] = &$input;
            let k = $k;
            let codec_spec = $codec_spec;

            // Build the SIntVec
            let sintvec = SIntVec::<$endianness>::builder(input)
                .k(k)
                .codec(codec_spec)
                .build()
                .unwrap();

            // Basic property checks
            assert_eq!(sintvec.len(), input.len());
            assert_eq!(sintvec.is_empty(), input.is_empty());

            // Test full decompression
            assert_eq!(&sintvec.iter().collect::<Vec<i64>>(), input, "iter failed");

            if !input.is_empty() {
                let mut rng = StdRng::seed_from_u64(42);
                let num_indices = 100.min(input.len());
                let indices: Vec<usize> = (0..num_indices)
                    .map(|_| rng.random_range(0..input.len()))
                    .collect();
                let expected: Vec<i64> = indices.iter().map(|&i| input[i]).collect();

                // Test safe accessors
                for &idx in &indices {
                    assert_eq!(sintvec.get(idx), Some(input[idx]), "get failed");
                }
                assert_eq!(
                    sintvec.get_many(&indices).unwrap(),
                    expected,
                    "get_many failed"
                );

                // Test unsafe accessors
                unsafe {
                    for &idx in &indices {
                        assert_eq!(
                            sintvec.get_unchecked(idx),
                            input[idx],
                            "get_unchecked failed"
                        );
                    }
                    assert_eq!(
                        sintvec.get_many_unchecked(&indices),
                        expected,
                        "get_many_unchecked failed"
                    );
                }

                // Parallel tests
                #[cfg(feature = "parallel")]
                {
                    assert_eq!(
                        &sintvec.par_iter().collect::<Vec<i64>>(),
                        input,
                        "par_iter failed"
                    );
                    assert_eq!(
                        sintvec.par_get_many(&indices).unwrap(),
                        expected,
                        "par_get_many failed"
                    );
                    assert_eq!(
                        sintvec.par_get_many_unchecked(&indices),
                        expected,
                        "par_get_many_unchecked failed"
                    );
                }
            } else {
                // Special checks for empty vec
                assert!(sintvec.get(0).is_none());
                assert_eq!(sintvec.get_many(&[]).unwrap(), Vec::<i64>::new());
                unsafe {
                    assert_eq!(sintvec.get_many_unchecked(&[]), Vec::<i64>::new());
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
    32,
    VariableCodecSpec::Gamma
);

// Single Element Vector
test_sintvec_configuration!(
    test_single_gamma_le,
    LE,
    vec![-42i64],
    1,
    VariableCodecSpec::Gamma
);

// Mixed positive and negative values
test_sintvec_configuration!(
    test_mixed_values_delta_be,
    BE,
    generate_random_signed_vec(1000, 1_000_000),
    32,
    VariableCodecSpec::Delta
);

// Test k that is NOT a power of two
test_sintvec_configuration!(
    test_k_non_power_of_two,
    LE,
    generate_random_signed_vec(1000, 1000),
    24, // Not a power of two
    VariableCodecSpec::Gamma
);

#[test]
fn test_sintvec_all_codecs_systematic() {
    let data = generate_random_signed_vec(1000, 1000);
    let mut rng = StdRng::seed_from_u64(42);
    let test_indices: Vec<usize> = (0..50).map(|_| rng.random_range(0..data.len())).collect();

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
    ];

    for codec_spec in codecs_to_test {
        // Skip Unary for this test as it's too slow for non-trivial values
        if codec_spec == VariableCodecSpec::Unary {
            continue;
        }

        let sintvec = SIntVec::<LE>::builder(&data)
            .codec(codec_spec)
            .build()
            .unwrap_or_else(|e| panic!("Build failed for {:?}: {:?}", codec_spec, e));

        assert_eq!(
            &sintvec.iter().collect::<Vec<i64>>(),
            &data,
            "iter failed for {:?}",
            codec_spec
        );

        let expected: Vec<i64> = test_indices.iter().map(|&i| data[i]).collect();
        assert_eq!(
            sintvec.get_many(&test_indices).unwrap(),
            expected,
            "get_many failed for {:?}",
            codec_spec
        );
    }
}

#[test]
fn test_sintvec_builder_rejects_auto_codecs() {
    let data = vec![-10i64, 20, 100];
    let codecs_with_auto_params = vec![
        VariableCodecSpec::Auto,
        VariableCodecSpec::Rice { log2_b: None },
        VariableCodecSpec::Zeta { k: None },
    ];
    for codec in codecs_with_auto_params {
        let result = LESIntVec::builder(&data).codec(codec).build();
        // The SIntVec builder uses the from_iter_builder of the underlying IntVec,
        // which rejects auto codecs.
        assert!(matches!(result, Err(IntVecError::InvalidParameters(_))));
    }
}

#[test]
fn test_zigzag_identities() {
    let values: Vec<i64> = vec![0, -1, 1, -2, 2, i64::MAX, i64::MIN];
    for &v in &values {
        assert_eq!(v.to_nat().to_int(), v);
    }
}
