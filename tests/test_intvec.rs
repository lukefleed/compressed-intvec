//! Integration tests for `IntVec`.

use compressed_intvec::prelude::*;
use dsi_bitstream::prelude::{len_delta, len_gamma, Codes, BE, LE};
use rand::{rngs::StdRng, Rng, SeedableRng};

// Import helper functions from the common module.
#[path = "common/mod.rs"]
mod common;
use common::helpers::{generate_random_vec, generate_with_distribution};

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

macro_rules! test_configuration {
    ($test_name:ident, $endianness:ty, $input:expr, $k:expr, $codec_spec:expr) => {
        #[test]
        fn $test_name() {
            let input = &$input;
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

            // Check metadata based on encoding type
            if let Encoding::Fixed { .. } = intvec.encoding() {
                assert_eq!(intvec.get_sampling_rate(), None);
                assert_eq!(intvec.get_num_samples(), 0);
            } else {
                assert_eq!(intvec.get_sampling_rate(), Some(k));
                if input.is_empty() {
                    assert_eq!(intvec.get_num_samples(), 0);
                } else {
                    let expected_samples = input.len().div_ceil(k);
                    assert_eq!(intvec.get_num_samples(), expected_samples);
                }
            }

            // Test full decompression via iterator
            let decompressed: Vec<u64> = intvec.iter().collect();
            assert_eq!(&decompressed, input, "Iterator decompression failed");

            // Test full decompression via into_vec
            let decompressed_into_vec = intvec.clone().into_vec();
            assert_eq!(
                &decompressed_into_vec, input,
                "into_vec decompression failed"
            );

            if !input.is_empty() {
                // Test `get` for single random access
                let indices_to_test = [0, input.len() / 2, input.len() - 1];
                for &idx in &indices_to_test {
                    assert_eq!(intvec.get(idx), Some(input[idx]), "get({}) failed", idx);
                }

                // Test `get_many` for batched random access
                let mut rng = StdRng::seed_from_u64(42);
                let num_indices = 100.min(input.len());
                let indices: Vec<usize> = (0..num_indices)
                    .map(|_| rng.random_range(0..input.len()))
                    .collect();
                let expected: Vec<u64> = indices.iter().map(|&i| input[i]).collect();
                let results = intvec.get_many(&indices).unwrap();
                assert_eq!(results, expected, "get_many failed");

                // Parallel tests
                #[cfg(feature = "parallel")]
                {
                    // Test parallel iterator
                    let par_decompressed: Vec<u64> = intvec.par_iter().collect();
                    assert_eq!(&par_decompressed, input, "Parallel iterator failed");

                    // Test parallel get_many
                    let par_results = intvec.par_get_many(&indices).unwrap();
                    assert_eq!(par_results, expected, "par_get_many failed");
                }
            } else {
                // Special checks for empty vec
                assert!(intvec.get(0).is_none());
                assert_eq!(intvec.get_many(&[]).unwrap(), vec![]);
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
    CodecSpec::Auto
);
test_configuration!(
    test_empty_be,
    BE,
    generate_random_vec(0, 0),
    32,
    CodecSpec::Auto
);
test_configuration!(
    test_empty_fixed_le,
    LE,
    generate_random_vec(0, 0),
    32, // k is ignored
    CodecSpec::FixedLength { num_bits: Some(8) }
);

// Single Element Vector
test_configuration!(test_single_gamma_le, LE, vec![42], 1, CodecSpec::Gamma);
test_configuration!(test_single_gamma_be, BE, vec![42], 1, CodecSpec::Gamma);
test_configuration!(
    test_single_fixed_le,
    LE,
    vec![42],
    1, // k is ignored
    CodecSpec::FixedLength { num_bits: Some(8) }
);

// Zeros Vector
test_configuration!(test_zeros_auto_le, LE, vec![0; 1000], 16, CodecSpec::Auto);
test_configuration!(
    test_zeros_unary_be,
    BE,
    vec![0; 1000],
    16,
    CodecSpec::Explicit(Codes::Unary)
);
test_configuration!(
    test_zeros_fixed_le,
    LE,
    vec![0; 1000],
    16, // k is ignored
    CodecSpec::FixedLength { num_bits: Some(1) }
);

// Uniform Distributions
test_configuration!(
    test_uniform_small_auto_le,
    LE,
    generate_random_vec(1000, 100),
    32,
    CodecSpec::Auto
);
test_configuration!(
    test_uniform_large_auto_be,
    BE,
    generate_random_vec(1000, 1_000_000),
    32,
    CodecSpec::Auto
);

// Specific Code Distributions
test_configuration!(
    test_gamma_dist_le,
    LE,
    generate_with_distribution(1000, len_gamma),
    32,
    CodecSpec::Gamma
);
test_configuration!(
    test_delta_dist_be,
    BE,
    generate_with_distribution(1000, len_delta),
    32,
    CodecSpec::Delta
);

// Explicit Code Parameters
test_configuration!(
    test_explicit_zeta_le,
    LE,
    generate_random_vec(500, 1000),
    16,
    CodecSpec::Zeta { k: Some(5) }
);
test_configuration!(
    test_explicit_rice_be,
    BE,
    generate_random_vec(500, 1000),
    16,
    CodecSpec::Rice { log2_b: Some(4) }
);

// FixedLength specific tests
test_configuration!(
    test_fixed_auto_bits_le,
    LE,
    generate_random_vec(100, 1000), // max_val is 999, needs 10 bits
    32,                             // k is ignored
    CodecSpec::FixedLength { num_bits: None }
);
test_configuration!(
    test_fixed_full_64_bits_be,
    BE,
    vec![u64::MAX - 1, u64::MAX],
    32, // k is ignored
    CodecSpec::FixedLength { num_bits: Some(64) }
);

// Edge case for sampling rate (only for DSI codes)
test_configuration!(
    test_k_equals_len,
    LE,
    generate_random_vec(100, 100),
    100,
    CodecSpec::Auto
);

#[test]
fn test_invalid_parameters() {
    let input = vec![1, 2, 3];
    // k=0 is invalid for DSI codecs
    let result = IntVec::<LE>::builder(&input)
        .k(0)
        .codec(CodecSpec::Gamma)
        .build();
    assert!(matches!(result, Err(IntVecError::InvalidParameters(_))));

    // Value too large for FixedLength
    let input_large = vec![10, 20, 256]; // 256 requires 9 bits
    let result_large = IntVec::<LE>::builder(&input_large)
        .codec(CodecSpec::FixedLength { num_bits: Some(8) })
        .build();
    assert!(matches!(
        result_large,
        Err(IntVecError::InvalidParameters(_))
    ));
}

#[test]
fn test_out_of_bounds() {
    let input = vec![10, 20, 30];
    let intvec = IntVec::<LE>::builder(&input).build().unwrap();
    assert!(matches!(
        intvec.get_many(&[0, 1, 3]),
        Err(IntVecError::IndexOutOfBounds(3))
    ));
}

#[test]
fn test_all_codecs_systematic() {
    let input = generate_random_vec(10_000, 1000);
    let mut rng = StdRng::seed_from_u64(42);
    let test_indices: Vec<usize> = (0..100).map(|_| rng.random_range(0..input.len())).collect();
    let expected_values: Vec<u64> = test_indices.iter().map(|&i| input[i]).collect();

    let codecs_to_test = vec![
        ("Gamma", CodecSpec::Gamma),
        ("Delta", CodecSpec::Delta),
        ("Auto", CodecSpec::Auto),
        (
            "FixedLength_auto",
            CodecSpec::FixedLength { num_bits: None },
        ),
    ];

    for (codec_name, codec_spec) in codecs_to_test {
        let intvec = IntVec::<LE>::builder(&input)
            .codec(codec_spec)
            .build()
            .unwrap_or_else(|e| panic!("Build failed for {}: {:?}", codec_name, e));

        assert_eq!(
            intvec.clone().into_vec(),
            input,
            "into_vec failed for {}",
            codec_name
        );
        let get_many_results = intvec.get_many(&test_indices).unwrap();
        assert_eq!(
            get_many_results, expected_values,
            "get_many failed for {}",
            codec_name
        );
    }
}

#[test]
fn test_from_iter_builder() {
    let data: Vec<u64> = (0..1000).collect();

    // --- Success Cases ---
    let intvec_gamma = LEIntVec::from_iter_builder(data.clone())
        .codec(CodecSpec::Gamma)
        .k(16)
        .build()
        .unwrap();
    assert_eq!(intvec_gamma.len(), data.len());
    assert_eq!(intvec_gamma.get(500), Some(500));
    assert_eq!(intvec_gamma.clone().into_vec(), data);

    let intvec_fixed = LEIntVec::from_iter_builder(data.clone())
        .codec(CodecSpec::FixedLength { num_bits: Some(10) }) // 1000 fits in 10 bits
        .build()
        .unwrap();
    assert_eq!(intvec_fixed.len(), data.len());
    assert_eq!(intvec_fixed.get(999), Some(999));
    assert_eq!(intvec_fixed.clone().into_vec(), data);

    // --- Failure Cases: Automatic parameter selection ---
    let codecs_with_auto_params = vec![
        CodecSpec::Auto,
        CodecSpec::FixedLength { num_bits: None },
        CodecSpec::Rice { log2_b: None },
        CodecSpec::Zeta { k: None },
    ];
    for codec in codecs_with_auto_params {
        let result = LEIntVec::from_iter_builder(data.clone().into_iter())
            .codec(codec)
            .build();
        assert!(matches!(result, Err(IntVecError::InvalidParameters(_))));
    }

    // --- Failure Case: Value too large for FixedLength ---
    let data_too_large = vec![10, 20, 256];
    let result = LEIntVec::from_iter_builder(data_too_large)
        .codec(CodecSpec::FixedLength { num_bits: Some(8) })
        .build();
    assert!(matches!(result, Err(IntVecError::InvalidParameters(_))));
}
