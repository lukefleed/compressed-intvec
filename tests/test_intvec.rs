//! Integration tests for `IntVec`.

use compressed_intvec::prelude::*;
use dsi_bitstream::prelude::{len_delta, len_gamma, BE, LE};
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
test_configuration!(test_zeros_unary_be, BE, vec![0; 1000], 16, CodecSpec::Unary);
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

    // Expanded list to test all first-class codec variants.
    let codecs_to_test = vec![
        ("Gamma", CodecSpec::Gamma),
        ("Delta", CodecSpec::Delta),
        ("Unary", CodecSpec::Unary),
        ("Omega", CodecSpec::Omega),
        ("VByteLe", CodecSpec::VByteLe),
        ("VByteBe", CodecSpec::VByteBe),
        ("Rice_fixed", CodecSpec::Rice { log2_b: Some(4) }),
        ("Zeta_fixed", CodecSpec::Zeta { k: Some(3) }),
        ("Golomb_fixed", CodecSpec::Golomb { b: Some(8) }),
        ("Pi_fixed", CodecSpec::Pi { k: Some(3) }),
        ("ExpGolomb_fixed", CodecSpec::ExpGolomb { k: Some(2) }),
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
    // The iterator-based builder cannot analyze the data, so codecs requiring
    // auto-parameter selection must fail.
    let codecs_with_auto_params = vec![
        CodecSpec::Auto,
        CodecSpec::FixedLength { num_bits: None },
        CodecSpec::Rice { log2_b: None },
        CodecSpec::Zeta { k: None },
        CodecSpec::Golomb { b: None },
        CodecSpec::Pi { k: None },
        CodecSpec::ExpGolomb { k: None },
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

    // --- Failure Case: Value too large for FixedLength ---
    let data_too_large = vec![10, 20, 256];
    let result = LEIntVec::from_iter_builder(data_too_large)
        .codec(CodecSpec::FixedLength { num_bits: Some(8) })
        .build();
    assert!(matches!(result, Err(IntVecError::InvalidParameters(_))));
}

macro_rules! test_get_many_optimizations {
    ($test_name:ident, $k:expr) => {
        #[test]
        fn $test_name() {
            let k_val = $k;
            let data_size = k_val * 10;
            let data = generate_random_vec(data_size, 1_000_000);
            let intvec = LEIntVec::builder(&data).k(k_val).build().unwrap();

            // Test case 1: Indices tightly clustered within a single sample block
            let clustered_indices = vec![k_val + 2, k_val + 3, k_val + 5];
            let expected1: Vec<u64> = clustered_indices.iter().map(|&i| data[i]).collect();
            let result1 = intvec.get_many(&clustered_indices).unwrap();
            assert_eq!(
                result1, expected1,
                "get_many failed for clustered indices with k={}",
                k_val
            );

            // Test case 2: Indices crossing sample block boundaries
            let cross_block_indices = vec![k_val - 1, k_val, k_val + 1, 2 * k_val - 1, 2 * k_val];
            let expected2: Vec<u64> = cross_block_indices.iter().map(|&i| data[i]).collect();
            let result2 = intvec.get_many(&cross_block_indices).unwrap();
            assert_eq!(
                result2, expected2,
                "get_many failed for cross-block indices with k={}",
                k_val
            );

            // Test case 3: Unordered indices with duplicates
            let unordered_indices = vec![3 * k_val, k_val / 2, 3 * k_val];
            let expected3: Vec<u64> = unordered_indices.iter().map(|&i| data[i]).collect();
            let result3 = intvec.get_many(&unordered_indices).unwrap();
            assert_eq!(
                result3, expected3,
                "get_many failed for unordered indices with k={}",
                k_val
            );

            // Test case 4: Using get_many_from_iter with a range
            let range = (data_size / 2)..(data_size / 2 + 10);
            let expected4: Vec<u64> = range.clone().map(|i| data[i]).collect();
            let result4 = intvec.get_many_from_iter(range).unwrap();
            assert_eq!(
                result4, expected4,
                "get_many_from_iter failed for range with k={}",
                k_val
            );
        }
    };
}

// Test with k being a power of two (fast path)
test_get_many_optimizations!(test_get_many_k_power_of_two, 16);

// Test with k NOT being a power of two (fallback path)
test_get_many_optimizations!(test_get_many_k_not_power_of_two, 24);

#[test]
fn test_seq_reader_stateful_access() {
    let k = 16;
    let data_size = k * 5;
    let data = generate_random_vec(data_size, 1_000_000);
    let intvec = LEIntVec::builder(&data).k(k).build().unwrap();

    let mut seq_reader = intvec.seq_reader();

    // 1. Forward access within a block
    assert_eq!(seq_reader.get(2).unwrap(), Some(data[2]));
    assert_eq!(seq_reader.get(5).unwrap(), Some(data[5])); // Should be fast

    // 2. Forward access crossing a block
    assert_eq!(seq_reader.get(k + 1).unwrap(), Some(data[k + 1]));

    // 3. Backward access, forcing a seek
    assert_eq!(seq_reader.get(3).unwrap(), Some(data[3]));

    // 4. Multiple sequential reads
    for i in 3..10 {
        assert_eq!(seq_reader.get(i).unwrap(), Some(data[i]));
    }
}
