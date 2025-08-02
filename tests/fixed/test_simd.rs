//! Integration tests for SIMD-accelerated batch access methods.
//!
//! This entire module is conditionally compiled and will only be included in
//! the test suite when the `simd` feature is enabled. It is designed to
//! validate the correctness of the SIMD implementations in `get_many` and
//! `par_get_many` across various access patterns, bit-widths, and vector types.

#![cfg(all(test, feature = "simd"))]

mod test_simd {
    use compressed_intvec::prelude::*;
    use rand::{rng, seq::SliceRandom};

    // Import helper functions from the common module.
    use crate::common::helpers::{generate_random_signed_vec, generate_random_vec};

    /// A powerful macro to generate a comprehensive test case for a specific
    /// SIMD configuration. It validates both sequential and parallel `get_many`
    /// against a variety of challenging index access patterns.
    macro_rules! test_simd_config {
        (
            $test_name:ident,
            $vec_type:ty,
            $data_type:ty,
            $data_gen_fn:expr,
            $num_bits:expr
        ) => {
            #[test]
            fn $test_name() {
                // 1. SETUP: Generate data and build the vector.
                const VECTOR_SIZE: usize = 2048;
                let data = $data_gen_fn(VECTOR_SIZE);
                let vec = <$vec_type>::builder(&data)
                    .bit_width(BitWidth::Explicit($num_bits))
                    .build()
                    .unwrap();

                // 2. DEFINE ACCESS PATTERNS: Create different sets of indices to test.
                let patterns = [
                    ("contiguous_run", (100..228).collect::<Vec<_>>()), // 128 elements, ideal for SIMD
                    ("scattered", {
                        let mut indices: Vec<usize> = (0..VECTOR_SIZE).collect();
                        indices.shuffle(&mut rng());
                        indices.truncate(128);
                        indices
                    }),
                    ("mixed_runs", {
                        let mut indices = Vec::new();
                        indices.extend(10..20); // Short run
                        indices.push(500); // Scattered
                        indices.extend(300..428); // Long run
                        indices.push(1000); // Scattered
                        indices.extend(100..105); // Short overlapping run
                        indices.shuffle(&mut rng());
                        indices
                    }),
                    ("full_vector", (0..VECTOR_SIZE).collect()),
                    ("empty", vec![]),
                ];

                // 3. VALIDATE: For each pattern, test both get_many and par_get_many.
                for (name, indices) in patterns {
                    let expected: Vec<$data_type> = indices.iter().map(|&i| data[i]).collect();

                    // Test sequential `get_many`.
                    let result_seq = vec.get_many(&indices).unwrap();
                    assert_eq!(
                        result_seq, expected,
                        "Sequential get_many failed for pattern '{}'",
                        name
                    );

                    // Test parallel `par_get_many`.
                    let result_par = vec.par_get_many(&indices).unwrap();
                    assert_eq!(
                        result_par, expected,
                        "Parallel par_get_many failed for pattern '{}'",
                        name
                    );
                }
            }
        };
    }

    // --- Instantiate tests for LEFixedVec (u64) ---
    test_simd_config!(
        test_le_fixed_vec_u8_simd,
        LEFixedVec,
        u64,
        |n| generate_random_vec(n, 1 << 8),
        8
    );
    test_simd_config!(
        test_le_fixed_vec_u16_simd,
        LEFixedVec,
        u64,
        |n| generate_random_vec(n, 1 << 16),
        16
    );
    test_simd_config!(
        test_le_fixed_vec_u32_simd,
        LEFixedVec,
        u64,
        |n| generate_random_vec(n, 1 << 32),
        32
    );
    test_simd_config!(
        test_le_fixed_vec_u64_simd,
        LEFixedVec,
        u64,
        |n| generate_random_vec(n, u64::MAX),
        64
    );

    // --- Instantiate tests for BEFixedVec (u64) ---
    // These test the scalar fallback paths in the SIMD module for Big Endian.
    test_simd_config!(
        test_be_fixed_vec_u8_simd,
        BEFixedVec,
        u64,
        |n| generate_random_vec(n, 1 << 8),
        8
    );
    test_simd_config!(
        test_be_fixed_vec_u16_simd,
        BEFixedVec,
        u64,
        |n| generate_random_vec(n, 1 << 16),
        16
    );
    test_simd_config!(
        test_be_fixed_vec_u32_simd,
        BEFixedVec,
        u64,
        |n| generate_random_vec(n, 1 << 32),
        32
    );
    test_simd_config!(
        test_be_fixed_vec_u64_simd,
        BEFixedVec,
        u64,
        |n| generate_random_vec(n, u64::MAX),
        64
    );

    // --- Instantiate tests for LESFixedVec (i64) ---
    // These test both the underlying gather and the SIMD zigzag decoding.
    test_simd_config!(
        test_le_sfixed_vec_i8_simd,
        LESFixedVec,
        i64,
        |n| generate_random_signed_vec(n, 1 << 7),
        8
    );
    test_simd_config!(
        test_le_sfixed_vec_i16_simd,
        LESFixedVec,
        i64,
        |n| generate_random_signed_vec(n, 1 << 15),
        16
    );
    test_simd_config!(
        test_le_sfixed_vec_i32_simd,
        LESFixedVec,
        i64,
        |n| generate_random_signed_vec(n, 1 << 31),
        32
    );
    test_simd_config!(
        test_le_sfixed_vec_i64_simd,
        LESFixedVec,
        i64,
        |n| generate_random_signed_vec(n, i64::MAX),
        64
    );

    // --- Instantiate tests for BESFixedVec (i64) ---
    test_simd_config!(
        test_be_sfixed_vec_i8_simd,
        BESFixedVec,
        i64,
        |n| generate_random_signed_vec(n, 1 << 7),
        8
    );
    test_simd_config!(
        test_be_sfixed_vec_i16_simd,
        BESFixedVec,
        i64,
        |n| generate_random_signed_vec(n, 1 << 15),
        16
    );
    test_simd_config!(
        test_be_sfixed_vec_i32_simd,
        BESFixedVec,
        i64,
        |n| generate_random_signed_vec(n, 1 << 31),
        32
    );
    test_simd_config!(
        test_be_sfixed_vec_i64_simd,
        BESFixedVec,
        i64,
        |n| generate_random_signed_vec(n, i64::MAX),
        64
    );
}
