//! Integration tests for parallel functionality in [`SeqVec`].

#![cfg(feature = "parallel")]

use compressed_intvec::seq::SeqVec;
use dsi_bitstream::prelude::{BE, LE};
use rayon::prelude::*;

// --- Helper function for parallel testing ---

fn run_parallel_tests<T, E>(sequences: Vec<Vec<T>>, type_name: &str)
where
    T: compressed_intvec::variable::traits::Storable
        + PartialEq
        + std::fmt::Debug
        + Copy
        + Send
        + Sync
        + 'static,
    E: dsi_bitstream::traits::Endianness + std::fmt::Debug + Sync + Send + Clone + 'static,
    for<'a> compressed_intvec::seq::iter::SeqVecBitReader<'a, E>: dsi_bitstream::prelude::BitRead<E, Error = core::convert::Infallible>
        + dsi_bitstream::prelude::CodesRead<E>
        + dsi_bitstream::prelude::BitSeek<Error = core::convert::Infallible>,
    dsi_bitstream::impls::BufBitWriter<E, dsi_bitstream::impls::MemWordWriterVec<u64, Vec<u64>>>:
        dsi_bitstream::prelude::BitWrite<E, Error = core::convert::Infallible>
            + dsi_bitstream::prelude::CodesWrite<E>,
{
    let context = format!(
        "parallel tests for {} in {}",
        type_name,
        std::any::type_name::<E>()
    );

    let vec: SeqVec<T, E> =
        SeqVec::from_slices(&sequences).expect(&format!("Build failed: {}", context));

    // --- Test 1: par_iter() ---
    // Collect all sequences in parallel and compare with original
    let collected_par: Vec<Vec<T>> = vec.par_iter().map(|seq| seq.collect::<Vec<T>>()).collect();

    assert_eq!(
        collected_par, sequences,
        "par_iter() results mismatch for {}",
        context
    );

    // --- Test 2: par_into_vecs() ---
    // Consume the SeqVec and decode all sequences in parallel
    let vec_for_into: SeqVec<T, E> =
        SeqVec::from_slices(&sequences).expect(&format!("Build failed: {}", context));
    let collected_into: Vec<Vec<T>> = vec_for_into.par_into_vecs();

    assert_eq!(
        collected_into, sequences,
        "par_into_vecs() results mismatch for {}",
        context
    );

    // --- Test 3: par_decode_many() with safe bounds checking ---
    if !sequences.is_empty() {
        // Generate some indices (evens, if available)
        let indices: Vec<usize> = (0..sequences.len()).filter(|x| x % 2 == 0).collect();

        if !indices.is_empty() {
            let expected_subset: Vec<Vec<T>> =
                indices.iter().map(|&i| sequences[i].clone()).collect();

            // Safe version with bounds checking
            let results = vec
                .par_decode_many(&indices)
                .expect("par_decode_many failed");
            let results_vec: Vec<Vec<T>> = results.into_iter().map(|s| s.collect()).collect();

            assert_eq!(
                results_vec, expected_subset,
                "par_decode_many mismatch for {}",
                context
            );
        }
    }

    // --- Test 4: par_decode_many() with out-of-bounds should fail ---
    if !sequences.is_empty() {
        let mut indices = vec![0];
        if !sequences.is_empty() {
            indices.push(sequences.len()); // This is out-of-bounds
        }

        let result = vec.par_decode_many(&indices);
        assert!(
            result.is_err(),
            "par_decode_many should fail with out-of-bounds index in {}",
            context
        );
    }
}

// --- Macro for Type-Parameterized Parallel Testing ---

macro_rules! test_parallel_all_types {
    ($test_name:ident, $E:ty) => {
        #[test]
        fn $test_name() {
            // Test with u32
            {
                let sequences: Vec<Vec<u32>> = vec![
                    vec![1, 2, 3],
                    vec![10, 20],
                    vec![],
                    vec![100, 200, 300, 400, 500],
                ];
                run_parallel_tests::<u32, $E>(sequences, stringify!(u32));
            }

            // Test with i16
            {
                let sequences: Vec<Vec<i16>> =
                    vec![vec![-1, 2, -3], vec![10, -20], vec![], vec![-100, 200]];
                run_parallel_tests::<i16, $E>(sequences, stringify!(i16));
            }

            // Test with u64
            {
                let sequences: Vec<Vec<u64>> = vec![vec![1, 2, 3], vec![100, 200, 300]];
                run_parallel_tests::<u64, $E>(sequences, stringify!(u64));
            }
        }
    };
}

test_parallel_all_types!(test_parallel_le, LE);
test_parallel_all_types!(test_parallel_be, BE);
