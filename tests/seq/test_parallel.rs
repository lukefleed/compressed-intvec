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
    let collected_par: Vec<Vec<T>> = vec.par_iter().map(|seq| seq.to_vec()).collect();

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
            let results_vec: Vec<Vec<T>> = results.into_iter().collect();

            assert_eq!(
                results_vec, expected_subset,
                "par_decode_many() results mismatch for {}",
                context
            );

            // Unsafe version (same result expected)
            let results_unchecked = unsafe { vec.par_decode_many_unchecked(&indices) };
            let results_unchecked_vec: Vec<Vec<T>> = results_unchecked.into_iter().collect();

            assert_eq!(
                results_unchecked_vec, expected_subset,
                "par_decode_many_unchecked() results mismatch for {}",
                context
            );
        }
    }

    // --- Test 4: par_decode_many() with out-of-bounds error ---
    if !sequences.is_empty() {
        let bad_indices = vec![0, sequences.len()]; // Last index is out of bounds
        let result = vec.par_decode_many(&bad_indices);
        assert!(
            result.is_err(),
            "par_decode_many() should fail on out-of-bounds index for {}",
            context
        );
    }

    // --- Test 5: Empty indices edge case ---
    let empty_indices: Vec<usize> = vec![];
    let empty_results = vec
        .par_decode_many(&empty_indices)
        .expect("par_decode_many with empty indices should succeed");
    assert!(
        empty_results.is_empty(),
        "par_decode_many with empty indices should return empty for {}",
        context
    );
}

// --- par_for_each tests ---

fn run_par_for_each_tests<T, E>(sequences: Vec<Vec<T>>, type_name: &str)
where
    T: compressed_intvec::variable::traits::Storable
        + PartialEq
        + std::fmt::Debug
        + Copy
        + Send
        + Sync
        + Into<u64>
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
        "par_for_each tests for {} in {}",
        type_name,
        std::any::type_name::<E>()
    );

    let vec: SeqVec<T, E> =
        SeqVec::from_slices(&sequences).expect(&format!("Build failed: {}", context));

    // --- Test 1: par_for_each count ---
    let expected_counts: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
    let actual_counts: Vec<usize> = vec.par_for_each(|seq| seq.count());

    assert_eq!(
        actual_counts, expected_counts,
        "par_for_each count mismatch for {}",
        context
    );

    // --- Test 2: par_for_each sum ---
    let expected_sums: Vec<u64> = sequences
        .iter()
        .map(|s| s.iter().map(|&v| v.into()).sum())
        .collect();
    let actual_sums: Vec<u64> = vec.par_for_each(|seq| seq.map(|v| v.into()).sum());

    assert_eq!(
        actual_sums, expected_sums,
        "par_for_each sum mismatch for {}",
        context
    );

    // --- Test 3: par_for_each collect (equivalent to par_iter) ---
    let expected_collected: Vec<Vec<T>> = sequences.clone();
    let actual_collected: Vec<Vec<T>> = vec.par_for_each(|seq| seq.collect());

    assert_eq!(
        actual_collected, expected_collected,
        "par_for_each collect mismatch for {}",
        context
    );

    // --- Test 4: par_for_each_reduce ---
    let expected_total: u64 = sequences
        .iter()
        .flat_map(|s| s.iter())
        .map(|&v| v.into())
        .sum();
    let actual_total: u64 = vec.par_for_each_reduce(
        |seq| seq.map(|v| v.into()).sum::<u64>(),
        || 0u64,
        |a, b| a + b,
    );

    assert_eq!(
        actual_total, expected_total,
        "par_for_each_reduce sum mismatch for {}",
        context
    );

    // --- Test 5: par_for_each_many ---
    if !sequences.is_empty() {
        let indices: Vec<usize> = (0..sequences.len()).filter(|x| x % 2 == 0).collect();

        if !indices.is_empty() {
            let expected_subset_counts: Vec<usize> =
                indices.iter().map(|&i| sequences[i].len()).collect();

            let actual_subset_counts = vec
                .par_for_each_many(&indices, |seq| seq.count())
                .expect("par_for_each_many failed");

            assert_eq!(
                actual_subset_counts, expected_subset_counts,
                "par_for_each_many count mismatch for {}",
                context
            );
        }
    }

    // --- Test 6: par_for_each_many with out-of-bounds error ---
    if !sequences.is_empty() {
        let bad_indices = vec![0, sequences.len()];
        let result = vec.par_for_each_many(&bad_indices, |seq| seq.count());
        assert!(
            result.is_err(),
            "par_for_each_many should fail on out-of-bounds index for {}",
            context
        );
    }

    // --- Test 7: par_for_each on empty SeqVec ---
    let empty_vec: SeqVec<T, E> =
        SeqVec::from_slices::<&[T], _>(&[]).expect("Failed to build empty SeqVec");
    let empty_results: Vec<usize> = empty_vec.par_for_each(|seq| seq.count());
    assert!(
        empty_results.is_empty(),
        "par_for_each on empty SeqVec should return empty for {}",
        context
    );
}

// --- Tests with stored lengths ---

fn run_par_for_each_with_lengths_tests<T, E>(sequences: Vec<Vec<T>>, type_name: &str)
where
    T: compressed_intvec::variable::traits::Storable
        + PartialEq
        + std::fmt::Debug
        + Copy
        + Send
        + Sync
        + Into<u64>
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
        "par_for_each with lengths tests for {} in {}",
        type_name,
        std::any::type_name::<E>()
    );

    // Build with stored lengths
    let vec: SeqVec<T, E> = SeqVec::builder()
        .store_lengths(true)
        .build(&sequences)
        .expect(&format!("Build failed: {}", context));

    assert!(
        vec.has_stored_lengths(),
        "SeqVec should have stored lengths for {}",
        context
    );

    // --- Test 1: par_for_each count with stored lengths ---
    let expected_counts: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
    let actual_counts: Vec<usize> = vec.par_for_each(|seq| seq.count());

    assert_eq!(
        actual_counts, expected_counts,
        "par_for_each count with lengths mismatch for {}",
        context
    );

    // --- Test 2: par_for_each sum with stored lengths ---
    let expected_sums: Vec<u64> = sequences
        .iter()
        .map(|s| s.iter().map(|&v| v.into()).sum())
        .collect();
    let actual_sums: Vec<u64> = vec.par_for_each(|seq| seq.map(|v| v.into()).sum());

    assert_eq!(
        actual_sums, expected_sums,
        "par_for_each sum with lengths mismatch for {}",
        context
    );

    // --- Test 3: par_iter with stored lengths ---
    let collected_par: Vec<Vec<T>> = vec.par_iter().map(|seq| seq.to_vec()).collect();

    assert_eq!(
        collected_par, sequences,
        "par_iter with lengths results mismatch for {}",
        context
    );
}

// --- Test data generators ---

fn small_u32_sequences() -> Vec<Vec<u32>> {
    vec![
        vec![1, 2, 3],
        vec![10, 20, 30, 40],
        vec![100],
        vec![],
        vec![5, 6],
    ]
}

fn medium_u32_sequences() -> Vec<Vec<u32>> {
    (0..100)
        .map(|i| (0..((i % 20) + 1)).map(|j| (i * 100 + j) as u32).collect())
        .collect()
}

fn large_u32_sequences() -> Vec<Vec<u32>> {
    (0..1000)
        .map(|i| (0..((i % 50) + 1)).map(|j| (i * 1000 + j) as u32).collect())
        .collect()
}

fn small_u64_sequences() -> Vec<Vec<u64>> {
    vec![
        vec![1, 2, 3],
        vec![10, 20, 30, 40],
        vec![1000000000],
        vec![],
        vec![5, 6],
    ]
}

// --- Parallel test runners ---

#[test]
fn test_parallel_small_u32_le() {
    run_parallel_tests::<u32, LE>(small_u32_sequences(), "u32");
}

#[test]
fn test_parallel_small_u32_be() {
    run_parallel_tests::<u32, BE>(small_u32_sequences(), "u32");
}

#[test]
fn test_parallel_medium_u32_le() {
    run_parallel_tests::<u32, LE>(medium_u32_sequences(), "u32");
}

#[test]
fn test_parallel_large_u32_le() {
    run_parallel_tests::<u32, LE>(large_u32_sequences(), "u32");
}

#[test]
fn test_parallel_small_u64_le() {
    run_parallel_tests::<u64, LE>(small_u64_sequences(), "u64");
}

// --- par_for_each test runners ---

#[test]
fn test_par_for_each_small_u32_le() {
    run_par_for_each_tests::<u32, LE>(small_u32_sequences(), "u32");
}

#[test]
fn test_par_for_each_small_u32_be() {
    run_par_for_each_tests::<u32, BE>(small_u32_sequences(), "u32");
}

#[test]
fn test_par_for_each_medium_u32_le() {
    run_par_for_each_tests::<u32, LE>(medium_u32_sequences(), "u32");
}

#[test]
fn test_par_for_each_large_u32_le() {
    run_par_for_each_tests::<u32, LE>(large_u32_sequences(), "u32");
}

#[test]
fn test_par_for_each_small_u64_le() {
    run_par_for_each_tests::<u64, LE>(small_u64_sequences(), "u64");
}

// --- par_for_each with stored lengths test runners ---

#[test]
fn test_par_for_each_with_lengths_small_u32_le() {
    run_par_for_each_with_lengths_tests::<u32, LE>(small_u32_sequences(), "u32");
}

#[test]
fn test_par_for_each_with_lengths_medium_u32_le() {
    run_par_for_each_with_lengths_tests::<u32, LE>(medium_u32_sequences(), "u32");
}

#[test]
fn test_par_for_each_with_lengths_large_u32_le() {
    run_par_for_each_with_lengths_tests::<u32, LE>(large_u32_sequences(), "u32");
}

// --- Edge case tests ---

#[test]
fn test_parallel_empty_seqvec() {
    let empty_sequences: Vec<Vec<u32>> = vec![];
    let vec: SeqVec<u32, LE> = SeqVec::from_slices(&empty_sequences).unwrap();

    // par_iter on empty
    let collected: Vec<Vec<u32>> = vec.par_iter().collect();
    assert!(collected.is_empty());

    // par_for_each on empty
    let counts: Vec<usize> = vec.par_for_each(|seq| seq.count());
    assert!(counts.is_empty());

    // par_decode_many with empty indices
    let results = vec.par_decode_many(&[]).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_parallel_single_sequence() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4, 5]];
    let vec: SeqVec<u32, LE> = SeqVec::from_slices(&sequences).unwrap();

    // par_iter
    let collected: Vec<Vec<u32>> = vec.par_iter().collect();
    assert_eq!(collected, sequences);

    // par_for_each
    let sums: Vec<u64> = vec.par_for_each(|seq| seq.map(|v| v as u64).sum());
    assert_eq!(sums, vec![15]);

    // par_decode_many
    let results = vec.par_decode_many(&[0]).unwrap();
    assert_eq!(results, sequences);
}

#[test]
fn test_parallel_all_empty_sequences() {
    let sequences: Vec<Vec<u32>> = vec![vec![], vec![], vec![]];
    let vec: SeqVec<u32, LE> = SeqVec::from_slices(&sequences).unwrap();

    // par_iter
    let collected: Vec<Vec<u32>> = vec.par_iter().collect();
    assert_eq!(collected, sequences);

    // par_for_each
    let counts: Vec<usize> = vec.par_for_each(|seq| seq.count());
    assert_eq!(counts, vec![0, 0, 0]);

    let sums: Vec<u64> = vec.par_for_each(|seq| seq.map(|v| v as u64).sum());
    assert_eq!(sums, vec![0, 0, 0]);
}

#[test]
fn test_par_for_each_early_termination() {
    // Test operations that can terminate early (any, find, position)
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20, 30], vec![100, 200, 300]];
    let vec: SeqVec<u32, LE> = SeqVec::from_slices(&sequences).unwrap();

    // any() with early termination
    let has_large: Vec<bool> = vec.par_for_each(|seq| seq.any(|v| v > 50));
    assert_eq!(has_large, vec![false, false, true]);

    // find() with early termination
    let first_gt_15: Vec<Option<u32>> = vec.par_for_each(|seq| seq.find(|&v| v > 15));
    assert_eq!(first_gt_15, vec![None, Some(20), Some(100)]);

    // position() with early termination
    let pos_gt_15: Vec<Option<usize>> = vec.par_for_each(|seq| seq.position(|v| v > 15));
    assert_eq!(pos_gt_15, vec![None, Some(1), Some(0)]);
}

#[test]
fn test_par_for_each_reduce_associativity() {
    // Verify that reduction produces correct results regardless of partition
    let sequences: Vec<Vec<u32>> = (0..100).map(|i| vec![i as u32, (i * 2) as u32]).collect();
    let vec: SeqVec<u32, LE> = SeqVec::from_slices(&sequences).unwrap();

    // Sum reduction
    let expected: u64 = sequences
        .iter()
        .flat_map(|s| s.iter())
        .map(|&v| v as u64)
        .sum();

    let actual: u64 = vec.par_for_each_reduce(
        |seq| seq.map(|v| v as u64).sum::<u64>(),
        || 0u64,
        |a, b| a + b,
    );

    assert_eq!(actual, expected);

    // Max reduction
    let expected_max: u32 = sequences
        .iter()
        .flat_map(|s| s.iter())
        .copied()
        .max()
        .unwrap_or(0);

    let actual_max: u32 =
        vec.par_for_each_reduce(|seq| seq.max().unwrap_or(0), || 0u32, |a, b| a.max(b));

    assert_eq!(actual_max, expected_max);
}

#[test]
fn test_par_for_each_many_preserves_order() {
    // Verify that results are returned in the same order as indices
    let sequences: Vec<Vec<u32>> = vec![vec![1], vec![2], vec![3], vec![4], vec![5]];
    let vec: SeqVec<u32, LE> = SeqVec::from_slices(&sequences).unwrap();

    // Reverse order indices
    let indices = vec![4, 3, 2, 1, 0];
    let sums = vec
        .par_for_each_many(&indices, |seq| seq.map(|v| v as u64).sum::<u64>())
        .unwrap();

    // Results should be in index order: [5, 4, 3, 2, 1]
    assert_eq!(sums, vec![5, 4, 3, 2, 1]);
}

#[test]
fn test_par_decode_many_preserves_order() {
    // Verify that results are returned in the same order as indices
    let sequences: Vec<Vec<u32>> = vec![
        vec![1, 10],
        vec![2, 20],
        vec![3, 30],
        vec![4, 40],
        vec![5, 50],
    ];
    let vec: SeqVec<u32, LE> = SeqVec::from_slices(&sequences).unwrap();

    // Reverse order indices
    let indices = vec![4, 2, 0];
    let results = vec.par_decode_many(&indices).unwrap();

    // Results should be in index order
    assert_eq!(results[0], vec![5, 50]);
    assert_eq!(results[1], vec![3, 30]);
    assert_eq!(results[2], vec![1, 10]);
}

#[test]
fn test_par_for_each_with_duplicate_indices() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20]];
    let vec: SeqVec<u32, LE> = SeqVec::from_slices(&sequences).unwrap();

    // Duplicate indices
    let indices = vec![0, 0, 1, 0];
    let sums = vec
        .par_for_each_many(&indices, |seq| seq.map(|v| v as u64).sum::<u64>())
        .unwrap();

    assert_eq!(sums, vec![6, 6, 30, 6]);
}

#[test]
fn test_par_into_vecs_consumes_seqvec() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20]];
    let vec: SeqVec<u32, LE> = SeqVec::from_slices(&sequences).unwrap();

    // par_into_vecs consumes the SeqVec
    let results = vec.par_into_vecs();
    assert_eq!(results, sequences);

    // vec is now consumed and cannot be used
    // (This is a compile-time check, not a runtime test)
}

#[test]
fn test_par_for_each_sum() {
    let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
    let vec: SeqVec<u32, LE, Vec<u64>> = SeqVec::from_slices(sequences).unwrap();

    let sums: Vec<u64> = vec.par_for_each(|seq| seq.map(|v| v as u64).sum());
    assert_eq!(sums, vec![6, 30, 100]);
}

#[test]
fn test_par_for_each_count() {
    let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
    let vec: SeqVec<u32, LE, Vec<u64>> = SeqVec::from_slices(sequences).unwrap();

    let counts: Vec<usize> = vec.par_for_each(|seq| seq.count());
    assert_eq!(counts, vec![3, 2, 1]);
}

#[test]
fn test_par_for_each_reduce_total_sum() {
    let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
    let vec: SeqVec<u32, LE, Vec<u64>> = SeqVec::from_slices(sequences).unwrap();

    let total: u64 = vec.par_for_each_reduce(
        |seq| seq.map(|v| v as u64).sum::<u64>(),
        || 0u64,
        |a, b| a + b,
    );
    assert_eq!(total, 136);
}

#[test]
fn test_par_for_each_many() {
    let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100], &[1000, 2000]];
    let vec: SeqVec<u32, LE, Vec<u64>> = SeqVec::from_slices(sequences).unwrap();

    let sums = vec
        .par_for_each_many(&[0, 2], |seq| seq.map(|v| v as u64).sum::<u64>())
        .unwrap();
    assert_eq!(sums, vec![6, 100]);
}

#[test]
fn test_par_for_each_empty() {
    let sequences: &[&[u32]] = &[];
    let vec: SeqVec<u32, LE, Vec<u64>> = SeqVec::from_slices(sequences).unwrap();

    let sums: Vec<u64> = vec.par_for_each(|seq| seq.map(|v| v as u64).sum());
    assert!(sums.is_empty());
}

#[test]
fn test_par_for_each_with_empty_sequences() {
    let sequences: &[&[u32]] = &[&[], &[1, 2], &[]];
    let vec: SeqVec<u32, LE, Vec<u64>> = SeqVec::from_slices(sequences).unwrap();

    let sums: Vec<u64> = vec.par_for_each(|seq| seq.map(|v| v as u64).sum());
    assert_eq!(sums, vec![0, 3, 0]);
}
