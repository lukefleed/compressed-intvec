//! Integration tests for parallel functionality in `FixedVec`.

#![cfg(feature = "parallel")]

use compressed_intvec::fixed::{BitWidth, Error as FixedVecError, FixedVec, SFixedVec, UFixedVec};
use rand::{RngExt, rngs::StdRng, seq::SliceRandom, SeedableRng};
use rayon::iter::ParallelIterator;

fn get_test_data_unsigned() -> Vec<u32> {
    (0..10_000).collect()
}

fn get_test_data_signed() -> Vec<i16> {
    (-5_000..5_000).collect()
}

#[test]
fn test_par_iter() {
    // Unsigned test
    let data_u32 = get_test_data_unsigned();
    let vec_u32: UFixedVec<u32> = FixedVec::builder().build(&data_u32).unwrap();

    let collected_par: Vec<u32> = vec_u32.par_iter().collect();
    assert_eq!(
        collected_par, data_u32,
        "par_iter for UFixedVec<u32> failed"
    );

    // Signed test
    let data_i16 = get_test_data_signed();
    let vec_i16: SFixedVec<i16> = FixedVec::builder().build(&data_i16).unwrap();

    let collected_par_signed: Vec<i16> = vec_i16.par_iter().collect();
    assert_eq!(
        collected_par_signed, data_i16,
        "par_iter for SFixedVec<i16> failed"
    );
}

#[test]
fn test_par_get_many() {
    let data = get_test_data_unsigned();
    let vec: UFixedVec<u32> = FixedVec::builder()
        .bit_width(BitWidth::Minimal)
        .build(&data)
        .unwrap();

    // Create a set of random, shuffled indices to access
    let mut rng = StdRng::seed_from_u64(1337);
    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.shuffle(&mut rng);
    let indices_to_get = &indices[0..1000];

    // Expected results in the order of the requested indices
    let expected: Vec<u32> = indices_to_get.iter().map(|&i| data[i]).collect();

    // Test the safe `par_get_many`
    let results = vec.par_get_many(indices_to_get).unwrap();
    assert_eq!(results, expected, "par_get_many results are incorrect");

    // Test `par_get_many` with an out-of-bounds index
    let mut invalid_indices = indices_to_get.to_vec();
    invalid_indices.push(data.len()); // Add an invalid index
    let result_err = vec.par_get_many(&invalid_indices);
    assert!(
        matches!(result_err, Err(FixedVecError::InvalidParameters(_))),
        "par_get_many should fail on out-of-bounds index"
    );
}

#[test]
fn test_par_get_many_unchecked() {
    let data = get_test_data_signed();
    let vec: SFixedVec<i16> = FixedVec::builder().build(&data).unwrap();

    // Create a set of random indices
    let mut rng = StdRng::seed_from_u64(1337);
    let indices_to_get: Vec<usize> = (0..1000).map(|_| rng.random_range(0..data.len())).collect();

    let expected: Vec<i16> = indices_to_get.iter().map(|&i| data[i]).collect();

    // Test the unsafe `par_get_many_unchecked`
    // SAFETY: We are generating indices that are guaranteed to be in-bounds.
    let results = unsafe { vec.par_get_many_unchecked(&indices_to_get) };
    assert_eq!(
        results, expected,
        "par_get_many_unchecked results are incorrect"
    );

    // Test with an empty slice of indices
    let empty_results = unsafe { vec.par_get_many_unchecked(&[]) };
    assert!(
        empty_results.is_empty(),
        "par_get_many_unchecked with empty indices should return empty vec"
    );
}
