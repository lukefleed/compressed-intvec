//! Integration tests for `SIntVec`.

use compressed_intvec::prelude::*;
use dsi_bitstream::prelude::{ToInt, ToNat};
use rand::{rngs::StdRng, Rng, SeedableRng};

// Import helper functions from the common module.
#[path = "common/mod.rs"]
mod common;
use common::helpers::generate_random_signed_vec;

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

#[test]
fn test_sintvec_construction_and_get() {
    let data = generate_random_signed_vec(1000, 500);

    // Build the SIntVec using Gamma codec (a safe choice).
    let sintvec = LESIntVec::builder(&data)
        .codec(CodecSpec::Gamma)
        .k(16)
        .build()
        .unwrap();

    // Basic property checks
    assert_eq!(sintvec.len(), data.len());
    assert!(!sintvec.is_empty());
    assert_eq!(sintvec.get_sampling_rate(), Some(16));

    // Test `get` for single random access
    let indices_to_test = [0, data.len() / 2, data.len() - 1];
    for &idx in &indices_to_test {
        assert_eq!(sintvec.get(idx), Some(data[idx]), "get({}) failed", idx);
    }
}

#[test]
fn test_sintvec_iter() {
    let data = generate_random_signed_vec(1000, 500);
    let sintvec = LESIntVec::builder(&data)
        .codec(CodecSpec::Gamma)
        .build()
        .unwrap();

    // Test full decompression via iterator
    let decompressed: Vec<i64> = sintvec.iter().collect();
    assert_eq!(decompressed, data, "Iterator decompression failed");
}

#[test]
fn test_sintvec_with_fixed_length() {
    let data = vec![-500, 499, 0, -1, 1];
    // ZigZag encoding of -500 is 999. 999 fits in 10 bits.
    let sintvec = LESIntVec::builder(&data)
        .codec(CodecSpec::FixedLength { num_bits: Some(10) })
        .build()
        .unwrap();

    assert_eq!(sintvec.len(), data.len());
    assert_eq!(sintvec.get(0), Some(-500));
    assert_eq!(sintvec.get(1), Some(499));
    assert_eq!(sintvec.iter().collect::<Vec<_>>(), data);
    assert_eq!(sintvec.get_sampling_rate(), None);
}

#[test]
fn test_sintvec_invalid_parameters() {
    let data = vec![-10, 20, 1000];
    // ZigZag of 1000 is 1999, which requires 11 bits.
    let result = LESIntVec::builder(&data)
        .codec(CodecSpec::FixedLength { num_bits: Some(10) })
        .build();
    assert!(matches!(result, Err(IntVecError::InvalidParameters(_))));
}

#[test]
fn test_sintvec_builder_rejects_auto_codecs() {
    let data = vec![-10, 20, 100];
    let result = LESIntVec::builder(&data).codec(CodecSpec::Auto).build();
    // The SIntVec builder uses the from_iter_builder internally, which rejects auto codecs.
    assert!(matches!(result, Err(IntVecError::InvalidParameters(_))));
}

#[test]
#[cfg(feature = "parallel")]
fn test_sintvec_parallel_methods() {
    let data = generate_random_signed_vec(10_000, 5000);
    let sintvec = LESIntVec::builder(&data)
        .codec(CodecSpec::Gamma)
        .k(32)
        .build()
        .unwrap();

    // Test par_iter
    let par_decompressed: Vec<i64> = sintvec.par_iter().collect();
    assert_eq!(par_decompressed, data, "Parallel iterator failed");

    // Test par_get_many
    let mut rng = StdRng::seed_from_u64(1337);
    let indices: Vec<usize> = (0..200).map(|_| rng.random_range(0..data.len())).collect();
    let expected: Vec<i64> = indices.iter().map(|&i| data[i]).collect();
    let par_results = sintvec.par_get_many(&indices).unwrap();
    assert_eq!(par_results, expected, "par_get_many failed");
}

#[test]
fn test_zigzag_identities() {
    let values: Vec<i64> = vec![0, -1, 1, -2, 2, i64::MAX, i64::MIN];
    for &v in &values {
        assert_eq!(v.to_nat().to_int(), v);
    }
}
