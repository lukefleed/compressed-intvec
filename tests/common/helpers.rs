//! Helper functions for integration tests.
#![allow(dead_code)] // Allow unused functions as this is a helper library for tests
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Generates a vector with uniformly random `u64` values.
pub fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    if max_val_exclusive == 0 {
        return vec![0; size];
    }
    let mut rng = StdRng::seed_from_u64(42);
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

/// Generates a vector with uniformly random `i64` values.
pub fn generate_random_signed_vec(size: usize, max_val_exclusive: i64) -> Vec<i64> {
    if max_val_exclusive == 0 {
        return vec![0; size];
    }
    let mut rng = StdRng::seed_from_u64(42);
    (0..size)
        .map(|_| rng.random_range(-max_val_exclusive..max_val_exclusive))
        .collect()
}

/// Generates a `u64` vector with a specific distribution based on a code's length function.
pub fn generate_with_distribution(size: usize, len_fn: impl Fn(u64) -> usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    dsi_bitstream::utils::sample_implied_distribution(len_fn, &mut rng)
        .take(size)
        .collect()
}
