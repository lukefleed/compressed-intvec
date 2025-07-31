use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;
// Importa i tipi necessari da sux
use sux::prelude::{BitFieldSlice, BitFieldVec};

/// Generates a vector with uniformly random values.
fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    if max_val_exclusive == 0 {
        return vec![0; size];
    }
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

/// The main benchmark function to compare checked vs. unchecked access.
fn benchmark_checked_vs_unchecked(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 10_000_000;
    const NUM_ACCESSES: usize = 1_000_000;

    let data = generate_random_vec(VECTOR_SIZE, 1 << 8);

    // --- 2. Setup FixedVec ---
    let fixed_vec = LEFixedVec::builder(&data)
        .build()
        .expect("Failed to build FixedVec");

    // --- 3. Setup sux::BitFieldVec ---
    let sux_bfv = BitFieldVec::<u64>::from_slice(&data).unwrap();

    // --- 4. Setup Indices for Access ---
    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();

    // a) Our FixedVec (checked)
    c.bench_function("FixedVec_get_loop_(checked)", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(fixed_vec.get(index));
            }
        })
    });

    // b) Our FixedVec (unchecked)
    c.bench_function("FixedVec_get_unchecked_loop", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(unsafe { fixed_vec.get_unchecked(index) });
            }
        })
    });

    // c) sux::BitFieldVec (checked)
    c.bench_function("sux::BitFieldVec_get_loop_(checked)", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(sux_bfv.get(index));
            }
        })
    });

    // d) sux::BitFieldVec (unchecked)
    c.bench_function("sux::BitFieldVec_get_unchecked_loop", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(unsafe { sux_bfv.get_unchecked(index) });
            }
        })
    });

    // Standard Vec
    c.bench_function("Vec_get_loop_(checked)", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(data[index]);
            }
        })
    });

    c.bench_function("Vec_get_unchecked_loop", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                // SAFETY: We assume the indices are valid as per our setup.
                black_box(unsafe { data.get_unchecked(index) });
            }
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = benchmark_checked_vs_unchecked
}
criterion_main!(benches);
