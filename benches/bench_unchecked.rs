use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;

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
    const K_VALUE: usize = 64;

    let data = generate_random_vec(VECTOR_SIZE, 1 << 20);

    // --- 1. Setup IntVec (Variable-Length) ---
    let intvec = LEIntVec::builder(&data)
        .k(K_VALUE)
        .codec(VariableCodecSpec::Auto)
        .build()
        .expect("Failed to build IntVec");

    // --- 2. Setup FixedVec ---
    let fixed_vec = LEFixedVec::builder(&data)
        .build()
        .expect("Failed to build FixedVec");

    // --- 3. Setup Indices for Access ---
    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();

    // --- 4. Benchmark IntVec ---
    let mut group_intvec = c.benchmark_group("CheckedVsUnchecked/IntVec");
    group_intvec.throughput(Throughput::Elements(NUM_ACCESSES as u64));

    // a) `get` in a loop
    group_intvec.bench_function("get_loop_(checked)", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(intvec.get(index));
            }
        })
    });

    // b) `get_unchecked` in a loop
    group_intvec.bench_function("get_unchecked_loop", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                // SAFETY: Indices are generated within bounds for the benchmark.
                black_box(unsafe { intvec.get_unchecked(index) });
            }
        })
    });

    // c) `get_many`
    group_intvec.bench_function("get_many_(checked)", |b| {
        b.iter(|| {
            black_box(intvec.get_many(black_box(&access_indices))).unwrap();
        })
    });

    // d) `get_many_unchecked`
    group_intvec.bench_function("get_many_unchecked", |b| {
        b.iter(|| {
            // SAFETY: Indices are generated within bounds for the benchmark.
            black_box(unsafe { intvec.get_many_unchecked(black_box(&access_indices)) });
        })
    });
    group_intvec.finish();

    // --- 5. Benchmark FixedVec ---
    let mut group_fixedvec = c.benchmark_group("CheckedVsUnchecked/FixedVec");
    group_fixedvec.throughput(Throughput::Elements(NUM_ACCESSES as u64));

    // a) `get` in a loop
    group_fixedvec.bench_function("get_loop_(checked)", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(fixed_vec.get(index));
            }
        })
    });

    // b) `get_unchecked` in a loop
    group_fixedvec.bench_function("get_unchecked_loop", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                // SAFETY: Indices are generated within bounds for the benchmark.
                black_box(unsafe { fixed_vec.get_unchecked(index) });
            }
        })
    });

    // c) `get_many`
    group_fixedvec.bench_function("get_many_(checked)", |b| {
        b.iter(|| {
            black_box(fixed_vec.get_many(black_box(&access_indices))).unwrap();
        })
    });

    // d) `get_many_unchecked`
    group_fixedvec.bench_function("get_many_unchecked", |b| {
        b.iter(|| {
            // SAFETY: Indices are generated within bounds for the benchmark.
            black_box(unsafe { fixed_vec.get_many_unchecked(black_box(&access_indices)) });
        })
    });
    group_fixedvec.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_secs(5));
    targets = benchmark_checked_vs_unchecked
}
criterion_main!(benches);
