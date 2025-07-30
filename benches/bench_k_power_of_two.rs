//! # Micro-Benchmark for `k` Power-of-Two Optimization
//!
//! This benchmark is specifically designed to measure the performance impact of the
//! "smart dispatch" optimization in `IntVec::get_many`. It directly compares the
//! performance of a `get_many` call when the sampling rate `k` is a power of
//! two (enabling a fast path with bit-shifts) versus when it is not (requiring
//! a fallback to slower division operations).
//!
//! ## Methodology
//!
//! To isolate the effect of this specific optimization, the benchmark simplifies
//! all other variables:
//! - It uses a single, consistent data distribution (`UniformLow`).
//! - It uses a single access pattern: a large batch of pre-sorted indices. This
//!   maximizes the time spent within the core loop of `get_many`, making the
//!   difference between bit-shifts and divisions more prominent.
//! - It compares two nearly identical `IntVec` instances, differing only in their
//!   `k` value: `k=32` (power of two) and `k=31` (not a power of two).
//!
//! The expected outcome is to demonstrate a clear and quantifiable performance
//! advantage for `k=32`, thus justifying the additional code complexity of the
//! smart dispatch mechanism.

use compressed_intvec::{codec_spec::CodecSpec, intvec::LEIntVec};
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

/// The main benchmark function.
fn benchmark_k_optimization(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 10_000_000;
    const NUM_ACCESSES: usize = 1_000_000;

    // --- 1. Setup Data and Indices ---

    // Generate a moderately large vector of data.
    let data = generate_random_vec(VECTOR_SIZE, 1 << 20);

    // Generate a large number of random indices to access.
    let mut rng = SmallRng::seed_from_u64(1337);
    let mut access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();
    // Pre-sort the indices to simulate the ideal sequential scan pattern,
    // which forces the benchmark to spend maximum time in the optimized loop.
    access_indices.sort_unstable();

    // --- 2. Build IntVec Instances ---

    // Build the IntVec with k = 32 (a power of two, enabling the fast path).
    let intvec_k32 = LEIntVec::builder(&data)
        .k(32)
        .codec(CodecSpec::Delta) // A typical, fast codec
        .build()
        .expect("Failed to build IntVec with k=32");

    // Build the IntVec with k = 31 (NOT a power of two, forcing the fallback path).
    let intvec_k31 = LEIntVec::builder(&data)
        .k(31)
        .codec(CodecSpec::Delta)
        .build()
        .expect("Failed to build IntVec with k=31");

    // --- 3. Run Benchmarks ---

    let mut group = c.benchmark_group("K_PowerOfTwo_Optimization");
    group.throughput(Throughput::Elements(NUM_ACCESSES as u64));

    // Benchmark the k=32 (power of two) case.
    group.bench_function("get_many_k=32_(fast_path_bitshift)", |b| {
        b.iter(|| {
            let _ = black_box(intvec_k32.get_many(black_box(&access_indices)));
        })
    });

    // Benchmark the k=31 (non-power of two) case.
    group.bench_function("get_many_k=31_(fallback_path_division)", |b| {
        b.iter(|| {
            let _ = black_box(intvec_k31.get_many(black_box(&access_indices)));
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(5));
    targets = benchmark_k_optimization
}
criterion_main!(benches);
