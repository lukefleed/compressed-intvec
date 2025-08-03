use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use std::time::Duration;

const VECTOR_SIZE: usize = 10_000_000;
const NUM_ACCESSES: usize = 1_000_000;

/// Generates a vector with uniformly random signed values.
fn generate_random_signed_vec(size: usize, bit_width: u32) -> Vec<i64> {
    if bit_width == 0 {
        return vec![0; size];
    }
    let mut rng = SmallRng::seed_from_u64(42);

    if bit_width == 64 {
        // Special case for 64-bit to avoid integer overflow when calculating the range.
        // We generate values across the full i64 range.
        (0..size).map(|_| rng.random::<i64>()).collect()
    } else {
        // For other bit widths, the range calculation is safe.
        // Generate values within the range that fits the bit width after ZigZag encoding.
        let max_val = 1i64 << (bit_width - 1);
        (0..size)
            .map(|_| rng.random_range(-max_val..max_val))
            .collect()
    }
}

/// Generates a fixed vector of random indices for access tests.
fn generate_indices(num_accesses: usize, vector_size: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(1337);
    (0..num_accesses)
        .map(|_| rng.random_range(0..vector_size))
        .collect()
}

/// The main benchmark function.
fn bench_zigzag_decode_performance(c: &mut Criterion) {
    // Generate a single set of random indices to be used across all benchmarks.
    let indices = generate_indices(NUM_ACCESSES, VECTOR_SIZE);

    // Test the byte-aligned bit widths where SIMD is most relevant.
    let bit_widths = [8, 16, 32, 64];

    for &bit_width in &bit_widths {
        let mut group = c.benchmark_group(format!("SFixedVec/BitWidth={}", bit_width));
        group.throughput(Throughput::Elements(NUM_ACCESSES as u64));

        let data_i64 = generate_random_signed_vec(VECTOR_SIZE, bit_width);
        let sfixed_vec = LESFixedVec::builder(&data_i64)
            .bit_width(BitWidth::Explicit(bit_width as usize))
            .build()
            .unwrap();

        // Benchmark the sequential batch method.
        group.bench_function("get_many_unchecked", |b| {
            b.iter(|| unsafe {
                // This will use SIMD for ZigZag decoding if the feature is enabled.
                black_box(sfixed_vec.get_many_unchecked(black_box(&indices)));
            })
        });

        // Benchmark the parallel batch method.
        #[cfg(feature = "parallel")]
        group.bench_function("par_get_many_unchecked", |b| {
            b.iter(|| unsafe {
                // This also benefits from SIMD ZigZag decoding on each thread's results.
                black_box(sfixed_vec.par_get_many_unchecked(black_box(&indices)));
            })
        });

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(15));
    targets = bench_zigzag_decode_performance
}
criterion_main!(benches);
