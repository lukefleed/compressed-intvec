use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;

const DATA_SIZE: usize = 10_000_000; // 10M elements
const INDEX_COUNT: usize = 1_000_000; // 1M random indices

fn generate_test_data() -> (Vec<u64>, Vec<i64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(42);

    // Generate 10M random unsigned integers
    let unsigned_data: Vec<u64> = (0..DATA_SIZE)
        .map(|_| rng.random_range(0..1_000_000))
        .collect();

    // Generate 10M random signed integers
    let signed_data: Vec<i64> = (0..DATA_SIZE)
        .map(|_| rng.random_range(-500_000..500_000))
        .collect();

    // Generate 1M random indices
    let indices: Vec<usize> = (0..INDEX_COUNT)
        .map(|_| rng.random_range(0..DATA_SIZE))
        .collect();
    (unsigned_data, signed_data, indices)
}

fn bench_unsigned_access(c: &mut Criterion) {
    let (unsigned_data, _, indices) = generate_test_data();

    // Create FixedVec with minimal bit width
    let fixed_vec = LEFixedVec::from_slice(&unsigned_data).unwrap();

    let mut group = c.benchmark_group("unsigned_access");
    group.sample_size(10);

    // Benchmark: Individual get_unchecked calls in a loop
    group.bench_function("loop_get_unchecked", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(INDEX_COUNT);
            for &idx in &indices {
                let value = unsafe { fixed_vec.get_unchecked(idx) };
                results.push(black_box(value));
            }
            results
        })
    });

    // Benchmark: get_many_unchecked without SIMD (compile without simd feature)
    group.bench_function("get_many_unchecked", |b| {
        b.iter(|| {
            let results = unsafe { fixed_vec.get_many_unchecked(black_box(&indices)) };
            black_box(results)
        })
    });

    #[cfg(feature = "parallel")]
    {
        // Benchmark: par_get_many_unchecked
        group.bench_function("par_get_many_unchecked", |b| {
            b.iter(|| {
                let results = unsafe { fixed_vec.par_get_many_unchecked(black_box(&indices)) };
                black_box(results)
            })
        });
    }

    group.finish();
}

fn bench_signed_access(c: &mut Criterion) {
    let (_, signed_data, indices) = generate_test_data();

    // Create SFixedVec with minimal bit width
    let sfixed_vec = LESFixedVec::from_slice(&signed_data).unwrap();

    let mut group = c.benchmark_group("signed_access");
    group.sample_size(10);

    // Benchmark: Individual get_unchecked calls in a loop
    group.bench_function("loop_get_unchecked", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(INDEX_COUNT);
            for &idx in &indices {
                let value = unsafe { sfixed_vec.get_unchecked(idx) };
                results.push(black_box(value));
            }
            results
        })
    });

    // Benchmark: get_many_unchecked (with SIMD zigzag decoding if enabled)
    group.bench_function("get_many_unchecked", |b| {
        b.iter(|| {
            let results = unsafe { sfixed_vec.get_many_unchecked(black_box(&indices)) };
            black_box(results)
        })
    });

    #[cfg(feature = "parallel")]
    {
        // Benchmark: par_get_many_unchecked
        group.bench_function("par_get_many_unchecked", |b| {
            b.iter(|| {
                let results = unsafe { sfixed_vec.par_get_many_unchecked(black_box(&indices)) };
                black_box(results)
            })
        });
    }

    group.finish();
}

fn bench_simd_comparison(c: &mut Criterion) {
    let (unsigned_data, signed_data, indices) = generate_test_data();

    // Create test vectors
    let fixed_vec = LEFixedVec::from_slice(&unsigned_data).unwrap();
    let sfixed_vec = LESFixedVec::from_slice(&signed_data).unwrap();

    let mut group = c.benchmark_group("simd_comparison");
    group.sample_size(10);

    // Compare different data sizes to see SIMD scaling
    for &size in &[1_000, 10_000, 100_000, 1_000_000] {
        let subset_indices = &indices[..size];

        group.bench_with_input(
            BenchmarkId::new("unsigned_get_many", size),
            &subset_indices,
            |b, indices| {
                b.iter(|| {
                    let results = unsafe { fixed_vec.get_many_unchecked(black_box(indices)) };
                    black_box(results)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("signed_get_many", size),
            &subset_indices,
            |b, indices| {
                b.iter(|| {
                    let results = unsafe { sfixed_vec.get_many_unchecked(black_box(indices)) };
                    black_box(results)
                })
            },
        );

        #[cfg(feature = "parallel")]
        {
            group.bench_with_input(
                BenchmarkId::new("unsigned_par_get_many", size),
                &subset_indices,
                |b, indices| {
                    b.iter(|| {
                        let results =
                            unsafe { fixed_vec.par_get_many_unchecked(black_box(indices)) };
                        black_box(results)
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("signed_par_get_many", size),
                &subset_indices,
                |b, indices| {
                    b.iter(|| {
                        let results =
                            unsafe { sfixed_vec.par_get_many_unchecked(black_box(indices)) };
                        black_box(results)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_zigzag_simd(c: &mut Criterion) {
    let (_, signed_data, indices) = generate_test_data();
    let sfixed_vec = LESFixedVec::from_slice(&signed_data).unwrap();

    let mut group = c.benchmark_group("zigzag_decoding");
    group.sample_size(10);

    // Test different vector sizes to see SIMD benefits
    for &size in &[64, 256, 1024, 8192, 65536] {
        let subset_indices = &indices[..size];

        group.bench_with_input(
            BenchmarkId::new("zigzag_decode", size),
            &subset_indices,
            |b, indices| {
                b.iter(|| {
                    let results = unsafe { sfixed_vec.get_many_unchecked(black_box(indices)) };
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

fn bench_memory_patterns(c: &mut Criterion) {
    let (unsigned_data, _, _) = generate_test_data();
    let fixed_vec = LEFixedVec::from_slice(&unsigned_data).unwrap();

    let mut group = c.benchmark_group("memory_patterns");
    group.sample_size(10);

    // Sequential access pattern
    let sequential_indices: Vec<usize> = (0..INDEX_COUNT).collect();

    // Random access pattern
    let mut rng = StdRng::seed_from_u64(123);
    let random_indices: Vec<usize> = (0..INDEX_COUNT)
        .map(|_| rng.random_range(0..DATA_SIZE))
        .collect();

    // Clustered access pattern (groups of nearby indices)
    let mut clustered_indices = Vec::with_capacity(INDEX_COUNT);
    for _ in 0..INDEX_COUNT / 64 {
        let base = rng.random_range(0..DATA_SIZE - 64);
        for offset in 0..64 {
            clustered_indices.push(base + offset);
        }
    } // Benchmark different access patterns
    group.bench_function("sequential_access", |b| {
        b.iter(|| {
            let results = unsafe { fixed_vec.get_many_unchecked(black_box(&sequential_indices)) };
            black_box(results)
        })
    });

    group.bench_function("random_access", |b| {
        b.iter(|| {
            let results = unsafe { fixed_vec.get_many_unchecked(black_box(&random_indices)) };
            black_box(results)
        })
    });

    group.bench_function("clustered_access", |b| {
        b.iter(|| {
            let results = unsafe { fixed_vec.get_many_unchecked(black_box(&clustered_indices)) };
            black_box(results)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    // bench_unsigned_access,
    // bench_signed_access,
    bench_simd_comparison,
    // bench_zigzag_simd,
    // bench_memory_patterns
);
criterion_main!(benches);
