use compressed_intvec::prelude::*;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::IndexedRandom};
use rand_distr::{Distribution as RandDistribution, Uniform};
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

/// Defines the different access patterns to be benchmarked.
#[derive(Debug, Clone, Copy)]
enum AccessPattern {
    /// Indices are grouped into several "hot" clusters.
    Clustered,
    /// Indices are perfectly sequential.
    Sorted,
    /// Indices are fully random and uncorrelated.
    Random,
    /// Indices read one block, skip one block, and repeat.
    Strided,
}

impl AccessPattern {
    /// Returns a string representation for use in benchmark names.
    fn name(&self) -> &'static str {
        match self {
            AccessPattern::Clustered => "Clustered",
            AccessPattern::Sorted => "Sorted",
            AccessPattern::Random => "Random",
            AccessPattern::Strided => "Strided",
        }
    }

    /// Generates a vector of indices corresponding to the access pattern.
    fn generate_indices(
        &self,
        rng: &mut SmallRng,
        num_accesses: usize,
        vector_size: usize,
        k: usize,
    ) -> Vec<usize> {
        match self {
            AccessPattern::Random => (0..num_accesses)
                .map(|_| rng.random_range(0..vector_size))
                .collect(),
            AccessPattern::Sorted => {
                let mut indices: Vec<usize> = (0..num_accesses)
                    .map(|_| rng.random_range(0..vector_size))
                    .collect();
                indices.sort_unstable();
                indices
            }
            AccessPattern::Clustered => {
                let num_clusters = (num_accesses / 100).max(1);
                let mut centroids = vec![0; num_clusters];
                let uniform_centroid = Uniform::new(0, vector_size.saturating_sub(2 * k)).unwrap();
                for centroid in &mut centroids {
                    *centroid = uniform_centroid.sample(rng);
                }

                let mut indices = Vec::with_capacity(num_accesses);
                let uniform_offset = Uniform::new(0, 2 * k).unwrap();
                for _ in 0..num_accesses {
                    let centroid = centroids.choose(rng).unwrap();
                    let offset = uniform_offset.sample(rng);
                    indices.push((centroid + offset).min(vector_size - 1));
                }
                indices
            }
            AccessPattern::Strided => {
                let mut indices = Vec::new();
                let mut current_pos = 0;
                while current_pos < vector_size && indices.len() < num_accesses {
                    // Read a block of k indices.
                    let end_read_block = (current_pos + k).min(vector_size);
                    indices.extend(current_pos..end_read_block);
                    // Skip the next block.
                    current_pos += 2 * k;
                }
                indices.truncate(num_accesses);
                indices
            }
        }
    }
}

fn benchmark_iter_access(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 10_000_000;
    const NUM_ACCESSES: usize = 100_000;
    const K_VALUE: usize = 32;

    let data = generate_random_vec(VECTOR_SIZE, 1 << 20);
    let intvec = LEVarVec::builder()
        .k(K_VALUE)
        .codec(Codec::Delta)
        .build(&data)
        .expect("Failed to build VarVec");

    let patterns = [
        AccessPattern::Sorted,
        AccessPattern::Clustered,
        AccessPattern::Random,
        AccessPattern::Strided,
    ];

    for pattern in patterns {
        let mut group = c.benchmark_group(format!("IterAccess/{}", pattern.name()));
        group.throughput(Throughput::Elements(NUM_ACCESSES as u64));

        let mut rng = SmallRng::seed_from_u64(1337);
        let access_indices = pattern.generate_indices(&mut rng, NUM_ACCESSES, VECTOR_SIZE, K_VALUE);

        // 1. Benchmark `get_many_from_iter`
        group.bench_function("get_many_from_iter", |b| {
            b.iter(|| {
                let results = intvec
                    .get_many_from_iter(black_box(access_indices.iter().copied()))
                    .unwrap();
                black_box(results);
            })
        });

        // 2. Benchmark the loop with a reusable and STATEFUL `VarVecSeqReader`.
        group.bench_function("loop_seq_reader_get", |b| {
            b.iter(|| {
                let mut results = Vec::with_capacity(access_indices.len());
                let mut reader = intvec.seq_reader();
                for index in black_box(access_indices.iter().copied()) {
                    results.push(reader.get(index).unwrap().unwrap());
                }
                black_box(results);
            })
        });

        // 3. Benchmark the loop with a reusable but STATELESS `VarVecReader`.
        group.bench_function("loop_reader_get", |b| {
            b.iter(|| {
                let mut results = Vec::with_capacity(access_indices.len());
                let mut reader = intvec.reader();
                for index in black_box(access_indices.iter().copied()) {
                    results.push(reader.get(index).unwrap().unwrap());
                }
                black_box(results);
            })
        });

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(10))
        .measurement_time(Duration::from_secs(2));
    targets = benchmark_iter_access
}
criterion_main!(benches);
