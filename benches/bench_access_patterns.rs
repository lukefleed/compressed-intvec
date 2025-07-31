use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, seq::IndexedRandom, Rng, SeedableRng};
use rand_distr::{Distribution as RandDistribution, Uniform};
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

/// The main benchmark function.
fn benchmark_access_patterns(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 10_000_000;
    const NUM_ACCESSES: usize = 1_000_000;
    const K_VALUE: usize = 32;

    // --- Setup Data and IntVec ---
    let data = generate_random_vec(VECTOR_SIZE, 1 << 20);
    let intvec = LEIntVec::builder(&data)
        .k(K_VALUE)
        .codec(VariableCodecSpec::Delta)
        .build()
        .expect("Failed to build IntVec");

    // --- Setup sux::BitFieldVec for comparison ---
    let mut sux_bfv = BitFieldVec::<u64>::new(
        (u64::BITS - data.iter().max().unwrap_or(&0).leading_zeros()) as usize,
        0,
    );
    for &val in &data {
        sux_bfv.push(val);
    }

    let patterns = [
        AccessPattern::Clustered,
        AccessPattern::Sorted,
        AccessPattern::Random,
        AccessPattern::Strided,
    ];

    for pattern in patterns {
        let mut group = c.benchmark_group(format!("AccessPatterns/{}", pattern.name()));

        let mut rng = SmallRng::seed_from_u64(1337);
        let access_indices = pattern.generate_indices(&mut rng, NUM_ACCESSES, VECTOR_SIZE, K_VALUE);

        // --- Our IntVec benchmarks ---
        group.bench_function("IntVec/get_unchecked_loop", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // SAFETY: Indices are generated within bounds.
                    black_box(unsafe { intvec.get_unchecked(index) });
                }
            })
        });

        group.bench_function("IntVec/get_many_unchecked", |b| {
            b.iter(|| {
                // SAFETY: Indices are generated within bounds.
                let _ = black_box(unsafe { intvec.get_many_unchecked(black_box(&access_indices)) });
            })
        });

        #[cfg(feature = "parallel")]
        group.bench_function("IntVec/par_get_many_unchecked", |b| {
            b.iter(|| {
                // SAFETY: Indices are generated within bounds.
                let _ =
                    black_box(unsafe { intvec.par_get_many_unchecked(black_box(&access_indices)) });
            })
        });

        // --- sux::BitFieldVec benchmarks ---
        group.bench_function("sux::BitFieldVec/get_unchecked_loop", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // SAFETY: Indices are generated within bounds.
                    black_box(unsafe { sux_bfv.get_unchecked(index) });
                }
            })
        });

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = benchmark_access_patterns
}
criterion_main!(benches);
