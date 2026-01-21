use std::time::Duration;

// benches/fixed/bench_access_patterns.rs
use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, seq::IndexedRandom, Rng, SeedableRng};
use rand_distr::{Distribution as RandDistribution, Uniform};
use succinct::int_vec::{IntVec as SuccinctIntVec, IntVector};
use sux::prelude::{BitFieldSlice, BitFieldVec};

/// Generates a vector with uniformly random values up to a given maximum.
///
/// # Arguments
/// * `size` - The number of elements to generate.
/// * `max_val_exclusive` - The exclusive upper bound for the random values. A value of 0
///   indicates that the full range of `u64` should be used.
fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    if max_val_exclusive == 0 {
        // This case occurs if the requested bit width is 64.
        // We generate full-range u64 values.
        return (0..size).map(|_| rng.random::<u64>()).collect();
    }
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

/// Defines the different access patterns to be benchmarked.
#[derive(Debug, Clone, Copy)]
enum AccessPattern {
    /// Indices are fully random and uncorrelated.
    Random,
    /// Indices are sorted, simulating a sequential scan.
    Sorted,
    /// Indices are grouped into several "hot" clusters.
    Clustered,
    /// Indices read one block, skip one block, and repeat.
    Strided,
}

impl AccessPattern {
    /// Returns a string representation for use in benchmark names.
    fn name(&self) -> &'static str {
        match self {
            AccessPattern::Random => "Random",
            AccessPattern::Sorted => "Sorted",
            AccessPattern::Clustered => "Clustered",
            AccessPattern::Strided => "Strided",
        }
    }

    /// Generates a vector of indices corresponding to the access pattern.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    /// * `num_accesses` - The total number of indices to generate.
    /// * `vector_size` - The size of the vector being accessed.
    /// * `block_size` - A parameter used for strided and clustered patterns.
    fn generate_indices(
        &self,
        rng: &mut SmallRng,
        num_accesses: usize,
        vector_size: usize,
        block_size: usize,
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
                let num_clusters = (num_accesses / 100).max(2);
                let cluster_radius = 2 * block_size;
                let mut centroids = vec![0; num_clusters];
                let uniform_centroid =
                    Uniform::new(0, vector_size.saturating_sub(cluster_radius)).unwrap();
                for centroid in &mut centroids {
                    *centroid = uniform_centroid.sample(rng);
                }

                let mut indices = Vec::with_capacity(num_accesses);
                let uniform_offset = Uniform::new(0, cluster_radius).unwrap();
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
                    // Read a block of indices.
                    let end_read_block = (current_pos + block_size).min(vector_size);
                    indices.extend(current_pos..end_read_block);
                    // Skip the next block.
                    current_pos += 2 * block_size;
                }
                indices.truncate(num_accesses);
                indices
            }
        }
    }
}

/// The main benchmark function for different access patterns.
fn benchmark_access_patterns(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 10_000_000;
    const NUM_ACCESSES: usize = 1_000_000;
    const BIT_WIDTH: u32 = 16; // A non-power-of-two width to stress the logic.
    const BLOCK_SIZE: usize = 64; // Used for clustered/strided patterns.

    // --- Setup Data and Compressed Vectors ---
    let data = generate_random_vec(VECTOR_SIZE, 1u64 << BIT_WIDTH);
    let intvec = LEFixedVec::builder()
        .bit_width(BitWidth::Explicit(BIT_WIDTH as usize))
        .build(&data)
        .expect("Failed to build LEFixedVec");
    let sux_bfv = BitFieldVec::<u64>::from_slice(&data).unwrap();
    // Create succinct::IntVector by pushing elements.
    let mut succinct_iv = IntVector::<u64>::new(BIT_WIDTH as usize);
    for &val in &data {
        succinct_iv.push(val);
    }

    let patterns = [
        AccessPattern::Random,
        AccessPattern::Sorted,
        AccessPattern::Clustered,
        AccessPattern::Strided,
    ];

    for pattern in patterns {
        let mut group = c.benchmark_group(format!(
            "AccessPatterns/{}bit/{}",
            BIT_WIDTH,
            pattern.name()
        ));

        let mut rng = SmallRng::seed_from_u64(1337);
        let access_indices =
            pattern.generate_indices(&mut rng, NUM_ACCESSES, VECTOR_SIZE, BLOCK_SIZE);

        group.bench_function("Vec<u64>/get_unchecked_loop", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // SAFETY: Indices are generated within bounds.
                    black_box(unsafe { data.get_unchecked(index) });
                }
            })
        });

        group.bench_function("LEFixedVec/get_unchecked_loop", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // SAFETY: Indices are generated within bounds.
                    black_box(unsafe { intvec.get_unchecked(index) });
                }
            })
        });

        group.bench_function("sux::BitFieldVec/get_unchecked_loop", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // SAFETY: Indices are generated within bounds.
                    black_box(unsafe { sux_bfv.get_unchecked(index) });
                }
            })
        });

        group.bench_function("succinct::IntVector/get_loop", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // The `get` method in succinct performs bounds checking.
                    black_box(SuccinctIntVec::get(&succinct_iv, index as u64));
                }
            })
        });

        #[cfg(feature = "parallel")]
        group.bench_function("LEFixedVec/par_get_many_unchecked", |b| {
            b.iter(|| {
                // SAFETY: Indices are generated within bounds.
                let _ =
                    black_box(unsafe { intvec.par_get_many_unchecked(black_box(&access_indices)) });
            })
        });
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_secs(2));

    targets = benchmark_access_patterns
}
criterion_main!(benches);
