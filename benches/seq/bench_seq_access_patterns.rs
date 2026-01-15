// benches/seq/bench_seq_access_patterns.rs
//
// Benchmarks for SeqVec access patterns. Compares different access patterns
// (sequential, clustered, random) to measure the impact of spatial locality
// on compressed sequence decoding performance.

use compressed_intvec::seq::{LESeqVec, SeqVec, VariableCodecSpec};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, seq::IndexedRandom, Rng, SeedableRng};
use rand_distr::{Distribution as RandDistribution, Uniform};
use std::time::Duration;

const NUM_SEQUENCES: usize = 100_000;
const NUM_ACCESSES: usize = 50_000;

/// Defines the order in which sequences are accessed.
#[derive(Debug, Clone, Copy)]
enum AccessPattern {
    /// Sequences accessed in strictly increasing order.
    Sequential,
    /// Sequences accessed in random, uncorrelated order.
    Random,
    /// Sequences grouped into "hot" clusters with high locality.
    Clustered,
}

impl AccessPattern {
    fn name(&self) -> &'static str {
        match self {
            AccessPattern::Sequential => "Sequential",
            AccessPattern::Random => "Random",
            AccessPattern::Clustered => "Clustered",
        }
    }

    fn generate_indices(
        &self,
        rng: &mut SmallRng,
        num_accesses: usize,
        num_sequences: usize,
    ) -> Vec<usize> {
        match self {
            AccessPattern::Sequential => (0..num_accesses).map(|i| i % num_sequences).collect(),
            AccessPattern::Random => (0..num_accesses)
                .map(|_| rng.random_range(0..num_sequences))
                .collect(),
            AccessPattern::Clustered => {
                let num_clusters = 10;
                let cluster_radius = num_sequences / 50;
                let centroids: Vec<usize> = (0..num_clusters)
                    .map(|_| rng.random_range(0..num_sequences))
                    .collect();

                let offset_dist = Uniform::new(0, cluster_radius.max(1)).unwrap();
                (0..num_accesses)
                    .map(|_| {
                        let centroid = *centroids.choose(rng).unwrap();
                        let offset = offset_dist.sample(rng);
                        (centroid + offset).min(num_sequences - 1)
                    })
                    .collect()
            }
        }
    }
}

/// Generates sequences with power-law length distribution (realistic for graphs).
fn generate_sequences(rng: &mut SmallRng, num_sequences: usize) -> Vec<Vec<u32>> {
    let max_value = 10_000u32;
    (0..num_sequences)
        .map(|_| {
            // Power-law-ish: many short, few long
            let r: f64 = rng.random();
            let len = if r < 0.5 {
                rng.random_range(1..=5)
            } else if r < 0.85 {
                rng.random_range(5..=20)
            } else if r < 0.97 {
                rng.random_range(20..=100)
            } else {
                rng.random_range(100..=500)
            };
            (0..len).map(|_| rng.random_range(1..=max_value)).collect()
        })
        .collect()
}

fn count_total_elements(sequences: &[Vec<u32>], indices: &[usize]) -> u64 {
    indices.iter().map(|&i| sequences[i].len() as u64).sum()
}

/// Core benchmark: compares reader types across access patterns.
fn benchmark_access_patterns(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let sequences = generate_sequences(&mut rng, NUM_SEQUENCES);

    let seqvec: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences)
        .expect("Failed to build SeqVec");

    let patterns = [
        AccessPattern::Sequential,
        AccessPattern::Clustered,
        AccessPattern::Random,
    ];

    for pattern in patterns {
        let mut group = c.benchmark_group(format!("SeqAccessPattern/{}", pattern.name()));

        let mut pattern_rng = SmallRng::seed_from_u64(1337);
        let indices = pattern.generate_indices(&mut pattern_rng, NUM_ACCESSES, NUM_SEQUENCES);
        let total_elements = count_total_elements(&sequences, &indices);
        group.throughput(Throughput::Elements(total_elements));

        // Baseline: uncompressed Vec<Vec<u32>>
        group.bench_function("Baseline_VecVec", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    for &val in &sequences[idx] {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // SeqVecReader (stateless) with get_into
        group.bench_function("Reader_get_into", |b| {
            b.iter(|| {
                let mut reader = seqvec.reader();
                let mut buffer = Vec::with_capacity(64);
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    reader.get_into(idx, &mut buffer).unwrap();
                    for &val in &buffer {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        group.finish();
    }
}

/// Benchmark: sorted indices vs unsorted for batch access.
fn benchmark_sorted_batch(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let sequences = generate_sequences(&mut rng, NUM_SEQUENCES);

    let seqvec: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences)
        .expect("Failed to build SeqVec");

    let indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..NUM_SEQUENCES))
        .collect();

    let mut sorted_indices = indices.clone();
    sorted_indices.sort_unstable();

    let total_elements = count_total_elements(&sequences, &indices);

    let mut group = c.benchmark_group("SeqBatchAccess");
    group.throughput(Throughput::Elements(total_elements));

    // Random order with SeqVecReader
    group.bench_function("Unsorted_Reader", |b| {
        b.iter(|| {
            let mut reader = seqvec.reader();
            let mut buffer = Vec::with_capacity(64);
            let mut sum = 0u64;
            for &idx in black_box(&indices) {
                reader.get_into(idx, &mut buffer).unwrap();
                for &val in &buffer {
                    sum += val as u64;
                }
            }
            black_box(sum)
        })
    });

    // Sorted order with SeqVecReader
    group.bench_function("Sorted_Reader", |b| {
        b.iter(|| {
            let mut reader = seqvec.reader();
            let mut buffer = Vec::with_capacity(64);
            let mut sum = 0u64;
            for &idx in black_box(&sorted_indices) {
                reader.get_into(idx, &mut buffer).unwrap();
                for &val in &buffer {
                    sum += val as u64;
                }
            }
            black_box(sum)
        })
    });

    // get_many_sequences (internally sorts)
    group.bench_function("get_many_sequences", |b| {
        b.iter(|| {
            let results = seqvec
                .get_many_sequences(black_box(&indices))
                .expect("get_many_sequences failed");
            let sum: u64 = results
                .iter()
                .flat_map(|seq| seq.iter())
                .map(|&v| v as u64)
                .sum();
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(5));
    targets = benchmark_access_patterns, benchmark_sorted_batch
}

criterion_main!(benches);
