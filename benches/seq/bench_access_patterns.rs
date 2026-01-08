use compressed_intvec::{prelude::*, seq::LESeqVec};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, seq::IndexedRandom, Rng, SeedableRng};
use std::time::Duration;

/// Defines different access patterns to sequences.
#[derive(Debug, Clone, Copy)]
enum AccessPattern {
    /// Sequentially access indices 0, 1, 2, ...
    Sequential,
    /// Access indices grouped into clusters of nearby values.
    Clustered,
    /// Fully random uncorrelated access.
    Random,
}

impl AccessPattern {
    fn name(&self) -> &'static str {
        match self {
            AccessPattern::Sequential => "Sequential",
            AccessPattern::Clustered => "Clustered",
            AccessPattern::Random => "Random",
        }
    }

    fn generate_indices(
        &self,
        num_sequences: usize,
        num_accesses: usize,
        rng_seed: u64,
    ) -> Vec<usize> {
        let mut rng = SmallRng::seed_from_u64(rng_seed);

        match self {
            AccessPattern::Sequential => (0..num_accesses).map(|i| i % num_sequences).collect(),
            AccessPattern::Clustered => {
                // Create 5 clusters, randomly select within each cluster
                let num_clusters = 5;
                let cluster_size = num_sequences / num_clusters;
                (0..num_accesses)
                    .map(|_| {
                        let cluster = rng.random_range(0..num_clusters);
                        let offset = rng.random_range(0..cluster_size);
                        cluster * cluster_size + offset
                    })
                    .collect()
            }
            AccessPattern::Random => (0..num_sequences)
                .collect::<Vec<_>>()
                .choose_multiple(&mut rng, num_accesses)
                .copied()
                .collect(),
        }
    }
}

fn generate_sequences(num_sequences: usize, avg_len: usize, rng_seed: u64) -> Vec<Vec<u64>> {
    let mut rng = SmallRng::seed_from_u64(rng_seed);
    (0..num_sequences)
        .map(|_| {
            let seq_len = std::cmp::max(1, rng.random_range((avg_len / 2)..=(avg_len * 2)));
            (0..seq_len).map(|_| rng.random_range(0..100_000)).collect()
        })
        .collect()
}

fn benchmark_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("seq_access_patterns");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(2));

    let num_sequences = 10_000;
    let num_accesses = 1_000;

    let sequences = generate_sequences(num_sequences, 50, 42);
    let seq_vec = {
        let slice_refs: Vec<&[u64]> = sequences.iter().map(|s| s.as_slice()).collect();
        LESeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .build(&slice_refs)
            .expect("Failed to build SeqVec")
    };

    group.throughput(Throughput::Elements(num_accesses as u64));

    let patterns = [
        AccessPattern::Sequential,
        AccessPattern::Clustered,
        AccessPattern::Random,
    ];

    for pattern in patterns {
        let indices = pattern.generate_indices(num_sequences, num_accesses, 123);

        // 1. Stateless reader (fresh seek for each access)
        group.bench_function(format!("{}/reader", pattern.name()), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                let reader = black_box(&seq_vec).reader();
                for &idx in black_box(indices.iter()) {
                    if let Some(seq_iter) = reader.get(idx) {
                        sum = sum.wrapping_add(seq_iter.count() as u64);
                    }
                }
                black_box(sum);
            })
        });

        // 2. Stateful seq_reader (optimized for sequential-like patterns)
        group.bench_function(format!("{}/seq_reader", pattern.name()), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                let mut seq_reader = black_box(&seq_vec).seq_reader();
                for &idx in black_box(indices.iter()) {
                    if let Some(seq_iter) = seq_reader.get(idx) {
                        sum = sum.wrapping_add(seq_iter.count() as u64);
                    }
                }
                black_box(sum);
            })
        });
    }

    group.finish();
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
