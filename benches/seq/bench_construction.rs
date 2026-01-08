use compressed_intvec::{prelude::*, seq::LESeqVec};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;

/// Defines the distribution of sequence lengths and element values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Distribution {
    /// Uniform small elements, uniform short sequences.
    UniformShort,
    /// Uniform large elements, uniform long sequences.
    UniformLong,
    /// Mixed: some very short, some very long sequences.
    Mixed,
}

impl Distribution {
    fn name(&self) -> &'static str {
        match self {
            Distribution::UniformShort => "UniformShort",
            Distribution::UniformLong => "UniformLong",
            Distribution::Mixed => "Mixed",
        }
    }

    /// Generates a vector of sequences according to the distribution.
    fn generate(&self, num_sequences: usize, rng_seed: u64) -> Vec<Vec<u64>> {
        let mut rng = SmallRng::seed_from_u64(rng_seed);

        match self {
            Distribution::UniformShort => (0..num_sequences)
                .map(|_| {
                    let seq_len = rng.random_range(1..=10);
                    (0..seq_len).map(|_| rng.random_range(0..1_000)).collect()
                })
                .collect(),
            Distribution::UniformLong => (0..num_sequences)
                .map(|_| {
                    let seq_len = rng.random_range(50..=200);
                    (0..seq_len)
                        .map(|_| rng.random_range(0..1_000_000))
                        .collect()
                })
                .collect(),
            Distribution::Mixed => {
                (0..num_sequences)
                    .map(|_| {
                        // Alternate between very short and very long
                        let seq_len = if rng.random() { 2 } else { 100 };
                        (0..seq_len)
                            .map(|_| rng.random_range(0..1_000_000))
                            .collect()
                    })
                    .collect()
            }
        }
    }
}

fn benchmark_construction(c: &mut Criterion) {
    let num_sequences = 10_000;
    let distributions = [
        Distribution::UniformShort,
        Distribution::UniformLong,
        Distribution::Mixed,
    ];

    for dist in distributions {
        let mut group = c.benchmark_group(format!("seq_construction/{}", dist.name()));
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(100));
        group.measurement_time(Duration::from_secs(2));

        let sequences = dist.generate(num_sequences, 42);

        // Count total elements for throughput reporting
        let total_elements: usize = sequences.iter().map(|s| s.len()).sum();
        group.throughput(Throughput::Elements(total_elements as u64));

        // 1. Benchmark from_slices (2-pass with Auto codec)
        group.bench_function(format!("{}/from_slices_auto", dist.name()), |b| {
            b.iter(|| {
                let slice_refs: Vec<&[u64]> = sequences.iter().map(|s| s.as_slice()).collect();
                let _seq_vec = black_box(LESeqVec::from_slices(&slice_refs))
                    .expect("Failed to build from_slices");
            })
        });

        // 2. Benchmark builder with explicit Delta codec (single-pass, fast)
        group.bench_function(format!("{}/builder_delta", dist.name()), |b| {
            b.iter(|| {
                let slice_refs: Vec<&[u64]> = sequences.iter().map(|s| s.as_slice()).collect();
                let _seq_vec = black_box(
                    LESeqVec::builder()
                        .codec(VariableCodecSpec::Delta)
                        .build(&slice_refs),
                )
                .expect("Failed to build with Delta");
            })
        });

        // 3. Benchmark builder with explicit Gamma codec (single-pass, reasonable)
        group.bench_function(format!("{}/builder_gamma", dist.name()), |b| {
            b.iter(|| {
                let slice_refs: Vec<&[u64]> = sequences.iter().map(|s| s.as_slice()).collect();
                let _seq_vec = black_box(
                    LESeqVec::builder()
                        .codec(VariableCodecSpec::Gamma)
                        .build(&slice_refs),
                )
                .expect("Failed to build with Gamma");
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
    targets = benchmark_construction
}
criterion_main!(benches);
