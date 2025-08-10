use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use dsi_bitstream::codes::len_rice;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;

/// Defines the data distributions for testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Distribution {
    UniformLow,
    RiceImplied,
}

impl Distribution {
    /// Generates a vector of data according to the distribution.
    fn generate(&self, size: usize) -> Vec<u64> {
        match self {
            Distribution::UniformLow => {
                let mut rng = SmallRng::seed_from_u64(42);
                (0..size).map(|_| rng.random_range(0..1_000)).collect()
            }
            Distribution::RiceImplied => {
                let mut rng = SmallRng::seed_from_u64(42);
                dsi_bitstream::utils::sample_implied_distribution(|v| len_rice(v, 4), &mut rng)
                    .take(size)
                    .collect()
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Distribution::UniformLow => "UniformLow",
            Distribution::RiceImplied => "RiceImplied",
        }
    }
}

/// The main benchmark function for construction speed.
fn benchmark_construction(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 1_000_000;
    const K_VALUE: usize = 32;

    let distributions = [Distribution::UniformLow, Distribution::RiceImplied];

    let codecs_to_test = [
        ("Auto", VariableCodecSpec::Auto),
        ("Gamma", VariableCodecSpec::Gamma),
        ("Delta", VariableCodecSpec::Delta),
        ("Unary", VariableCodecSpec::Unary),
        ("Rice_Auto", VariableCodecSpec::Rice { log2_b: None }),
        ("Zeta_Auto", VariableCodecSpec::Zeta { k: None }),
        ("Omega", VariableCodecSpec::Omega),
        ("VByteLe", VariableCodecSpec::VByteLe),
        ("VByteBe", VariableCodecSpec::VByteBe),
        ("Pi", VariableCodecSpec::Pi { k: Some(3) }),
        ("Golomb_Auto", VariableCodecSpec::Golomb { b: None }),
        ("ExpGolomb", VariableCodecSpec::ExpGolomb { k: Some(2) }),
    ];

    for distribution in distributions {
        let mut group = c.benchmark_group(format!("ConstructionSpeed/{}", distribution.name()));
        group.throughput(Throughput::Elements(VECTOR_SIZE as u64));

        let data = distribution.generate(VECTOR_SIZE);

        // --- Benchmark `build` from slice for various codecs ---
        for (spec_name, codec_spec) in codecs_to_test {
            // Skip Unary codec for RiceImplied distribution as it's too slow
            if distribution == Distribution::RiceImplied
                && matches!(codec_spec, VariableCodecSpec::Unary)
            {
                println!(
                    "Skipping codec {} for {} distribution (impractical).",
                    spec_name,
                    distribution.name()
                );
                continue;
            }

            group.bench_function(format!("from_slice/{}", spec_name), |b| {
                b.iter(|| {
                    black_box(
                        LEIntVec::builder(black_box(&data))
                            .k(K_VALUE)
                            .codec(codec_spec)
                            .build()
                            .unwrap(),
                    );
                })
            });
        }

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(10))
        .measurement_time(Duration::from_secs(2));
    targets = benchmark_construction
}
criterion_main!(benches);
