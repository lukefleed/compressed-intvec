use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use dsi_bitstream::{
    codes::{len_rice, len_zeta_param},
    utils::sample_implied_distribution,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;

#[cfg(feature = "parallel")]
use compressed_intvec::{codec_spec::CodecSpec, intvec::LEIntVec};

/// Enum to define the data distributions for testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Distribution {
    UniformLow,
    UniformHigh,
    Geometric,
    PowerLaw,
}

impl Distribution {
    /// Generates a vector of data according to the distribution.
    fn generate(&self, size: usize) -> Vec<u64> {
        match self {
            Distribution::UniformLow => generate_random_vec(size, 1_000),
            Distribution::UniformHigh => generate_random_vec(size, 1 << 32),
            Distribution::Geometric => {
                let mut rng = SmallRng::seed_from_u64(42);
                sample_implied_distribution(|v| len_rice(v, 4), &mut rng)
                    .take(size)
                    .collect()
            }
            Distribution::PowerLaw => {
                let mut rng = SmallRng::seed_from_u64(42);
                sample_implied_distribution(|v| len_zeta_param::<false>(v, 3), &mut rng)
                    .take(size)
                    .collect()
            }
        }
    }
}

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

/// The main benchmark function that orchestrates all tests.
fn benchmark_random_access(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 1_000_000;
    const NUM_ACCESSES: usize = 10_000;
    const K_VALUES: [usize; 4] = [16, 32, 64, 128];

    let distributions = [
        (Distribution::UniformLow, "UniformLow"),
        (Distribution::UniformHigh, "UniformHigh"),
        (Distribution::Geometric, "Geometric"),
        (Distribution::PowerLaw, "PowerLaw"),
    ];

    // Codecs that are dependent on the `k` parameter of the integer vector.
    let k_dependent_codecs = [
        ("Gamma", CodecSpec::Gamma),
        ("Delta", CodecSpec::Delta),
        ("Unary", CodecSpec::Unary),
        ("Rice", CodecSpec::Rice { log2_b: None }),
        ("Zeta", CodecSpec::Zeta { k: None }),
        ("Explicit_Omega", CodecSpec::Omega),
        ("Explicit_VByteLe", CodecSpec::VByteLe),
        ("Explicit_VByteBe", CodecSpec::VByteBe),
        ("Explicit_Pi", CodecSpec::Pi { k: Some(3) }),
        ("Explicit_Golomb", CodecSpec::Golomb { b: Some(8) }),
        ("Explicit_ExpGolomb", CodecSpec::ExpGolomb { k: Some(2) }),
    ];

    // Codecs that are not dependent on `k`.
    let k_independent_codecs = [("FixedLength", CodecSpec::FixedLength { num_bits: None })];

    // Prepare a vector of random indices for access tests.
    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();

    for (distribution, dist_name) in distributions {
        let mut group = c.benchmark_group(format!("RandomAccess/{}", dist_name));

        // Configure the benchmark group settings.
        group
            .sample_size(10)
            .warm_up_time(Duration::from_millis(1))
            .throughput(Throughput::Elements(NUM_ACCESSES as u64));

        let data = distribution.generate(VECTOR_SIZE);

        // --- Baseline benchmark on the original Vec<u64> ---
        group.bench_function("Baseline/get", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // Accessing the original, uncompressed vector.
                    black_box(data[index]);
                }
            })
        });

        // Benchmark k-dependent codecs
        for (spec_name, codec_spec) in k_dependent_codecs {
            if matches!(
                distribution,
                Distribution::UniformHigh | Distribution::PowerLaw
            ) {
                if matches!(
                    codec_spec,
                    CodecSpec::Unary | CodecSpec::Rice { .. } | CodecSpec::Golomb { .. }
                ) {
                    println!(
                        "\n- Skipping codec: {} for {} distribution",
                        spec_name, dist_name
                    );
                    continue;
                }
            }

            for &k_value in &K_VALUES {
                let intvec = LEIntVec::builder(&data)
                    .k(k_value)
                    .codec(codec_spec)
                    .build()
                    .expect("Failed to build IntVec");

                group.bench_function(format!("{}/k={}/get", spec_name, k_value), |b| {
                    b.iter(|| {
                        for &index in black_box(&access_indices) {
                            black_box(intvec.get(index));
                        }
                    })
                });
            }
        }

        // Benchmark k-independent codecs
        for (spec_name, codec_spec) in k_independent_codecs {
            // This codec is not dependent on k, so we build it once.
            let intvec = LEIntVec::builder(&data)
                .k(K_VALUES[0]) // This k value is ignored by the builder.
                .codec(codec_spec)
                .build()
                .expect("Failed to build IntVec");

            group.bench_function(format!("{}/get", spec_name), |b| {
                b.iter(|| {
                    for &index in black_box(&access_indices) {
                        black_box(intvec.get(index));
                    }
                })
            });
        }

        group.finish();
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(1))
        .measurement_time(Duration::from_secs(10));
    targets = benchmark_random_access
);

criterion_main!(benches);
