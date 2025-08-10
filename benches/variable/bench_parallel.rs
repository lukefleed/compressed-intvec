use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use dsi_bitstream::{
    codes::{len_rice, len_zeta_param},
    utils::sample_implied_distribution,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;

#[cfg(feature = "parallel")]
use compressed_intvec::prelude::*;

/// Enum to define the data distributions for testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Distribution {
    UniformLow,
    UniformHigh,
    RiceImplied,
    ZetaImplied,
}

impl Distribution {
    /// Generates a vector of data according to the distribution.
    fn generate(&self, size: usize) -> Vec<u64> {
        match self {
            Distribution::UniformLow => generate_random_vec(size, 1_000),
            Distribution::UniformHigh => generate_random_vec(size, 1 << 32),
            Distribution::RiceImplied => {
                let mut rng = SmallRng::seed_from_u64(42);
                sample_implied_distribution(|v| len_rice(v, 4), &mut rng)
                    .take(size)
                    .collect()
            }
            Distribution::ZetaImplied => {
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

fn benchmark_access(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 1_000_000;
    const NUM_ACCESSES: usize = 100_000;
    const K_VALUE: usize = 32; // Fixed k for this benchmark suite

    let distributions = [
        (Distribution::UniformLow, "UniformLow"),
        (Distribution::UniformHigh, "UniformHigh"),
        (Distribution::RiceImplied, "RiceImplied"),
        (Distribution::ZetaImplied, "ZetaImplied"),
    ];

    let codecs_to_test = [
        ("Gamma", VariableCodecSpec::Gamma),
        ("Delta", VariableCodecSpec::Delta),
        ("Unary", VariableCodecSpec::Unary),
        ("Rice", VariableCodecSpec::Rice { log2_b: None }),
        ("Zeta", VariableCodecSpec::Zeta { k: None }),
        ("Explicit_Omega", VariableCodecSpec::Omega),
        ("Explicit_VByteLe", VariableCodecSpec::VByteLe),
        ("Explicit_VByteBe", VariableCodecSpec::VByteBe),
        ("Explicit_Pi", VariableCodecSpec::Pi { k: Some(3) }),
        ("Explicit_Golomb", VariableCodecSpec::Golomb { b: Some(8) }),
        (
            "Explicit_ExpGolomb",
            VariableCodecSpec::ExpGolomb { k: Some(2) },
        ),
    ];

    // Prepare a vector of random indices for access tests.
    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();

    for (distribution, dist_name) in distributions {
        let mut group = c.benchmark_group(dist_name);
        let data = distribution.generate(VECTOR_SIZE);

        for (spec_name, codec_spec) in codecs_to_test {
            // Skip combinations known to cause issues.
            if matches!(
                distribution,
                Distribution::UniformHigh | Distribution::ZetaImplied
            ) && matches!(
                codec_spec,
                VariableCodecSpec::Unary
                    | VariableCodecSpec::Rice { .. }
                    | VariableCodecSpec::Golomb { .. }
            ) {
                println!(
                    "\n- Skipping codec: {} for {} distribution",
                    spec_name, dist_name
                );
                continue;
            }

            let intvec = LEIntVec::builder(&data)
                .k(K_VALUE)
                .codec(codec_spec)
                .build()
                .expect("Failed to build IntVec");

            // 1. Benchmark 'get_unchecked' in a loop.
            group.bench_function(format!("{}/get_unchecked_loop", spec_name), |b| {
                b.iter(|| {
                    for &index in black_box(&access_indices) {
                        // SAFETY: Indices are generated within bounds for the benchmark.
                        black_box(unsafe { intvec.get_unchecked(index) });
                    }
                })
            });

            // 2. Benchmark 'get_many_unchecked' (sequential batch).
            group.bench_function(format!("{}/get_many_unchecked", spec_name), |b| {
                b.iter(|| {
                    // SAFETY: Indices are generated within bounds for the benchmark.
                    let _ =
                        black_box(unsafe { intvec.get_many_unchecked(black_box(&access_indices)) });
                })
            });

            // 3. Benchmark 'par_get_many_unchecked' (parallel batch).
            group.bench_function(format!("{}/par_get_many_unchecked", spec_name), |b| {
                b.iter(|| {
                    // SAFETY: Indices are generated within bounds for the benchmark.
                    let _ = black_box(unsafe {
                        intvec.par_get_many_unchecked(black_box(&access_indices))
                    });
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
        .warm_up_time(Duration::from_millis(10))
        .measurement_time(Duration::from_secs(2));
    targets = benchmark_access
);

criterion_main!(benches);
