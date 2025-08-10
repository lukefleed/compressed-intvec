// benches/variable/bench_sequential_ops.rs

//! # Benchmark for Sequential and Parallel Iteration Performance
//!
//! This benchmark suite is designed to measure the throughput of iterating over
//! an `IntVec`, both sequentially with `iter()` and in parallel with `par_iter()`.
//! It provides a comprehensive comparison of all sensible variable-length codecs
//! across a range of different data distributions.
//!
//! ## Methodology
//!
//! The benchmark evaluates performance along two key dimensions:
//!
//! 1.  **Data Distribution**: To understand how codecs perform under different
//!     conditions, several data distributions are tested:
//!     - `UniformLow`: Small, uniformly distributed integers.
//!     - `UniformHigh`: Large, uniformly distributed integers.
//!     - `RiceImplied`: Data with a geometric-like distribution, ideal for
//!       Rice and Golomb codes.
//!     - `ZetaImplied`: Data with a power-law distribution, ideal for Zeta codes.
//!
//! 2.  **Compression Codec**: A wide variety of codecs from `VariableCodecSpec` are
//!     tested to measure their raw decoding speed. This includes fundamental codes
//!     like Gamma and Delta, as well as specialized ones like Zeta and fast,
//!     byte-aligned codes like VByte.
//!
//! The results are compared against a baseline of iterating over an uncompressed
//! `Vec<u64>`, providing a clear measure of the decompression overhead for each
//! codec and scenario.

use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use dsi_bitstream::{
    codes::{len_rice, len_zeta_param},
    utils::sample_implied_distribution,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Defines the data distributions for testing.
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
            Distribution::UniformLow => {
                let mut rng = SmallRng::seed_from_u64(42);
                (0..size).map(|_| rng.random_range(0..1_000)).collect()
            }
            Distribution::UniformHigh => {
                let mut rng = SmallRng::seed_from_u64(42);
                (0..size).map(|_| rng.random_range(0..1 << 32)).collect()
            }
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

    fn name(&self) -> &'static str {
        match self {
            Distribution::UniformLow => "UniformLow",
            Distribution::UniformHigh => "UniformHigh",
            Distribution::RiceImplied => "RiceImplied",
            Distribution::ZetaImplied => "ZetaImplied",
        }
    }
}

/// The main benchmark function for sequential and parallel iteration.
fn benchmark_sequential_ops(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 1_000_000;
    const K_VALUE: usize = 32; // A typical k value

    let distributions = [
        Distribution::UniformLow,
        Distribution::UniformHigh,
        Distribution::RiceImplied,
        Distribution::ZetaImplied,
    ];

    let codecs_to_test = [
        ("Gamma", VariableCodecSpec::Gamma),
        ("Delta", VariableCodecSpec::Delta),
        ("Unary", VariableCodecSpec::Unary),
        ("Rice", VariableCodecSpec::Rice { log2_b: None }),
        ("Zeta", VariableCodecSpec::Zeta { k: None }),
        ("Omega", VariableCodecSpec::Omega),
        ("VByteLe", VariableCodecSpec::VByteLe),
        ("VByteBe", VariableCodecSpec::VByteBe),
        ("Pi", VariableCodecSpec::Pi { k: Some(3) }),
        ("Golomb", VariableCodecSpec::Golomb { b: Some(8) }),
        ("ExpGolomb", VariableCodecSpec::ExpGolomb { k: Some(2) }),
    ];

    for distribution in distributions {
        let mut group = c.benchmark_group(format!("SequentialOps/{}", distribution.name()));
        group.throughput(Throughput::Elements(VECTOR_SIZE as u64));
        let data = distribution.generate(VECTOR_SIZE);

        // --- Baseline benchmark on the original Vec<u64> ---
        group.bench_function("Baseline/iter_sum", |b| {
            b.iter(|| {
                // The sum operation ensures the compiler cannot optimize away the loop.
                black_box(black_box(&data).iter().sum::<u64>());
            })
        });

        #[cfg(feature = "parallel")]
        group.bench_function("Baseline/par_iter_sum", |b| {
            b.iter(|| {
                black_box(black_box(&data).par_iter().sum::<u64>());
            })
        });

        // --- Benchmarks for each IntVec codec ---
        for (spec_name, codec_spec) in codecs_to_test {
            // Skip combinations known to be extremely slow or impractical.
            if (matches!(
                distribution,
                Distribution::UniformHigh | Distribution::ZetaImplied
            ) && matches!(
                codec_spec,
                VariableCodecSpec::Unary
                    | VariableCodecSpec::Rice { .. }
                    | VariableCodecSpec::Golomb { .. }
            )) || (matches!(distribution, Distribution::RiceImplied)
                && matches!(codec_spec, VariableCodecSpec::Unary))
            {
                println!(
                    "Skipping codec {} for {} distribution (impractical).",
                    spec_name,
                    distribution.name()
                );
                continue;
            }

            let intvec = LEIntVec::builder(&data)
                .k(K_VALUE)
                .codec(codec_spec)
                .build()
                .expect("Failed to build IntVec");

            // 1. Sequential Iteration
            group.bench_function(format!("{}/iter_sum", spec_name), |b| {
                b.iter(|| {
                    black_box(black_box(&intvec).iter().sum::<u64>());
                })
            });

            // 2. Parallel Iteration
            #[cfg(feature = "parallel")]
            group.bench_function(format!("{}/par_iter_sum", spec_name), |b| {
                b.iter(|| {
                    black_box(black_box(&intvec).par_iter().sum::<u64>());
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
    targets = benchmark_sequential_ops
}
criterion_main!(benches);
