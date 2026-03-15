use compressed_intvec::prelude::*;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use dsi_bitstream::{
    codes::{len_rice, len_zeta},
    utils::sample_implied_distribution,
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::time::Duration;
use succinct::int_vec::{IntVec as SuccinctIntVec, IntVector as SuccinctIntVector};
use sux::prelude::BitFieldVec;
use value_traits::slices::SliceByValue;

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
                sample_implied_distribution(|v| len_zeta(v, 3), &mut rng)
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

fn benchmark_random_access(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 1_000_000;
    const NUM_ACCESSES: usize = 100_000;
    const K_VALUES: [usize; 8] = [2, 4, 8, 16, 32, 64, 128, 256];

    let distributions = [
        Distribution::UniformLow,
        Distribution::UniformHigh,
        Distribution::RiceImplied,
        Distribution::ZetaImplied,
    ];

    // Codecs for VarVec (k-dependent).
    let variable_codecs = [
        ("Gamma", Codec::Gamma),
        ("Delta", Codec::Delta),
        ("Unary", Codec::Unary),
        ("Rice", Codec::Rice { log2_b: None }),
        ("Zeta", Codec::Zeta { k: None }),
        ("Omega", Codec::Omega),
        ("VByteLe", Codec::VByteLe),
    ];

    // Prepare a vector of random indices for access tests.
    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();

    for distribution in distributions {
        let mut group = c.benchmark_group(format!("RandomAccess/{}", distribution.name()));
        group.throughput(Throughput::Elements(NUM_ACCESSES as u64));
        let data = distribution.generate(VECTOR_SIZE);

        // --- Baseline benchmark on the original Vec<u64> ---
        group.bench_function("Baseline/get_unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // Accessing the original, uncompressed vector.
                    black_box(unsafe { data.get_unchecked(index) });
                }
            })
        });

        // --- Benchmark FixedVec (k-independent) ---
        let fixed_vec = LEFixedVec::builder()
            .bit_width(BitWidth::PowerOfTwo)
            .build(&data)
            .expect("Failed to build FixedVec");

        group.bench_function("FixedVec/get_unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(unsafe { fixed_vec.get_unchecked(index) });
                }
            })
        });

        // --- Benchmark sux::BitFieldVec (k-independent) ---
        if !data.is_empty() {
            let sux_bfv = BitFieldVec::<u64>::from_slice(&data).unwrap();
            group.bench_function("sux::BitFieldVec/get_unchecked", |b| {
                b.iter(|| {
                    for &index in black_box(&access_indices) {
                        black_box(unsafe { sux_bfv.get_value_unchecked(index) });
                    }
                })
            });
        }

        // --- Benchmark succinct::IntVector (k-independent) ---
        if !data.is_empty() {
            let mut succinct_iv =
                SuccinctIntVector::<u64>::with_capacity(fixed_vec.bit_width(), VECTOR_SIZE as u64);
            for &val in &data {
                succinct_iv.push(val);
            }
            group.bench_function("succinct::IntVector/get", |b| {
                b.iter(|| {
                    for &index in black_box(&access_indices) {
                        // The `get` method in succinct performs bounds checking.
                        // We use UFCS here to avoid ambiguity with compressed_intvec's VarVec struct.
                        black_box(SuccinctIntVec::get(&succinct_iv, index as u64));
                    }
                })
            });
        }

        // --- Benchmark VarVec (k-dependent codecs) ---
        for (spec_name, codec_spec) in variable_codecs {
            if matches!(
                distribution,
                Distribution::UniformHigh | Distribution::ZetaImplied
            ) && matches!(codec_spec, Codec::Unary | Codec::Rice { .. })
            {
                continue;
            }

            for &k_value in &K_VALUES {
                let intvec = LEVarVec::builder()
                    .k(k_value)
                    .codec(codec_spec)
                    .build(&data)
                    .expect("Failed to build VarVec");

                let mut reader = intvec.reader();

                let bench_name = format!("{}/k={}/get_unchecked", spec_name, k_value);
                group.bench_function(bench_name, |b| {
                    b.iter(|| {
                        for &index in black_box(&access_indices) {
                            black_box(unsafe { reader.get_unchecked(index) });
                        }
                    })
                });
            }
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(3));
    targets = benchmark_random_access
}

criterion_main!(benches);
