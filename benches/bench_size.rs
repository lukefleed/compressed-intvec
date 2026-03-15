//! # Benchmark for Measuring Memory Space.
//!
//! This utility generates `VarVec` and `FixedVec` instances with various
//! configurations to measure their memory footprint. It is intended to be run
//! as a benchmark: `cargo bench --bench bench_size`.
//!
//! ## Output
//!
//! A `size_results.csv` file is generated in the `bench_results/` directory.

use compressed_intvec::prelude::*;
use criterion::{Criterion, criterion_group, criterion_main};
use dsi_bitstream::{
    codes::{len_rice, len_zeta},
    utils::sample_implied_distribution,
};
use mem_dbg::{DbgFlags, MemDbg, MemSize, SizeFlags};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::{
    collections::HashSet,
    fmt::{Display, Formatter},
    fs::{self, File},
    io::Write,
    sync::Once,
};
use sux::prelude::BitFieldVec;

// --- Data Generation Utilities ---

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

/// Generates a vector with a specific distribution based on a code's length function.
fn generate_with_distribution(size: usize, len_fn: impl Fn(u64) -> usize) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    sample_implied_distribution(len_fn, &mut rng)
        .take(size)
        .collect()
}

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
            Distribution::RiceImplied => generate_with_distribution(size, |v| len_rice(v, 4)),
            Distribution::ZetaImplied => {
                generate_with_distribution(size, |v| len_zeta(v, 3))
            }
        }
    }
}

/// Holds the results for a single space benchmark configuration.
#[derive(Debug)]
struct BenchResult {
    name: String,
    k: usize,
    space_bytes: usize,
    original_data_bytes: usize,
    data_distribution: String,
}

impl Display for BenchResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\"{}\",{},{},{},\"{}\"",
            self.name, self.k, self.space_bytes, self.original_data_bytes, self.data_distribution
        )
    }
}

// A static Once to ensure the measurement logic runs only one time.
static BENCH_ONCE: Once = Once::new();

/// Runs the complete space measurement suite.
fn run_space_measurements() {
    BENCH_ONCE.call_once(|| {
        const VECTOR_SIZE: usize = 1_000_000;
        let k_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        let distributions = [
            Distribution::UniformLow,
            Distribution::UniformHigh,
            Distribution::RiceImplied,
            Distribution::ZetaImplied,
        ];
        let dsi_codecs_to_test = [
            ("Gamma", Codec::Gamma),
            ("Delta", Codec::Delta),
            ("Unary", Codec::Unary),
            ("Rice_auto", Codec::Rice { log2_b: None }),
            ("Zeta_auto", Codec::Zeta { k: None }),
            ("Omega", Codec::Omega),
            ("VByteLe", Codec::VByteLe),
            ("VByteBe", Codec::VByteBe),
            ("Pi", Codec::Pi { k: None }),
            ("ExpGolomb", Codec::ExpGolomb { k: None }),
        ];

        let mut all_results: Vec<BenchResult> = Vec::new();

        for &distribution in &distributions {
            let dist_name = format!("{:?}_{}", distribution, VECTOR_SIZE);
            println!("\n--- Processing Distribution: {} ---", dist_name);
            let data = distribution.generate(VECTOR_SIZE);
            let original_size_bytes = data.mem_size(SizeFlags::default());
            let _ = data.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::COLOR | DbgFlags::PERCENTAGE);

            // Baseline: Vec<u64>
            all_results.push(BenchResult {
                name: "Vec<u64>".to_string(),
                k: 0,
                space_bytes: original_size_bytes,
                original_data_bytes: original_size_bytes,
                data_distribution: dist_name.clone(),
            });

            // Baseline: UFixedVec (auto-bits) using TryFrom<&[T]>
            let fixed_vec = UFixedVec::<u64>::try_from(data.as_slice()).unwrap();
            let _ = fixed_vec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::COLOR | DbgFlags::PERCENTAGE);
            all_results.push(BenchResult {
                name: format!("LEFixedVec(bits={})", fixed_vec.bit_width()),
                k: 0, // k is not applicable
                space_bytes: fixed_vec.mem_size(SizeFlags::default()),
                original_data_bytes: original_size_bytes,
                data_distribution: dist_name.clone(),
            });

            // Baseline: sux::BitFieldVec
            if !data.is_empty() {
                let sux_bfv = BitFieldVec::<u64>::from_slice(&data).unwrap();
                let _ =
                    sux_bfv.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::COLOR | DbgFlags::PERCENTAGE);
                all_results.push(BenchResult {
                    name: format!("sux::BitFieldVec(bits={})", sux_bfv.bit_width()),
                    k: 0,
                    space_bytes: sux_bfv.mem_size(SizeFlags::default()),
                    original_data_bytes: original_size_bytes,
                    data_distribution: dist_name.clone(),
                });
            }

            // DSI Codecs (VarVec)
            for &(spec_name, ref codec_spec) in &dsi_codecs_to_test {
                if (matches!(
                    distribution,
                    Distribution::UniformHigh | Distribution::ZetaImplied
                ) && matches!(codec_spec, Codec::Unary | Codec::Rice { .. }))
                    || (matches!(distribution, Distribution::UniformLow)
                        && matches!(codec_spec, Codec::Unary))
                {
                    println!("- Skipping {} for distribution {}", spec_name, dist_name);
                    continue;
                }

                for &k in &k_values {
                    let intvec = LEVarVec::builder()
                        .k(k)
                        .codec(*codec_spec)
                        .build(&data)
                        .unwrap();

                    let name = format!("{:?}", intvec.encoding())
                        .replace([' ', '{', '}'], "")
                        .replace(':', "=");

                    println!("  - Measured {} (k={})", name, k);
                    let _ =
                        intvec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::COLOR | DbgFlags::PERCENTAGE);

                    all_results.push(BenchResult {
                        name,
                        k,
                        space_bytes: intvec.mem_size(SizeFlags::default()),
                        original_data_bytes: original_size_bytes,
                        data_distribution: dist_name.clone(),
                    });
                }
            }
        }

        // --- Write Results to CSV File ---
        let output_dir = "bench_results";
        fs::create_dir_all(output_dir).expect("Could not create benchmark results directory.");
        let output_path = format!("{}/size_results.csv", output_dir);
        let mut file = File::create(output_path).expect("Could not create results CSV file.");
        writeln!(file, "name,k,space_bytes,original_bytes,distribution")
            .expect("Could not write CSV header.");

        let mut unique_keys = HashSet::new();
        for result in all_results {
            let key = (
                result.name.clone(),
                result.k,
                result.data_distribution.clone(),
            );
            if unique_keys.insert(key) {
                writeln!(file, "{}", result).expect("Could not write result row to CSV.");
            }
        }
        println!("\nSpace measurement results written to bench_results/size_results.csv");
    });
}

// --- Criterion Runner Setup ---
fn criterion_benchmark_runner(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpaceMeasurementSuite");
    group.bench_function("GenerateSpaceCSV", |b| b.iter(run_space_measurements));
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark_runner
}
criterion_main!(benches);
