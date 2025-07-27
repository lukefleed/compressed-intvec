#![allow(clippy::all)]
//! # Benchmark for Measuring Memory Space of `IntVec`.
//!
//! This utility generates `IntVec` instances with various configurations to measure
//! their memory footprint. The configurations are aligned with `bench_random_access`.
//! It is intended to be run as a benchmark: `cargo bench --bench bench_size`.
//!
//! ## Output
//!
//! A `size_results.csv` file is generated in the `bench_results/` directory.

use compressed_intvec::{
    codec_spec::{resolve_codec, CodecSpec},
    intvec::LEIntVec,
};
use criterion::{criterion_group, criterion_main, Criterion};
use dsi_bitstream::{
    codes::{len_rice, len_zeta_param},
    utils::sample_implied_distribution,
};
use mem_dbg::{MemSize, SizeFlags};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{
    collections::HashSet,
    fmt::{Display, Formatter},
    fs::{self, File},
    io::Write,
    sync::Once,
};

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
    Geometric,
    PowerLaw,
}

impl Distribution {
    /// Generates a vector of data according to the distribution.
    fn generate(&self, size: usize) -> Vec<u64> {
        match self {
            Distribution::UniformLow => generate_random_vec(size, 1_000),
            Distribution::UniformHigh => generate_random_vec(size, 1 << 32),
            Distribution::Geometric => generate_with_distribution(size, |v| len_rice(v, 4)),
            Distribution::PowerLaw => {
                generate_with_distribution(size, |v| len_zeta_param::<false>(v, 3))
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
            Distribution::Geometric,
            Distribution::PowerLaw,
        ];
        let dsi_codecs_to_test = [
            ("Gamma", CodecSpec::Gamma),
            ("Delta", CodecSpec::Delta),
            ("Unary", CodecSpec::Unary),
            ("Rice_auto", CodecSpec::Rice { log2_b: None }),
            ("Zeta_auto", CodecSpec::Zeta { k: None }),
            ("Omega", CodecSpec::Omega),
            ("VByteLe", CodecSpec::VByteLe),
            ("VByteBe", CodecSpec::VByteBe),
            ("Pi", CodecSpec::Pi { k: Some(3) }),
            ("Golomb", CodecSpec::Golomb { b: Some(8) }),
            ("ExpGolomb", CodecSpec::ExpGolomb { k: Some(2) }),
        ];

        let mut all_results: Vec<BenchResult> = Vec::new();

        for &distribution in &distributions {
            let dist_name = format!("{:?}_{}", distribution, VECTOR_SIZE);
            println!("\n--- Processing Distribution: {} ---", dist_name);
            let data = distribution.generate(VECTOR_SIZE);
            let original_size_bytes = data.mem_size(SizeFlags::default());

            // Baseline: Vec<u64>
            all_results.push(BenchResult {
                name: "Vec<u64>".to_string(),
                k: 0,
                space_bytes: original_size_bytes,
                original_data_bytes: original_size_bytes,
                data_distribution: dist_name.clone(),
            });

            // Baseline: FixedLength
            let codec_spec = CodecSpec::FixedLength { num_bits: None };
            let resolved_encoding = resolve_codec(&data, codec_spec.clone()).unwrap();
            let name = format!("{:?}", resolved_encoding)
                .replace([' ', '{', '}'], "")
                .replace(':', "=");
            let intvec = LEIntVec::builder(&data).codec(codec_spec).build().unwrap();
            all_results.push(BenchResult {
                name,
                k: 1, // `k` is not used, but set to 1 for consistency.
                space_bytes: intvec.mem_size(SizeFlags::default()),
                original_data_bytes: original_size_bytes,
                data_distribution: dist_name.clone(),
            });

            // DSI Codecs
            for &(spec_name, ref codec_spec) in &dsi_codecs_to_test {
                if (matches!(
                    distribution,
                    Distribution::UniformHigh | Distribution::PowerLaw
                ) && matches!(
                    codec_spec,
                    CodecSpec::Unary | CodecSpec::Rice { .. } | CodecSpec::Golomb { .. }
                )) || (matches!(distribution, Distribution::UniformLow)
                    && matches!(codec_spec, CodecSpec::Unary))
                {
                    println!("- Skipping {} for distribution {}", spec_name, dist_name);
                    continue;
                }

                for &k in &k_values {
                    // Resolve the codec to get its actual name and parameters.
                    let resolved = resolve_codec(&data, codec_spec.clone()).unwrap();
                    let name = format!("{:?}", resolved)
                        .replace([' ', '{', '}'], "")
                        .replace(':', "=");

                    // Build the IntVec with the original spec.
                    let intvec = LEIntVec::builder(&data)
                        .k(k)
                        .codec(codec_spec.clone())
                        .build()
                        .unwrap();
                    println!("  - Measured {} (k={})", name, k);

                    // Store the result using the resolved name.
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

        // Use a HashSet to ensure that we only write unique rows to the CSV.
        // This handles cases where different CodecSpecs resolve to the same underlying code.
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
// This setup ensures that the space measurement logic is run as part of `cargo bench`,
// making it consistent with the other benchmarks.
fn criterion_benchmark_runner(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpaceMeasurementSuite");
    // We only need one iteration to generate the file.
    group.bench_function("GenerateSpaceCSV", |b| b.iter(run_space_measurements));
    group.finish();
}

criterion_group! {
    name = benches;
    // We only need a small sample size because the core logic is inside a `Once` block.
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark_runner
}
criterion_main!(benches);
