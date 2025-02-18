use compressed_intvec::codecs::{
    DeltaCodec, ExpGolombCodec, GammaCodec, MinimalBinaryCodec, ParamDeltaCodec, ParamGammaCodec,
    RiceCodec,
};
use compressed_intvec::intvec::{BEIntVec, LEIntVec};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use std::u64;

/// Generates a vector of `size` u64 values sampled from the distribution `dist`
/// (which produces values of type `T`), converting them to u64 using the provided `convert` function.
fn generate_vec_with_distribution<D, T>(
    size: usize,
    dist: D,
    convert: impl Fn(T) -> u64,
) -> Vec<u64>
where
    D: Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(42);
    (0..size).map(|_| convert(dist.sample(&mut rng))).collect()
}

/// Generates a list of random indices in the interval [0, max).
fn generate_random_indexes(n: usize, max: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.random_range(0..max)).collect()
}

/// Benchmarks random access by both using Criterion and by direct timing.
///
/// - `results`: vector to store (benchmark name, parameter k, elapsed time in seconds).
/// - `c`: the Criterion benchmark context.
/// - `name`: benchmark name.
/// - `input`: the input vector on which the benchmark is executed.
/// - `k`: a parameter used for building the compressed data structure.
/// - `param`: a specific parameter for the codec.
/// - `build_vec`: a closure to build the compressed structure from the input data.
/// - `get`: a function to perform random access on the compressed structure.
fn benchmark_random_access<T, C: Copy>(
    results: &mut Vec<(String, usize, f64)>,
    c: &mut Criterion,
    name: &str,
    input: Vec<u64>,
    k: usize,
    param: C,
    build_vec: impl Fn(Vec<u64>, usize, C) -> T,
    get: impl Fn(&T, usize) -> Option<u64>,
) {
    // Run Criterion benchmark
    c.bench_function(name, |b| {
        b.iter(|| {
            let vec = build_vec(input.clone(), k, param);
            let indexes = generate_random_indexes(input.len(), input.len());
            for &i in &indexes {
                black_box(get(&vec, i).unwrap());
            }
        });
    });

    // Measure execution time directly to record results in a CSV file
    let vec = build_vec(input.clone(), k, param);
    let indexes = generate_random_indexes(input.len(), input.len());
    let start = Instant::now();
    for &i in &indexes {
        black_box(get(&vec, i).unwrap());
    }
    let elapsed = start.elapsed().as_secs_f64();
    results.push((name.to_string(), k, elapsed));
}

/// Main entry point for running all benchmarks.
fn bench_all(c: &mut Criterion) {
    let input_size = 10_000;
    let max_value = u64::MAX;
    let ks = vec![4, 8, 16, 32, 64, 128];

    // Vector to store benchmark results
    let mut results: Vec<(String, usize, f64)> = Vec::new();

    // Example 1: Using a distribution that produces u64 values (Uniform)
    let dist_vec = Uniform::new(0, max_value).unwrap();
    let uniform = generate_vec_with_distribution(input_size, dist_vec, |x| x);

    // Reference benchmark on Vec<u64> random access (uniform)
    let indexes = generate_random_indexes(input_size, input_size);
    c.bench_function("Vec<u64> random access (uniform)", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(uniform[i]);
            }
        });
    });

    // Run benchmarks for different data structures and codecs
    for &k in &ks {
        // LEIntVec benchmarks
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec GammaCodec"),
            uniform.clone(),
            k,
            &GammaCodec,
            |data: Vec<u64>, k: usize, _codec: &GammaCodec| {
                LEIntVec::<GammaCodec>::from_with_param(&data, k, ())
            },
            LEIntVec::<_>::get,
        );
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec DeltaCodec"),
            uniform.clone(),
            k,
            &DeltaCodec,
            |data: Vec<u64>, k: usize, _codec: &DeltaCodec| {
                LEIntVec::<DeltaCodec>::from_with_param(&data, k, ())
            },
            LEIntVec::<_>::get,
        );
        let exp_k = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec ExpGolombCodec"),
            uniform.clone(),
            k,
            exp_k,
            |data: Vec<u64>, k: usize, param: usize| {
                LEIntVec::<ExpGolombCodec>::from_with_param(&data, k, param)
            },
            LEIntVec::<_>::get,
        );
        let rice_k = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec RiceCodec"),
            uniform.clone(),
            k,
            rice_k,
            |data: Vec<u64>, k: usize, param: usize| {
                LEIntVec::<RiceCodec>::from_with_param(&data, k, param)
            },
            LEIntVec::<_>::get,
        );

        let min_param = *uniform.iter().max().unwrap();
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec MinimalBinaryCodec"),
            uniform.clone(),
            k,
            min_param,
            |data: Vec<u64>, k: usize, param: u64| {
                LEIntVec::<MinimalBinaryCodec>::from_with_param(&data, k, param)
            },
            LEIntVec::<_>::get,
        );

        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec ParamDeltaCodec"),
            uniform.clone(),
            k,
            &ParamDeltaCodec::<true, true>,
            |data: Vec<u64>, k: usize, _codec: &ParamDeltaCodec<true, true>| {
                LEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(&data, k, ())
            },
            LEIntVec::<_>::get,
        );
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec ParamGammaCodec"),
            uniform.clone(),
            k,
            &ParamGammaCodec::<true>,
            |data: Vec<u64>, k: usize, _codec: &ParamGammaCodec<true>| {
                LEIntVec::<ParamGammaCodec<true>>::from_with_param(&data, k, ())
            },
            LEIntVec::<_>::get,
        );

        // BEIntVec benchmarks
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec GammaCodec"),
            uniform.clone(),
            k,
            &GammaCodec,
            |data: Vec<u64>, k: usize, _codec: &GammaCodec| {
                BEIntVec::<GammaCodec>::from_with_param(&data, k, ())
            },
            BEIntVec::<_>::get,
        );
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec DeltaCodec"),
            uniform.clone(),
            k,
            &DeltaCodec,
            |data: Vec<u64>, k: usize, _codec: &DeltaCodec| {
                BEIntVec::<DeltaCodec>::from_with_param(&data, k, ())
            },
            BEIntVec::<_>::get,
        );
        let exp_k = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec ExpGolombCodec"),
            uniform.clone(),
            k,
            exp_k,
            |data: Vec<u64>, k: usize, param: usize| {
                BEIntVec::<ExpGolombCodec>::from_with_param(&data, k, param)
            },
            BEIntVec::<_>::get,
        );
        let rice_k = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec RiceCodec"),
            uniform.clone(),
            k,
            rice_k,
            |data: Vec<u64>, k: usize, param: usize| {
                BEIntVec::<RiceCodec>::from_with_param(&data, k, param)
            },
            BEIntVec::<_>::get,
        );
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec ParamDeltaCodec"),
            uniform.clone(),
            k,
            &ParamDeltaCodec::<true, true>,
            |data: Vec<u64>, k: usize, _codec: &ParamDeltaCodec<true, true>| {
                BEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(&data, k, ())
            },
            BEIntVec::<ParamDeltaCodec<true, true>>::get,
        );
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec ParamGammaCodec"),
            uniform.clone(),
            k,
            &ParamGammaCodec::<true>,
            |data: Vec<u64>, k: usize, _codec: &ParamGammaCodec<true>| {
                BEIntVec::<ParamGammaCodec<true>>::from_with_param(&data, k, ())
            },
            BEIntVec::<ParamGammaCodec<true>>::get,
        );
    }

    // Write the benchmark results into a CSV file
    let mut file =
        File::create("bench_results/bench_random_access.csv").expect("Unable to create CSV file");
    writeln!(file, "name,k,elapsed").expect("Error writing CSV header");
    for (name, k, elapsed) in results {
        writeln!(file, "{},{},{}", name, k, elapsed).expect("Error writing CSV data");
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
