use compressed_intvec::codecs::{
    DeltaCodec, ExpGolombCodec, GammaCodec, ParamDeltaCodec, ParamGammaCodec, RiceCodec,
};
use compressed_intvec::intvec::{BEIntVec, LEIntVec};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::distr::{Distribution, Uniform};
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use std::u64;

/// Generate a vector of random u64 values in the range [0, max) using a uniform distribution.
fn generate_uniform_vec(size: usize, max: u64) -> Vec<u64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let uniform = Uniform::new(0, max).unwrap();
    (0..size).map(|_| uniform.sample(&mut rng)).collect()
}

/// Generate a list of random indexes within the range [0, max).
fn generate_random_indexes(n: usize, max: usize) -> Vec<usize> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.random_range(0..max)).collect()
}

/// Benchmark random access per varianti, raccogliendo il tempo di esecuzione in results.
fn benchmark_random_access<T, C: Copy>(
    results: &mut Vec<(String, usize, f64)>, // (nome benchmark, valore di k, tempo in sec)
    c: &mut Criterion,
    name: &str,
    input: Vec<u64>,
    k: usize,
    param: C,
    build_vec: impl Fn(Vec<u64>, usize, C) -> Result<T, Box<dyn std::error::Error>>,
    get: impl Fn(&T, usize) -> Option<u64>,
) {
    // Usato anche per Criterion
    c.bench_function(name, |b| {
        b.iter(|| {
            let vec = build_vec(input.clone(), k, param).unwrap();
            let indexes = generate_random_indexes(input.len(), input.len());
            for &i in &indexes {
                black_box(get(&vec, i).unwrap());
            }
        });
    });

    // Misurazione extra per registrare il tempo in un CSV
    let vec = build_vec(input.clone(), k, param).unwrap();
    let indexes = generate_random_indexes(input.len(), input.len());
    let start = Instant::now();
    for &i in &indexes {
        black_box(get(&vec, i).unwrap());
    }
    let elapsed = start.elapsed().as_secs_f64();
    results.push((name.to_string(), k, elapsed));
}

/// Benchmarks per LE e BE, salvando i risultati in un CSV al termine.
fn bench_all(c: &mut Criterion) {
    let input_size = 10_000;
    let max_value = u64::MAX;
    let ks = vec![4, 8, 16, 32, 64, 128]; // diversi valori di k

    // Vettore per raccogliere risultati per CSV.
    let mut results: Vec<(String, usize, f64)> = Vec::new();
    let uniform = generate_uniform_vec(input_size, max_value);

    // Benchmark su Vec<u64> (base di riferimento)
    let indexes = generate_random_indexes(input_size, input_size);
    c.bench_function("Vec<u64> random access (uniform)", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(uniform[i]);
            }
        });
    });

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
                LEIntVec::<GammaCodec>::from_with_param(data, k, ())
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
                LEIntVec::<DeltaCodec>::from_with_param(data, k, ())
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
                LEIntVec::<ExpGolombCodec>::from_with_param(data, k, param)
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
                LEIntVec::<RiceCodec>::from_with_param(data, k, param)
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
                LEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(data, k, ())
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
                LEIntVec::<ParamGammaCodec<true>>::from_with_param(data, k, ())
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
                BEIntVec::<GammaCodec>::from_with_param(data, k, ())
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
                BEIntVec::<DeltaCodec>::from_with_param(data, k, ())
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
                BEIntVec::<ExpGolombCodec>::from_with_param(data, k, param)
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
                BEIntVec::<RiceCodec>::from_with_param(data, k, param)
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
                BEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(data, k, ())
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
                BEIntVec::<ParamGammaCodec<true>>::from_with_param(data, k, ())
            },
            BEIntVec::<ParamGammaCodec<true>>::get,
        );
    }

    // Scrittura dei risultati in CSV
    let mut file =
        File::create("benchmark_random_access.csv").expect("Impossibile creare il file CSV");
    writeln!(file, "name,k,elapsed").expect("Errore di scrittura header CSV");
    for (name, k, elapsed) in results {
        writeln!(file, "{},{},{}", name, k, elapsed).expect("Errore di scrittura nel CSV");
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
