use mem_dbg::{MemSize, SizeFlags};
use rand::{
    distr::{Distribution, Uniform},
    SeedableRng,
};
use std::io::Write;
use std::{fs::File, time::Duration};

// Assume these are re-exported from the relevant module.
use compressed_intvec::{
    codecs::{
        DeltaCodec, ExpGolombCodec, GammaCodec, MinimalBinaryCodec, ParamDeltaCodec,
        ParamGammaCodec, RiceCodec,
    },
    intvec::{BEIntVec, LEIntVec},
};
use criterion::{criterion_group, criterion_main, Criterion};

/// Generate a vector of random u64 values in the range [0, max) using a uniform distribution.
fn generate_uniform_vec(size: usize, max: u64) -> Vec<u64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let uniform = Uniform::new(0, max).unwrap();
    (0..size).map(|_| uniform.sample(&mut rng)).collect()
}

/// Benchmark the space occupancy (in bytes) of a compressed intvec, storing the result in the CSV results vector.
fn benchmark_space<T, C: Copy>(
    results: &mut Vec<(String, usize, usize)>, // (benchmark name, k, space in bytes)
    c: &mut Criterion,
    input: Vec<u64>,
    k: usize,
    codec_param: C,
    build_vec: impl Fn(Vec<u64>, usize, C) -> T,
    size: impl Fn(&T) -> usize,
    benchmark_name: &str,
) {
    let bench_label = format!("{}", benchmark_name);

    let intvec = build_vec(input.clone(), k, codec_param);
    let space = size(&intvec);
    results.push((bench_label.clone(), k, space));

    c.bench_function(&bench_label, |b| {
        b.iter(|| {
            std::hint::black_box(&intvec);
        });
    });
}

fn bench_all(c: &mut Criterion) {
    let input_size = 10_000;
    let max_value = u64::MAX;
    let uniform = generate_uniform_vec(input_size, max_value);
    let ks = vec![4, 8, 16, 32, 64, 128];

    let mut results: Vec<(String, usize, usize)> = Vec::new();

    // Benchmark LEIntVec with GammaCodec (no extra codec parameter).
    for &k in &ks {
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            (), // GammaCodec uses no extra runtime parameter
            |data, k, param| LEIntVec::<GammaCodec>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec GammaCodec",
        );
    }

    // Benchmark LEIntVec with DeltaCodec (no extra codec parameter).
    for &k in &ks {
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            (), // DeltaCodec uses no extra runtime parameter
            |data, k, param| LEIntVec::<DeltaCodec>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec DeltaCodec",
        );
    }

    // Benchmark LEIntVec with ExpGolombCodec (using a runtime parameter calculated from the input).
    for &k in &ks {
        let exp_param = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            exp_param, // Use the calculated exp_param for ExpGolombCodec
            |data, k, param| LEIntVec::<ExpGolombCodec>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec ExpGolombCodec",
        );
    }

    // Benchmark LEIntVec with RiceCodec (using a runtime parameter calculated from the input).
    for &k in &ks {
        let rice_param = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            rice_param,
            |data, k, param| LEIntVec::<RiceCodec>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec RiceCodec",
        );
    }

    // Benchmark LEIntVec with MinimalBinaryCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            16,
            |data, k, param| LEIntVec::<MinimalBinaryCodec>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec MinimalBinaryCodec",
        );
    }

    // Benchmark LEIntVec with ParamDeltaCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            (), // ParamDeltaCodec has no extra runtime parameter
            |data, k, param| {
                LEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(&data, k, param)
            },
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec ParamDeltaCodec",
        );
    }

    // Benchmark LEIntVec with ParamGammaCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            (), // ParamGammaCodec has no extra runtime parameter
            |data, k, param| LEIntVec::<ParamGammaCodec<true>>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec ParamGammaCodec",
        );
    }

    // Benchmark BEIntVec with GammaCodec (no extra codec parameter).
    for &k in &ks {
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            (), // GammaCodec uses no extra runtime parameter
            |data, k, param| BEIntVec::<GammaCodec>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec GammaCodec",
        );
    }

    // Benchmark BEIntVec with DeltaCodec (no extra codec parameter).
    for &k in &ks {
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            (), // DeltaCodec uses no extra runtime parameter
            |data, k, param| BEIntVec::<DeltaCodec>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec DeltaCodec",
        );
    }

    // Benchmark BEIntVec with ExpGolombCodec (using a runtime parameter calculated from the input).
    for &k in &ks {
        let exp_param = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            exp_param, // Use the calculated exp_param for ExpGolombCodec
            |data, k, param| BEIntVec::<ExpGolombCodec>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec ExpGolombCodec",
        );
    }

    // Benchmark BEIntVec with RiceCodec (using a runtime parameter calculated from the input).
    for &k in &ks {
        let rice_param = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            rice_param,
            |data, k, param| BEIntVec::<RiceCodec>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec RiceCodec",
        );
    }

    // Benchmark LEIntVec with MinimalBinaryCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            16,
            |data, k, param| BEIntVec::<MinimalBinaryCodec>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec MinimalBinaryCodec",
        );
    }

    // Benchmark BEIntVec with ParamDeltaCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            (), // ParamDeltaCodec has no extra runtime parameter
            |data, k, param| {
                BEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(&data, k, param)
            },
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec ParamDeltaCodec",
        );
    }

    // Benchmark BEIntVec with ParamGammaCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            c,
            uniform.clone(),
            k,
            (), // ParamGammaCodec has no extra runtime parameter
            |data, k, param| BEIntVec::<ParamGammaCodec<true>>::from_with_param(&data, k, param),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec ParamGammaCodec",
        );
    }

    // Write the accumulated benchmark results to a CSV file.
    let mut file = File::create("bench_results/bench_space.csv").expect("Cannot create CSV file");
    writeln!(file, "name,k,space").expect("Error writing CSV header");
    // the second line is just name,0,space and it's the space of the uniform vector
    writeln!(
        file,
        "Standard Vec,0,{}",
        uniform.mem_size(SizeFlags::default())
    )
    .expect("Error writing CSV line");
    for (name, k, space) in results {
        writeln!(file, "{},{},{}", name, k, space).expect("Error writing CSV line");
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::from_secs(1));
    targets = bench_all
}
criterion_main!(benches);
