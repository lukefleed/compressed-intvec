use mem_dbg::{MemSize, SizeFlags};
use rand::{
    distr::{Distribution, Uniform},
    SeedableRng,
};
use std::{fs::File, time::Duration};
use std::{io::Write, u64};

use compressed_intvec::{
    codecs::{
        DeltaCodec, ExpGolombCodec, GammaCodec, MinimalBinaryCodec, ParamDeltaCodec,
        ParamGammaCodec, RiceCodec,
    },
    intvec::{BEIntVec, LEIntVec},
};
use criterion::{criterion_group, criterion_main, Criterion};
use std::fs;

fn generate_uniform_vec(size: usize, max: u64) -> Vec<u64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let uniform = Uniform::new(0, max).unwrap();
    (0..size).map(|_| uniform.sample(&mut rng)).collect()
}

fn benchmark_space<T, C: Copy>(
    results: &mut Vec<(String, usize, usize)>,
    input: &[u64],
    k: usize,
    codec_param: C,
    build_vec: impl Fn(&[u64], usize, C) -> T,
    size: impl Fn(&T) -> usize,
    benchmark_name: &str,
) {
    let bench_label = format!("{}", benchmark_name);
    let intvec = build_vec(input, k, codec_param);
    let space = size(&intvec);
    results.push((bench_label, k, space));
}

fn bench_all(c: &mut Criterion) {
    let input_size = 10_000;
    let max_value = 10_000;
    let uniform = generate_uniform_vec(input_size, max_value);
    let ks = vec![4, 8, 16, 32, 64, 128];
    let mut results = Vec::new();

    // Benchmark LEIntVec codecs
    for &k in &ks {
        benchmark_space(
            &mut results,
            &uniform,
            k,
            (),
            |data, k, param| LEIntVec::<GammaCodec>::from_with_param(data, k, param).unwrap(),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec GammaCodec",
        );
    }

    // Benchmark LEIntVec with DeltaCodec (no extra codec parameter).
    for &k in &ks {
        benchmark_space(
            &mut results,
            &uniform,
            k,
            (),
            |data, k, param| LEIntVec::<DeltaCodec>::from_with_param(data, k, param).unwrap(),
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
            &uniform,
            k,
            exp_param, // Use the calculated exp_param for ExpGolombCodec
            |data, k, param| LEIntVec::<ExpGolombCodec>::from_with_param(data, k, param).unwrap(),
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
            &uniform,
            k,
            rice_param,
            |data, k, param| LEIntVec::<RiceCodec>::from_with_param(data, k, param).unwrap(),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec RiceCodec",
        );
    }

    // Benchmark LEIntVec with MinimalBinaryCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            &uniform,
            k,
            16,
            |data, k, param| {
                LEIntVec::<MinimalBinaryCodec>::from_with_param(data, k, param).unwrap()
            },
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec MinimalBinaryCodec",
        );
    }

    // Benchmark LEIntVec with ParamDeltaCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            &uniform,
            k,
            (), // ParamDeltaCodec has no extra runtime parameter
            |data, k, param| {
                LEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(data, k, param).unwrap()
            },
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec ParamDeltaCodec",
        );
    }

    // Benchmark LEIntVec with ParamGammaCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            &uniform,
            k,
            (), // ParamGammaCodec has no extra runtime parameter
            |data, k, param| {
                LEIntVec::<ParamGammaCodec<true>>::from_with_param(data, k, param).unwrap()
            },
            |intvec| intvec.mem_size(SizeFlags::default()),
            "LEIntVec ParamGammaCodec",
        );
    }

    // Benchmark BEIntVec with GammaCodec (no extra codec parameter).
    for &k in &ks {
        benchmark_space(
            &mut results,
            &uniform,
            k,
            (), // GammaCodec uses no extra runtime parameter
            |data, k, param| BEIntVec::<GammaCodec>::from_with_param(data, k, param).unwrap(),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec GammaCodec",
        );
    }

    // Benchmark BEIntVec with DeltaCodec (no extra codec parameter).
    for &k in &ks {
        benchmark_space(
            &mut results,
            &uniform,
            k,
            (), // DeltaCodec uses no extra runtime parameter
            |data, k, param| BEIntVec::<DeltaCodec>::from_with_param(data, k, param).unwrap(),
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
            &uniform,
            k,
            exp_param, // Use the calculated exp_param for ExpGolombCodec
            |data, k, param| BEIntVec::<ExpGolombCodec>::from_with_param(data, k, param).unwrap(),
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
            &uniform,
            k,
            rice_param,
            |data, k, param| BEIntVec::<RiceCodec>::from_with_param(data, k, param).unwrap(),
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec RiceCodec",
        );
    }

    // Benchmark BEIntVec with MinimalBinaryCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            &uniform,
            k,
            16,
            |data, k, param| {
                BEIntVec::<MinimalBinaryCodec>::from_with_param(data, k, param).unwrap()
            },
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec MinimalBinaryCodec",
        );
    }

    // Benchmark BEIntVec with ParamDeltaCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            &uniform,
            k,
            (), // ParamDeltaCodec has no extra runtime parameter
            |data, k, param| {
                BEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(data, k, param).unwrap()
            },
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec ParamDeltaCodec",
        );
    }

    // Benchmark BEIntVec with ParamGammaCodec.
    for &k in &ks {
        benchmark_space(
            &mut results,
            &uniform,
            k,
            (), // ParamGammaCodec has no extra runtime parameter
            |data, k, param| {
                BEIntVec::<ParamGammaCodec<true>>::from_with_param(data, k, param).unwrap()
            },
            |intvec| intvec.mem_size(SizeFlags::default()),
            "BEIntVec ParamGammaCodec",
        );
    }

    // Write the accumulated benchmark results to a CSV file.
    let dir = "bench_results";
    fs::create_dir_all(dir).expect("Cannot create benchmark results directory");
    let file_path = format!("{}/bench_space.csv", dir);
    let mut file = File::create(&file_path).expect("Cannot create CSV file");
    writeln!(file, "name,k,space").expect("Error writing CSV header");
    writeln!(
        file,
        "Standard Vec,0,{}",
        uniform.mem_size(SizeFlags::default())
    )
    .expect("Error writing CSV line");
    for (name, k, space) in results {
        writeln!(file, "{},{},{}", name, k, space).expect("Error writing CSV line");
    }

    // Benchmark dummy per Criterion
    c.bench_function("dummy", |b| b.iter(|| {}));
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)  // Riduci ulteriormente il tempo
        .warm_up_time(Duration::from_millis(1))
        .measurement_time(Duration::from_secs(1));
    targets = bench_all
}
criterion_main!(benches);
