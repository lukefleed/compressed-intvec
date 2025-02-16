use compressed_intvec::{
    BEIntVec, DeltaCodec, ExpGolombCodec, GammaCodec, LEIntVec, ParamDeltaCodec, ParamGammaCodec,
    RiceCodec,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion, PlotConfiguration};
use rand::distr::{Distribution, Uniform};
use rand::{Rng, SeedableRng};

pub fn plot_config() -> PlotConfiguration {
    PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic)
}

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

/// Benchmark random access for a given vector and codec.
///
/// Fixed: unwrap the result of `build_vec` so that the `get` function receives the expected type.
fn benchmark_random_access<T, C>(
    c: &mut Criterion,
    name: &str,
    input: Vec<u64>,
    k: usize,
    param: C,
    build_vec: impl Fn(Vec<u64>, usize, C) -> Result<T, Box<dyn std::error::Error>>,
    get: impl Fn(&T, usize) -> Option<u64>,
) {
    let vec = build_vec(input.clone(), k, param).unwrap();
    let indexes = generate_random_indexes(input.len(), input.len());
    c.bench_function(name, |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(get(&vec, i).unwrap());
            }
        });
    });
}

/// Benchmarks for different codecs and distributions
fn bench_all(c: &mut Criterion) {
    let input_size = 10_000;
    let max_value = 1_000;
    let ks = vec![4, 8, 16, 32, 64]; // Different values of k to benchmark

    // Generate different data distributions
    let uniform = generate_uniform_vec(input_size, max_value);

    // Benchmark Vec<u64> for different distributions
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
            c,
            &format!("LEIntVec GammaCodec random access (uniform, k={})", k),
            uniform.clone(),
            k,
            GammaCodec,
            |arg0: Vec<u64>, arg1: usize, _arg2: GammaCodec| {
                LEIntVec::<GammaCodec>::from_with_param(arg0, arg1, ())
            },
            LEIntVec::<_>::get,
        );
        benchmark_random_access(
            c,
            &format!("LEIntVec DeltaCodec random access (uniform, k={})", k),
            uniform.clone(),
            k,
            DeltaCodec,
            |arg0: Vec<u64>, arg1: usize, _arg2: DeltaCodec| {
                LEIntVec::<DeltaCodec>::from_with_param(arg0, arg1, ())
            },
            LEIntVec::<_>::get,
        );
        benchmark_random_access(
            c,
            &format!("LEIntVec ExpGolombCodec random access (uniform, k={})", k),
            uniform.clone(),
            k,
            3,
            |arg0: Vec<u64>, arg1: usize, arg2: usize| {
                LEIntVec::<ExpGolombCodec>::from_with_param(arg0, arg1, arg2)
            },
            LEIntVec::<_>::get,
        );
        benchmark_random_access(
            c,
            &format!("LEIntVec RiceCodec random access (uniform, k={})", k),
            uniform.clone(),
            k,
            3,
            |arg0: Vec<u64>, arg1: usize, arg2: usize| {
                LEIntVec::<RiceCodec>::from_with_param(arg0, arg1, arg2)
            },
            LEIntVec::<_>::get,
        );
        benchmark_random_access(
            c,
            &format!(
                "LEIntVec ParamDeltaCodec (true,true) random access (uniform, k={})",
                k
            ),
            uniform.clone(),
            k,
            ParamDeltaCodec::<true, true>,
            |arg0: Vec<u64>, arg1: usize, _arg2: ParamDeltaCodec<true, true>| {
                LEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(arg0, arg1, ())
            },
            LEIntVec::<_>::get,
        );
        benchmark_random_access(
            c,
            &format!(
                "LEIntVec ParamGammaCodec (true) random access (uniform, k={})",
                k
            ),
            uniform.clone(),
            k,
            ParamGammaCodec::<true>,
            |arg0: Vec<u64>, arg1: usize, _arg2: ParamGammaCodec<true>| {
                LEIntVec::<ParamGammaCodec<true>>::from_with_param(arg0, arg1, ())
            },
            LEIntVec::<_>::get,
        );

        // BEIntVec benchmarks
        benchmark_random_access(
            c,
            &format!("BEIntVec GammaCodec random access (uniform, k={})", k),
            uniform.clone(),
            k,
            GammaCodec,
            |arg0: Vec<u64>, arg1: usize, _arg2: GammaCodec| {
                BEIntVec::<GammaCodec>::from_with_param(arg0, arg1, ())
            },
            BEIntVec::<_>::get,
        );
        benchmark_random_access(
            c,
            &format!("BEIntVec DeltaCodec random access (uniform, k={})", k),
            uniform.clone(),
            k,
            DeltaCodec,
            |arg0: Vec<u64>, arg1: usize, _arg2: DeltaCodec| {
                BEIntVec::<DeltaCodec>::from_with_param(arg0, arg1, ())
            },
            BEIntVec::<_>::get,
        );
        benchmark_random_access(
            c,
            &format!("BEIntVec ExpGolombCodec random access (uniform, k={})", k),
            uniform.clone(),
            k,
            3,
            |arg0: Vec<u64>, arg1: usize, arg2: usize| {
                BEIntVec::<ExpGolombCodec>::from_with_param(arg0, arg1, arg2)
            },
            BEIntVec::<_>::get,
        );
        benchmark_random_access(
            c,
            &format!("BEIntVec RiceCodec random access (uniform, k={})", k),
            uniform.clone(),
            k,
            3,
            |arg0: Vec<u64>, arg1: usize, arg2: usize| {
                BEIntVec::<RiceCodec>::from_with_param(arg0, arg1, arg2)
            },
            BEIntVec::<_>::get,
        );
        benchmark_random_access(
            c,
            &format!(
                "BEIntVec ParamDeltaCodec (true,true) random access (uniform, k={})",
                k
            ),
            uniform.clone(),
            k,
            ParamDeltaCodec::<true, true>,
            |arg0: Vec<u64>, arg1: usize, _arg2: ParamDeltaCodec<true, true>| {
                BEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(arg0, arg1, ())
            },
            BEIntVec::<ParamDeltaCodec<true, true>>::get,
        );
        benchmark_random_access(
            c,
            &format!(
                "BEIntVec ParamGammaCodec (true) random access (uniform, k={})",
                k
            ),
            uniform.clone(),
            k,
            ParamGammaCodec::<true>,
            |arg0: Vec<u64>, arg1: usize, _arg2: ParamGammaCodec<true>| {
                BEIntVec::<ParamGammaCodec<true>>::from_with_param(arg0, arg1, ())
            },
            BEIntVec::<ParamGammaCodec<true>>::get,
        );
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
