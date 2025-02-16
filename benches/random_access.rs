use compressed_intvec::{
    BEIntVec, DeltaCodec, ExpGolombCodec, GammaCodec, LEIntVec, ParamDeltaCodec, ParamGammaCodec,
    RiceCodec,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};

/// Generate a vector of random u64 values in the range [0, 1000).
fn generate_random_vec(size: usize) -> Vec<u64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..size).map(|_| rng.random_range(0..1000)).collect()
}

/// Generate a list of random indexes within the range [0, max).
fn generate_random_indexes(n: usize, max: usize) -> Vec<usize> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.random_range(0..max)).collect()
}

fn bench_standard_vec_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let indexes = generate_random_indexes(10_000, 10_000);
    c.bench_function("Vec<u64> random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(input[i]);
            }
        })
    });
}

/// =============================
/// LEIntVec Random Access Benchmarks
/// =============================

fn bench_le_gamma_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let vec = LEIntVec::<GammaCodec>::from_with_param(input, k, ()).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("LEIntVec GammaCodec random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

fn bench_le_delta_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let vec = LEIntVec::<DeltaCodec>::from_with_param(input, k, ()).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("LEIntVec DeltaCodec random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

fn bench_le_exp_golomb_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let codec_param = 3;
    let vec = LEIntVec::<ExpGolombCodec>::from_with_param(input, k, codec_param).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("LEIntVec ExpGolombCodec random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

fn bench_le_rice_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let codec_param = 3;
    let vec = LEIntVec::<RiceCodec>::from_with_param(input, k, codec_param).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("LEIntVec RiceCodec random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

fn bench_le_param_delta_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    // Using ParamDeltaCodec with variant (true, true)
    let vec = LEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(input, k, ()).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("LEIntVec ParamDeltaCodec (true,true) random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

fn bench_le_param_gamma_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    // Using ParamGammaCodec with variant (true)
    let vec = LEIntVec::<ParamGammaCodec<true>>::from_with_param(input, k, ()).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("LEIntVec ParamGammaCodec (true) random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

/// =============================
/// BEIntVec Random Access Benchmarks
/// =============================
///
/// Note: This assumes that BEIntVec now provides a `get()` method similar to LEIntVec.

fn bench_be_gamma_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let vec = BEIntVec::<GammaCodec>::from_with_param(input, k, ()).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("BEIntVec GammaCodec random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

fn bench_be_delta_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let vec = BEIntVec::<DeltaCodec>::from_with_param(input, k, ()).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("BEIntVec DeltaCodec random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

fn bench_be_exp_golomb_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let codec_param = 3;
    let vec = BEIntVec::<ExpGolombCodec>::from_with_param(input, k, codec_param).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("BEIntVec ExpGolombCodec random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

fn bench_be_rice_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let codec_param = 3;
    let vec = BEIntVec::<RiceCodec>::from_with_param(input, k, codec_param).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("BEIntVec RiceCodec random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

fn bench_be_param_delta_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let vec = BEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(input, k, ()).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("BEIntVec ParamDeltaCodec (true,true) random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

fn bench_be_param_gamma_random_access(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let vec = BEIntVec::<ParamGammaCodec<true>>::from_with_param(input, k, ()).unwrap();
    let indexes = generate_random_indexes(vec.len(), vec.len());
    c.bench_function("BEIntVec ParamGammaCodec (true) random access", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(vec.get(i).unwrap());
            }
        })
    });
}

criterion_group!(
    benches,
    bench_standard_vec_random_access,
    bench_le_gamma_random_access,
    bench_le_delta_random_access,
    bench_le_exp_golomb_random_access,
    bench_le_rice_random_access,
    bench_le_param_delta_random_access,
    bench_le_param_gamma_random_access,
    bench_be_gamma_random_access,
    bench_be_delta_random_access,
    bench_be_exp_golomb_random_access,
    bench_be_rice_random_access,
    bench_be_param_delta_random_access,
    bench_be_param_gamma_random_access,
);
criterion_main!(benches);
