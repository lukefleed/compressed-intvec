use compressed_intvec::{
    BEIntVec, DeltaCodec, ExpGolombCodec, GammaCodec, LEIntVec, ParamDeltaCodec, ParamGammaCodec,
    RiceCodec,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use once_cell::sync::Lazy;
use rand::{Rng, SeedableRng};

/// Global fixture: generate a vector of random u64 values in the range [0, 1000)
static INPUT_VEC: Lazy<Vec<u64>> = Lazy::new(|| {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..10_000).map(|_| rng.random_range(0..1000)).collect()
});

/// Global fixture: generate a list of random indexes in the range [0, 10_000)
static INDEXES: Lazy<Vec<usize>> = Lazy::new(|| {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..10_000).map(|_| rng.random_range(0..10_000)).collect()
});

/// Values for k to test.
static K_VALUES: &[usize] = &[2, 4, 8, 16, 32, 64];

//
// LEIntVec benchmarks
//

fn bench_le_gamma_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("LEIntVec GammaCodec random access (varying k)");
    for &k in K_VALUES {
        let vec = LEIntVec::<GammaCodec>::from_with_param(INPUT_VEC.clone(), k, ()).unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_le_delta_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("LEIntVec DeltaCodec random access (varying k)");
    for &k in K_VALUES {
        let vec = LEIntVec::<DeltaCodec>::from_with_param(INPUT_VEC.clone(), k, ()).unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_le_exp_golomb_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("LEIntVec ExpGolombCodec random access (varying k)");
    let codec_param = 3;
    for &k in K_VALUES {
        let vec =
            LEIntVec::<ExpGolombCodec>::from_with_param(INPUT_VEC.clone(), k, codec_param).unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_le_rice_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("LEIntVec RiceCodec random access (varying k)");
    let codec_param = 3;
    for &k in K_VALUES {
        let vec =
            LEIntVec::<RiceCodec>::from_with_param(INPUT_VEC.clone(), k, codec_param).unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_le_param_delta_varying_k(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("LEIntVec ParamDeltaCodec (true,true) random access (varying k)");
    for &k in K_VALUES {
        let vec =
            LEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(INPUT_VEC.clone(), k, ())
                .unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_le_param_gamma_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("LEIntVec ParamGammaCodec (true) random access (varying k)");
    for &k in K_VALUES {
        let vec =
            LEIntVec::<ParamGammaCodec<true>>::from_with_param(INPUT_VEC.clone(), k, ()).unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

//
// BEIntVec benchmarks (assuming BEIntVec has a similar `get()` method)
//

fn bench_be_gamma_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("BEIntVec GammaCodec random access (varying k)");
    for &k in K_VALUES {
        let vec = BEIntVec::<GammaCodec>::from_with_param(INPUT_VEC.clone(), k, ()).unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_be_delta_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("BEIntVec DeltaCodec random access (varying k)");
    for &k in K_VALUES {
        let vec = BEIntVec::<DeltaCodec>::from_with_param(INPUT_VEC.clone(), k, ()).unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_be_exp_golomb_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("BEIntVec ExpGolombCodec random access (varying k)");
    let codec_param = 3;
    for &k in K_VALUES {
        let vec =
            BEIntVec::<ExpGolombCodec>::from_with_param(INPUT_VEC.clone(), k, codec_param).unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_be_rice_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("BEIntVec RiceCodec random access (varying k)");
    let codec_param = 3;
    for &k in K_VALUES {
        let vec =
            BEIntVec::<RiceCodec>::from_with_param(INPUT_VEC.clone(), k, codec_param).unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_be_param_delta_varying_k(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("BEIntVec ParamDeltaCodec (true,true) random access (varying k)");
    for &k in K_VALUES {
        let vec =
            BEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(INPUT_VEC.clone(), k, ())
                .unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

fn bench_be_param_gamma_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("BEIntVec ParamGammaCodec (true) random access (varying k)");
    for &k in K_VALUES {
        let vec =
            BEIntVec::<ParamGammaCodec<true>>::from_with_param(INPUT_VEC.clone(), k, ()).unwrap();
        group.bench_function(format!("k = {}", k), |b| {
            b.iter(|| {
                for &i in INDEXES.iter() {
                    black_box(vec.get(i).unwrap());
                }
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_le_gamma_varying_k,
    bench_be_gamma_varying_k,
    bench_le_delta_varying_k,
    bench_be_delta_varying_k,
    bench_le_exp_golomb_varying_k,
    bench_be_exp_golomb_varying_k,
    bench_le_rice_varying_k,
    bench_be_rice_varying_k,
    bench_le_param_delta_varying_k,
    bench_be_param_delta_varying_k,
    bench_le_param_gamma_varying_k,
    bench_be_param_gamma_varying_k,
);
criterion_main!(benches);
