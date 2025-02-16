use compressed_intvec::{
    BEIntVec, DeltaCodec, ExpGolombCodec, GammaCodec, LEIntVec, ParamDeltaCodec, ParamGammaCodec,
    RiceCodec,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};

/// Genera un vettore di `u64` casuali compresi tra 0 e 10000,
/// in modo da evitare valori troppo elevati.
fn generate_random_vec(size: usize) -> Vec<u64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..size).map(|_| rng.random_range(0..10000)).collect()
}

/// ----------------------------
/// Benchmarks per vettori LE
/// ----------------------------

/// Benchmark per la creazione di un LEIntVec con GammaCodec.
fn bench_le_gamma_creation(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    c.bench_function("LEIntVec GammaCodec creation", |b| {
        b.iter(|| {
            let vec = LEIntVec::<GammaCodec>::from_with_param(input.clone(), k, ()).unwrap();
            black_box(vec);
        })
    });
}

/// Benchmark per la creazione di un LEIntVec con DeltaCodec.
fn bench_le_delta_creation(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    c.bench_function("LEIntVec DeltaCodec creation", |b| {
        b.iter(|| {
            let vec = LEIntVec::<DeltaCodec>::from_with_param(input.clone(), k, ()).unwrap();
            black_box(vec);
        })
    });
}

/// Benchmark per la creazione di un LEIntVec con ExpGolombCodec.
/// Il parametro del codec è fissato a 3.
fn bench_le_exp_golomb_creation(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let codec_param = 3;
    c.bench_function("LEIntVec ExpGolombCodec creation", |b| {
        b.iter(|| {
            let vec =
                LEIntVec::<ExpGolombCodec>::from_with_param(input.clone(), k, codec_param).unwrap();
            black_box(vec);
        })
    });
}

/// Benchmark per la creazione di un LEIntVec con RiceCodec.
/// Il parametro del codec è fissato a 3.
fn bench_le_rice_creation(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let codec_param = 3;
    c.bench_function("LEIntVec RiceCodec creation", |b| {
        b.iter(|| {
            let vec =
                LEIntVec::<RiceCodec>::from_with_param(input.clone(), k, codec_param).unwrap();
            black_box(vec);
        })
    });
}

/// Benchmark per la creazione di un LEIntVec con ParamDeltaCodec.
/// In questo esempio viene utilizzata la variante `ParamDeltaCodec<true, true>`.
fn bench_le_param_delta_creation(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    c.bench_function("LEIntVec ParamDeltaCodec (true,true) creation", |b| {
        b.iter(|| {
            let vec =
                LEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(input.clone(), k, ())
                    .unwrap();
            black_box(vec);
        })
    });
}

/// ----------------------------
/// Benchmarks per vettori BE
/// ----------------------------
/// Poiché `BEIntVec` non espone un metodo `get`, vengono misurate solo le prestazioni di creazione.

fn bench_be_gamma_creation(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    c.bench_function("BEIntVec GammaCodec creation", |b| {
        b.iter(|| {
            let vec = BEIntVec::<GammaCodec>::from_with_param(input.clone(), k, ()).unwrap();
            black_box(vec);
        })
    });
}

fn bench_be_delta_creation(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    c.bench_function("BEIntVec DeltaCodec creation", |b| {
        b.iter(|| {
            let vec = BEIntVec::<DeltaCodec>::from_with_param(input.clone(), k, ()).unwrap();
            black_box(vec);
        })
    });
}

fn bench_be_exp_golomb_creation(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let codec_param = 3;
    c.bench_function("BEIntVec ExpGolombCodec creation", |b| {
        b.iter(|| {
            let vec =
                BEIntVec::<ExpGolombCodec>::from_with_param(input.clone(), k, codec_param).unwrap();
            black_box(vec);
        })
    });
}

fn bench_be_rice_creation(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    let codec_param = 3;
    c.bench_function("BEIntVec RiceCodec creation", |b| {
        b.iter(|| {
            let vec =
                BEIntVec::<RiceCodec>::from_with_param(input.clone(), k, codec_param).unwrap();
            black_box(vec);
        })
    });
}

fn bench_be_param_delta_creation(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    c.bench_function("BEIntVec ParamDeltaCodec (true,true) creation", |b| {
        b.iter(|| {
            let vec =
                BEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(input.clone(), k, ())
                    .unwrap();
            black_box(vec);
        })
    });
}

fn bench_be_param_gamma_creation(c: &mut Criterion) {
    let input = generate_random_vec(10_000);
    let k = 16;
    c.bench_function("BEIntVec ParamGammaCodec (true) creation", |b| {
        b.iter(|| {
            let vec =
                BEIntVec::<ParamGammaCodec<true>>::from_with_param(input.clone(), k, ()).unwrap();
            black_box(vec);
        })
    });
}

criterion_group!(
    benches,
    bench_le_gamma_creation,
    bench_le_delta_creation,
    bench_le_exp_golomb_creation,
    bench_le_rice_creation,
    bench_le_param_delta_creation,
    bench_be_gamma_creation,
    bench_be_delta_creation,
    bench_be_exp_golomb_creation,
    bench_be_rice_creation,
    bench_be_param_delta_creation,
    bench_be_param_gamma_creation,
);
criterion_main!(benches);
