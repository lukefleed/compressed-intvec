use compressed_intvec::{DeltaCodec, ExpGolombCodec, GammaCodec, IntVec};
use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

const DATA_SIZE: usize = 100000;
const ACCESS_COUNT: usize = 1000;

fn generate_vec(size: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..size).map(|_| rng.random_range(0..10000)).collect()
}

fn prepare_indices(size: usize, range: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..size).map(|_| rng.random_range(0..range)).collect()
}

fn bench_all(c: &mut Criterion) {
    let data = generate_vec(DATA_SIZE);
    let delta_vec = IntVec::<DeltaCodec>::from(data.clone(), 32).unwrap();
    let gamma_vec = IntVec::<GammaCodec>::from(data.clone(), 32).unwrap();
    let exp_golomb_vec = IntVec::<ExpGolombCodec>::from_with_param(data.clone(), 3, 9).unwrap();
    let indices = prepare_indices(ACCESS_COUNT, DATA_SIZE);

    c.bench_function("Random Access Standard Vec", |b| {
        b.iter(|| {
            for idx in indices.iter() {
                let _ = data[*idx];
            }
        });
    });

    c.bench_function("Random Access IntVec Gamma", |b| {
        b.iter(|| {
            for idx in indices.iter() {
                let _ = gamma_vec.get(*idx);
            }
        });
    });

    c.bench_function("Random Access IntVec Delta", |b| {
        b.iter(|| {
            for idx in indices.iter() {
                let _ = delta_vec.get(*idx);
            }
        });
    });

    c.bench_function("Random Access IntVec Exp-Golomb", |b| {
        b.iter(|| {
            for idx in indices.iter() {
                let _ = exp_golomb_vec.get(*idx);
            }
        });
    });
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
