use compressed_intvec::{DeltaCodec, GammaCodec, IntVec};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

fn prepare_data(size: usize) -> Vec<u64> {
    let mut rng = rand::rng();
    (0..size).map(|_| rng.random_range(0..10000)).collect()
}

fn bench_vec(c: &mut Criterion) {
    c.bench_function("Random Access standard Vec", |b| {
        let data = prepare_data(100000);
        // do 1000 random accesses, use the get method
        b.iter(|| {
            for _ in 0..1000 {
                let idx = black_box(rand::rng().random_range(0..data.len()));
                let _ = data[idx];
            }
        });
    });
}

fn bench_intvec_gamma_64(c: &mut Criterion) {
    c.bench_function("Random Access IntVec Gamma", |b| {
        let data = prepare_data(100000);
        let compressed_data = IntVec::<GammaCodec>::from(data.clone(), 64).unwrap();
        // do 1000 random accesses, use the get method
        b.iter(|| {
            for _ in 0..1000 {
                let idx = black_box(rand::rng().random_range(0..compressed_data.len()));
                let _ = compressed_data.get(idx);
            }
        });
    });
}

fn bench_intvec_delta_64(c: &mut Criterion) {
    c.bench_function("Random Access IntVec Delta", |b| {
        let data = prepare_data(100000);
        let compressed_data = IntVec::<DeltaCodec>::from(data.clone(), 64).unwrap();
        // do 1000 random accesses, use the get method
        b.iter(|| {
            for _ in 0..1000 {
                let idx = black_box(rand::rng().random_range(0..compressed_data.len()));
                let _ = compressed_data.get(idx);
            }
        });
    });
}

criterion_group!(
    benches,
    bench_vec,
    bench_intvec_gamma_64,
    bench_intvec_delta_64
);
criterion_main!(benches);
