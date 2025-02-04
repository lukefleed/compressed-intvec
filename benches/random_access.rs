use compressed_intvec::{DeltaCodec, GammaCodec, IntVec};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

const DATA_SIZE: usize = 100000;
const ACCESS_COUNT: usize = 1000;

fn prepare_data(size: usize) -> Vec<u64> {
    let mut rng = rand::rng();
    (0..size).map(|_| rng.random_range(0..10000)).collect()
}

fn prepare_indices(size: usize, range: usize) -> Vec<usize> {
    let mut rng = rand::rng();
    (0..size).map(|_| rng.random_range(0..range)).collect()
}

fn bench_vec(c: &mut Criterion, data: &Vec<u64>, indices: &Vec<usize>) {
    c.bench_function("Random Access standard Vec", |b| {
        b.iter(|| {
            for &idx in indices {
                let _ = black_box(data[idx]);
            }
        });
    });
}

fn bench_intvec_gamma_64(c: &mut Criterion, data: &Vec<u64>, indices: &Vec<usize>) {
    let compressed_data = IntVec::<GammaCodec>::from(data.clone(), 64).unwrap();
    c.bench_function("Random Access IntVec Gamma", |b| {
        b.iter(|| {
            for &idx in indices {
                let _ = black_box(compressed_data.get(idx));
            }
        });
    });
}

fn bench_intvec_delta_64(c: &mut Criterion, data: &Vec<u64>, indices: &Vec<usize>) {
    let compressed_data = IntVec::<DeltaCodec>::from(data.clone(), 64).unwrap();
    c.bench_function("Random Access IntVec Delta", |b| {
        b.iter(|| {
            for &idx in indices {
                let _ = black_box(compressed_data.get(idx));
            }
        });
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let data = prepare_data(DATA_SIZE);
    let indices = prepare_indices(ACCESS_COUNT, DATA_SIZE);

    bench_vec(c, &data, &indices);
    bench_intvec_gamma_64(c, &data, &indices);
    bench_intvec_delta_64(c, &data, &indices);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
