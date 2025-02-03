use compressed_intvec::{DeltaCodec, GammaCodec, IntVec};
use criterion::{criterion_group, criterion_main, Criterion};

// Benchmark the gamma codec
fn bench_gamma_codec(c: &mut Criterion) {
    c.bench_function("gamma_codec", |b| {
        b.iter(|| {
            // create a random vector with 1000 elements from 0 to 10000
            let input: Vec<u64> = (0..1000).map(|_| rand::random::<u64>() % 10000).collect();
            let compressed_input = IntVec::<GammaCodec>::from(input.clone(), 64).unwrap();

            for i in 0..input.len() {
                assert_eq!(input[i], compressed_input.get(i).unwrap());
            }
        });
    });
}

// Benchmark the delta codec
fn bench_delta_codec(c: &mut Criterion) {
    c.bench_function("delta_codec", |b| {
        b.iter(|| {
            // create a random vector with 1000 elements from 0 to 10000
            let input: Vec<u64> = (0..1000).map(|_| rand::random::<u64>() % 10000).collect();
            let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64).unwrap();

            for i in 0..input.len() {
                assert_eq!(input[i], compressed_input.get(i).unwrap());
            }
        });
    });
}

criterion_group!(benches, bench_gamma_codec, bench_delta_codec);
criterion_main!(benches);
