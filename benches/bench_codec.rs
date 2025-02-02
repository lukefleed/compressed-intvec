use compressed_intvec::{DeltaCodec, GammaCodec, IntVec};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

/// Genera un vettore di `n` valori casuali nell'intervallo [0, max)
fn generate_input(n: usize, max: u64) -> Vec<u64> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random_range(0..max)).collect()
}

/// Benchmark per il codec Gamma:
fn bench_gamma(c: &mut Criterion) {
    // Genera un input di 100.000 numeri casuali
    let input = generate_input(100_000, 10_000);
    let sampling_rate = 64;

    // Benchmark per la costruzione del vettore compresso
    c.bench_function("GammaCodec from", |b| {
        b.iter(|| {
            let compressed = IntVec::<GammaCodec>::from(black_box(input.clone()), sampling_rate);
            black_box(compressed);
        })
    });

    // Pre-costruisce il vettore compresso per testare il metodo `get`
    let compressed = IntVec::<GammaCodec>::from(input.clone(), sampling_rate);
    c.bench_function("GammaCodec get", |b| {
        b.iter(|| {
            for i in 0..input.len() {
                black_box(compressed.get(i));
            }
        })
    });
}

/// Benchmark per il codec Delta:
fn bench_delta(c: &mut Criterion) {
    // Genera un input di 100.000 numeri casuali
    let input = generate_input(100_000, 10_000);
    let sampling_rate = 64;

    // Benchmark per la costruzione del vettore compresso
    c.bench_function("DeltaCodec from", |b| {
        b.iter(|| {
            let compressed = IntVec::<DeltaCodec>::from(black_box(input.clone()), sampling_rate);
            black_box(compressed);
        })
    });

    // Pre-costruisce il vettore compresso per testare il metodo `get`
    let compressed = IntVec::<DeltaCodec>::from(input.clone(), sampling_rate);
    c.bench_function("DeltaCodec get", |b| {
        b.iter(|| {
            for i in 0..input.len() {
                black_box(compressed.get(i));
            }
        })
    });
}

criterion_group!(benches, bench_gamma, bench_delta);
criterion_main!(benches);
