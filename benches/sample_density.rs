use compressed_intvec::{DeltaCodec, GammaCodec, IntVec};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

fn prepare_data(size: usize) -> Vec<u64> {
    let mut rng = rand::rng();
    (0..size).map(|_| rng.random_range(0..10000)).collect()
}

// bench for different values of k the IntVec with the different codecs
fn bench_intvec(c: &mut Criterion) {
    let data = prepare_data(100000);
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256].iter() {
        c.bench_function(&format!("Random Access IntVec Gamma k={}", k), |b| {
            let compressed_data = IntVec::<GammaCodec>::from(data.clone(), *k).unwrap();
            // do 1000 random accesses, use the get method
            b.iter(|| {
                for _ in 0..1000 {
                    let idx = black_box(rand::rng().random_range(0..compressed_data.len()));
                    let _ = compressed_data.get(idx);
                }
            });
        });

        c.bench_function(&format!("Random Access IntVec Delta k={}", k), |b| {
            let compressed_data = IntVec::<DeltaCodec>::from(data.clone(), *k).unwrap();
            // do 1000 random accesses, use the get method
            b.iter(|| {
                for _ in 0..1000 {
                    let idx = black_box(rand::rng().random_range(0..compressed_data.len()));
                    let _ = compressed_data.get(idx);
                }
            });
        });
    }
}

criterion_group!(benches, bench_intvec);
criterion_main!(benches);
