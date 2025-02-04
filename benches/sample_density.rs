use compressed_intvec::{DeltaCodec, ExpGolombCodec, GammaCodec, IntVec};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

fn prepare_data(size: usize) -> Vec<u64> {
    let mut rng = rand::rng();
    (0..size).map(|_| rng.random_range(0..10000)).collect()
}

// bench for different values of k the IntVec with the different codecs
fn bench_intvec(c: &mut Criterion) {
    let data = prepare_data(100000);
    let delta_vec = IntVec::<DeltaCodec>::from(data.clone(), 32).unwrap();
    let gamma_vec = IntVec::<GammaCodec>::from(data.clone(), 32).unwrap();
    let exp_golomb_vec = IntVec::<ExpGolombCodec>::from(data.clone(), 32).unwrap();

    for k in [4, 8, 16, 32, 64, 128, 256].iter() {
        c.bench_function(&format!("Random Access IntVec Gamma k={}", k), |b| {
            b.iter(|| {
                for _ in 0..1000 {
                    let idx = black_box(rand::rng().random_range(0..gamma_vec.len()));
                    let _ = gamma_vec.get(idx);
                }
            });
        });

        c.bench_function(&format!("Random Access IntVec Delta k={}", k), |b| {
            // do 1000 random accesses, use the get method
            b.iter(|| {
                for _ in 0..1000 {
                    let idx = black_box(rand::rng().random_range(0..delta_vec.len()));
                    let _ = delta_vec.get(idx);
                }
            });
        });

        c.bench_function(&format!("Random Access IntVec Exp-Golomb k={}", k), |b| {
            // do 1000 random accesses, use the get method
            b.iter(|| {
                for _ in 0..1000 {
                    let idx = black_box(rand::rng().random_range(0..exp_golomb_vec.len()));
                    let _ = exp_golomb_vec.get(idx);
                }
            });
        });
    }
}

criterion_group!(benches, bench_intvec);
criterion_main!(benches);
