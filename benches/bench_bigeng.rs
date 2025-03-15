use compressed_intvec::codecs::{DeltaCodec, ExpGolombCodec, GammaCodec, MinimalBinaryCodec, ParamDeltaCodec, ParamGammaCodec, RiceCodec};
use compressed_intvec::intvec::BEIntVec;
use criterion::{criterion_group, criterion_main, Criterion};
use qwt::{AccessUnsigned, QWT512};
use rand::{Rng, SeedableRng};
use std::fs;
use std::path::Path;

fn load_ascii_from_file(path: &Path) -> Vec<u64> {
    // Read the file as bytes
    let text = fs::read(path).expect("Failed to read file");
    
    // Convert each byte to u32 (ASCII value)
    let tmp: Vec<u32> = text.into_iter().map(|b| b as u32).collect();

    // Return a Vec of u64
    tmp.into_iter().map(|x| x as u64).collect()
}

// fn bench_intvec_construction(c: &mut Criterion) {
//     let path = Path::new("dataset/big_english");
//     let values = load_ascii_from_file(path);
//     let rice_k = (values.iter().sum::<u64>() as f64 / values.len() as f64)
//             .log2()
//             .floor() as usize;
    
//     let mut group = c.benchmark_group("intvec_construction_big_english");
//     group.sample_size(10);
//     group.bench_function("intvec_construction_big_english", |b| {
//         b.iter(|| {
//             let _ = BEIntVec::<RiceCodec>::from_with_param(black_box(&values), 32, rice_k).unwrap();
//         });
//     });
//     group.finish();
// }

// fn bench_qwt_construction(c: &mut Criterion) {
//     let path = Path::new("dataset/big_english");
//     let values = load_ascii_from_file(path);
    
//     let mut group = c.benchmark_group("qwt_construction_big_english");
//     group.sample_size(10);
//     group.bench_function("qwt_construction_big_english", |b| {
//         b.iter_with_setup(
//             || values.clone(), // Setup: clone outside the measured iteration
//             |values_clone| {
//                 let _ = QWT512::from(black_box(values_clone));
//             }
//         );
//     });
//     group.finish();
// }

fn generate_random_query_indices(len: usize, num_queries: usize) -> Vec<usize> {
    // use same seed for reproducibility
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    (0..num_queries).map(|_| rng.random_range(0..len)).collect()
}

fn bench_standard_vec_access(c: &mut Criterion) {
    let path = Path::new("dataset/big_english");
    let values = load_ascii_from_file(path);

    let query_indices = generate_random_query_indices(values.len(), 1000);

    let mut group = c.benchmark_group("standard_vec_access_big_english");
    group.sample_size(10);
    group.bench_function("standard_vec_access_big_english", |b| {
        b.iter(|| {
            for i in &query_indices {
                let _ = values[*i];
            }
        });
    });
    group.finish();
}


fn bench_intvec_access(c: &mut Criterion) {
    let path = Path::new("dataset/big_english");
    let values = load_ascii_from_file(path);
    
    let query_indices = generate_random_query_indices(values.len(), 1000);
    let mut group = c.benchmark_group("intvec_access_big_english");

    let intvec = BEIntVec::<DeltaCodec>::from(&values, 32).unwrap();
    group.sample_size(10);
    group.bench_function("Access Delta Codec", |b| {
        b.iter(|| {
            for i in &query_indices {
                let _ = intvec.get(*i);
            }
        });
    });

    let intvec = BEIntVec::<ParamDeltaCodec<true, true>>::from(&values, 32).unwrap();
    group.sample_size(10);
    group.bench_function("Access Param Delta Codec", |b| {
        b.iter(|| {
            for i in &query_indices {
                let _ = intvec.get(*i);
            }
        });
    });

    let intvec = BEIntVec::<GammaCodec>::from(&values, 32).unwrap();
    group.bench_function("Access Gamma Codec", |b| {
        b.iter(|| {
            for i in &query_indices {
                let _ = intvec.get(*i);
            }
        });
    });

    let intvec = BEIntVec::<ParamGammaCodec<true>>::from(&values, 32).unwrap();
    group.bench_function("Access Param Gamma Codec", |b| {
        b.iter(|| {
            for i in &query_indices {
                let _ = intvec.get(*i);
            }
        });
    });


    let intvec = BEIntVec::<MinimalBinaryCodec>::from_with_param(&values, 32, 16).unwrap();
    group.bench_function("Access Minimal Binary Codec", |b| {
        b.iter(|| {
            for i in &query_indices {
                let _ = intvec.get(*i);
            }
        });
    });


    let rice_k = (values.iter().sum::<u64>() as f64 / values.len() as f64)
            .log2()
            .floor() as usize;
    let intvec = BEIntVec::<RiceCodec>::from_with_param(&values, 32, rice_k).unwrap();
    group.bench_function("Access RiceCodec", |b| {
        b.iter(|| {
            for i in &query_indices {
                let _ = intvec.get(*i);
            }
        });
    });

    let exp_k = (values.iter().sum::<u64>() as f64 / values.len() as f64)
    .log2()
    .floor() as usize;
    let intvec = BEIntVec::<ExpGolombCodec>::from_with_param(&values, 32, exp_k).unwrap();
    group.bench_function("Access ExpGolombCodec", |b| {
        b.iter(|| {
            for i in &query_indices {
                let _ = intvec.get(*i);
            }
        });
    });

    group.finish();
}

fn bench_qwt_access(c: &mut Criterion) {
    let path = Path::new("dataset/big_english");
    let values = load_ascii_from_file(path);
    let qwt = QWT512::from(values.clone());

    let query_indices = generate_random_query_indices(values.len(), 1000);

    let mut group = c.benchmark_group("qwt_access_big_english");
    group.sample_size(10);
    group.bench_function("qwt_access_big_english", |b| {
        b.iter(|| {
            for i in &query_indices {
                let _ = qwt.get(*i);
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_intvec_access,  bench_qwt_access, bench_standard_vec_access);
criterion_main!(benches);