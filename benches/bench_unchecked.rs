use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;
// Importa i tipi necessari da sux
use sux::prelude::{BitFieldSlice, BitFieldVec};

/// Generates a vector with uniformly random values.
fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    if max_val_exclusive == 0 {
        return vec![0; size];
    }
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size)
        .map(|_| rng.gen_range(0..max_val_exclusive))
        .collect()
}

/// The main benchmark function to compare checked vs. unchecked access.
fn benchmark_checked_vs_unchecked(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 10_000_000;
    const NUM_ACCESSES: usize = 1_000_000;
    const K_VALUE: usize = 64;

    let data = generate_random_vec(VECTOR_SIZE, 1 << 10);

    // // --- 1. Setup IntVec (Variable-Length) ---
    // let intvec = LEIntVec::builder(&data)
    //     .k(K_VALUE)
    //     .codec(VariableCodecSpec::Auto)
    //     .build()
    //     .expect("Failed to build IntVec");

    // --- 2. Setup FixedVec ---
    let fixed_vec = LEFixedVec::builder(&data)
        .build()
        .expect("Failed to build FixedVec");

    // --- 3. Setup sux::BitFieldVec ---
    let sux_bfv = BitFieldVec::<u64>::from_slice(&data).unwrap();

    // --- 4. Setup Indices for Access ---
    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.gen_range(0..VECTOR_SIZE))
        .collect();

    // // --- 5. Benchmark IntVec ---
    // let mut group_intvec = c.benchmark_group("CheckedVsUnchecked/IntVec");

    // group_intvec.bench_function("get_loop_(checked)", |b| {
    //     b.iter(|| {
    //         for &index in black_box(&access_indices) {
    //             black_box(intvec.get(index));
    //         }
    //     })
    // });

    // group_intvec.bench_function("get_unchecked_loop", |b| {
    //     b.iter(|| {
    //         for &index in black_box(&access_indices) {
    //             black_box(unsafe { intvec.get_unchecked(index) });
    //         }
    //     })
    // });

    // group_intvec.finish();

    // --- 6. Benchmark Fixed-Width Implementations ---
    let mut group_fixed = c.benchmark_group("CheckedVsUnchecked/Fixed");

    // a) Our FixedVec (checked)
    group_fixed.bench_function("FixedVec_get_loop_(checked)", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(fixed_vec.get(index));
            }
        })
    });

    // b) Our FixedVec (unchecked)
    group_fixed.bench_function("FixedVec_get_unchecked_loop", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(unsafe { fixed_vec.get_unchecked(index) });
            }
        })
    });

    // c) sux::BitFieldVec (checked)
    group_fixed.bench_function("sux::BitFieldVec_get_loop_(checked)", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(sux_bfv.get(index));
            }
        })
    });

    // d) sux::BitFieldVec (unchecked)
    group_fixed.bench_function("sux::BitFieldVec_get_unchecked_loop", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(unsafe { sux_bfv.get_unchecked(index) });
            }
        })
    });

    group_fixed.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_secs(5));
    targets = benchmark_checked_vs_unchecked
}
criterion_main!(benches);
