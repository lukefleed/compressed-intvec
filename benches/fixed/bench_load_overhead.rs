use compressed_intvec::fixed::atomic::AtomicFixedVec;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::atomic::Ordering;
use sux::prelude::AtomicBitFieldSlice;

const VECTOR_SIZE: usize = 1_000_000;
const NUM_ACCESSES: usize = 100_000;
const BIT_WIDTH: usize = 16;

/// The main benchmark function for load overhead.
fn benchmark_load_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("LoadOverhead/{}bit", BIT_WIDTH));

    // --- 1. Setup Data and Indices ---
    // Generate a vector of random data.
    let mut rng = SmallRng::seed_from_u64(42);
    let data: Vec<u64> = (0..VECTOR_SIZE)
        .map(|_| rng.random_range(0..1u64 << BIT_WIDTH))
        .collect();

    // Generate a consistent set of random indices for a fair comparison.
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();

    // --- 2. Setup Atomic Vectors ---
    // Our AtomicFixedVec
    let our_vec = AtomicFixedVec::<u64, u64>::new(BIT_WIDTH, VECTOR_SIZE).unwrap();
    for (i, &val) in data.iter().enumerate() {
        our_vec.store(i, val, Ordering::Relaxed);
    }

    // sux::bits::AtomicBitFieldVec
    let sux_atomic_storage: Vec<std::sync::atomic::AtomicU64> = (0..(VECTOR_SIZE * BIT_WIDTH)
        .div_ceil(u64::BITS as usize))
        .map(|_| std::sync::atomic::AtomicU64::new(0))
        .collect();

    let sux_vec = unsafe {
        sux::bits::AtomicBitFieldVec::<u64, _>::from_raw_parts(
            sux_atomic_storage.as_slice(),
            BIT_WIDTH,
            VECTOR_SIZE,
        )
    };
    for (i, &val) in data.iter().enumerate() {
        unsafe {
            sux_vec.set_atomic_unchecked(i, val, Ordering::Relaxed);
        }
    }

    // --- 3. Run Benchmarks ---
    group.bench_function("Our_AtomicFixedVec/load_loop", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                black_box(our_vec.load(black_box(index), Ordering::Relaxed));
            }
        })
    });

    group.bench_function("Sux_AtomicBitFieldVec/get_atomic_unchecked_loop", |b| {
        b.iter(|| {
            for &index in black_box(&access_indices) {
                // SAFETY: Indices are generated within bounds.
                black_box(unsafe {
                    sux_vec.get_atomic_unchecked(black_box(index), Ordering::Relaxed)
                });
            }
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_load_overhead);
criterion_main!(benches);