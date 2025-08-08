use compressed_intvec::fixed::atomic::AtomicFixedVec;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use sux::prelude::{AtomicBitFieldSlice};

/// The main benchmark function for atomic operations.
///
/// This suite measures the performance of `load`, `store`, `swap`, and `compare_exchange`
/// for our `AtomicFixedVec` and compares it against two baselines:
/// 1. `Vec<AtomicU64>`: The theoretical maximum speed for full-word atomic operations.
/// 2. `sux::bits::AtomicBitFieldVec`: A mature, high-performance reference implementation.
///
/// Scenarios tested:
/// - Single-threaded vs. Multi-threaded random access.
/// - Power-of-two bit width (16-bit) to test the lock-free path.
/// - Non-power-of-two bit width (21-bit) to test the hybrid seqlock/mutex path.
fn benchmark_atomic_ops(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 1_000_000;
    const NUM_ACCESSES: usize = 100_000;
    const NUM_THREADS: usize = 4;

    // Generate a single, consistent set of random indices for all benchmarks.
    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();
    let access_values: Vec<u64> = (0..NUM_ACCESSES).map(|_| rng.random()).collect();

    // Test both a lock-free and a locked configuration.
    for &bit_width in &[16, 21] {
        let max_val = (1u64 << bit_width) - 1;

        // --- 1. Single-Threaded Benchmarks ---
        {
            let mut group =
                c.benchmark_group(format!("AtomicOps/{}bit/SingleThread", bit_width));

            // Baseline: std::sync::atomic
            let std_vec: Vec<AtomicU64> = (0..VECTOR_SIZE).map(|_| AtomicU64::new(0)).collect();
            // Our AtomicFixedVec
            let our_vec = AtomicFixedVec::<u64, u64>::new(bit_width, VECTOR_SIZE).unwrap();
            // sux::bits::AtomicBitFieldVec
            let sux_vec_storage: Vec<AtomicU64> =
                (0..(VECTOR_SIZE * bit_width).div_ceil(u64::BITS.try_into().unwrap()) + 2)
                    .map(|_| AtomicU64::new(0))
                    .collect();
            let sux_vec = unsafe {
                sux::bits::AtomicBitFieldVec::<u64, _>::from_raw_parts(
                    sux_vec_storage.as_slice(),
                    bit_width,
                    VECTOR_SIZE,
                )
            };

            // Benchmark Load
            group.bench_function("Baseline_Vec<AtomicU64>/load", |b| {
                b.iter(|| {
                    for &idx in black_box(&access_indices) {
                        black_box(std_vec[idx].load(Ordering::SeqCst));
                    }
                })
            });
            group.bench_function("Our_AtomicFixedVec/load", |b| {
                b.iter(|| {
                    for &idx in black_box(&access_indices) {
                        black_box(our_vec.load(idx, Ordering::SeqCst));
                    }
                })
            });
            group.bench_function("Sux_AtomicBitFieldVec/load", |b| {
                b.iter(|| {
                    for &idx in black_box(&access_indices) {
                        black_box(unsafe { sux_vec.get_atomic_unchecked(idx, Ordering::SeqCst) });
                    }
                })
            });

            // Benchmark Store
            group.bench_function("Our_AtomicFixedVec/store", |b| {
                b.iter(|| {
                    for i in 0..NUM_ACCESSES {
                        our_vec.store(access_indices[i], access_values[i] & max_val, Ordering::SeqCst);
                    }
                })
            });
            group.bench_function("Sux_AtomicBitFieldVec/store", |b| {
                b.iter(|| {
                    for i in 0..NUM_ACCESSES {
                        unsafe {
                            sux_vec.set_atomic_unchecked(
                                access_indices[i],
                                access_values[i] & max_val,
                                Ordering::SeqCst,
                            );
                        }
                    }
                })
            });

            // Benchmark Compare-Exchange
            group.bench_function("Our_AtomicFixedVec/cas", |b| {
                b.iter(|| {
                    for i in 0..NUM_ACCESSES {
                        let idx = access_indices[i];
                        let current = our_vec.load(idx, Ordering::Relaxed);
                        let _ = our_vec.compare_exchange(
                            idx,
                            current,
                            access_values[i] & max_val,
                            Ordering::SeqCst,
                            Ordering::Relaxed,
                        );
                    }
                })
            });
            group.finish();
        }

        // --- 2. Multi-Threaded Benchmarks ---
        {
            let mut group = c.benchmark_group(format!("AtomicOps/{}bit/MultiThread", bit_width));
            let indices_chunks: Vec<_> = access_indices.chunks(NUM_ACCESSES / NUM_THREADS).collect();
            let values_chunks: Vec<_> = access_values.chunks(NUM_ACCESSES / NUM_THREADS).collect();

            // Our AtomicFixedVec
            let our_vec = Arc::new(AtomicFixedVec::<u64, u64>::new(bit_width, VECTOR_SIZE).unwrap());
            // sux::bits::AtomicBitFieldVec
            let sux_vec_storage: Arc<Vec<AtomicU64>> = Arc::new(
                (0..(VECTOR_SIZE * bit_width).div_ceil(u64::BITS.try_into().unwrap()) + 2)
                    .map(|_| AtomicU64::new(0))
                    .collect(),
            );

            // Benchmark Load (Multi-threaded)
            group.bench_function("Our_AtomicFixedVec/load", |b| {
                b.iter(|| {
                    let barrier = Arc::new(Barrier::new(NUM_THREADS));
                    thread::scope(|s| {
                        for chunk in &indices_chunks {
                            let vec_clone = Arc::clone(&our_vec);
                            let barrier_clone = Arc::clone(&barrier);
                            s.spawn(move || {
                                barrier_clone.wait();
                                for &idx in *chunk {
                                    black_box(vec_clone.load(idx, Ordering::SeqCst));
                                }
                            });
                        }
                    });
                })
            });

            group.bench_function("Sux_AtomicBitFieldVec/load", |b| {
                b.iter(|| {
                    let barrier = Arc::new(Barrier::new(NUM_THREADS));
                    thread::scope(|s| {
                        for chunk in &indices_chunks {
                            let barrier_clone = Arc::clone(&barrier);
                            let storage_clone = Arc::clone(&sux_vec_storage);
                            s.spawn(move || {
                                let sux_vec = unsafe {
                                    sux::bits::AtomicBitFieldVec::<u64, _>::from_raw_parts(
                                        storage_clone.as_slice(),
                                        bit_width,
                                        VECTOR_SIZE,
                                    )
                                };
                                barrier_clone.wait();
                                for &idx in *chunk {
                                    black_box(unsafe {
                                        sux_vec.get_atomic_unchecked(idx, Ordering::SeqCst)
                                    });
                                }
                            });
                        }
                    });
                })
            });
            
            // Benchmark Store (Multi-threaded)
            group.bench_function("Our_AtomicFixedVec/store", |b| {
                b.iter(|| {
                    let barrier = Arc::new(Barrier::new(NUM_THREADS));
                    thread::scope(|s| {
                        for (idx_chunk, val_chunk) in indices_chunks.iter().zip(&values_chunks) {
                            let vec_clone = Arc::clone(&our_vec);
                            let barrier_clone = Arc::clone(&barrier);
                            s.spawn(move || {
                                barrier_clone.wait();
                                for i in 0..idx_chunk.len() {
                                    vec_clone.store(idx_chunk[i], val_chunk[i] & max_val, Ordering::SeqCst);
                                }
                            });
                        }
                    });
                })
            });
            
            group.finish();
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_secs(2));

    targets = benchmark_atomic_ops
}
criterion_main!(benches);