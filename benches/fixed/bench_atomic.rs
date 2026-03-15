use compressed_intvec::fixed::atomic::UAtomicFixedVec;
use compressed_intvec::fixed::BitWidth;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use sux::prelude::bit_field_slice::AtomicBitFieldSlice;

const VECTOR_SIZE: usize = 1_000_000;
const NUM_ACCESSES: usize = 100_000;

/// Generates a vector with uniformly random values up to a given maximum.
///
/// # Arguments
/// * `size` - The number of elements to generate.
/// * `max_val_exclusive` - The exclusive upper bound for the random values.
fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    let limit = if max_val_exclusive == 0 {
        u64::MAX
    } else {
        max_val_exclusive
    };

    (0..size).map(|_| rng.random_range(0..limit)).collect()
}

/// Defines the contention pattern for multi-threaded benchmarks.
#[derive(Debug, Clone, Copy)]
enum Contention {
    /// Threads access random, uncorrelated indices.
    Random,
    /// All threads access the same, single index.
    High,
}

impl Contention {
    fn name(&self) -> &'static str {
        match self {
            Contention::Random => "RandomContention",
            Contention::High => "HighContention",
        }
    }
}

/// Helper to convert `Ordering` to a string for benchmark names.
fn ordering_to_str(order: Ordering) -> &'static str {
    match order {
        Ordering::Relaxed => "Relaxed",
        Ordering::Release => "Release",
        Ordering::Acquire => "Acquire",
        Ordering::AcqRel => "AcqRel",
        Ordering::SeqCst => "SeqCst",
        _ => "Unknown",
    }
}

/// Registers and runs all single-threaded benchmarks for a given configuration.
fn register_single_thread_benches(
    c: &mut Criterion,
    bit_width: usize,
    access_indices: &[usize],
    access_values: &[u64],
) {
    let max_val = if bit_width == 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    // Generate a single set of initial data for all structures in this benchmark group.
    let initial_data = generate_random_vec(VECTOR_SIZE, 1 << bit_width as u32);

    for &order in &[Ordering::SeqCst, Ordering::Relaxed] {
        let mut group = c.benchmark_group(format!(
            "AtomicOps/{}bit/SingleThread/{}",
            bit_width,
            ordering_to_str(order)
        ));
        group.throughput(Throughput::Elements(NUM_ACCESSES as u64));

        // Baseline: std::sync::atomic, initialized with random data.
        let std_vec: Vec<AtomicU64> = initial_data
            .iter()
            .map(|&val| AtomicU64::new(val))
            .collect();

        // Our AtomicFixedVec, initialized with the same random data.
        let vec = UAtomicFixedVec::<u64>::builder()
            .bit_width(BitWidth::Explicit(bit_width))
            .build(&initial_data)
            .unwrap();

        // sux::bits::AtomicBitFieldVec, initialized with the same random data.
        let sux_vec_storage: Vec<AtomicU64> =
            (0..(VECTOR_SIZE * bit_width).div_ceil(u64::BITS as usize) + 2)
                .map(|_| AtomicU64::new(0))
                .collect();
        let sux_vec = unsafe {
            sux::bits::AtomicBitFieldVec::<u64, _>::from_raw_parts(
                sux_vec_storage.as_slice(),
                bit_width,
                VECTOR_SIZE,
            )
        };
        for (i, &val) in initial_data.iter().enumerate() {
            unsafe { sux_vec.set_atomic_unchecked(i, val, Ordering::Relaxed) };
        }

        // --- Benchmark Load ---
        group.bench_function("Baseline_Vec<AtomicU64>/load", |b| {
            b.iter(|| {
                for &idx in black_box(access_indices) {
                    black_box(std_vec[idx].load(order));
                }
            })
        });
        group.bench_function("AtomicFixedVec/load", |b| {
            b.iter(|| {
                for &idx in black_box(access_indices) {
                    black_box(vec.load(idx, order));
                }
            })
        });
        group.bench_function("Sux_AtomicBitFieldVec/load", |b| {
            b.iter(|| {
                for &idx in black_box(access_indices) {
                    black_box(unsafe { sux_vec.get_atomic_unchecked(idx, order) });
                }
            })
        });

        // --- Benchmark Store ---
        group.bench_function("Baseline_Vec<AtomicU64>/store", |b| {
            b.iter(|| {
                for i in 0..NUM_ACCESSES {
                    std_vec[access_indices[i]].store(access_values[i], order);
                }
            })
        });
        group.bench_function("AtomicFixedVec/store", |b| {
            b.iter(|| {
                for i in 0..NUM_ACCESSES {
                    vec.store(access_indices[i], access_values[i] & max_val, order);
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
                            order,
                        );
                    }
                }
            })
        });

        // --- Benchmark Compare-Exchange ---
        group.bench_function("AtomicFixedVec/cas", |b| {
            b.iter(|| {
                for i in 0..NUM_ACCESSES {
                    let idx = access_indices[i];
                    let current = vec.load(idx, Ordering::Relaxed);
                    let _ = vec.compare_exchange(
                        idx,
                        current,
                        access_values[i] & max_val,
                        order,
                        Ordering::Relaxed,
                    );
                }
            })
        });
        group.finish();
    }
}

/// Registers and runs all multi-threaded benchmarks for a given configuration.
#[allow(clippy::too_many_lines)]
fn register_multi_thread_benches(
    c: &mut Criterion,
    bit_width: usize,
    num_threads: usize,
    access_indices: &[usize],
    access_values: &[u64],
) {
    let max_val = if bit_width == 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };
    // Generate a single set of initial data for all structures in this benchmark group.
    let initial_data = generate_random_vec(VECTOR_SIZE, 1 << bit_width as u32);
    let indices_chunks: Vec<_> = access_indices.chunks(NUM_ACCESSES / num_threads).collect();
    let values_chunks: Vec<_> = access_values.chunks(NUM_ACCESSES / num_threads).collect();
    let high_contention_index = VECTOR_SIZE / 2;

    for &contention in &[Contention::Random, Contention::High] {
        for &order in &[Ordering::SeqCst, Ordering::Relaxed] {
            let mut group = c.benchmark_group(format!(
                "AtomicOps/{}bit/{}Threads/{}/{}",
                bit_width,
                num_threads,
                contention.name(),
                ordering_to_str(order)
            ));
            group.throughput(Throughput::Elements(NUM_ACCESSES as u64));

            // Baseline: std::sync::atomic, initialized with shared random data.
            let std_vec = Arc::new(
                initial_data
                    .iter()
                    .map(|&v| AtomicU64::new(v))
                    .collect::<Vec<_>>(),
            );

            // Our AtomicFixedVec, initialized with shared random data.
            let vec = Arc::new(
                UAtomicFixedVec::<u64>::builder()
                    .bit_width(BitWidth::Explicit(bit_width))
                    .build(&initial_data)
                    .unwrap(),
            );

            // sux::bits::AtomicBitFieldVec, initialized with shared random data.
            let sux_vec_storage: Arc<Vec<AtomicU64>> = Arc::new({
                let storage = (0..(VECTOR_SIZE * bit_width).div_ceil(u64::BITS as usize) + 2)
                    .map(|_| AtomicU64::new(0))
                    .collect::<Vec<_>>();
                let sux_temp = unsafe {
                    sux::bits::AtomicBitFieldVec::<u64, _>::from_raw_parts(
                        storage.as_slice(),
                        bit_width,
                        VECTOR_SIZE,
                    )
                };
                for (i, &val) in initial_data.iter().enumerate() {
                    unsafe { sux_temp.set_atomic_unchecked(i, val, Ordering::Relaxed) };
                }
                storage
            });

            // --- Benchmark Load ---
            group.bench_function("Baseline_Vec<AtomicU64>/load", |b| {
                b.iter(|| {
                    let barrier = Arc::new(Barrier::new(num_threads));
                    thread::scope(|s| {
                        for chunk in &indices_chunks {
                            let std_vec_clone = Arc::clone(&std_vec);
                            let barrier_clone = Arc::clone(&barrier);
                            s.spawn(move || {
                                barrier_clone.wait();
                                match contention {
                                    Contention::Random => {
                                        for &idx in *chunk {
                                            black_box(std_vec_clone[idx].load(order));
                                        }
                                    }
                                    Contention::High => {
                                        for _ in *chunk {
                                            black_box(
                                                std_vec_clone[high_contention_index].load(order),
                                            );
                                        }
                                    }
                                }
                            });
                        }
                    });
                })
            });
            group.bench_function("AtomicFixedVec/load", |b| {
                b.iter(|| {
                    let barrier = Arc::new(Barrier::new(num_threads));
                    thread::scope(|s| {
                        for chunk in &indices_chunks {
                            let vec_clone = Arc::clone(&vec);
                            let barrier_clone = Arc::clone(&barrier);
                            s.spawn(move || {
                                barrier_clone.wait();
                                match contention {
                                    Contention::Random => {
                                        for &idx in *chunk {
                                            black_box(vec_clone.load(idx, order));
                                        }
                                    }
                                    Contention::High => {
                                        for _ in *chunk {
                                            black_box(vec_clone.load(high_contention_index, order));
                                        }
                                    }
                                }
                            });
                        }
                    });
                })
            });
            group.bench_function("Sux_AtomicBitFieldVec/load", |b| {
                b.iter(|| {
                    let barrier = Arc::new(Barrier::new(num_threads));
                    thread::scope(|s| {
                        for chunk in &indices_chunks {
                            let storage_clone = Arc::clone(&sux_vec_storage);
                            let barrier_clone = Arc::clone(&barrier);
                            s.spawn(move || {
                                let sux_vec = unsafe {
                                    sux::bits::AtomicBitFieldVec::<u64, _>::from_raw_parts(
                                        storage_clone.as_slice(),
                                        bit_width,
                                        VECTOR_SIZE,
                                    )
                                };
                                barrier_clone.wait();
                                match contention {
                                    Contention::Random => {
                                        for &idx in *chunk {
                                            black_box(unsafe {
                                                sux_vec.get_atomic_unchecked(idx, order)
                                            });
                                        }
                                    }
                                    Contention::High => {
                                        for _ in *chunk {
                                            black_box(unsafe {
                                                sux_vec.get_atomic_unchecked(
                                                    high_contention_index,
                                                    order,
                                                )
                                            });
                                        }
                                    }
                                }
                            });
                        }
                    });
                })
            });

            // --- Benchmark Store ---
            group.bench_function("AtomicFixedVec/store", |b| {
                b.iter(|| {
                    let barrier = Arc::new(Barrier::new(num_threads));
                    thread::scope(|s| {
                        for (thread_id, (idx_chunk, val_chunk)) in
                            indices_chunks.iter().zip(&values_chunks).enumerate()
                        {
                            let vec_clone = Arc::clone(&vec);
                            let barrier_clone = Arc::clone(&barrier);
                            s.spawn(move || {
                                barrier_clone.wait();
                                match contention {
                                    Contention::Random => {
                                        for i in 0..idx_chunk.len() {
                                            vec_clone.store(
                                                idx_chunk[i],
                                                val_chunk[i] & max_val,
                                                order,
                                            );
                                        }
                                    }
                                    Contention::High => {
                                        for _ in 0..idx_chunk.len() {
                                            vec_clone.store(
                                                high_contention_index,
                                                thread_id as u64,
                                                order,
                                            );
                                        }
                                    }
                                }
                            });
                        }
                    });
                })
            });

            // --- Benchmark Compare-Exchange (High Contention Stress Test) ---
            if matches!(contention, Contention::High) {
                group.bench_function("AtomicFixedVec/cas_increment", |b| {
                    b.iter_batched(
                        || vec.store(high_contention_index, 0, Ordering::Relaxed),
                        |_| {
                            let barrier = Arc::new(Barrier::new(num_threads));
                            thread::scope(|s| {
                                for chunk in &indices_chunks {
                                    let vec_clone = Arc::clone(&vec);
                                    let barrier_clone = Arc::clone(&barrier);
                                    s.spawn(move || {
                                        barrier_clone.wait();
                                        for _ in 0..chunk.len() {
                                            let mut current = vec_clone
                                                .load(high_contention_index, Ordering::Relaxed);
                                            loop {
                                                match vec_clone.compare_exchange(
                                                    high_contention_index,
                                                    current,
                                                    current.wrapping_add(1),
                                                    order,
                                                    Ordering::Relaxed,
                                                ) {
                                                    Ok(_) => break,
                                                    Err(actual) => current = actual,
                                                }
                                            }
                                        }
                                    });
                                }
                            });
                        },
                        criterion::BatchSize::SmallInput,
                    )
                });
            }
            group.finish();
        }
    }
}

/// The main benchmark function that orchestrates all tests.
fn benchmark_atomic_ops(c: &mut Criterion) {
    // Generate a single, consistent set of random indices for all benchmarks.
    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();
    let access_values: Vec<u64> = (0..NUM_ACCESSES).map(|_| rng.random()).collect();

    // Test both a lock-free and a locked configuration.
    for &bit_width in &[16, 21] {
        // --- 1. Single-Threaded Benchmarks ---
        register_single_thread_benches(c, bit_width, &access_indices, &access_values);

        // --- 2. Multi-Threaded Benchmarks ---
        for &num_threads in &[2, 4, 8] {
            register_multi_thread_benches(
                c,
                bit_width,
                num_threads,
                &access_indices,
                &access_values,
            );
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(10))
        .measurement_time(Duration::from_secs(2));

    targets = benchmark_atomic_ops
}
criterion_main!(benches);
