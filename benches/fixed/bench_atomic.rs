//! # Comprehensive Benchmark Suite for Atomic Operations
//!
//! This suite provides an exhaustive performance analysis of `AtomicFixedVec`,
//! comparing it against `sux::bits::AtomicBitFieldVec` and a `Vec<AtomicU64>`
//! baseline under a wide range of conditions.
//!
//! ## Methodology
//!
//! To provide a complete performance picture, the benchmarks are structured
//! along several key dimensions:
//!
//! 1.  **Bit Width**:
//!     -   **16-bit**: Tests the highly optimized, lock-free path for power-of-two
//!         widths where elements are guaranteed to fit within a single `u64`.
//!     -   **21-bit**: Tests the more complex (but correct) hybrid path for
//!         non-power-of-two widths, which uses 128-bit atomics for values
//!         that span word boundaries.
//!
//! 2.  **Concurrency Level (Scalability)**:
//!     -   **Single-Thread**: Establishes a baseline for raw, uncontended throughput.
//!     -   **Multi-Thread (2, 4, 8 threads)**: Measures performance scaling as
//!         the number of concurrent threads increases.
//!
//! 3.  **Contention Pattern**:
//!     -   **Random Access (Diffuse Contention)**: Simulates a workload where
//!         threads access random, uniformly distributed locations. This is a
//!         common case with low probability of multiple threads hitting the same
//!         atomic word simultaneously.
//!     -   **High Contention**: A stress test where all threads repeatedly target
//!         the *exact same* memory location. This is the worst-case scenario and
//!         is critical for evaluating the efficiency of the underlying
//_compare-and-swap_
//!         loops and cache coherency protocols.
//!
//! 4.  **Memory Ordering**:
//!     -   **`Ordering::SeqCst`**: The strongest, most expensive ordering, which
//!         guarantees a single global order of operations.
//!     -   **`Ordering::Relaxed`**: The weakest, fastest ordering, which provides
//!         no ordering guarantees between threads but ensures atomicity. This is
//!         common in algorithms like counters where only the final atomic value matters.

use compressed_intvec::fixed::atomic::UAtomicFixedVec;
use compressed_intvec::fixed::BitWidth;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use sux::prelude::AtomicBitFieldSlice;

const VECTOR_SIZE: usize = 1_000_000;
const NUM_ACCESSES: usize = 100_000;

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

    for &order in &[Ordering::SeqCst, Ordering::Relaxed] {
        let mut group = c.benchmark_group(format!(
            "AtomicOps/{}bit/SingleThread/{}",
            bit_width,
            ordering_to_str(order)
        ));

        // Baseline: std::sync::atomic
        let std_vec: Vec<AtomicU64> = (0..VECTOR_SIZE).map(|_| AtomicU64::new(0)).collect();
        // Our AtomicFixedVec
        let initial_data = vec![0u64; VECTOR_SIZE];
        let vec = UAtomicFixedVec::<u64>::builder()
            .bit_width(BitWidth::Explicit(bit_width))
            .build(&initial_data)
            .unwrap();
        // sux::bits::AtomicBitFieldVec
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

            // Baseline: std::sync::atomic
            let std_vec = Arc::new(
                (0..VECTOR_SIZE)
                    .map(|_| AtomicU64::new(0))
                    .collect::<Vec<_>>(),
            );
            // Our AtomicFixedVec
            let initial_data = vec![0u64; VECTOR_SIZE];
            let vec = Arc::new(
                UAtomicFixedVec::<u64>::builder()
                    .bit_width(BitWidth::Explicit(bit_width))
                    .build(&initial_data)
                    .unwrap(),
            );
            // sux::bits::AtomicBitFieldVec
            let sux_vec_storage: Arc<Vec<AtomicU64>> = Arc::new(
                (0..(VECTOR_SIZE * bit_width).div_ceil(u64::BITS as usize) + 2)
                    .map(|_| AtomicU64::new(0))
                    .collect(),
            );

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
                    b.iter_with_setup(
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
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_secs(3));

    targets = benchmark_atomic_ops
}
criterion_main!(benches);