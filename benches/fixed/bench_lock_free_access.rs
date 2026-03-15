use compressed_intvec::fixed::atomic::UAtomicFixedVec;
use compressed_intvec::fixed::BitWidth;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use sux::prelude::{bit_field_slice::AtomicBitFieldSlice, AtomicBitFieldVec};

const VECTOR_SIZE: usize = 10_000;
const OPS_PER_THREAD: usize = 100_000;
const BIT_WIDTH: usize = 16; // Power of two for the lock-free path

fn benchmark_lock_free_scaling(c: &mut Criterion) {
    // Determine the number of logical cores available.
    let num_cores = std::thread::available_parallelism().unwrap().get();
    let mut thread_counts: Vec<usize> = (1..=num_cores).filter(|n| n.is_power_of_two()).collect();
    if !thread_counts.contains(&num_cores) {
        thread_counts.push(num_cores);
    }
    thread_counts.sort_unstable();
    thread_counts.dedup();

    for &num_threads in &thread_counts {
        let total_ops = (OPS_PER_THREAD * num_threads) as u64;
        let mut group =
            c.benchmark_group(format!("LockFreeScaling_Diffuse/{}Threads", num_threads));
        group.throughput(Throughput::Elements(total_ops));

        // Pre-generate a single set of random indices for this benchmark configuration.
        let mut rng = SmallRng::seed_from_u64(42);
        let access_indices: Vec<usize> = (0..total_ops as usize)
            .map(|_| rng.random_range(0..VECTOR_SIZE))
            .collect();

        // --- Setup Data Structures Once ---
        let baseline_u16 = Arc::new(
            (0..VECTOR_SIZE)
                .map(|_| AtomicU16::new(0))
                .collect::<Vec<_>>(),
        );
        let afv_16bit = Arc::new(
            UAtomicFixedVec::<u64>::builder()
                .bit_width(BitWidth::Explicit(BIT_WIDTH))
                .build(&vec![0; VECTOR_SIZE])
                .unwrap(),
        );
        let sux_storage_16bit = Arc::new(
            (0..(VECTOR_SIZE * BIT_WIDTH).div_ceil(64) + 2)
                .map(|_| AtomicU64::new(0))
                .collect(),
        );

        // --- Benchmark Runs ---
        group.bench_function("Baseline_Vec<AtomicU16>/store", |b| {
            b.iter(|| run_store_on_atomic_u16(&baseline_u16, num_threads, &access_indices));
        });

        group.bench_function("AtomicFixedVec/store", |b| {
            b.iter(|| run_store_on_atomic_fixed_vec(&afv_16bit, num_threads, &access_indices));
        });

        group.bench_function("sux::AtomicBitFieldVec/store", |b| {
            b.iter(|| run_store_on_sux_vec(&sux_storage_16bit, num_threads, &access_indices));
        });

        group.finish();
    }
}

fn run_store_on_atomic_u16(vec: &Arc<Vec<AtomicU16>>, num_threads: usize, indices: &[usize]) {
    let barrier = Arc::new(Barrier::new(num_threads));
    let chunks: Vec<_> = indices.chunks(OPS_PER_THREAD).collect();

    thread::scope(|s| {
        for (thread_id, chunk) in chunks.iter().enumerate() {
            let vec_clone = Arc::clone(vec);
            let barrier_clone = Arc::clone(&barrier);
            s.spawn(move || {
                barrier_clone.wait();
                for &index in *chunk {
                    vec_clone[index].store(thread_id as u16, Ordering::SeqCst);
                }
            });
        }
    });
}

fn run_store_on_atomic_fixed_vec(
    vec: &Arc<UAtomicFixedVec<u64>>,
    num_threads: usize,
    indices: &[usize],
) {
    let barrier = Arc::new(Barrier::new(num_threads));
    let chunks: Vec<_> = indices.chunks(OPS_PER_THREAD).collect();

    thread::scope(|s| {
        for (thread_id, chunk) in chunks.iter().enumerate() {
            let vec_clone = Arc::clone(vec);
            let barrier_clone = Arc::clone(&barrier);
            s.spawn(move || {
                barrier_clone.wait();
                for &index in *chunk {
                    vec_clone.store(index, thread_id as u64, Ordering::SeqCst);
                }
            });
        }
    });
}

fn run_store_on_sux_vec(storage: &Arc<Vec<AtomicU64>>, num_threads: usize, indices: &[usize]) {
    let barrier = Arc::new(Barrier::new(num_threads));
    let chunks: Vec<_> = indices.chunks(OPS_PER_THREAD).collect();

    thread::scope(|s| {
        for (thread_id, chunk) in chunks.iter().enumerate() {
            let storage_clone = Arc::clone(storage);
            let barrier_clone = Arc::clone(&barrier);
            s.spawn(move || {
                let sux_vec = unsafe {
                    AtomicBitFieldVec::<u64, _>::from_raw_parts(
                        storage_clone.as_slice(),
                        BIT_WIDTH,
                        VECTOR_SIZE,
                    )
                };
                barrier_clone.wait();
                for &index in *chunk {
                    unsafe {
                        sux_vec.set_atomic_unchecked(index, thread_id as u64, Ordering::SeqCst);
                    }
                }
            });
        }
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    targets = benchmark_lock_free_scaling
}
criterion_main!(benches);
