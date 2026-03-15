// benches/fixed/bench_random_access.rs
use std::time::Duration;

use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use simple_sds_sbwt::{int_vector::IntVector as SdsIntVector, ops::Access};
use succinct::int_vec::{IntVec, IntVector as SuccinctIntVector};
use sux::prelude::BitFieldVec;
use value_traits::slices::SliceByValue;

/// Generates a vector with uniformly random values up to a given maximum.
fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

/// The main benchmark function for random access performance.
fn benchmark_random_access(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 10_000_000;
    const NUM_ACCESSES: usize = 1_000_000;

    // Test a range of bit widths.
    let bit_widths_to_test: Vec<u32> = (4..=64).step_by(4).collect();

    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();

    for &bit_width in &bit_widths_to_test {
        let mut group = c.benchmark_group(format!("RandomAccess/{}bit", bit_width));
        group.throughput(Throughput::Elements(NUM_ACCESSES as u64));

        let data = if bit_width == 64 {
            let mut rng = SmallRng::seed_from_u64(42);
            (0..VECTOR_SIZE).map(|_| rng.random::<u64>()).collect()
        } else {
            generate_random_vec(VECTOR_SIZE, 1u64 << bit_width)
        };

        // --- 1. Baseline: Standard Vec with the smallest fitting type ---
        match bit_width {
            bw if bw <= 8 => {
                let baseline_data: Vec<u8> = data.iter().map(|&v| v as u8).collect();
                group.bench_function("Baseline_Vec<u8>/Unchecked", |b| {
                    b.iter(|| {
                        for &index in black_box(&access_indices) {
                            black_box(unsafe { baseline_data.get_unchecked(index) });
                        }
                    })
                });
            }
            bw if bw <= 16 => {
                let baseline_data: Vec<u16> = data.iter().map(|&v| v as u16).collect();
                group.bench_function("Baseline_Vec<u16>/Unchecked", |b| {
                    b.iter(|| {
                        for &index in black_box(&access_indices) {
                            black_box(unsafe { baseline_data.get_unchecked(index) });
                        }
                    })
                });
            }
            bw if bw <= 32 => {
                let baseline_data: Vec<u32> = data.iter().map(|&v| v as u32).collect();
                group.bench_function("Baseline_Vec<u32>/Unchecked", |b| {
                    b.iter(|| {
                        for &index in black_box(&access_indices) {
                            black_box(unsafe { baseline_data.get_unchecked(index) });
                        }
                    })
                });
            }
            _ => {
                group.bench_function("Baseline_Vec<u64>/Unchecked", |b| {
                    b.iter(|| {
                        for &index in black_box(&access_indices) {
                            black_box(unsafe { data.get_unchecked(index) });
                        }
                    })
                });
            }
        };

        // --- 2. Setup Compressed Vectors ---
        let le_fixed_vec = LEFixedVec::builder()
            .bit_width(BitWidth::Explicit(bit_width as usize))
            .build(&data)
            .unwrap();

        let sux_bfv = BitFieldVec::<u64>::from_slice(&data).unwrap();

        let mut succinct_iv = SuccinctIntVector::<u64>::new(bit_width as usize);
        succinct_iv.reserve_exact(data.len().try_into().unwrap());
        for &val in &data {
            succinct_iv.push(val);
        }

        // Setup for simple-sds-sbwt
        let mut sds_iv = SdsIntVector::with_len(data.len(), bit_width as usize, 0).unwrap();
        for (i, &val) in data.iter().enumerate() {
            sds_iv.set(i, val);
        }

        // --- 3. Benchmark compressed-intvec ---
        group.bench_function("LEFixedVec/Unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(unsafe { le_fixed_vec.get_unchecked(index) });
                }
            })
        });

        group.bench_function("LEFixedVec/Unaligned-Unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(unsafe { le_fixed_vec.get_unaligned_unchecked(index) });
                }
            })
        });

        // --- 4. Benchmark sux::BitFieldVec ---
        group.bench_function("sux::BitFieldVec/Unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(unsafe { sux_bfv.get_value_unchecked(index) });
                }
            })
        });

        group.bench_function("sux::BitFieldVec/Unaligned-Unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(unsafe { sux_bfv.get_unaligned_unchecked(index) });
                }
            })
        });

        // --- 5. Benchmark succinct::VarVector ---
        group.bench_function("succinct::VarVector/get", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(succinct_iv.get(index as u64));
                }
            })
        });

        // --- 6. Benchmark simple-sds-sbwt::VarVector ---
        group.bench_function("simple-sds-sbwt::VarVector/get", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // `get` performs bounds checking.
                    black_box(sds_iv.get(index));
                }
            })
        });

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(10));

    targets = benchmark_random_access
}
criterion_main!(benches);
