// benches/fixed/bench_random_write.rs
use std::time::Duration;

use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use simple_sds_sbwt::{int_vector::IntVector as SdsIntVector, ops::Access};
use succinct::int_vec::{IntVecMut, IntVector as SuccinctIntVector};
use sux::prelude::BitFieldVec;
use value_traits::slices::SliceByValueMut;

/// Generates a vector with uniformly random values up to a given maximum.
fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

/// The main benchmark function for random write performance.
fn benchmark_random_write(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 10_000_000;
    const NUM_WRITES: usize = 1_000_000;

    // Test a range of bit widths.
    let bit_widths_to_test: Vec<u32> = (4..=64).step_by(4).collect();

    // Generate a single, consistent set of random indices for all benchmarks.
    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_WRITES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();

    for &bit_width in &bit_widths_to_test {
        let mut group = c.benchmark_group(format!("RandomWrite/{}bit", bit_width));
        group.throughput(Throughput::Elements(NUM_WRITES as u64));

        // Generate the initial data and the values to be written.
        let initial_data = generate_random_vec(VECTOR_SIZE, 1u64 << bit_width);
        let access_values = generate_random_vec(NUM_WRITES, 1u64 << bit_width);

        // --- 1. Baseline: Standard Vec with the smallest fitting type ---
        match bit_width {
            bw if bw <= 8 => {
                let baseline_data: Vec<u8> = initial_data.iter().map(|&v| v as u8).collect();
                let values_typed: Vec<u8> = access_values.iter().map(|&v| v as u8).collect();
                group.bench_function("Baseline_Vec<u8>/set_unchecked", |b| {
                    b.iter_batched(
                        || baseline_data.clone(),
                        |mut vec| {
                            for i in 0..NUM_WRITES {
                                unsafe {
                                    *vec.get_unchecked_mut(access_indices[i]) = values_typed[i]
                                };
                            }
                            black_box(vec);
                        },
                        criterion::BatchSize::LargeInput,
                    );
                });
            }
            bw if bw <= 16 => {
                let baseline_data: Vec<u16> = initial_data.iter().map(|&v| v as u16).collect();
                let values_typed: Vec<u16> = access_values.iter().map(|&v| v as u16).collect();
                group.bench_function("Baseline_Vec<u16>/set_unchecked", |b| {
                    b.iter_batched(
                        || baseline_data.clone(),
                        |mut vec| {
                            for i in 0..NUM_WRITES {
                                unsafe {
                                    *vec.get_unchecked_mut(access_indices[i]) = values_typed[i]
                                };
                            }
                            black_box(vec);
                        },
                        criterion::BatchSize::LargeInput,
                    );
                });
            }
            bw if bw <= 32 => {
                let baseline_data: Vec<u32> = initial_data.iter().map(|&v| v as u32).collect();
                let values_typed: Vec<u32> = access_values.iter().map(|&v| v as u32).collect();
                group.bench_function("Baseline_Vec<u32>/set_unchecked", |b| {
                    b.iter_batched(
                        || baseline_data.clone(),
                        |mut vec| {
                            for i in 0..NUM_WRITES {
                                unsafe {
                                    *vec.get_unchecked_mut(access_indices[i]) = values_typed[i]
                                };
                            }
                            black_box(vec);
                        },
                        criterion::BatchSize::LargeInput,
                    );
                });
            }
            _ => {
                let baseline_data: Vec<u64> = initial_data.to_vec();
                group.bench_function("Baseline_Vec<u64>/set_unchecked", |b| {
                    b.iter_batched(
                        || baseline_data.clone(),
                        |mut vec| {
                            for i in 0..NUM_WRITES {
                                unsafe {
                                    *vec.get_unchecked_mut(access_indices[i]) = access_values[i]
                                };
                            }
                            black_box(vec);
                        },
                        criterion::BatchSize::LargeInput,
                    );
                });
            }
        };

        // --- 2. Setup Compressed Vectors ---
        let le_fixed_vec = LEFixedVec::builder()
            .bit_width(BitWidth::Explicit(bit_width as usize))
            .build(&initial_data)
            .unwrap();

        let sux_bfv = BitFieldVec::<u64>::from_slice(&initial_data).unwrap();

        let mut succinct_iv = SuccinctIntVector::<u64>::new(bit_width as usize);
        succinct_iv.reserve_exact(initial_data.len().try_into().unwrap());
        for &val in &initial_data {
            succinct_iv.push(val);
        }

        let mut sds_iv = SdsIntVector::with_len(initial_data.len(), bit_width as usize, 0).unwrap();
        for (i, &val) in initial_data.iter().enumerate() {
            sds_iv.set(i, val);
        }

        // --- 3. Benchmark compressed-intvec ---
        group.bench_function("LEFixedVec/set_unchecked", |b| {
            b.iter_batched(
                || le_fixed_vec.clone(),
                |mut vec| {
                    for i in 0..NUM_WRITES {
                        unsafe { vec.set_unchecked(access_indices[i], access_values[i]) };
                    }
                    black_box(vec);
                },
                criterion::BatchSize::LargeInput,
            );
        });

        // --- 4. Benchmark sux::BitFieldVec ---
        group.bench_function("sux::BitFieldVec/set_unchecked", |b| {
            b.iter_batched(
                || sux_bfv.clone(),
                |mut vec| {
                    for i in 0..NUM_WRITES {
                        unsafe { vec.set_value_unchecked(access_indices[i], access_values[i]) };
                    }
                    black_box(vec);
                },
                criterion::BatchSize::LargeInput,
            );
        });

        // --- 5. Benchmark succinct::VarVector ---
        group.bench_function("succinct::VarVector/set", |b| {
            b.iter_batched(
                || succinct_iv.clone(),
                |mut vec| {
                    for i in 0..NUM_WRITES {
                        // `set` performs bounds checking.
                        vec.set(access_indices[i] as u64, access_values[i]);
                    }
                    black_box(vec);
                },
                criterion::BatchSize::LargeInput,
            );
        });

        // --- 6. Benchmark simple-sds-sbwt::VarVector ---
        group.bench_function("simple-sds-sbwt::VarVector/set", |b| {
            b.iter_batched(
                || sds_iv.clone(),
                |mut vec| {
                    for i in 0..NUM_WRITES {
                        // `set` performs bounds checking.
                        vec.set(access_indices[i], access_values[i]);
                    }
                    black_box(vec);
                },
                criterion::BatchSize::LargeInput,
            );
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

    targets = benchmark_random_write
}
criterion_main!(benches);
