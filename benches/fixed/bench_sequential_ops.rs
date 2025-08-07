use std::time::Duration;

// benches/fixed/bench_sequential_ops.rs
use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sux::prelude::{BitFieldSliceMut, BitFieldVec};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    if max_val_exclusive == 0 {
        return (0..size).map(|_| rng.random::<u64>()).collect();
    }
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

/// The main benchmark function for sequential operations.
fn benchmark_sequential_ops(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 5_000_000;

    let bit_widths_to_test: Vec<u32> = (8..64).step_by(8).collect();

    for &bit_width in &bit_widths_to_test {
        let mut group = c.benchmark_group(format!("SequentialOps/{}bit", bit_width));
        group.sample_size(10);

        let data = generate_random_vec(VECTOR_SIZE, 1u64.wrapping_shl(bit_width));

        // --- Setup Compressed Vectors ---
        let le_fixed_vec = LEFixedVec::builder()
            .bit_width(BitWidth::Explicit(bit_width as usize))
            .build(&data)
            .unwrap();
        let sux_bfv = BitFieldVec::<u64>::from_slice(&data).unwrap();

        // --- 1. Sequential Iteration Benchmarks ---
        group.bench_function("Baseline_Vec<u64>/iter_sum", |b| {
            b.iter(|| {
                // The sum operation ensures the compiler cannot optimize away the loop.
                black_box(black_box(&data).iter().sum::<u64>());
            })
        });

        group.bench_function("LEFixedVec/iter_sum", |b| {
            b.iter(|| {
                black_box(black_box(&le_fixed_vec).iter().sum::<u64>());
            })
        });

        group.bench_function("sux::BitFieldVec/iter_sum", |b| {
            b.iter(|| {
                black_box(black_box(&sux_bfv).iter().sum::<u64>());
            })
        });


        // --- 2. Parallel Iteration Benchmarks ---
        #[cfg(feature = "parallel")]
        {
            group.bench_function("Baseline_Vec<u64>/par_iter_sum", |b| {
                b.iter(|| {
                    black_box(black_box(&data).par_iter().sum::<u64>());
                })
            });

            group.bench_function("LEFixedVec/par_iter_sum", |b| {
                b.iter(|| {
                    black_box(black_box(&le_fixed_vec).par_iter().sum::<u64>());
                })
            });
        }

        // --- 3. In-place Mapping Benchmarks ---
        let mask = if bit_width == 64 {
            u64::MAX
        } else {
            (1u64 << bit_width) - 1
        };
        // Define an inherently safe map_fn by applying the mask directly.
        let map_fn = |x: u64| (x ^ 0x5555_5555_5555_5555) & mask;

        group.bench_function("Baseline_Vec<u64>/map_in_place", |b| {
            b.iter_with_setup(
                || data.clone(),
                |mut d| {
                    d.iter_mut().for_each(|x| *x = map_fn(*x));
                    black_box(d);
                },
            );
        });

        group.bench_function("LEFixedVec/map_in_place_unchecked", |b| {
            b.iter_with_setup(
                || le_fixed_vec.clone(),
                |mut vec| {
                    unsafe { vec.map_in_place_unchecked(map_fn) };
                    black_box(vec);
                },
            );
        });

        group.bench_function("sux::BitFieldVec/map_in_place_unchecked", |b| {
            b.iter_with_setup(
                || sux_bfv.clone(),
                |mut vec| {
                    unsafe {
                        vec.apply_in_place_unchecked(map_fn);
                    }
                    black_box(vec);
                },
            );
        });

        group.finish();
    }
}


criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_secs(2));

    targets = benchmark_sequential_ops
}
criterion_main!(benches);