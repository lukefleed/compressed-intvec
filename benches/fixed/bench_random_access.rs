// benches/fixed/bench_random_access.rs
use std::time::Duration;

use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sux::prelude::{BitFieldSlice, BitFieldSliceMut, BitFieldVec};

/// Generates a vector with uniformly random values up to a given maximum.
///
/// # Arguments
/// * `size` - The number of elements to generate.
/// * `max_val_exclusive` - The exclusive upper bound for the random values.
fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

/// The main benchmark function for random access performance.
///
/// This suite measures the speed of `get_unchecked` and `get_unaligned_unchecked`
/// for both our `FixedVec` and `sux::BitFieldVec`, comparing them against a `Vec<u64>` baseline.
fn benchmark_random_access(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 10_000_000;
    const NUM_ACCESSES: usize = 1_000_000;

    // Test a range of bit widths, including powers of two and others.
    let bit_widths_to_test: Vec<u32> = (8..=64).step_by(4).collect();

    // Pre-generate the random indices that will be used for all benchmarks to ensure
    // a fair comparison.
    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();

    for &bit_width in &bit_widths_to_test {
        let mut group = c.benchmark_group(format!("RandomAccess/{}bit", bit_width));

        // Generate a single data vector for this bit_width to be used by all structures.
        // The 64-bit case is handled explicitly to generate full-range u64 values.
        let data = if bit_width == 64 {
            let mut rng = SmallRng::seed_from_u64(42);
            (0..VECTOR_SIZE).map(|_| rng.random::<u64>()).collect()
        } else {
            generate_random_vec(VECTOR_SIZE, 1u64 << bit_width)
        };

        // --- 1. Baseline: Standard Vec<u64> ---
        group.bench_function("Baseline_Vec<u64>/Unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // SAFETY: Indices are generated within bounds for this benchmark.
                    black_box(unsafe { data.get_unchecked(index) });
                }
            })
        });

        // --- 2. Setup Compressed Vectors ---
        // Our library's vectors. The builder adds padding automatically.
        let le_fixed_vec = LEFixedVec::builder()
            .bit_width(BitWidth::Explicit(bit_width as usize))
            .build(&data)
            .unwrap();
        // let be_fixed_vec = BEFixedVec::builder()
        //     .bit_width(BitWidth::Explicit(bit_width as usize))
        //     .build(&data)
        //     .unwrap();

        // `from_slice` correctly infers the bit width from the data.
        let sux_bfv = BitFieldVec::<u64>::from_slice(&data).unwrap();
        assert_eq!(sux_bfv.bit_width(), le_fixed_vec.bit_width());

        // --- 3. Benchmark Our LEFixedVec ---
        group.bench_function("LEFixedVec/Unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // SAFETY: Indices are generated within bounds.
                    black_box(unsafe { le_fixed_vec.get_unchecked(index) });
                }
            })
        });

        group.bench_function("LEFixedVec/UnalignedUnchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // SAFETY: Indices are generated within bounds and builder adds padding.
                    black_box(unsafe { le_fixed_vec.get_unaligned_unchecked(index) });
                }
            })
        });

        // // --- 4. Benchmark Our BEFixedVec ---
        // group.bench_function("BEFixedVec/Unchecked", |b| {
        //     b.iter(|| {
        //         for &index in black_box(&access_indices) {
        //             // SAFETY: Indices are generated within bounds.
        //             black_box(unsafe { be_fixed_vec.get_unchecked(index) });
        //         }
        //     })
        // });

        // --- 5. Benchmark sux::BitFieldVec ---
        group.bench_function("sux::BitFieldVec/Unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // SAFETY: Indices are generated within bounds.
                    black_box(unsafe { sux_bfv.get_unchecked(index) });
                }
            })
        });

        // sux::BitFieldVec's unaligned access has specific constraints and requires padding.
        let w_bits = 64;
        let can_use_unaligned =
            bit_width <= w_bits - 8 + 2 || bit_width == w_bits - 8 + 4 || bit_width == w_bits;

        if can_use_unaligned {
            let mut sux_bfv_unaligned =
                sux::prelude::BitFieldVec::<u64>::new_unaligned(bit_width as usize, VECTOR_SIZE);
            for (i, &v) in data.iter().enumerate() {
                sux_bfv_unaligned.set(i, v);
            }
            group.bench_function("sux::BitFieldVec/UnalignedUnchecked", |b| {
                b.iter(|| {
                    for &index in black_box(&access_indices) {
                        // SAFETY: Indices are in bounds, vector was created with `new_unaligned`.
                        black_box(unsafe { sux_bfv_unaligned.get_unaligned_unchecked(index) });
                    }
                })
            });
        }

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_secs(2));

    targets = benchmark_random_access
}
criterion_main!(benches);
