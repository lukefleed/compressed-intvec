//! # Micro-Benchmark for Checked vs. Unchecked Random Access
//!
//! This benchmark suite is designed to measure and compare the raw performance
//! of random element access across different fixed-width integer vector
//! implementations. It aims to answer key performance questions:
//!
//! 1.  **Implementation Overhead**: How does the access speed of `LEFixedVec`
//!     and `BEFixedVec` compare to a standard `Vec<u64>` (our baseline) and
//!     `sux::BitFieldVec` (a well-regarded external library)?
//!
//! 2.  **Checked vs. Unchecked**: What is the performance cost of bounds checking?
//!     This is measured by comparing `get()` against `get_unchecked()`.
//!
//! 3.  **Endianness Impact**: Is there a significant performance difference
//!     between the Little-Endian (`LE`) and Big-Endian (`BE`) implementations
//!     of `FixedVec`?
//!
//! 4.  **Bit-Width Optimization**: How does performance change with the number
//!     of bits used per integer? The benchmark specifically tests values that
//!     are powers of two (e.g., 8, 16, 32, 64), which should enable fast-path
//!     optimizations (bit-shifts), against values that are not (e.g., 7, 15, 31, 63),
//!     forcing fallback to slower arithmetic (division/modulo).
//!
//! ## Methodology
//!
//! - A large vector (`VECTOR_SIZE`) is created with uniformly random data.
//! - A significant number of random indices (`NUM_ACCESSES`) are pre-generated.
//! - For each `bit_width` in a predefined set, a benchmark group is created.
//! - Inside each group, every implementation is benchmarked for both checked
//!   and unchecked access patterns using the same set of random indices.

use std::time::Duration;

use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sux::prelude::{BitFieldSlice, BitFieldVec};

/// Generates a vector with uniformly random values.
fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    if max_val_exclusive == 0 {
        return vec![0; size];
    }
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

/// The main benchmark function.
fn benchmark_random_access(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 10_000_000;
    const NUM_ACCESSES: usize = 1_000_000;

    // A list of bit widths to test, designed to highlight the
    // performance impact of power-of-two optimizations.
    let bit_widths_to_test = [7, 8, 15, 16, 31, 32];

    // Pre-generate the random indices that will be used for all benchmarks.
    let mut rng = SmallRng::seed_from_u64(1337);
    let access_indices: Vec<usize> = (0..NUM_ACCESSES)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();

    // --- Parametric Benchmarks for different bit widths ---
    for &bit_width in &bit_widths_to_test {
        // --- Baseline: Standard Vec<u64> ---
        let baseline_data = generate_random_vec(VECTOR_SIZE, 1 << bit_width);
        let mut group =
            c.benchmark_group(format!("RandomAccess/{}bit_Baseline_Vec<u64>", bit_width));

        group.bench_function("Checked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(baseline_data.get(index));
                }
            })
        });
        group.bench_function("Unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    // SAFETY: Indices are generated within bounds for this benchmark.
                    black_box(unsafe { baseline_data.get_unchecked(index) });
                }
            })
        });
        group.finish();

        let mut group = c.benchmark_group(format!("RandomAccess/{}bit", bit_width));

        // Setup data and structures for this specific bit_width.
        let data = generate_random_vec(VECTOR_SIZE, 1 << bit_width.min(63));
        let le_fixed_vec = LEFixedVec::builder(&data)
            .bit_width(BitWidth::Explicit(bit_width))
            .build()
            .unwrap();
        let be_fixed_vec = BEFixedVec::builder(&data)
            .bit_width(BitWidth::Explicit(bit_width))
            .build()
            .unwrap();
        let sux_bfv = BitFieldVec::<u64>::from_slice(&data).unwrap();

        // 1. Our LEFixedVec
        group.bench_function("LEFixedVec/Checked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(le_fixed_vec.get(index));
                }
            })
        });
        group.bench_function("LEFixedVec/Unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(unsafe { le_fixed_vec.get_unchecked(index) });
                }
            })
        });

        // 2. Our BEFixedVec
        group.bench_function("BEFixedVec/Checked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(be_fixed_vec.get(index));
                }
            })
        });
        group.bench_function("BEFixedVec/Unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(unsafe { be_fixed_vec.get_unchecked(index) });
                }
            })
        });

        // 3. sux::BitFieldVec
        group.bench_function("sux::BitFieldVec/Checked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(sux_bfv.get(index));
                }
            })
        });
        group.bench_function("sux::BitFieldVec/Unchecked", |b| {
            b.iter(|| {
                for &index in black_box(&access_indices) {
                    black_box(unsafe { sux_bfv.get_unchecked(index) });
                }
            })
        });

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    targets = benchmark_random_access
}
criterion_main!(benches);
