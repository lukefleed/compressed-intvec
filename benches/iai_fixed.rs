use compressed_intvec::prelude::*;
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

const VECTOR_SIZE: usize = 10_000;
const ACCESS_COUNT: usize = 1_000;

// --- Setup Functions ---

fn setup_fixed_bw16() -> (LEFixedVec, Vec<usize>) {
    let data: Vec<u64> = (0..VECTOR_SIZE as u64).map(|i| i % (1 << 16)).collect();
    let vec = FixedVec::builder()
        .bit_width(BitWidth::Explicit(16))
        .build(&data)
        .unwrap();
    let indices: Vec<usize> = (0..ACCESS_COUNT).map(|i| i * 10).collect();
    (vec, indices)
}

fn setup_fixed_bw32() -> (LEFixedVec, Vec<usize>) {
    let data: Vec<u64> = (0..VECTOR_SIZE as u64).collect();
    let vec = FixedVec::builder()
        .bit_width(BitWidth::Explicit(32))
        .build(&data)
        .unwrap();
    let indices: Vec<usize> = (0..ACCESS_COUNT).map(|i| i * 10).collect();
    (vec, indices)
}

fn setup_fixed_iter_bw16() -> LEFixedVec {
    let data: Vec<u64> = (0..VECTOR_SIZE as u64).map(|i| i % (1 << 16)).collect();
    FixedVec::builder()
        .bit_width(BitWidth::Explicit(16))
        .build(&data)
        .unwrap()
}

fn setup_fixed_set_bw16() -> (LEFixedVec, Vec<usize>, Vec<u64>) {
    let data: Vec<u64> = (0..VECTOR_SIZE as u64).map(|i| i % (1 << 16)).collect();
    let vec = FixedVec::builder()
        .bit_width(BitWidth::Explicit(16))
        .build(&data)
        .unwrap();
    let indices: Vec<usize> = (0..ACCESS_COUNT).map(|i| i * 10).collect();
    let values: Vec<u64> = (0..ACCESS_COUNT as u64).map(|i| i % (1 << 16)).collect();
    (vec, indices, values)
}

// --- Benchmarks ---

#[library_benchmark]
#[bench::bw16(setup_fixed_bw16())]
#[bench::bw32(setup_fixed_bw32())]
fn bench_random_get(input: (LEFixedVec, Vec<usize>)) -> u64 {
    let (vec, indices) = input;
    let mut sum = 0u64;
    for &i in &indices {
        // SAFETY: all indices are in bounds (i < VECTOR_SIZE).
        sum = sum.wrapping_add(unsafe { vec.get_unchecked(i) });
    }
    black_box(sum)
}

#[library_benchmark]
#[bench::bw16(setup_fixed_iter_bw16())]
fn bench_iter_sum(vec: LEFixedVec) -> u64 {
    black_box(vec.iter().sum::<u64>())
}

#[library_benchmark]
#[bench::bw16(setup_fixed_set_bw16())]
fn bench_random_set(input: (LEFixedVec, Vec<usize>, Vec<u64>)) -> u64 {
    let (mut vec, indices, values) = input;
    for (&i, &v) in indices.iter().zip(values.iter()) {
        // SAFETY: all indices are in bounds and values fit in 16 bits.
        unsafe { vec.set_unchecked(i, v) };
    }
    // Read back one element to prevent the writes from being optimized away.
    black_box(unsafe { vec.get_unchecked(0) })
}

// --- Groups and Main ---

library_benchmark_group!(
    name = fixed_group;
    benchmarks = bench_random_get, bench_iter_sum, bench_random_set
);

main!(library_benchmark_groups = fixed_group);
