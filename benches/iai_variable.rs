use compressed_intvec::prelude::*;
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

const VECTOR_SIZE: usize = 10_000;
const ACCESS_COUNT: usize = 1_000;

// --- Setup Functions ---

fn setup_var_gamma() -> (LEVarVec, Vec<usize>) {
    let data: Vec<u64> = (0..VECTOR_SIZE as u64).map(|i| i % 1000).collect();
    let vec: LEVarVec = VarVec::builder()
        .k(32)
        .codec(Codec::Gamma)
        .build(&data)
        .unwrap();
    let indices: Vec<usize> = (0..ACCESS_COUNT).map(|i| i * 10).collect();
    (vec, indices)
}

fn setup_var_delta() -> (LEVarVec, Vec<usize>) {
    let data: Vec<u64> = (0..VECTOR_SIZE as u64).map(|i| i % 1000).collect();
    let vec: LEVarVec = VarVec::builder()
        .k(32)
        .codec(Codec::Delta)
        .build(&data)
        .unwrap();
    let indices: Vec<usize> = (0..ACCESS_COUNT).map(|i| i * 10).collect();
    (vec, indices)
}

fn setup_var_iter_gamma() -> LEVarVec {
    let data: Vec<u64> = (0..VECTOR_SIZE as u64).map(|i| i % 1000).collect();
    VarVec::builder()
        .k(32)
        .codec(Codec::Gamma)
        .build(&data)
        .unwrap()
}

fn setup_var_reader_gamma() -> (LEVarVec, Vec<usize>) {
    let data: Vec<u64> = (0..VECTOR_SIZE as u64).map(|i| i % 1000).collect();
    let vec: LEVarVec = VarVec::builder()
        .k(32)
        .codec(Codec::Gamma)
        .build(&data)
        .unwrap();
    let indices: Vec<usize> = (0..ACCESS_COUNT).map(|i| i * 10).collect();
    (vec, indices)
}

// --- Benchmarks ---

#[library_benchmark]
#[bench::gamma(setup_var_gamma())]
#[bench::delta(setup_var_delta())]
fn bench_random_get(input: (LEVarVec, Vec<usize>)) -> u64 {
    let (vec, indices) = input;
    let mut sum = 0u64;
    for &i in &indices {
        // SAFETY: all indices are in bounds (i < VECTOR_SIZE).
        sum = sum.wrapping_add(unsafe { vec.get_unchecked(i) });
    }
    black_box(sum)
}

#[library_benchmark]
#[bench::gamma(setup_var_iter_gamma())]
fn bench_iter_sum(vec: LEVarVec) -> u64 {
    black_box(vec.iter().sum::<u64>())
}

#[library_benchmark]
#[bench::gamma(setup_var_reader_gamma())]
fn bench_reader_get(input: (LEVarVec, Vec<usize>)) -> u64 {
    let (vec, indices) = input;
    let mut reader = vec.reader();
    let mut sum = 0u64;
    for &i in &indices {
        // SAFETY: all indices are in bounds (i < VECTOR_SIZE).
        sum = sum.wrapping_add(unsafe { reader.get_unchecked(i) });
    }
    black_box(sum)
}

// --- Groups and Main ---

library_benchmark_group!(
    name = variable_group;
    benchmarks = bench_random_get, bench_iter_sum, bench_reader_get
);

main!(library_benchmark_groups = variable_group);
