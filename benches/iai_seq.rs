use compressed_intvec::prelude::*;
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

const SEQ_COUNT: usize = 1_000;
const SEQ_LEN: usize = 20;

// --- Setup Functions ---

fn setup_seq_delta() -> LESeqVec<u64> {
    let sequences: Vec<Vec<u64>> = (0..SEQ_COUNT)
        .map(|i| {
            (0..SEQ_LEN)
                .map(|j| ((i * SEQ_LEN + j) as u64) % 1000)
                .collect()
        })
        .collect();
    SeqVec::builder()
        .codec(Codec::Delta)
        .build(&sequences)
        .unwrap()
}

fn setup_seq_with_indices() -> (LESeqVec<u64>, Vec<usize>) {
    let vec = setup_seq_delta();
    let indices: Vec<usize> = (0..100).map(|i| i * 10).collect();
    (vec, indices)
}

fn setup_seq_gamma() -> LESeqVec<u64> {
    let sequences: Vec<Vec<u64>> = (0..SEQ_COUNT)
        .map(|i| {
            (0..SEQ_LEN)
                .map(|j| ((i * SEQ_LEN + j) as u64) % 1000)
                .collect()
        })
        .collect();
    SeqVec::builder()
        .codec(Codec::Gamma)
        .build(&sequences)
        .unwrap()
}

fn setup_seq_gamma_with_indices() -> (LESeqVec<u64>, Vec<usize>) {
    let vec = setup_seq_gamma();
    let indices: Vec<usize> = (0..100).map(|i| i * 10).collect();
    (vec, indices)
}

// --- Benchmarks ---

#[library_benchmark]
#[bench::delta(setup_seq_with_indices())]
#[bench::gamma(setup_seq_gamma_with_indices())]
fn bench_decode_sequences(input: (LESeqVec<u64>, Vec<usize>)) -> u64 {
    let (vec, indices) = input;
    let mut sum = 0u64;
    for &i in &indices {
        if let Some(seq_iter) = vec.get(i) {
            for val in seq_iter {
                sum = sum.wrapping_add(val);
            }
        }
    }
    black_box(sum)
}

#[library_benchmark]
#[bench::delta(setup_seq_delta())]
#[bench::gamma(setup_seq_gamma())]
fn bench_iter_all(vec: LESeqVec<u64>) -> u64 {
    let mut sum = 0u64;
    for seq in vec.iter() {
        for val in seq {
            sum = sum.wrapping_add(val);
        }
    }
    black_box(sum)
}

// --- Groups and Main ---

library_benchmark_group!(
    name = seq_group;
    benchmarks = bench_decode_sequences, bench_iter_all
);

main!(library_benchmark_groups = seq_group);
