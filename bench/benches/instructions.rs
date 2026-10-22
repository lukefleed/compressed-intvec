//! Instruction counts of the hot paths, with gungraun under Callgrind (Linux, Valgrind and
//! `gungraun-runner` 0.20.0). Counts do not vary between runs, so they detect regressions
//! smaller than the noise of a timing, and CI gates on them; wall-clock times never gate.
//!
//! Each benchmark receives its structure from a setup function, which is not counted, and
//! returns it, so that freeing it is not counted either. The teardown checks the result.

use compressed_intvec::{AtomicFixedVec, FixedVec, SeqVec, VarVec};
use compressed_intvec_bench::{
    Code, Distribution, INDEX_SEED, Lengths, Mode, Shape, VALUE_SEED, indices, uniform_word,
};
use gungraun::{library_benchmark, library_benchmark_group, main};
use std::{hint::black_box, sync::atomic::Ordering::Relaxed};

/// The elements of the `FixedVec`, `AtomicFixedVec` and `VarVec` structures; the `SeqVec` has
/// `LEN / 16` sequences.
const LEN: usize = 1 << 14;
/// The accesses of one benchmark.
const ACCESSES: usize = 1 << 10;
/// The seed of the values that the `set` benchmarks write.
const WRITE_SEED: u64 = 0x5EED_0010;

/// A structure, the indices and values of the operation, and its expected and actual result.
struct Case<S> {
    structure: S,
    indices: Vec<u32>,
    values: Vec<u64>,
    expected: u64,
    result: u64,
}

fn check<S>(case: Case<S>) {
    assert_eq!(case.result, case.expected, "wrong result");
}

fn word(index: usize, width: u32) -> u64 {
    uniform_word(VALUE_SEED, index as u64, width)
}

fn fixed(width: u32) -> Case<FixedVec<u64>> {
    let mut structure = FixedVec::with_capacity(width, LEN).expect("width at most 64");
    structure.extend((0..LEN).map(|i| word(i, width)));
    let indices = indices(INDEX_SEED, ACCESSES, LEN);
    let expected = indices
        .iter()
        .fold(0u64, |s, &i| s.wrapping_add(word(i as usize, width)));
    let values = indices
        .iter()
        .map(|&i| uniform_word(WRITE_SEED, i.into(), width))
        .collect();
    Case {
        structure,
        indices,
        values,
        expected,
        result: 0,
    }
}

fn fixed_all(width: u32) -> Case<FixedVec<u64>> {
    let mut case = fixed(width);
    case.expected = (0..LEN).fold(0u64, |s, i| s.wrapping_add(word(i, width)));
    case
}

fn fixed_written(width: u32) -> Case<FixedVec<u64>> {
    let mut case = fixed(width);
    case.expected = case.indices.iter().fold(0u64, |s, &i| {
        s.wrapping_add(uniform_word(WRITE_SEED, i.into(), width))
    });
    case
}

#[library_benchmark(setup = fixed, teardown = check)]
#[benches::width(args = [16, 21, 61])]
fn fixed_get(case: Case<FixedVec<u64>>) -> Case<FixedVec<u64>> {
    let mut case = black_box(case);
    let vec = &case.structure;
    case.result = Mode::Random.sum(&case.indices, |i| vec.get(i).expect("index in bounds"));
    case
}

#[library_benchmark(setup = fixed_written, teardown = check)]
#[benches::width(args = [16, 21])]
fn fixed_set(case: Case<FixedVec<u64>>) -> Case<FixedVec<u64>> {
    let mut case = black_box(case);
    for (&i, &x) in case.indices.iter().zip(&case.values) {
        case.structure.set(i as usize, x);
    }
    let vec = &case.structure;
    case.result = Mode::Random.sum(&case.indices, |i| vec.get(i).expect("index in bounds"));
    case
}

#[library_benchmark(setup = fixed_all, teardown = check)]
#[benches::width(args = [16, 21])]
fn fixed_iter(case: Case<FixedVec<u64>>) -> Case<FixedVec<u64>> {
    let mut case = black_box(case);
    case.result = case.structure.iter().fold(0u64, u64::wrapping_add);
    case
}

fn atomic(width: u32) -> Case<AtomicFixedVec<u64>> {
    let Case {
        structure,
        indices,
        values,
        expected,
        result,
    } = fixed(width);
    Case {
        structure: AtomicFixedVec::from(structure),
        indices,
        values,
        expected,
        result,
    }
}

#[library_benchmark(setup = atomic, teardown = check)]
#[benches::width(args = [16, 21])]
fn atomic_load(case: Case<AtomicFixedVec<u64>>) -> Case<AtomicFixedVec<u64>> {
    let mut case = black_box(case);
    let vec = &case.structure;
    case.result = Mode::Random.sum(&case.indices, |i| vec.load(i, Relaxed));
    case
}

#[library_benchmark(setup = atomic, teardown = check)]
#[benches::width(args = [16, 21])]
fn atomic_store(case: Case<AtomicFixedVec<u64>>) -> Case<AtomicFixedVec<u64>> {
    let mut case = black_box(case);
    let width = case.structure.bit_width();
    // Each store writes the value already there.
    for &i in &case.indices {
        case.structure
            .store(i as usize, word(i as usize, width), Relaxed);
    }
    let vec = &case.structure;
    case.result = Mode::Random.sum(&case.indices, |i| vec.load(i, Relaxed));
    case
}

fn var(code: Code) -> Case<VarVec<u32>> {
    let values = Distribution::PowerLaw.values(VALUE_SEED, LEN);
    let indices = indices(INDEX_SEED, ACCESSES, LEN);
    let expected = indices.iter().map(|&i| u64::from(values[i as usize])).sum();
    Case {
        structure: code.var_vec(&values, 32),
        indices,
        values: Vec::new(),
        expected,
        result: 0,
    }
}

fn var_all(code: Code) -> Case<VarVec<u32>> {
    let mut case = var(code);
    case.expected = Distribution::PowerLaw
        .values(VALUE_SEED, LEN)
        .iter()
        .map(|&x| u64::from(x))
        .sum();
    case
}

#[library_benchmark(setup = var, teardown = check)]
#[benches::code(args = [Code::Gamma, Code::Delta, Code::Zeta3, Code::VByteLe, Code::Automatic])]
fn var_get(case: Case<VarVec<u32>>) -> Case<VarVec<u32>> {
    let mut case = black_box(case);
    let vec = &case.structure;
    case.result = Mode::Random.sum(&case.indices, |i| {
        vec.get(i).expect("index in bounds").into()
    });
    case
}

#[library_benchmark(setup = var_all, teardown = check)]
#[benches::code(args = [Code::Gamma, Code::Delta, Code::Zeta3, Code::VByteLe, Code::Automatic])]
fn var_iter(case: Case<VarVec<u32>>) -> Case<VarVec<u32>> {
    let mut case = black_box(case);
    case.result = case.structure.iter().map(u64::from).sum();
    case
}

fn var_input(code: Code) -> (Code, Vec<u32>) {
    (code, Distribution::PowerLaw.values(VALUE_SEED, LEN))
}

#[library_benchmark]
#[benches::code(args = [Code::Gamma, Code::Automatic], setup = var_input)]
fn var_build(input: (Code, Vec<u32>)) -> (VarVec<u32>, Vec<u32>) {
    let (code, values) = black_box(input);
    let vec = code.var_vec(&values, 32);
    (vec, values)
}

fn seq(code: Code, lengths: Lengths) -> Case<SeqVec<u32>> {
    let sequences = Shape::Skewed.sequences(Distribution::PowerLaw, LEN / 16);
    let indices = indices(INDEX_SEED, ACCESSES, sequences.len());
    let expected = indices
        .iter()
        .flat_map(|&i| &sequences[i as usize])
        .map(|&x| u64::from(x))
        .sum();
    Case {
        structure: code.seq_vec(&sequences, lengths),
        indices,
        values: Vec::new(),
        expected,
        result: 0,
    }
}

#[library_benchmark(setup = seq, teardown = check)]
#[bench::gamma(Code::Gamma, Lengths::Absent)]
#[bench::gamma_lengths(Code::Gamma, Lengths::Stored)]
#[bench::vbyte(Code::VByteLe, Lengths::Absent)]
fn seq_get(case: Case<SeqVec<u32>>) -> Case<SeqVec<u32>> {
    let mut case = black_box(case);
    let vec = &case.structure;
    case.result = Mode::Random.sum(&case.indices, |i| {
        vec.get(i).expect("index in bounds").map(u64::from).sum()
    });
    case
}

library_benchmark_group!(
    name = fixed_group,
    benchmarks = [fixed_get, fixed_set, fixed_iter, atomic_load, atomic_store]
);
library_benchmark_group!(
    name = variable_group,
    benchmarks = [var_get, var_iter, var_build, seq_get]
);

main!(library_benchmark_groups = [fixed_group, variable_group]);
