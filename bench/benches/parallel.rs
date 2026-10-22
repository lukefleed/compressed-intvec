//! The parallel iterators of `FixedVec`, `VarVec` and `SeqVec` against their sequential
//! iterators on the same data, over sizes spaced by factors of 10, to find the size from which
//! the parallel one wins. The rayon pool is the global one, with as many threads as
//! `std::thread::available_parallelism` reports.
//!
//! Groups: `parallel/fixed`, `parallel/variable`, `parallel/seq`; the parameter is the number
//! of elements (of sequences for `seq`).

use compressed_intvec::FixedVec;
use compressed_intvec_bench::{
    Code, Distribution, Lengths, Shape, VALUE_SEED, bench, criterion, uniform_word,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rayon::prelude::*;

/// The width of the `FixedVec` data.
const WIDTH: u32 = 21;

fn fixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/fixed");
    for len in [1_000, 10_000, 100_000, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(len as u64));
        let build = || {
            let mut vec = FixedVec::<u64>::with_capacity(WIDTH, len).expect("width at most 64");
            vec.extend((0..len as u64).map(|i| uniform_word(VALUE_SEED, i, WIDTH)));
            vec
        };
        let expected = || {
            (0..len as u64)
                .map(|i| uniform_word(VALUE_SEED, i, WIDTH))
                .sum()
        };
        bench(&mut group, "FixedVec::iter", len, build, expected, |v| {
            v.iter().sum()
        });
        bench(
            &mut group,
            "FixedVec::par_iter",
            len,
            build,
            expected,
            |v| v.par_iter().sum(),
        );
    }
    group.finish();
}

fn variable(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/variable");
    for len in [1_000, 10_000, 100_000, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(len as u64));
        let values = Distribution::PowerLaw.values(VALUE_SEED, len);
        let build = || Code::Automatic.var_vec(&values, 32);
        let expected = || values.iter().map(|&x| u64::from(x)).sum();
        bench(&mut group, "VarVec::iter", len, build, expected, |v| {
            v.iter().map(u64::from).sum()
        });
        bench(&mut group, "VarVec::par_iter", len, build, expected, |v| {
            v.par_iter().map(u64::from).sum()
        });
    }
    group.finish();
}

fn seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel/seq");
    for len in [100, 1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(len as u64));
        let sequences = Shape::Skewed.sequences(Distribution::PowerLaw, len);
        let build = || Code::Automatic.seq_vec(&sequences, Lengths::Absent);
        let expected = || sequences.iter().flatten().map(|&x| u64::from(x)).sum();
        bench(&mut group, "SeqVec::iter", len, build, expected, |v| {
            v.iter().map(|s| s.map(u64::from).sum::<u64>()).sum()
        });
        bench(&mut group, "SeqVec::par_iter", len, build, expected, |v| {
            v.par_iter().map(|s| s.map(u64::from).sum::<u64>()).sum()
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = fixed, variable, seq
}
criterion_main!(benches);
