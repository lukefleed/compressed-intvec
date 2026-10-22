//! Serialization of every vector with bincode 1.3.3, whose fixed-width integers make the format
//! cost close to a copy, so that the numbers show this crate's serialization and the
//! validation on load. The baseline `Vec<u64>` holds as many words as the vector's storage:
//! the cost of the format alone.
//!
//! Groups: `serialization/serialize`, `serialization/deserialize`; the throughput is in
//! serialized bytes.

use compressed_intvec::{AtomicFixedVec, FixedVec, SeqVec, VarVec};
use compressed_intvec_bench::{
    Code, Distribution, Lengths, Shape, VALUE_SEED, bench_build, criterion, uniform_word,
};
use criterion::{
    BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use serde::{Serialize, de::DeserializeOwned};
use std::sync::atomic::Ordering::Relaxed;

/// The elements of the `FixedVec`, the `AtomicFixedVec` and the `VarVec`; the `SeqVec` has
/// `LEN / 16` sequences.
const LEN: usize = 1 << 20;
/// The width of the `FixedVec` data.
const WIDTH: u32 = 21;

fn fixed() -> FixedVec<u64> {
    let mut vec = FixedVec::with_capacity(WIDTH, LEN).expect("width at most 64");
    vec.extend((0..LEN as u64).map(|i| uniform_word(VALUE_SEED, i, WIDTH)));
    vec
}

fn sum_fixed(vec: &FixedVec<u64>) -> u64 {
    vec.iter().sum()
}

fn sum_atomic(vec: &AtomicFixedVec<u64>) -> u64 {
    (0..vec.len()).map(|i| vec.load(i, Relaxed)).sum()
}

fn variable() -> VarVec<u32> {
    Code::Automatic.var_vec(&Distribution::PowerLaw.values(VALUE_SEED, LEN), 32)
}

fn sum_variable(vec: &VarVec<u32>) -> u64 {
    vec.iter().map(u64::from).sum()
}

fn seq() -> SeqVec<u32> {
    let sequences = Shape::Skewed.sequences(Distribution::PowerLaw, LEN / 16);
    Code::Automatic.seq_vec(&sequences, Lengths::Absent)
}

fn sum_seq(vec: &SeqVec<u32>) -> u64 {
    vec.iter().map(|s| s.map(u64::from).sum::<u64>()).sum()
}

fn words(len: usize) -> Vec<u64> {
    (0..len as u64)
        .map(|i| uniform_word(VALUE_SEED, i, 64))
        .collect()
}

fn sum_words(words: &[u64]) -> u64 {
    words.iter().fold(0, |s, &x| s.wrapping_add(x))
}

/// The two directions of serialization, one group each.
#[derive(Clone, Copy)]
enum Direction {
    Serialize,
    Deserialize,
}

/// Registers the serialization or the deserialization of the value that `build` returns;
/// `sum` is the checksum of its elements.
fn case<T: Serialize + DeserializeOwned>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    direction: Direction,
    name: &str,
    build: impl Fn() -> T + Copy,
    sum: impl Fn(&T) -> u64 + Copy,
) {
    let encode = |value: &T| bincode::serialize(value).expect("serialization into memory");
    let decode = |bytes: &Vec<u8>| bincode::deserialize::<T>(bytes).expect("a valid payload");
    group.throughput(Throughput::Bytes(encode(&build()).len() as u64));
    let expected = || sum(&build());
    match direction {
        Direction::Serialize => bench_build(
            group,
            "bincode::serialize",
            name,
            build,
            expected,
            encode,
            |bytes| sum(&decode(bytes)),
        ),
        Direction::Deserialize => bench_build(
            group,
            "bincode::deserialize",
            name,
            || encode(&build()),
            expected,
            decode,
            sum,
        ),
    }
}

fn serialization(c: &mut Criterion) {
    for (direction, group) in [
        (Direction::Serialize, "serialization/serialize"),
        (Direction::Deserialize, "serialization/deserialize"),
    ] {
        let mut group = c.benchmark_group(group);
        case(&mut group, direction, "FixedVec<u64>", fixed, sum_fixed);
        case(
            &mut group,
            direction,
            "AtomicFixedVec<u64>",
            || AtomicFixedVec::from(fixed()),
            sum_atomic,
        );
        case(
            &mut group,
            direction,
            "Vec<u64> (as many words as the FixedVec)",
            || words(fixed().into_parts().0.len()),
            |words: &Vec<u64>| sum_words(words),
        );
        case(&mut group, direction, "VarVec<u32>", variable, sum_variable);
        case(&mut group, direction, "SeqVec<u32>", seq, sum_seq);
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = serialization
}
criterion_main!(benches);
