//! `VarVec` per code, distribution and sampling rate, against `Vec<u32>` and against
//! `FixedVec` and sux's `BitFieldVecU` (one unaligned load per element) at the smallest width
//! that holds the data, the same width for both.
//!
//! Groups:
//! - `variable/get/{uniform,geometric,power_law}/{cache,dram}/{random,chain}`: `get` at the
//!   default sampling rate, as throughput and as latency;
//! - `variable/sampling_rate/{cache,dram}`: `get` on `power_law` data for sampling rates from 4
//!   to 256, including 48, which is not a power of two;
//! - `variable/iter/{distribution}`: a full scan;
//! - `variable/run/power_law`: 64 consecutive elements from random starts, by skipping an
//!   iterator, which seeks;
//! - `variable/get_many/power_law`: sorted indices through `get_many` and through `get`;
//! - `variable/build/{distribution}`: construction with a chosen code and with the automatic
//!   choice;
//! - `variable/from_parts/{distribution}`: the validation of parts, on borrowed words, so that
//!   nothing is copied or allocated.

use compressed_intvec::{Codec, FixedVec, VarVec, VarVecParts};
use compressed_intvec_bench::{
    ACCESSES, Code, DRAM_DECODE_ACCESSES, Distribution, INDEX_SEED, Mode, VALUE_SEED, bench,
    bench_build, criterion, indices,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::{cell::OnceCell, num::NonZeroUsize};
use sux::{
    bits::{BitFieldVec, BitFieldVecU},
    traits::TryIntoUnaligned,
};
use value_traits::slices::{SliceByValue, SliceByValueMut};

/// The number of elements of a vector.
#[derive(Debug, Clone, Copy)]
enum Size {
    /// 2^15 elements: 20 to 90 KiB compressed, within L2.
    Cache,
    /// 2^26 elements: 60 to 200 MiB compressed, beyond any last-level cache.
    Dram,
}

impl Size {
    const ALL: [Size; 2] = [Size::Cache, Size::Dram];

    fn name(self) -> &'static str {
        match self {
            Size::Cache => "cache",
            Size::Dram => "dram",
        }
    }

    /// The accesses of one iteration of the `get` groups.
    fn accesses(self) -> usize {
        match self {
            Size::Cache => ACCESSES,
            Size::Dram => DRAM_DECODE_ACCESSES,
        }
    }

    fn len(self) -> usize {
        match self {
            Size::Cache => 1 << 15,
            Size::Dram => 1 << 26,
        }
    }
}

/// The default sampling rate of `VarVec`.
const SAMPLING_RATE: usize = 32;
/// The length of the vectors of the groups that have no size class.
const LEN: usize = 1 << 20;

fn sum(values: &[u32]) -> u64 {
    values.iter().map(|&x| u64::from(x)).sum()
}

fn sum_at(values: &[u32], indices: &[u32]) -> u64 {
    indices.iter().map(|&i| u64::from(values[i as usize])).sum()
}

fn build_sux(fixed: &FixedVec<u32>) -> BitFieldVecU<Box<[u64]>> {
    let mut vec = BitFieldVec::<Vec<u64>>::new(fixed.bit_width() as usize, fixed.len());
    for (i, x) in fixed.iter().enumerate() {
        vec.set_value(i, x.into());
    }
    vec.into_padded()
        .try_into_unaligned()
        .expect("a width of at most 32 supports unaligned reads")
}

fn get(c: &mut Criterion) {
    for distribution in Distribution::ALL {
        for size in Size::ALL {
            let values = OnceCell::new();
            let values = || values.get_or_init(|| distribution.values(VALUE_SEED, size.len()));
            let idx = indices(INDEX_SEED, size.accesses(), size.len());
            for mode in Mode::ALL {
                let mut group = c.benchmark_group(format!(
                    "variable/get/{}/{}/{}",
                    distribution.name(),
                    size.name(),
                    mode.name()
                ));
                group.throughput(Throughput::Elements(idx.len() as u64));
                let expected = || sum_at(values(), &idx);
                for code in Code::ALL {
                    bench(
                        &mut group,
                        "VarVec::get",
                        code.name(),
                        || code.var_vec(values(), SAMPLING_RATE),
                        expected,
                        |v| mode.sum(&idx, |i| v.get(i).expect("index in bounds").into()),
                    );
                }
                bench(
                    &mut group,
                    "Vec<u32>[i]",
                    "-",
                    || values().clone(),
                    expected,
                    |v| mode.sum(&idx, |i| v[i].into()),
                );
                bench(
                    &mut group,
                    "FixedVec::get",
                    "-",
                    || FixedVec::from_slice(values()),
                    expected,
                    |v| mode.sum(&idx, |i| v.get(i).expect("index in bounds").into()),
                );
                bench(
                    &mut group,
                    "sux::BitFieldVecU::index_value",
                    "-",
                    || build_sux(&FixedVec::from_slice(values())),
                    expected,
                    |v| mode.sum(&idx, |i| v.index_value(i)),
                );
                group.finish();
            }
        }
    }
}

fn sampling_rate(c: &mut Criterion) {
    let distribution = Distribution::PowerLaw;
    for size in Size::ALL {
        let values = OnceCell::new();
        let values = || values.get_or_init(|| distribution.values(VALUE_SEED, size.len()));
        let idx = indices(INDEX_SEED, size.accesses(), size.len());
        let expected = || sum_at(values(), &idx);
        let mut group = c.benchmark_group(format!("variable/sampling_rate/{}", size.name()));
        group.throughput(Throughput::Elements(idx.len() as u64));
        for code in [Code::Gamma, Code::Zeta3, Code::VByteLe] {
            for rate in [4, 8, 16, 32, 48, 64, 128, 256] {
                bench(
                    &mut group,
                    &format!("VarVec::get ({})", code.name()),
                    rate,
                    || code.var_vec(values(), rate),
                    expected,
                    |v| Mode::Random.sum(&idx, |i| v.get(i).expect("index in bounds").into()),
                );
            }
        }
        group.finish();
    }
}

fn iter(c: &mut Criterion) {
    for distribution in Distribution::ALL {
        let values = distribution.values(VALUE_SEED, LEN);
        let expected = || sum(&values);
        let mut group = c.benchmark_group(format!("variable/iter/{}", distribution.name()));
        group.throughput(Throughput::Elements(LEN as u64));
        for code in Code::ALL {
            bench(
                &mut group,
                "VarVec::iter",
                code.name(),
                || code.var_vec(&values, SAMPLING_RATE),
                expected,
                |v| v.iter().map(u64::from).sum(),
            );
        }
        bench(
            &mut group,
            "Vec<u32>::iter",
            "-",
            || values.clone(),
            expected,
            |v| v.iter().map(|&x| u64::from(x)).sum(),
        );
        bench(
            &mut group,
            "FixedVec::iter",
            "-",
            || FixedVec::from_slice(&values),
            expected,
            |v| v.iter().map(u64::from).sum(),
        );
        group.finish();
    }
}

/// The length of a run of consecutive elements in `variable/run`.
const RUN: usize = 64;
/// The number of runs per iteration in `variable/run`.
const RUNS: usize = 1024;

fn run(c: &mut Criterion) {
    let values = Distribution::PowerLaw.values(VALUE_SEED, LEN);
    let starts = indices(INDEX_SEED, RUNS, LEN - RUN);
    let expected = starts
        .iter()
        .map(|&a| sum(&values[a as usize..a as usize + RUN]))
        .sum::<u64>();
    let mut group = c.benchmark_group("variable/run/power_law");
    group.throughput(Throughput::Elements((RUN * RUNS) as u64));
    for code in Code::ALL {
        bench(
            &mut group,
            "VarVec::iter().skip(a).take(n)",
            code.name(),
            || code.var_vec(&values, SAMPLING_RATE),
            || expected,
            |v| {
                starts
                    .iter()
                    .map(|&a| {
                        v.iter()
                            .skip(a as usize)
                            .take(RUN)
                            .map(u64::from)
                            .sum::<u64>()
                    })
                    .sum()
            },
        );
    }
    bench(
        &mut group,
        "Vec<u32>[a..b].iter",
        "-",
        || values.clone(),
        || expected,
        |v| {
            starts
                .iter()
                .map(|&a| sum(&v[a as usize..a as usize + RUN]))
                .sum()
        },
    );
    bench(
        &mut group,
        "FixedVec::slice(a..b).iter",
        "-",
        || FixedVec::from_slice(&values),
        || expected,
        |v| {
            starts
                .iter()
                .map(|&a| {
                    let a = a as usize;
                    let slice = v.slice(a..a + RUN).expect("range in bounds");
                    slice.iter().map(u64::from).sum::<u64>()
                })
                .sum()
        },
    );
    group.finish();
}

fn get_many(c: &mut Criterion) {
    let values = Distribution::PowerLaw.values(VALUE_SEED, LEN);
    let mut sorted = indices(INDEX_SEED, ACCESSES, LEN);
    sorted.sort_unstable();
    let expected = sum_at(&values, &sorted);
    let mut group = c.benchmark_group("variable/get_many/power_law");
    group.throughput(Throughput::Elements(ACCESSES as u64));
    for code in Code::ALL {
        bench(
            &mut group,
            "VarVec::get_many (sorted indices)",
            code.name(),
            || code.var_vec(&values, SAMPLING_RATE),
            || expected,
            |v| {
                v.get_many(sorted.iter().map(|&i| i as usize))
                    .map(|x| u64::from(x.expect("index in bounds")))
                    .sum()
            },
        );
        bench(
            &mut group,
            "VarVec::get (sorted indices)",
            code.name(),
            || code.var_vec(&values, SAMPLING_RATE),
            || expected,
            |v| Mode::Random.sum(&sorted, |i| v.get(i).expect("index in bounds").into()),
        );
    }
    bench(
        &mut group,
        "Vec<u32>[i] (sorted indices)",
        "-",
        || values.clone(),
        || expected,
        |v| Mode::Random.sum(&sorted, |i| v[i].into()),
    );
    group.finish();
}

fn build(c: &mut Criterion) {
    for distribution in Distribution::ALL {
        let values = OnceCell::new();
        let values = || {
            values
                .get_or_init(|| distribution.values(VALUE_SEED, LEN))
                .clone()
        };
        let mut group = c.benchmark_group(format!("variable/build/{}", distribution.name()));
        group.throughput(Throughput::Elements(LEN as u64));
        let expected = || sum(&values());
        for code in Code::ALL {
            bench_build(
                &mut group,
                "VarVecBuilder::build",
                code.name(),
                values,
                expected,
                |data| code.var_vec(data, SAMPLING_RATE),
                |v| v.iter().map(u64::from).sum(),
            );
        }
        bench_build(
            &mut group,
            "FixedVec::from_slice",
            "-",
            values,
            expected,
            |data| FixedVec::from_slice(data),
            |v| v.iter().map(u64::from).sum(),
        );
        group.finish();
    }
}

/// The owned parts of a `VarVec`, from which each iteration of `variable/from_parts` checks a
/// borrowed view.
struct OwnedParts {
    data: Vec<u64>,
    samples: (Vec<u64>, usize, u32),
    sampling_rate: NonZeroUsize,
    len: usize,
    codec: Codec,
}

impl OwnedParts {
    fn new(vec: VarVec<u32>) -> Self {
        let VarVecParts {
            data,
            samples,
            sampling_rate,
            len,
            codec,
        } = vec.into_parts();
        Self {
            data,
            samples: samples.into_parts(),
            sampling_rate,
            len,
            codec,
        }
    }

    /// Checks the parts and returns the view: the routine of `variable/from_parts`.
    fn view(&self) -> VarVec<u32, &[u64]> {
        let (words, count, width) = &self.samples;
        VarVec::from_parts(VarVecParts {
            data: &self.data[..],
            samples: FixedVec::from_parts(&words[..], *count, *width).expect("valid samples"),
            sampling_rate: self.sampling_rate,
            len: self.len,
            codec: self.codec,
        })
        .expect("valid parts")
    }
}

fn from_parts(c: &mut Criterion) {
    for distribution in Distribution::ALL {
        let values = OnceCell::new();
        let values = || values.get_or_init(|| distribution.values(VALUE_SEED, LEN));
        let mut group = c.benchmark_group(format!("variable/from_parts/{}", distribution.name()));
        group.throughput(Throughput::Elements(LEN as u64));
        for code in Code::ALL {
            bench(
                &mut group,
                "VarVec::from_parts (borrowed)",
                code.name(),
                || {
                    let parts = OwnedParts::new(code.var_vec(values(), SAMPLING_RATE));
                    let read = parts.view().iter().map(u64::from).sum::<u64>();
                    assert_eq!(read, sum(values()), "from_parts: wrong elements");
                    parts
                },
                || LEN as u64,
                |parts| parts.view().len() as u64,
            );
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = get, sampling_rate, iter, run, get_many, build, from_parts
}
criterion_main!(benches);
