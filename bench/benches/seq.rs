//! `SeqVec` per code and shape, with and without stored lengths, against `Vec<Vec<u32>>` and
//! against `FlatSequences`, one flat array of values with an array of offsets: the uncompressed
//! layout that does not pay an allocation and a pointer per sequence.
//!
//! The values follow the power law of `Distribution::PowerLaw`, as the gaps of adjacency lists.
//!
//! Groups:
//! - `seq/get/{short,skewed}/{cache,dram}/{random,chain}`: the sum of a random sequence, as
//!   throughput and as latency (in `chain`, the next sequence depends on the previous sum, as
//!   in a graph traversal);
//! - `seq/scan/{short,skewed}`: every sequence in order;
//! - `seq/api/skewed`: the ways to consume a sequence (`sum`, `count`, extending a reused
//!   buffer, collecting a new `Vec`), with and without stored lengths;
//! - `seq/build/{short,skewed}`: construction;
//! - `seq/from_parts/{short,skewed}`: the validation of parts, on borrowed words.

use compressed_intvec::{Codec, FixedVec, SeqVec, SeqVecParts};
use compressed_intvec_bench::{
    ACCESSES, Code, DRAM_DECODE_ACCESSES, Distribution, FlatSequences, INDEX_SEED, Lengths, Mode,
    Shape, bench, bench_build, criterion, indices,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::cell::OnceCell;

/// The number of sequences of a vector.
#[derive(Debug, Clone, Copy)]
enum Size {
    /// 2^10 sequences: at most 100 KiB compressed, within L2.
    Cache,
    /// 2^21 sequences: 20 to 130 MiB compressed, beyond any last-level cache.
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
            Size::Cache => 1 << 10,
            Size::Dram => 1 << 21,
        }
    }
}

/// The number of sequences of the groups that have no size class.
const LEN: usize = 1 << 16;

fn sequences(shape: Shape, len: usize) -> Vec<Vec<u32>> {
    shape.sequences(Distribution::PowerLaw, len)
}

fn sum(values: &[u32]) -> u64 {
    values.iter().map(|&x| u64::from(x)).sum()
}

fn sum_all(sequences: &[Vec<u32>]) -> u64 {
    sequences.iter().map(|s| sum(s)).sum()
}

fn sum_at(sequences: &[Vec<u32>], indices: &[u32]) -> u64 {
    indices.iter().map(|&i| sum(&sequences[i as usize])).sum()
}

/// The sum of sequence `index` of `vec`.
#[inline]
fn seq_sum(vec: &SeqVec<u32>, index: usize) -> u64 {
    vec.get(index)
        .expect("index in bounds")
        .map(u64::from)
        .sum()
}

fn get(c: &mut Criterion) {
    for shape in Shape::ALL {
        for size in Size::ALL {
            let data = OnceCell::new();
            let data = || data.get_or_init(|| sequences(shape, size.len()));
            let idx = indices(INDEX_SEED, size.accesses(), size.len());
            for mode in Mode::ALL {
                let mut group = c.benchmark_group(format!(
                    "seq/get/{}/{}/{}",
                    shape.name(),
                    size.name(),
                    mode.name()
                ));
                group.throughput(Throughput::Elements(idx.len() as u64));
                let expected = || sum_at(data(), &idx);
                for code in Code::SEQ {
                    bench(
                        &mut group,
                        "SeqVec::get",
                        code.name(),
                        || code.seq_vec(data(), Lengths::Absent),
                        expected,
                        |v| mode.sum(&idx, |i| seq_sum(v, i)),
                    );
                }
                bench(
                    &mut group,
                    "Vec<Vec<u32>>[i]",
                    "-",
                    || data().clone(),
                    expected,
                    |v| mode.sum(&idx, |i| sum(&v[i])),
                );
                bench(
                    &mut group,
                    "FlatSequences::get",
                    "-",
                    || FlatSequences::new(data()),
                    expected,
                    |v| mode.sum(&idx, |i| sum(v.get(i))),
                );
                group.finish();
            }
        }
    }
}

fn scan(c: &mut Criterion) {
    for shape in Shape::ALL {
        let data = sequences(shape, LEN);
        let expected = || sum_all(&data);
        let mut group = c.benchmark_group(format!("seq/scan/{}", shape.name()));
        group.throughput(Throughput::Elements(LEN as u64));
        for lengths in Lengths::ALL {
            for code in Code::SEQ {
                bench(
                    &mut group,
                    &format!("SeqVec::iter ({})", lengths.name()),
                    code.name(),
                    || code.seq_vec(&data, lengths),
                    expected,
                    |v| v.iter().map(|s| s.map(u64::from).sum::<u64>()).sum(),
                );
            }
        }
        bench(
            &mut group,
            "Vec<Vec<u32>>::iter",
            "-",
            || data.clone(),
            expected,
            |v| sum_all(v),
        );
        bench(
            &mut group,
            "FlatSequences::get",
            "-",
            || FlatSequences::new(&data),
            expected,
            |v| (0..v.len()).map(|i| sum(v.get(i))).sum(),
        );
        group.finish();
    }
}

fn api(c: &mut Criterion) {
    let data = sequences(Shape::Skewed, LEN);
    let idx = indices(INDEX_SEED, ACCESSES, LEN);
    let total = sum_at(&data, &idx);
    let count = idx
        .iter()
        .map(|&i| data[i as usize].len() as u64)
        .sum::<u64>();
    let mut group = c.benchmark_group("seq/api/skewed");
    group.throughput(Throughput::Elements(ACCESSES as u64));
    let code = Code::Gamma;
    for lengths in Lengths::ALL {
        let build = || code.seq_vec(&data, lengths);
        bench(
            &mut group,
            "SeqVec::get(i).sum",
            lengths.name(),
            build,
            || total,
            |v| Mode::Random.sum(&idx, |i| seq_sum(v, i)),
        );
        bench(
            &mut group,
            "SeqVec::get(i).count",
            lengths.name(),
            build,
            || count,
            |v| Mode::Random.sum(&idx, |i| v.get(i).expect("index in bounds").count() as u64),
        );
        bench(
            &mut group,
            "Vec::extend(SeqVec::get(i)), reused",
            lengths.name(),
            || (build(), Vec::new()),
            || total,
            |(v, buffer)| {
                Mode::Random.sum(&idx, |i| {
                    buffer.clear();
                    buffer.extend(v.get(i).expect("index in bounds"));
                    sum(buffer)
                })
            },
        );
        bench(
            &mut group,
            "SeqVec::get(i).collect::<Vec<_>>",
            lengths.name(),
            build,
            || total,
            |v| {
                Mode::Random.sum(&idx, |i| {
                    let sequence: Vec<u32> = v.get(i).expect("index in bounds").collect();
                    sum(&sequence)
                })
            },
        );
    }
    bench(
        &mut group,
        "Vec<Vec<u32>>[i].iter().sum",
        "-",
        || data.clone(),
        || total,
        |v| Mode::Random.sum(&idx, |i| sum(&v[i])),
    );
    bench(
        &mut group,
        "Vec<Vec<u32>>[i].len",
        "-",
        || data.clone(),
        || count,
        |v| Mode::Random.sum(&idx, |i| v[i].len() as u64),
    );
    bench(
        &mut group,
        "Vec::extend_from_slice(FlatSequences::get(i)), reused",
        "-",
        || (FlatSequences::new(&data), Vec::new()),
        || total,
        |(v, buffer)| {
            Mode::Random.sum(&idx, |i| {
                buffer.clear();
                buffer.extend_from_slice(v.get(i));
                sum(buffer)
            })
        },
    );
    bench(
        &mut group,
        "FlatSequences::get(i).to_vec",
        "-",
        || FlatSequences::new(&data),
        || total,
        |v| {
            Mode::Random.sum(&idx, |i| {
                let sequence: Vec<u32> = v.get(i).to_vec();
                sum(&sequence)
            })
        },
    );
    group.finish();
}

fn build(c: &mut Criterion) {
    for shape in Shape::ALL {
        let data = OnceCell::new();
        let data = || data.get_or_init(|| sequences(shape, LEN)).clone();
        let expected = || sum_all(&data());
        let mut group = c.benchmark_group(format!("seq/build/{}", shape.name()));
        group.throughput(Throughput::Elements(LEN as u64));
        let checksum = |v: &SeqVec<u32>| v.iter().map(|s| s.map(u64::from).sum::<u64>()).sum();
        for code in Code::SEQ {
            bench_build(
                &mut group,
                "SeqVecBuilder::build (no lengths)",
                code.name(),
                data,
                expected,
                |d| code.seq_vec(d, Lengths::Absent),
                checksum,
            );
        }
        bench_build(
            &mut group,
            "SeqVecBuilder::build (stored lengths)",
            Code::Gamma.name(),
            data,
            expected,
            |d| Code::Gamma.seq_vec(d, Lengths::Stored),
            checksum,
        );
        bench_build(
            &mut group,
            "FlatSequences::new",
            "-",
            data,
            expected,
            |d| FlatSequences::new(d),
            |v| (0..v.len()).map(|i| sum(v.get(i))).sum(),
        );
        group.finish();
    }
}

/// The owned parts of a `SeqVec` without stored lengths, from which each iteration of
/// `seq/from_parts` checks a borrowed view.
struct OwnedParts {
    data: Vec<u64>,
    offsets: (Vec<u64>, usize, u32),
    codec: Codec,
}

impl OwnedParts {
    fn new(vec: SeqVec<u32>) -> Self {
        let SeqVecParts {
            data,
            offsets,
            lengths,
            codec,
        } = vec.into_parts();
        assert!(lengths.is_none(), "built without lengths");
        Self {
            data,
            offsets: offsets.into_parts(),
            codec,
        }
    }

    /// Checks the parts and returns the view: the routine of `seq/from_parts`.
    fn view(&self) -> SeqVec<u32, &[u64]> {
        let (words, count, width) = &self.offsets;
        SeqVec::from_parts(SeqVecParts {
            data: &self.data[..],
            offsets: FixedVec::from_parts(&words[..], *count, *width).expect("valid offsets"),
            lengths: None,
            codec: self.codec,
        })
        .expect("valid parts")
    }
}

fn from_parts(c: &mut Criterion) {
    for shape in Shape::ALL {
        let data = OnceCell::new();
        let data = || data.get_or_init(|| sequences(shape, LEN));
        let mut group = c.benchmark_group(format!("seq/from_parts/{}", shape.name()));
        group.throughput(Throughput::Elements(LEN as u64));
        for code in Code::SEQ {
            bench(
                &mut group,
                "SeqVec::from_parts (borrowed)",
                code.name(),
                || {
                    let parts = OwnedParts::new(code.seq_vec(data(), Lengths::Absent));
                    let read: u64 = parts
                        .view()
                        .iter()
                        .map(|s| s.map(u64::from).sum::<u64>())
                        .sum();
                    assert_eq!(read, sum_all(data()), "from_parts: wrong elements");
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
    targets = get, scan, api, build, from_parts
}
criterion_main!(benches);
