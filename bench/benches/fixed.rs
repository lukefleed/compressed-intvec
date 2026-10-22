//! `FixedVec` against `Vec` of the smallest primitive that holds the width, and against sux's
//! `BitFieldVec`, all at the same bit width.
//!
//! Each `FixedVec` operation is compared with the fastest equivalent that sux offers. Reads go
//! through `BitFieldVecU`, which reads each element with one unaligned load, at the widths that
//! support it (up to 58, 60 and 64), and through `BitFieldVec` at the others. Scans and runs
//! use sux's unchecked iterators, which do not check for the end.
//!
//! Groups:
//! - `fixed/get/{cache,dram}/{random,chain}`: checked reads, as throughput and as latency;
//! - `fixed/get_unchecked/{cache,dram}/random`: unchecked reads;
//! - `fixed/set/{cache,dram}`, `fixed/set_unchecked/{cache,dram}`: random writes;
//! - `fixed/iter/{cache,dram}`: a full scan;
//! - `fixed/run/cache`: 64 consecutive elements from random starts, through a slice view or
//!   by skipping an iterator;
//! - `fixed/build`: construction from a slice and by pushing.
//!
//! The size classes fix the packed footprint, not the length, so that a curve over the widths
//! does not cross a cache level.

use compressed_intvec::FixedVec;
use compressed_intvec_bench::{
    ACCESSES, DRAM_ACCESSES, INDEX_SEED, Lazy, Mode, VALUE_SEED, bench, bench_build, criterion,
    indices, uniform_word,
};
use criterion::{
    BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use sux::{
    bits::{BitFieldVec, BitFieldVecU},
    traits::{IntoUncheckedIterator, TryIntoUnaligned, UncheckedIterator},
};
use value_traits::slices::{SliceByValue, SliceByValueMut};

/// The bit widths: 8, 16, 32 and 64 divide the word; 13, 21 and 47 straddle words and `FixedVec`
/// reads them with one unaligned load; 60 is the widest width that it reads that way; 61 reads
/// two words.
const WIDTHS: [u32; 9] = [8, 13, 16, 21, 32, 47, 60, 61, 64];

/// The seed of the values written by the `set` groups.
const WRITE_SEED: u64 = 0x5EED_0010;

/// The packed footprint of a vector.
#[derive(Debug, Clone, Copy)]
enum Size {
    /// 32 KiB: together with the 64 KiB of indices of one iteration, it fits a 128 KiB L1D
    /// (Apple M-series performance cores); on cores with a 48 KiB L1D the data stays in L2.
    Cache,
    /// 256 MiB: far beyond any last-level cache, so random accesses go to DRAM.
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

    /// The accesses of one iteration of the access groups.
    fn accesses(self) -> usize {
        match self {
            Size::Cache => ACCESSES,
            Size::Dram => DRAM_ACCESSES,
        }
    }

    /// The number of elements of `width` bits in the footprint.
    fn len(self, width: u32) -> usize {
        let bytes: usize = match self {
            Size::Cache => 32 << 10,
            Size::Dram => 256 << 20,
        };
        bytes * 8 / width as usize
    }
}

/// Element `index` of every vector of `width` bits.
fn word(index: usize, width: u32) -> u64 {
    uniform_word(VALUE_SEED, index as u64, width)
}

/// The value that the `set` groups write at `index`.
fn new_word(index: usize, width: u32) -> u64 {
    uniform_word(WRITE_SEED, index as u64, width)
}

fn sum_words(indices: &[u32], word: impl Fn(usize) -> u64) -> u64 {
    indices
        .iter()
        .fold(0u64, |sum, &i| sum.wrapping_add(word(i as usize)))
}

fn build_fixed(len: usize, width: u32) -> FixedVec<u64> {
    let mut vec = FixedVec::with_capacity(width, len).expect("width at most 64");
    vec.extend((0..len).map(|i| word(i, width)));
    vec
}

fn build_sux(len: usize, width: u32) -> BitFieldVec<Vec<u64>> {
    let mut vec = BitFieldVec::<Vec<u64>>::new(width as usize, len);
    for i in 0..len {
        vec.set_value(i, word(i, width));
    }
    vec
}

/// Whether sux reads elements of this width with one unaligned load.
fn sux_unaligned(width: u32) -> bool {
    width <= 58 || width == 60 || width == 64
}

fn build_sux_unaligned(len: usize, width: u32) -> BitFieldVecU<Box<[u64]>> {
    build_sux(len, width)
        .into_padded()
        .try_into_unaligned()
        .expect("the width supports unaligned reads")
}

/// The element of the `Vec` baseline: the smallest primitive that holds the width.
trait Element: Copy + Into<u64> + TryFrom<u64> + 'static {
    const NAME: &'static str;
    fn from_word(word: u64) -> Self {
        Self::try_from(word)
            .ok()
            .expect("the word fits the element type")
    }
}
impl Element for u8 {
    const NAME: &'static str = "u8";
}
impl Element for u16 {
    const NAME: &'static str = "u16";
}
impl Element for u32 {
    const NAME: &'static str = "u32";
}
impl Element for u64 {
    const NAME: &'static str = "u64";
}

fn build_vec<T: Element>(len: usize, width: u32) -> Vec<T> {
    (0..len).map(|i| T::from_word(word(i, width))).collect()
}

/// Calls `$f::<T>($args)` with `T` the smallest primitive of at least `$width` bits.
macro_rules! by_element {
    ($width:expr, $f:ident($($arg:expr),* $(,)?)) => {
        match $width {
            0..=8 => $f::<u8>($($arg),*),
            9..=16 => $f::<u16>($($arg),*),
            17..=32 => $f::<u32>($($arg),*),
            _ => $f::<u64>($($arg),*),
        }
    };
}

type Group<'a> = BenchmarkGroup<'a, WallTime>;

fn get(c: &mut Criterion) {
    for size in Size::ALL {
        for mode in Mode::ALL {
            let mut group = c.benchmark_group(format!("fixed/get/{}/{}", size.name(), mode.name()));
            group.throughput(Throughput::Elements(size.accesses() as u64));
            for width in WIDTHS {
                let len = size.len(width);
                let idx = indices(INDEX_SEED, size.accesses(), len);
                let expected = sum_words(&idx, |i| word(i, width));
                by_element!(width, vec_get(&mut group, mode, len, width, &idx, expected));
                bench(
                    &mut group,
                    "FixedVec::get",
                    width,
                    || build_fixed(len, width),
                    || expected,
                    |v| mode.sum(&idx, |i| v.get(i).expect("index in bounds")),
                );
                if sux_unaligned(width) {
                    bench(
                        &mut group,
                        "sux::BitFieldVecU::index_value",
                        width,
                        || build_sux_unaligned(len, width),
                        || expected,
                        |v| mode.sum(&idx, |i| v.index_value(i)),
                    );
                } else {
                    bench(
                        &mut group,
                        "sux::BitFieldVec::index_value",
                        width,
                        || build_sux(len, width),
                        || expected,
                        |v| mode.sum(&idx, |i| v.index_value(i)),
                    );
                }
            }
            group.finish();
        }
    }
}

fn vec_get<T: Element>(
    group: &mut Group<'_>,
    mode: Mode,
    len: usize,
    width: u32,
    idx: &[u32],
    expected: u64,
) {
    bench(
        group,
        &format!("Vec<{}>[i]", T::NAME),
        width,
        || build_vec::<T>(len, width),
        || expected,
        |v| mode.sum(idx, |i| v[i].into()),
    );
}

fn get_unchecked(c: &mut Criterion) {
    for size in Size::ALL {
        let mut group = c.benchmark_group(format!("fixed/get_unchecked/{}/random", size.name()));
        group.throughput(Throughput::Elements(size.accesses() as u64));
        for width in WIDTHS {
            let len = size.len(width);
            let idx = indices(INDEX_SEED, size.accesses(), len);
            let expected = sum_words(&idx, |i| word(i, width));
            by_element!(
                width,
                vec_get_unchecked(&mut group, len, width, &idx, expected)
            );
            bench(
                &mut group,
                "FixedVec::get_unchecked",
                width,
                || build_fixed(len, width),
                || expected,
                // SAFETY: every index is below `len`.
                |v| Mode::Random.sum(&idx, |i| unsafe { v.get_unchecked(i) }),
            );
            if sux_unaligned(width) {
                bench(
                    &mut group,
                    "sux::BitFieldVecU::get_value_unchecked",
                    width,
                    || build_sux_unaligned(len, width),
                    || expected,
                    // SAFETY: every index is below `len`.
                    |v| Mode::Random.sum(&idx, |i| unsafe { v.get_value_unchecked(i) }),
                );
            } else {
                bench(
                    &mut group,
                    "sux::BitFieldVec::get_value_unchecked",
                    width,
                    || build_sux(len, width),
                    || expected,
                    // SAFETY: every index is below `len`.
                    |v| Mode::Random.sum(&idx, |i| unsafe { v.get_value_unchecked(i) }),
                );
            }
        }
        group.finish();
    }
}

fn vec_get_unchecked<T: Element>(
    group: &mut Group<'_>,
    len: usize,
    width: u32,
    idx: &[u32],
    expected: u64,
) {
    bench(
        group,
        &format!("Vec<{}>::get_unchecked", T::NAME),
        width,
        || build_vec::<T>(len, width),
        || expected,
        // SAFETY: every index is below `len`.
        |v| Mode::Random.sum(idx, |i| unsafe { *v.get_unchecked(i) }.into()),
    );
}

/// The indices and values of one pass of random writes.
struct Writes {
    indices: Vec<u32>,
    values: Vec<u64>,
    /// The sum of the elements at `indices` after the pass.
    expected: u64,
}

impl Writes {
    fn new(size: Size, width: u32) -> Self {
        let indices = indices(INDEX_SEED, size.accesses(), size.len(width));
        let values = indices
            .iter()
            .map(|&i| new_word(i as usize, width))
            .collect();
        let expected = sum_words(&indices, |i| new_word(i, width));
        Self {
            indices,
            values,
            expected,
        }
    }

    fn pairs(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.indices
            .iter()
            .zip(&self.values)
            .map(|(&i, &value)| (i as usize, value))
    }

    /// Builds a vector, writes once, and checks the written elements through `read`.
    fn checked<S>(
        &self,
        build: impl FnOnce() -> S,
        mut write: impl FnMut(&mut S),
        read: impl Fn(&S, usize) -> u64,
        name: &str,
    ) -> S {
        let mut vec = build();
        write(&mut vec);
        let sum = sum_words(&self.indices, |i| read(&vec, i));
        assert_eq!(
            sum, self.expected,
            "{name}: wrong elements after the writes"
        );
        vec
    }
}

fn set(c: &mut Criterion) {
    for size in Size::ALL {
        let mut group = c.benchmark_group(format!("fixed/set/{}", size.name()));
        group.throughput(Throughput::Elements(size.accesses() as u64));
        for width in WIDTHS {
            let len = size.len(width);
            let w = Writes::new(size, width);
            by_element!(width, vec_set(&mut group, len, width, &w));
            write_bench(
                &mut group,
                "FixedVec::set",
                width,
                &w,
                || build_fixed(len, width),
                |v, i, x| v.set(i, x),
                |v, i| v.get(i).expect("index in bounds"),
            );
            write_bench(
                &mut group,
                "sux::BitFieldVec::set_value",
                width,
                &w,
                || build_sux(len, width),
                |v, i, x| v.set_value(i, x),
                |v, i| v.index_value(i),
            );
        }
        group.finish();
    }
}

fn set_unchecked(c: &mut Criterion) {
    for size in Size::ALL {
        let mut group = c.benchmark_group(format!("fixed/set_unchecked/{}", size.name()));
        group.throughput(Throughput::Elements(size.accesses() as u64));
        for width in WIDTHS {
            let len = size.len(width);
            let w = Writes::new(size, width);
            by_element!(width, vec_set_unchecked(&mut group, len, width, &w));
            write_bench(
                &mut group,
                "FixedVec::set_unchecked (checks the value)",
                width,
                &w,
                || build_fixed(len, width),
                // SAFETY: every index is below `len`.
                |v, i, x| unsafe { v.set_unchecked(i, x) },
                |v, i| v.get(i).expect("index in bounds"),
            );
            write_bench(
                &mut group,
                "sux::BitFieldVec::set_value_unchecked",
                width,
                &w,
                || build_sux(len, width),
                // SAFETY: every index is below `len`, and every value fits the width.
                |v, i, x| unsafe { v.set_value_unchecked(i, x) },
                |v, i| v.index_value(i),
            );
        }
        group.finish();
    }
}

/// Registers a benchmark of one pass of `w` through `write`, checked once through `read`.
fn write_bench<S>(
    group: &mut Group<'_>,
    function: &str,
    width: u32,
    w: &Writes,
    build: impl FnOnce() -> S,
    write: impl Fn(&mut S, usize, u64),
    read: impl Fn(&S, usize) -> u64,
) {
    let pass = |v: &mut S| {
        for (i, x) in w.pairs() {
            write(v, i, x);
        }
    };
    bench(
        group,
        function,
        width,
        || w.checked(build, pass, read, function),
        || 0,
        |v| {
            pass(v);
            // The stores must happen: the vector escapes here.
            std::hint::black_box(&mut *v);
            0
        },
    );
}

fn vec_set<T: Element>(group: &mut Group<'_>, len: usize, width: u32, w: &Writes) {
    let values: Vec<T> = w.values.iter().map(|&x| T::from_word(x)).collect();
    vec_write(
        group,
        &format!("Vec<{}>[i] =", T::NAME),
        len,
        width,
        w,
        &values,
        |v, i, x| {
            v[i] = x;
        },
    );
}

fn vec_set_unchecked<T: Element>(group: &mut Group<'_>, len: usize, width: u32, w: &Writes) {
    let values: Vec<T> = w.values.iter().map(|&x| T::from_word(x)).collect();
    vec_write(
        group,
        &format!("Vec<{}>::get_unchecked_mut", T::NAME),
        len,
        width,
        w,
        &values,
        // SAFETY: every index is below `len`.
        |v, i, x| unsafe { *v.get_unchecked_mut(i) = x },
    );
}

fn vec_write<T: Element>(
    group: &mut Group<'_>,
    function: &str,
    len: usize,
    width: u32,
    w: &Writes,
    values: &[T],
    write: impl Fn(&mut Vec<T>, usize, T),
) {
    let pass = |v: &mut Vec<T>| {
        for (&i, &x) in w.indices.iter().zip(values) {
            write(v, i as usize, x);
        }
    };
    bench(
        group,
        function,
        width,
        || {
            w.checked(
                || build_vec::<T>(len, width),
                pass,
                |v, i| v[i].into(),
                function,
            )
        },
        || 0,
        |v| {
            pass(v);
            std::hint::black_box(&mut *v);
            0
        },
    );
}

fn iter(c: &mut Criterion) {
    for size in Size::ALL {
        let mut group = c.benchmark_group(format!("fixed/iter/{}", size.name()));
        for width in WIDTHS {
            let len = size.len(width);
            group.throughput(Throughput::Elements(len as u64));
            let total = Lazy::new();
            let expected =
                || total.get(|| (0..len).fold(0u64, |s, i| s.wrapping_add(word(i, width))));
            by_element!(width, vec_iter(&mut group, len, width, &expected));
            bench(
                &mut group,
                "FixedVec::iter",
                width,
                || build_fixed(len, width),
                expected,
                |v| v.iter().fold(0u64, u64::wrapping_add),
            );
            // The iterators of sux 0.14.0 shift by 64 at width 64 and return wrong values, so
            // the checksum would fail there.
            if width < 64 {
                bench(
                    &mut group,
                    "sux::BitFieldVec::into_unchecked_iter",
                    width,
                    || build_sux(len, width),
                    expected,
                    |v| {
                        let v = &*v;
                        let mut iter = v.into_unchecked_iter();
                        (0..len).fold(0u64, |sum, _| {
                            // SAFETY: the iterator starts at element 0 of `len` elements, and
                            // the loop takes `len` of them.
                            sum.wrapping_add(unsafe { iter.next_unchecked() })
                        })
                    },
                );
            }
        }
        group.finish();
    }
}

fn vec_iter<T: Element>(group: &mut Group<'_>, len: usize, width: u32, expected: &dyn Fn() -> u64) {
    bench(
        group,
        &format!("Vec<{}>::iter", T::NAME),
        width,
        || build_vec::<T>(len, width),
        expected,
        |v| v.iter().fold(0u64, |s, &x| s.wrapping_add(x.into())),
    );
}

/// The length of a run of consecutive elements in `fixed/run`.
const RUN: usize = 64;
/// The number of runs per iteration in `fixed/run`.
const RUNS: usize = 1024;

fn run(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed/run/cache");
    group.throughput(Throughput::Elements((RUN * RUNS) as u64));
    for width in [8, 21, 61] {
        let len = Size::Cache.len(width);
        let starts = indices(INDEX_SEED, RUNS, len - RUN);
        let expected = starts.iter().fold(0u64, |s, &start| {
            let start = start as usize;
            (start..start + RUN).fold(s, |s, i| s.wrapping_add(word(i, width)))
        });
        bench(
            &mut group,
            "Vec<u64>[a..b].iter",
            width,
            || build_vec::<u64>(len, width),
            || expected,
            |v| {
                starts.iter().fold(0u64, |s, &a| {
                    let a = a as usize;
                    v[a..a + RUN].iter().fold(s, |s, &x| s.wrapping_add(x))
                })
            },
        );
        bench(
            &mut group,
            "FixedVec::slice(a..b).iter",
            width,
            || build_fixed(len, width),
            || expected,
            |v| {
                starts.iter().fold(0u64, |s, &a| {
                    let a = a as usize;
                    let slice = v.slice(a..a + RUN).expect("range in bounds");
                    slice.iter().fold(s, u64::wrapping_add)
                })
            },
        );
        bench(
            &mut group,
            "FixedVec::iter().skip(a).take(n)",
            width,
            || build_fixed(len, width),
            || expected,
            |v| {
                starts.iter().fold(0u64, |s, &a| {
                    v.iter()
                        .skip(a as usize)
                        .take(RUN)
                        .fold(s, u64::wrapping_add)
                })
            },
        );
        bench(
            &mut group,
            "sux::BitFieldVec::into_unchecked_iter_from(a)",
            width,
            || build_sux(len, width),
            || expected,
            |v| {
                let v = &*v;
                starts.iter().fold(0u64, |s, &a| {
                    let mut iter = v.into_unchecked_iter_from(a as usize);
                    (0..RUN).fold(s, |s, _| {
                        // SAFETY: `a + RUN <= len`, so the iterator has `RUN` elements left.
                        s.wrapping_add(unsafe { iter.next_unchecked() })
                    })
                })
            },
        );
    }
    group.finish();
}

/// The length of the vectors of `fixed/build`.
const BUILD_LEN: usize = 1 << 20;

fn build(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed/build");
    group.throughput(Throughput::Elements(BUILD_LEN as u64));
    for width in [8, 21, 61] {
        let input = || (0..BUILD_LEN).map(|i| word(i, width)).collect::<Vec<u64>>();
        let total = Lazy::new();
        let expected =
            || total.get(|| (0..BUILD_LEN).fold(0u64, |s, i| s.wrapping_add(word(i, width))));
        let fixed_sum = |v: &FixedVec<u64>| {
            assert_eq!(v.bit_width(), width, "wrong bit width");
            v.iter().fold(0u64, u64::wrapping_add)
        };
        let sux_sum = |v: &BitFieldVec<Vec<u64>>| {
            assert_eq!(v.bit_width(), width as usize, "wrong bit width");
            (0..v.len()).fold(0u64, |s, i| s.wrapping_add(v.index_value(i)))
        };
        // Both choose the smallest width with a pass over the data, then write each value.
        bench_build(
            &mut group,
            "FixedVec::from_slice",
            width,
            input,
            expected,
            |data| FixedVec::from_slice(data),
            fixed_sum,
        );
        bench_build(
            &mut group,
            "sux::BitFieldVec::from_slice",
            width,
            input,
            expected,
            |data| BitFieldVec::<Vec<u64>>::from_slice(data).expect("width at most 64"),
            sux_sum,
        );
        // Growth from empty, one element at a time.
        bench_build(
            &mut group,
            "FixedVec::push",
            width,
            input,
            expected,
            |data| {
                let mut v = FixedVec::new(width).expect("width at most 64");
                for &x in data {
                    v.push(x);
                }
                v
            },
            fixed_sum,
        );
        bench_build(
            &mut group,
            "sux::BitFieldVec::push",
            width,
            input,
            expected,
            |data| {
                let mut v = BitFieldVec::<Vec<u64>>::with_capacity(width as usize, 0);
                for &x in data {
                    v.push(x);
                }
                v
            },
            sux_sum,
        );
        by_element!(width, vec_push(&mut group, width, &expected));
    }
    group.finish();
}

fn vec_push<T: Element>(group: &mut Group<'_>, width: u32, expected: &dyn Fn() -> u64) {
    bench_build(
        group,
        &format!("Vec<{}>::push", T::NAME),
        width,
        || build_vec::<T>(BUILD_LEN, width),
        expected,
        |data| {
            let mut v = Vec::new();
            for &x in data {
                v.push(x);
            }
            v
        },
        |v: &Vec<T>| v.iter().fold(0u64, |s, &x| s.wrapping_add(x.into())),
    );
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = get, get_unchecked, set, set_unchecked, iter, run, build
}
criterion_main!(benches);
