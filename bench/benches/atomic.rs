//! `AtomicFixedVec` against a `Vec` of the smallest std atomic that holds the width, and
//! against sux's `AtomicBitFieldVec`, at width 16 (no element straddles two words) and 21
//! (about a third of the elements straddle).
//!
//! `AtomicFixedVec` has only checked accessors, and so does indexing a `Vec`. sux has unchecked
//! ones, which are its fastest, so its timed benchmarks call `get_atomic_unchecked` and
//! `set_atomic_unchecked` (the untimed checks use the checked accessors). Every side uses the
//! same ordering.
//!
//! Groups:
//! - `atomic/{load,store}/{relaxed,seqcst}`, `atomic/{compare_exchange,fetch_add}/relaxed`:
//!   one thread, random indices;
//! - `atomic/disjoint/{store,read_mostly}/{16,21}`, `atomic/false_sharing/store/{16,21}`,
//!   `atomic/true_sharing/{store,read_mostly}/{16,21}`: the parameter is the number of threads,
//!   among 1, 2, 4 and 8 those that the machine can run in parallel;
//! - `atomic/round`: the cost of starting and finishing a timed round with no work.

use compressed_intvec::AtomicFixedVec;
use compressed_intvec_bench::{
    ACCESSES, INDEX_SEED, VALUE_SEED, bench, bench_threads, criterion, indices, thread_counts,
    uniform_word,
};
use criterion::{
    BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use std::{
    hint::black_box,
    sync::atomic::{
        AtomicU16, AtomicU32, AtomicU64,
        Ordering::{self, Relaxed, SeqCst},
    },
};
use sux::{bits::AtomicBitFieldVec, traits::AtomicBitFieldSlice};

/// The widths: 16 divides the word, 21 straddles it.
const WIDTHS: [u32; 2] = [16, 21];
/// The length of the vectors of the single-thread groups: at most 168 KiB, within L2.
const LEN: usize = 1 << 16;
/// The length of the vectors of the contended groups.
const CONTENDED_LEN: usize = 1 << 20;
/// The elements that the threads share in `false_sharing` and `true_sharing`: 1024 elements of
/// 21 bits take 2688 bytes, which are 21 cache lines of 128 bytes or 42 of 64 bytes.
const SHARED: usize = 1024;
/// The operations of one thread in one timed round.
const OPS: usize = 1 << 16;

fn word(index: usize, width: u32) -> u64 {
    uniform_word(VALUE_SEED, index as u64, width)
}

fn build_afv(len: usize, width: u32) -> AtomicFixedVec<u64> {
    let vec = AtomicFixedVec::new(width, len).expect("width at most 64");
    for i in 0..len {
        vec.store(i, word(i, width), Relaxed);
    }
    vec
}

fn build_sux(len: usize, width: u32) -> AtomicBitFieldVec<Vec<AtomicU64>> {
    let vec = AtomicBitFieldVec::<Vec<AtomicU64>>::new(width as usize, len);
    for i in 0..len {
        vec.set_atomic(i, word(i, width), Relaxed);
    }
    vec
}

/// The operations of the std atomic baseline, on `u64` values.
trait StdAtomic: Sync + Send + 'static {
    const NAME: &'static str;
    fn new(value: u64) -> Self;
    fn load(&self, order: Ordering) -> u64;
    fn store(&self, value: u64, order: Ordering);
    fn compare_exchange(&self, current: u64, new: u64, order: Ordering) -> Result<u64, u64>;
    fn fetch_add(&self, value: u64, order: Ordering) -> u64;
}

macro_rules! std_atomic {
    ($atomic:ty, $int:ty) => {
        impl StdAtomic for $atomic {
            const NAME: &'static str = stringify!($atomic);
            fn new(value: u64) -> Self {
                <$atomic>::new(value as $int)
            }
            #[inline]
            fn load(&self, order: Ordering) -> u64 {
                <$atomic>::load(self, order).into()
            }
            #[inline]
            fn store(&self, value: u64, order: Ordering) {
                <$atomic>::store(self, value as $int, order)
            }
            #[inline]
            fn compare_exchange(
                &self,
                current: u64,
                new: u64,
                order: Ordering,
            ) -> Result<u64, u64> {
                <$atomic>::compare_exchange(self, current as $int, new as $int, order, order)
                    .map(Into::into)
                    .map_err(Into::into)
            }
            #[inline]
            fn fetch_add(&self, value: u64, order: Ordering) -> u64 {
                <$atomic>::fetch_add(self, value as $int, order).into()
            }
        }
    };
}
std_atomic!(AtomicU16, u16);
std_atomic!(AtomicU32, u32);

fn build_std<A: StdAtomic>(len: usize, width: u32) -> Vec<A> {
    (0..len).map(|i| A::new(word(i, width))).collect()
}

/// Calls `$f::<A>($args)` with `A` the smallest std atomic of at least `$width` bits.
macro_rules! by_atomic {
    ($width:expr, $f:ident($($arg:expr),* $(,)?)) => {
        match $width {
            0..=16 => $f::<AtomicU16>($($arg),*),
            _ => $f::<AtomicU32>($($arg),*),
        }
    };
}

type Group<'a> = BenchmarkGroup<'a, WallTime>;

fn sum(indices: &[u32], get: impl Fn(usize) -> u64) -> u64 {
    indices
        .iter()
        .fold(0u64, |s, &i| s.wrapping_add(get(i as usize)))
}

/// Sums `exchange(i, x)` over the pairs of an index and its value.
fn exchange_sum(indices: &[u32], values: &[u64], exchange: impl Fn(usize, u64) -> u64) -> u64 {
    indices
        .iter()
        .zip(values)
        .fold(0u64, |s, (&i, &x)| s.wrapping_add(exchange(i as usize, x)))
}

fn order_name(order: Ordering) -> &'static str {
    match order {
        Relaxed => "relaxed",
        SeqCst => "seqcst",
        _ => unreachable!("only Relaxed and SeqCst are measured"),
    }
}

fn load(c: &mut Criterion) {
    for order in [Relaxed, SeqCst] {
        let mut group = c.benchmark_group(format!("atomic/load/{}", order_name(order)));
        group.throughput(Throughput::Elements(ACCESSES as u64));
        for width in WIDTHS {
            let idx = indices(INDEX_SEED, ACCESSES, LEN);
            let expected = sum(&idx, |i| word(i, width));
            bench(
                &mut group,
                "AtomicFixedVec::load",
                width,
                || build_afv(LEN, width),
                || expected,
                |v| sum(&idx, |i| v.load(i, order)),
            );
            bench(
                &mut group,
                "sux::AtomicBitFieldVec::get_atomic_unchecked",
                width,
                || build_sux(LEN, width),
                || expected,
                // SAFETY: every index is below `LEN`.
                |v| sum(&idx, |i| unsafe { v.get_atomic_unchecked(i, order) }),
            );
            by_atomic!(width, std_load(&mut group, width, order, &idx, expected));
        }
        group.finish();
    }
}

fn std_load<A: StdAtomic>(
    group: &mut Group<'_>,
    width: u32,
    order: Ordering,
    idx: &[u32],
    expected: u64,
) {
    bench(
        group,
        &format!("Vec<{}>::load", A::NAME),
        width,
        || build_std::<A>(LEN, width),
        || expected,
        |v| sum(idx, |i| v[i].load(order)),
    );
}

fn store(c: &mut Criterion) {
    for order in [Relaxed, SeqCst] {
        let mut group = c.benchmark_group(format!("atomic/store/{}", order_name(order)));
        group.throughput(Throughput::Elements(ACCESSES as u64));
        for width in WIDTHS {
            let idx = indices(INDEX_SEED, ACCESSES, LEN);
            // Each store writes the value already there, so every pass leaves the same data.
            let values: Vec<u64> = idx.iter().map(|&i| word(i as usize, width)).collect();
            let s = Stores {
                width,
                idx: &idx,
                values: &values,
            };
            s.bench(
                &mut group,
                "AtomicFixedVec::store",
                || build_afv(LEN, width),
                |v, i, x| v.store(i, x, order),
                |v, i| v.load(i, Relaxed),
            );
            s.bench(
                &mut group,
                "sux::AtomicBitFieldVec::set_atomic_unchecked",
                || build_sux(LEN, width),
                // SAFETY: every index is below `LEN`, and every value fits the width.
                |v, i, x| unsafe { v.set_atomic_unchecked(i, x, order) },
                |v, i| v.get_atomic(i, Relaxed),
            );
            by_atomic!(width, std_store(&mut group, &s, order));
        }
        group.finish();
    }
}

/// One pass of stores at random indices, each writing the value already there.
struct Stores<'a> {
    width: u32,
    idx: &'a [u32],
    values: &'a [u64],
}

impl Stores<'_> {
    /// Registers a benchmark of one pass through `store`; before timing, one pass runs and every
    /// element is checked through `load`.
    fn bench<S>(
        &self,
        group: &mut Group<'_>,
        function: &str,
        build: impl FnOnce() -> S,
        store: impl Fn(&S, usize, u64),
        load: impl Fn(&S, usize) -> u64,
    ) {
        let pass = |v: &S| {
            for (&i, &x) in self.idx.iter().zip(self.values) {
                store(v, i as usize, x);
            }
        };
        let width = self.width;
        bench(
            group,
            function,
            width,
            || {
                let v = build();
                pass(&v);
                let wrong = (0..LEN).find(|&i| load(&v, i) != word(i, width));
                assert_eq!(wrong, None, "{function}: wrong element after the stores");
                v
            },
            || 0,
            |v| {
                pass(v);
                black_box(&*v);
                0
            },
        );
    }
}

fn std_store<A: StdAtomic>(group: &mut Group<'_>, s: &Stores<'_>, order: Ordering) {
    s.bench(
        group,
        &format!("Vec<{}>::store", A::NAME),
        || build_std::<A>(LEN, s.width),
        |v, i, x| v[i].store(x, order),
        |v, i| v[i].load(Relaxed),
    );
}

fn read_modify_write(c: &mut Criterion) {
    let mut cas = c.benchmark_group("atomic/compare_exchange/relaxed");
    cas.throughput(Throughput::Elements(ACCESSES as u64));
    for width in WIDTHS {
        let idx = indices(INDEX_SEED, ACCESSES, LEN);
        let expected = sum(&idx, |i| word(i, width));
        // The values, computed outside the timed loop, as in the store benchmark.
        let values: Vec<u64> = idx.iter().map(|&i| word(i as usize, width)).collect();
        // Each exchange replaces the value with itself, so it always succeeds.
        bench(
            &mut cas,
            "AtomicFixedVec::compare_exchange",
            width,
            || build_afv(LEN, width),
            || expected,
            |v| {
                exchange_sum(&idx, &values, |i, x| {
                    v.compare_exchange(i, x, x, Relaxed, Relaxed)
                        .expect("the element holds x")
                })
            },
        );
        by_atomic!(width, std_cas(&mut cas, width, &idx, &values, expected));
    }
    cas.finish();

    let mut add = c.benchmark_group("atomic/fetch_add/relaxed");
    add.throughput(Throughput::Elements(ACCESSES as u64));
    for width in WIDTHS {
        let idx = indices(INDEX_SEED, ACCESSES, LEN);
        let expected = sum(&idx, |i| word(i, width));
        // Adding a zero that the compiler cannot see keeps the data, and the read-modify-write.
        let zero = black_box(0u64);
        bench(
            &mut add,
            "AtomicFixedVec::fetch_add",
            width,
            || build_afv(LEN, width),
            || expected,
            |v| sum(&idx, |i| v.fetch_add(i, zero, Relaxed)),
        );
        by_atomic!(width, std_fetch_add(&mut add, width, &idx, expected, zero));
    }
    add.finish();
}

fn std_cas<A: StdAtomic>(
    group: &mut Group<'_>,
    width: u32,
    idx: &[u32],
    values: &[u64],
    expected: u64,
) {
    bench(
        group,
        &format!("Vec<{}>::compare_exchange", A::NAME),
        width,
        || build_std::<A>(LEN, width),
        || expected,
        |v| {
            exchange_sum(idx, values, |i, x| {
                v[i].compare_exchange(x, x, Relaxed)
                    .expect("the element holds x")
            })
        },
    );
}

fn std_fetch_add<A: StdAtomic>(
    group: &mut Group<'_>,
    width: u32,
    idx: &[u32],
    expected: u64,
    zero: u64,
) {
    bench(
        group,
        &format!("Vec<{}>::fetch_add", A::NAME),
        width,
        || build_std::<A>(LEN, width),
        || expected,
        |v| sum(idx, |i| v[i].fetch_add(zero, Relaxed)),
    );
}

/// Which elements the threads of a contended benchmark access.
#[derive(Debug, Clone, Copy)]
enum Pattern {
    /// Each thread accesses its own contiguous part of a 2^20-element vector: no sharing.
    Disjoint,
    /// Thread `t` of `n` accesses the elements `t, t + n, t + 2n, ...` of the first 1024: no
    /// element is shared, but every word and cache line is.
    FalseSharing,
    /// Every thread accesses any of the first 1024 elements.
    TrueSharing,
}

impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Pattern::Disjoint => "disjoint",
            Pattern::FalseSharing => "false_sharing",
            Pattern::TrueSharing => "true_sharing",
        }
    }

    /// The indices of thread `thread` of `threads`.
    fn indices(self, thread: usize, threads: usize) -> Vec<u32> {
        let seed = INDEX_SEED + thread as u64;
        match self {
            Pattern::Disjoint => {
                let part = CONTENDED_LEN / threads;
                let base = (thread * part) as u32;
                indices(seed, OPS, part)
                    .into_iter()
                    .map(|i| base + i)
                    .collect()
            }
            Pattern::FalseSharing => indices(seed, OPS, SHARED / threads)
                .into_iter()
                .map(|j| j * threads as u32 + thread as u32)
                .collect(),
            Pattern::TrueSharing => indices(seed, OPS, SHARED),
        }
    }
}

/// What each thread does with its indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    /// Every operation stores the value already there.
    Store,
    /// One operation in 16 stores the value already there, the others load.
    ReadMostly,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::Store => "store",
            Op::ReadMostly => "read_mostly",
        }
    }

    fn is_store(self, k: usize) -> bool {
        match self {
            Op::Store => true,
            Op::ReadMostly => k % 16 == 15,
        }
    }
}

/// One thread's work on any vector: `load` and `store` stand for the vector's operations.
fn work(
    op: Op,
    indices: &[u32],
    width: u32,
    load: impl Fn(usize) -> u64,
    store: impl Fn(usize, u64),
) -> u64 {
    let mut sum = 0u64;
    for (k, &i) in indices.iter().enumerate() {
        let i = i as usize;
        if op.is_store(k) {
            store(i, word(i, width));
        } else {
            sum = sum.wrapping_add(load(i));
        }
    }
    sum
}

/// The checksum of `work`: the sum of the loaded values.
fn expected_work(op: Op, indices: &[Vec<u32>], width: u32) -> u64 {
    indices
        .iter()
        .flat_map(|ix| ix.iter().enumerate())
        .filter(|&(k, _)| !op.is_store(k))
        .fold(0u64, |s, (_, &i)| s.wrapping_add(word(i as usize, width)))
}

fn contended(c: &mut Criterion) {
    let combinations = [
        (Pattern::Disjoint, Op::Store),
        (Pattern::Disjoint, Op::ReadMostly),
        (Pattern::FalseSharing, Op::Store),
        (Pattern::TrueSharing, Op::Store),
        (Pattern::TrueSharing, Op::ReadMostly),
    ];
    for (pattern, op) in combinations {
        for width in WIDTHS {
            let mut group =
                c.benchmark_group(format!("atomic/{}/{}/{width}", pattern.name(), op.name()));
            for threads in thread_counts() {
                group.throughput(Throughput::Elements((threads * OPS) as u64));
                let ix: Vec<Vec<u32>> = (0..threads).map(|t| pattern.indices(t, threads)).collect();
                let expected = expected_work(op, &ix, width);
                let check = |load: &dyn Fn(usize) -> u64, name: &str| {
                    let wrong = (0..CONTENDED_LEN).find(|&i| load(i) != word(i, width));
                    assert_eq!(wrong, None, "{name}: an element changed under contention");
                };

                let v = build_afv(CONTENDED_LEN, width);
                bench_threads(&mut group, "AtomicFixedVec", threads, expected, |t| {
                    work(
                        op,
                        &ix[t],
                        width,
                        |i| v.load(i, Relaxed),
                        |i, x| v.store(i, x, Relaxed),
                    )
                });
                check(&|i| v.load(i, Relaxed), "AtomicFixedVec");
                drop(v);

                let v = build_sux(CONTENDED_LEN, width);
                bench_threads(
                    &mut group,
                    "sux::AtomicBitFieldVec",
                    threads,
                    expected,
                    |t| {
                        work(
                            op,
                            &ix[t],
                            width,
                            // SAFETY: every index is below `CONTENDED_LEN`.
                            |i| unsafe { v.get_atomic_unchecked(i, Relaxed) },
                            // SAFETY: every index is below `CONTENDED_LEN`, and every value
                            // fits the width.
                            |i, x| unsafe { v.set_atomic_unchecked(i, x, Relaxed) },
                        )
                    },
                );
                check(&|i| v.get_atomic(i, Relaxed), "sux::AtomicBitFieldVec");
                drop(v);

                by_atomic!(
                    width,
                    std_contended(&mut group, op, width, threads, &ix, expected)
                );
            }
            group.finish();
        }
    }
}

fn std_contended<A: StdAtomic>(
    group: &mut Group<'_>,
    op: Op,
    width: u32,
    threads: usize,
    ix: &[Vec<u32>],
    expected: u64,
) {
    let v = build_std::<A>(CONTENDED_LEN, width);
    bench_threads(
        group,
        &format!("Vec<{}>", A::NAME),
        threads,
        expected,
        |t| {
            work(
                op,
                &ix[t],
                width,
                |i| v[i].load(Relaxed),
                |i, x| v[i].store(x, Relaxed),
            )
        },
    );
    let wrong = (0..CONTENDED_LEN).find(|&i| v[i].load(Relaxed) != word(i, width));
    assert_eq!(
        wrong,
        None,
        "{}: an element changed under contention",
        A::NAME
    );
}

fn round(c: &mut Criterion) {
    let mut group = c.benchmark_group("atomic/round");
    for threads in thread_counts() {
        bench_threads(&mut group, "empty round", threads, 0, |_| 0);
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = load, store, read_modify_write, contended, round
}
criterion_main!(benches);
