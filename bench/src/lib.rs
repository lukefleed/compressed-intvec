//! Data, baselines and measurement helpers shared by the benchmark targets of `compressed-intvec`
//! and by the `space` report.
//!
//! Every value and index is a pure function of a seed and a position (`uniform_word`,
//! `Distribution::value`), computed with a SplitMix64-style mix. The data therefore never
//! changes with a release of a random number crate or with the platform, a structure of any
//! size can be built without first materializing its values, and the expected result of a
//! benchmark is computed from the same functions.

use compressed_intvec::{Codec, SeqVec, VarVec};
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use std::{
    cell::OnceCell,
    fmt::Display,
    hint::{self, black_box},
    num::NonZeroUsize,
    sync::atomic::{
        AtomicBool, AtomicU64, AtomicUsize,
        Ordering::{Acquire, Relaxed, Release},
    },
    thread,
    time::{Duration, Instant},
};

/// The seed of the values of every structure.
pub const VALUE_SEED: u64 = 0x5EED_0001;
/// The seed of the indices of random accesses.
pub const INDEX_SEED: u64 = 0x5EED_0002;
/// The seed of the sequence lengths of `SeqVec` data.
pub const LENGTH_SEED: u64 = 0x5EED_0003;

/// The number of accesses that one iteration of an access benchmark performs on a structure
/// that fits in cache.
///
/// As `u32` indices they take 64 KiB, so with the data of a `cache` size class the working set
/// stays within a private L2, and the 16384 random outcomes of a branch are more than a
/// predictor memorizes.
pub const ACCESSES: usize = 1 << 14;

/// The number of accesses that one iteration performs on a `FixedVec` of the `dram` size class.
///
/// Every iteration replays the same indices, so the cache lines that they touch must not fit
/// in any cache, or every iteration after the first one hits: 2^20 random accesses touch up to
/// 2^20 distinct lines, 64 to 128 MiB, beyond any last-level cache. With 2^14 accesses, the
/// latency of `FixedVec::get` on this class measured 15 to 19 ns, the latency of L2 hits.
pub const DRAM_ACCESSES: usize = 1 << 20;

/// The number of accesses that one iteration performs on a `VarVec` or a `SeqVec` of the `dram`
/// size class, where each access also decodes up to a block or a sequence.
///
/// Each access touches a sample or an offset and two or three lines of the stream, so 2^18
/// accesses touch about as many lines as the structure has. `VarVec::get` with gamma on
/// `power_law` data measured 264 ns per access with 2^18 accesses and 266 ns with 2^20, and
/// the dependent chain 299 and 300 ns, so the class stays in DRAM at a quarter of the time.
pub const DRAM_DECODE_ACCESSES: usize = 1 << 18;

/// The criterion configuration of every target.
///
/// The timings are set here, at the level of `Criterion`, and never on a group, so that the
/// command line (`--warm-up-time`, `--measurement-time`, `--sample-size`) overrides them.
pub fn criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(50)
}

/// The SplitMix64 output function: a bijective mix of a 64-bit state.
#[inline]
pub fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The `index`-th 64-bit word of a SplitMix64-style stream: the mix of
/// `seed + index * 0x9E37_79B9_7F4A_7C15`.
#[inline]
pub fn random_word(seed: u64, index: u64) -> u64 {
    mix(seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15)))
}

/// The `index`-th word of the stream `seed`, uniform in `0..2^bits`.
///
/// `bits` may be anything from 0 to 64: at 64 every word is possible (the high bits are taken
/// by a shift, because the exclusive bound `1 << 64` does not fit a `u64`).
#[inline]
pub fn uniform_word(seed: u64, index: u64, bits: u32) -> u64 {
    assert!(bits <= 64, "a word has at most 64 bits");
    if bits == 0 {
        0
    } else {
        random_word(seed, index) >> (64 - bits)
    }
}

/// A uniform value in `(0, 1]`, from 53 bits of `word`.
fn unit(word: u64) -> f64 {
    ((word >> 11) + 1) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// `count` indices, uniform in `0..len`, from the stream `seed`.
///
/// # Panics
///
/// If `len` is 0 or does not fit in `u32`.
pub fn indices(seed: u64, count: usize, len: usize) -> Vec<u32> {
    let len = u64::try_from(len).expect("usize fits in u64");
    assert!(len > 0 && len <= 1 << 32, "indices are u32");
    // Multiply-shift maps 32 uniform bits onto 0..len without a division.
    (0..count as u64)
        .map(|k| ((uniform_word(seed, k, 32) * len) >> 32) as u32)
        .collect()
}

/// The values of a variable-length vector, chosen so that different codes win.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distribution {
    /// Uniform below `2^20`. Every value takes about 20 bits, so variable-length codes cannot
    /// beat a fixed width: the worst case for `VarVec`.
    Uniform,
    /// Geometric with mean 15 (`P(x) = (15/16)^x / 16`): the distribution for which Rice and
    /// Golomb codes are optimal.
    Geometric,
    /// A power law with `P(X >= x) = (x + 1)^(-1/3)`, clamped to `u32::MAX`: the distribution
    /// for which the zeta code with `k = 3` is close to optimal, as for the gaps of web graphs.
    /// Half of the values are below 7, and about one in 1600 is clamped.
    PowerLaw,
}

impl Distribution {
    /// Every distribution, in report order.
    pub const ALL: [Distribution; 3] = [
        Distribution::Uniform,
        Distribution::Geometric,
        Distribution::PowerLaw,
    ];

    /// The name used in benchmark identifiers and in the space report.
    pub fn name(self) -> &'static str {
        match self {
            Distribution::Uniform => "uniform",
            Distribution::Geometric => "geometric",
            Distribution::PowerLaw => "power_law",
        }
    }

    /// The `index`-th value of the stream `seed`.
    #[inline]
    pub fn value(self, seed: u64, index: u64) -> u32 {
        let word = random_word(seed, index);
        match self {
            Distribution::Uniform => (word >> 44) as u32,
            Distribution::Geometric => (unit(word).ln() / (15.0f64 / 16.0).ln()) as u32,
            // Inverse transform of P(X >= x) = (x + 1)^(-1/3); the cast saturates at u32::MAX.
            Distribution::PowerLaw => (unit(word).powi(-3) - 1.0) as u32,
        }
    }

    /// The first `len` values of the stream `seed`.
    pub fn values(self, seed: u64, len: usize) -> Vec<u32> {
        (0..len as u64).map(|i| self.value(seed, i)).collect()
    }
}

/// The lengths of the sequences of a `SeqVec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Lengths uniform in `0..16` (mean 7.5), as short adjacency lists.
    Short,
    /// Half of the lengths in `1..=5`, 35% in `5..=20`, 12% in `20..=100`, 3% in `100..=1000`
    /// (mean about 30): many short sequences and a few long ones, as the degrees of a graph.
    Skewed,
}

impl Shape {
    /// Every shape, in report order.
    pub const ALL: [Shape; 2] = [Shape::Short, Shape::Skewed];

    /// The name used in benchmark identifiers and in the space report.
    pub fn name(self) -> &'static str {
        match self {
            Shape::Short => "short",
            Shape::Skewed => "skewed",
        }
    }

    /// The length of sequence `index`.
    fn length(self, index: u64) -> usize {
        let word = random_word(LENGTH_SEED, index);
        let pick = |low: u64, high: u64| (low + (word >> 32) % (high - low + 1)) as usize;
        match self {
            Shape::Short => (word >> 60) as usize,
            Shape::Skewed => match ((word & 0xFFFF_FFFF) * 100) >> 32 {
                0..50 => pick(1, 5),
                50..85 => pick(5, 20),
                85..97 => pick(20, 100),
                _ => pick(100, 1000),
            },
        }
    }

    /// `count` sequences of this shape, with values from `distribution`.
    pub fn sequences(self, distribution: Distribution, count: usize) -> Vec<Vec<u32>> {
        let mut next = 0;
        (0..count as u64)
            .map(|index| {
                let start = next;
                next += self.length(index) as u64;
                (start..next)
                    .map(|i| distribution.value(VALUE_SEED, i))
                    .collect()
            })
            .collect()
    }
}

/// The codes of the `VarVec` and `SeqVec` benchmarks: the universal codes, which encode any
/// of the distributions in a reasonable number of bits (so no pair of code and distribution
/// needs to be skipped), and the automatic choice that a builder without a code makes, which
/// can pick codes that the universal ones do not cover, such as Rice, Golomb and pi, when they
/// take the fewest bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Code {
    /// Elias gamma.
    Gamma,
    /// Elias delta.
    Delta,
    /// Zeta with `k = 3`.
    Zeta3,
    /// Elias omega.
    Omega,
    /// VByte, least significant group first.
    VByteLe,
    /// No code chosen: the builder picks the one with the fewest bits.
    Automatic,
}

impl Code {
    /// Every code of the `VarVec` benchmarks, in report order.
    pub const ALL: [Code; 6] = [
        Code::Gamma,
        Code::Delta,
        Code::Zeta3,
        Code::Omega,
        Code::VByteLe,
        Code::Automatic,
    ];

    /// The codes of the `SeqVec` benchmarks, in report order.
    pub const SEQ: [Code; 5] = [
        Code::Gamma,
        Code::Delta,
        Code::Zeta3,
        Code::VByteLe,
        Code::Automatic,
    ];

    /// The name used in benchmark identifiers and in the space report.
    pub fn name(self) -> &'static str {
        match self {
            Code::Gamma => "Gamma",
            Code::Delta => "Delta",
            Code::Zeta3 => "Zeta3",
            Code::Omega => "Omega",
            Code::VByteLe => "VByteLe",
            Code::Automatic => "automatic",
        }
    }

    /// The chosen code, or `None` for the automatic choice.
    pub fn codec(self) -> Option<Codec> {
        match self {
            Code::Gamma => Some(Codec::Gamma),
            Code::Delta => Some(Codec::Delta),
            Code::Zeta3 => Some(Codec::Zeta { k: 3 }),
            Code::Omega => Some(Codec::Omega),
            Code::VByteLe => Some(Codec::VByteLe),
            Code::Automatic => None,
        }
    }

    /// A `VarVec` of `values` with this code and `sampling_rate`.
    pub fn var_vec(self, values: &[u32], sampling_rate: usize) -> VarVec<u32> {
        let builder = VarVec::builder()
            .sampling_rate(NonZeroUsize::new(sampling_rate).expect("a sampling rate is positive"));
        match self.codec() {
            Some(codec) => builder
                .codec(codec)
                .build(values)
                .expect("universal codes encode every u32"),
            None => builder.build(values),
        }
    }

    /// A `SeqVec` of `sequences` with this code.
    pub fn seq_vec(self, sequences: &[Vec<u32>], lengths: Lengths) -> SeqVec<u32> {
        let builder = SeqVec::builder();
        let builder = match lengths {
            Lengths::Absent => builder,
            Lengths::Stored => builder.store_lengths(),
        };
        match self.codec() {
            Some(codec) => builder
                .codec(codec)
                .build(sequences)
                .expect("universal codes encode every u32"),
            None => builder.build(sequences),
        }
    }
}

/// Whether a `SeqVec` stores the length of each sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lengths {
    /// A sequence ends where the next one starts.
    Absent,
    /// Each sequence has its length stored, so `count` takes constant time.
    Stored,
}

impl Lengths {
    /// Both, in report order.
    pub const ALL: [Lengths; 2] = [Lengths::Absent, Lengths::Stored];

    /// The suffix used in benchmark identifiers and in the space report.
    pub fn name(self) -> &'static str {
        match self {
            Lengths::Absent => "no lengths",
            Lengths::Stored => "stored lengths",
        }
    }
}

/// Sequences in one flat array of values and one array of offsets: the uncompressed baseline
/// of `SeqVec` that does not pay one allocation per sequence, as `Vec<Vec<u32>>` does.
#[derive(Debug, Clone)]
pub struct FlatSequences {
    values: Vec<u32>,
    /// `offsets[i]..offsets[i + 1]` is sequence `i`.
    offsets: Vec<usize>,
}

impl FlatSequences {
    /// Copies `sequences`.
    pub fn new(sequences: &[Vec<u32>]) -> Self {
        let mut offsets = Vec::with_capacity(sequences.len() + 1);
        offsets.push(0);
        let mut values = Vec::new();
        for sequence in sequences {
            values.extend_from_slice(sequence);
            offsets.push(values.len());
        }
        Self { values, offsets }
    }

    /// The number of sequences.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Whether there is no sequence.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sequence `index`.
    ///
    /// # Panics
    ///
    /// If `index >= len`.
    #[inline]
    pub fn get(&self, index: usize) -> &[u32] {
        &self.values[self.offsets[index]..self.offsets[index + 1]]
    }

    /// The bytes of the two arrays, counting their capacity.
    pub fn allocated_bytes(&self) -> usize {
        self.values.capacity() * size_of::<u32>() + self.offsets.capacity() * size_of::<usize>()
    }
}

/// How an access benchmark issues its accesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Independent accesses, which the processor overlaps: throughput.
    Random,
    /// Each access waits for the value of the previous one: latency.
    Chain,
}

impl Mode {
    /// Both modes, in report order.
    pub const ALL: [Mode; 2] = [Mode::Random, Mode::Chain];

    /// The name used in benchmark identifiers.
    pub fn name(self) -> &'static str {
        match self {
            Mode::Random => "random",
            Mode::Chain => "chain",
        }
    }

    /// Returns the wrapping sum of `get(i)` over `indices`.
    ///
    /// In `Chain` mode, each index is combined with the previous value through a zero that the
    /// compiler cannot see, so every access depends on the previous one and the sum is the same
    /// as in `Random` mode. The combination adds an `and` and an `xor` to the chain of each
    /// access.
    #[inline(always)]
    pub fn sum(self, indices: &[u32], mut get: impl FnMut(usize) -> u64) -> u64 {
        match self {
            Mode::Random => indices
                .iter()
                .fold(0u64, |sum, &i| sum.wrapping_add(get(i as usize))),
            Mode::Chain => {
                let zero = black_box(0u64);
                let mut previous = 0u64;
                let mut sum = 0u64;
                for &i in indices {
                    previous = get(i as usize ^ (previous & zero) as usize);
                    sum = sum.wrapping_add(previous);
                }
                sum
            }
        }
    }
}

/// Registers a benchmark whose state is built on its first run and dropped when it ends, and
/// whose routine returns a checksum.
///
/// Building lazily means a benchmark excluded by a filter builds nothing, and only one large
/// structure exists at a time. Before timing, the routine runs once and its checksum must equal
/// `expected()`, so that a wrong implementation or a wrong baseline cannot go unnoticed.
/// Criterion passes the returned checksum through `black_box`, so the work is used once per
/// iteration instead of once per element.
pub fn bench<S>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    function: &str,
    parameter: impl Display,
    setup: impl FnOnce() -> S,
    expected: impl FnOnce() -> u64,
    mut routine: impl FnMut(&mut S) -> u64,
) {
    let message = format!("{function}/{parameter}: wrong checksum");
    let id = BenchmarkId::new(function, parameter);
    let mut setup = Some((setup, expected));
    let mut state = None;
    group.bench_function(id, |b| {
        let state = state.get_or_insert_with(|| {
            let (setup, expected) = setup.take().expect("the state is built once");
            let mut state = setup();
            assert_eq!(routine(&mut state), expected(), "{message}");
            state
        });
        b.iter(|| routine(state));
    });
}

/// Registers a benchmark of a routine that builds a value from an input, as `bench` does,
/// except that the routine's output is dropped after the timer stops, and the checksum is
/// computed from the output by `checksum`, outside the timed region.
pub fn bench_build<I, O>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    function: &str,
    parameter: impl Display,
    input: impl FnOnce() -> I,
    expected: impl FnOnce() -> u64,
    mut build: impl FnMut(&I) -> O,
    checksum: impl Fn(&O) -> u64,
) {
    let message = format!("{function}/{parameter}: wrong checksum");
    let id = BenchmarkId::new(function, parameter);
    let mut setup = Some((input, expected));
    let mut state = None;
    group.bench_function(id, |b| {
        let input = state.get_or_insert_with(|| {
            let (input, expected) = setup.take().expect("the input is built once");
            let input = input();
            assert_eq!(checksum(&build(&input)), expected(), "{message}");
            input
        });
        b.iter_with_large_drop(|| build(input));
    });
}

/// A value computed once, on first use, and shared by the benchmarks of one group.
///
/// Expected checksums of large structures are computed this way, so that a filtered-out
/// benchmark does not pay for them.
pub struct Lazy<T>(OnceCell<T>);

impl<T: Copy> Lazy<T> {
    /// An empty cell.
    pub fn new() -> Self {
        Self(OnceCell::new())
    }

    /// The value, computed by `init` on the first call.
    pub fn get(&self, init: impl FnOnce() -> T) -> T {
        *self.0.get_or_init(init)
    }
}

impl<T: Copy> Default for Lazy<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Registers a benchmark of `threads` threads that each run `work(thread)` once per round.
///
/// The threads are spawned before the benchmark and joined after it, so thread creation is
/// never timed. Between rounds they spin on a round counter instead of sleeping, so that
/// starting a round costs a cache-line transfer, not a wake-up by the operating system; a
/// round is timed from the main thread's increment of that counter to the last thread's
/// completion. The cost of a round with no work is measured by the same routine with `work`
/// doing nothing (`atomic/round`). `work` returns a checksum; the sum over the threads of one
/// round must equal `expected` before timing starts.
pub fn bench_threads(
    group: &mut BenchmarkGroup<'_, WallTime>,
    function: &str,
    threads: usize,
    expected: u64,
    work: impl Fn(usize) -> u64 + Sync,
) {
    let round_counter = AtomicU64::new(0);
    let finished = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let sum = AtomicU64::new(0);
    thread::scope(|scope| {
        for thread in 0..threads {
            let (round_counter, finished, stop, sum, work) =
                (&round_counter, &finished, &stop, &sum, &work);
            scope.spawn(move || {
                let mut seen = 0;
                loop {
                    // Acquire: pairs with the Release increment of `round_counter` in `round`,
                    // so the thread sees `stop` and the reset of `finished` that come before it.
                    while round_counter.load(Acquire) == seen {
                        hint::spin_loop();
                    }
                    seen += 1;
                    if stop.load(Relaxed) {
                        return;
                    }
                    sum.fetch_add(work(thread), Relaxed);
                    // Release: pairs with the Acquire load in `round`, which then sees `sum`.
                    finished.fetch_add(1, Release);
                }
            });
        }
        let round = || {
            finished.store(0, Relaxed);
            let begin = Instant::now();
            round_counter.fetch_add(1, Release);
            while finished.load(Acquire) < threads {
                hint::spin_loop();
            }
            begin.elapsed()
        };
        round();
        assert_eq!(
            sum.swap(0, Relaxed),
            expected,
            "{function}/{threads}: wrong checksum"
        );
        group.bench_function(BenchmarkId::new(function, threads), |b| {
            b.iter_custom(|rounds| (0..rounds).map(|_| round()).sum());
        });
        stop.store(true, Relaxed);
        round_counter.fetch_add(1, Release);
    });
}

/// The thread counts of the contention benchmarks that this machine can run in parallel.
///
/// The list is fixed, so that the name of a benchmark does not depend on the machine; counts
/// above the available parallelism are skipped.
pub fn thread_counts() -> Vec<usize> {
    let available = thread::available_parallelism().map_or(1, usize::from);
    [1, 2, 4, 8]
        .into_iter()
        .filter(|&threads| threads <= available)
        .collect()
}
