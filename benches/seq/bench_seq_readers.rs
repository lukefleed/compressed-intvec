// benches/seq/bench_seq_readers.rs
//
// Benchmarks for SeqVec reader methods and API variants.
//
// Measures:
// 1. API methods: get() iteration vs decode_vec() vs decode_into()
// 2. Buffer reuse benefit: allocating each time vs reusing a buffer
// 3. Codec read performance: Gamma vs Delta vs Zeta3
// 4. store_lengths optimization: bit-position termination vs count-based termination

use compressed_intvec::seq::{LESeqVec, SeqVec, VariableCodecSpec};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;

const NUM_SEQUENCES: usize = 50_000;
const NUM_ACCESSES: usize = 20_000;

/// Generates sequences with a fixed length and uniform random values.
fn generate_fixed_length_sequences(
    rng: &mut SmallRng,
    num_sequences: usize,
    seq_length: usize,
) -> Vec<Vec<u32>> {
    let max_value = 10_000u32;
    (0..num_sequences)
        .map(|_| {
            (0..seq_length)
                .map(|_| rng.random_range(1..=max_value))
                .collect()
        })
        .collect()
}

/// Generates sequences with power-law length distribution.
///
/// This distribution is realistic for graph adjacency lists: many nodes have
/// few neighbors, few nodes have many neighbors.
fn generate_power_law_sequences(rng: &mut SmallRng, num_sequences: usize) -> Vec<Vec<u32>> {
    let max_value = 10_000u32;
    (0..num_sequences)
        .map(|_| {
            let r: f64 = rng.random();
            let len = if r < 0.5 {
                rng.random_range(1..=5)
            } else if r < 0.85 {
                rng.random_range(5..=20)
            } else if r < 0.97 {
                rng.random_range(20..=100)
            } else {
                rng.random_range(100..=500)
            };
            (0..len).map(|_| rng.random_range(1..=max_value)).collect()
        })
        .collect()
}

/// Generates sequential access indices (wrapping around).
fn generate_sequential_indices(num_accesses: usize, num_sequences: usize) -> Vec<usize> {
    (0..num_accesses).map(|i| i % num_sequences).collect()
}

/// Compares API methods with different sequence lengths.
///
/// Tests three access patterns:
/// - get() + iteration: Zero-allocation lazy decoding
/// - decode_vec(): Allocates a new Vec per call
/// - decode_into(): Reuses a buffer across calls
fn benchmark_api_methods(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    // Short sequences emphasize per-call overhead.
    // Long sequences emphasize decode throughput.
    let sequence_lengths = [5, 50];

    for &seq_len in &sequence_lengths {
        let sequences = generate_fixed_length_sequences(&mut rng, NUM_SEQUENCES, seq_len);

        let seqvec: LESeqVec<u32> = SeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .build(&sequences)
            .expect("Failed to build SeqVec");

        let indices = generate_sequential_indices(NUM_ACCESSES, NUM_SEQUENCES);
        let total_elements = (NUM_ACCESSES * seq_len) as u64;

        let mut group = c.benchmark_group(format!("SeqApiMethods/len_{}", seq_len));
        group.throughput(Throughput::Elements(total_elements));

        // Baseline: uncompressed Vec<Vec<u32>>
        group.bench_function("Baseline_VecVec", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    for &val in &sequences[idx] {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // get() + inline iteration (zero allocation per access)
        group.bench_function("get_iter", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    for val in seqvec.get(idx).unwrap() {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // decode_vec() (allocates Vec per call)
        group.bench_function("decode_vec", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    let seq = seqvec.decode_vec(idx).unwrap();
                    for val in seq {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // decode_into() with buffer reuse
        group.bench_function("decode_into_reuse", |b| {
            b.iter(|| {
                let mut buffer = Vec::with_capacity(seq_len);
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    seqvec.decode_into(idx, &mut buffer).unwrap();
                    for &val in &buffer {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        group.finish();
    }
}

/// Measures the benefit of buffer reuse vs fresh allocation.
fn benchmark_buffer_reuse(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let seq_len = 50;

    let sequences = generate_fixed_length_sequences(&mut rng, NUM_SEQUENCES, seq_len);

    let seqvec: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences)
        .expect("Failed to build SeqVec");

    let indices = generate_sequential_indices(NUM_ACCESSES, NUM_SEQUENCES);
    let total_elements = (NUM_ACCESSES * seq_len) as u64;

    let mut group = c.benchmark_group("SeqBufferReuse");
    group.throughput(Throughput::Elements(total_elements));

    // Allocate new Vec each time via collect()
    group.bench_function("Allocate_Each", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for &idx in black_box(&indices) {
                let seq: Vec<u32> = seqvec.get(idx).unwrap().collect();
                for val in seq {
                    sum += val as u64;
                }
            }
            black_box(sum)
        })
    });

    // Reuse buffer with decode_into
    group.bench_function("Reuse_Buffer", |b| {
        b.iter(|| {
            let mut buffer = Vec::with_capacity(seq_len);
            let mut sum = 0u64;
            for &idx in black_box(&indices) {
                seqvec.decode_into(idx, &mut buffer).unwrap();
                for &val in &buffer {
                    sum += val as u64;
                }
            }
            black_box(sum)
        })
    });

    group.finish();
}

/// Compares codecs on read performance.
fn benchmark_codec_read(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let seq_len = 30;

    let sequences = generate_fixed_length_sequences(&mut rng, NUM_SEQUENCES, seq_len);
    let indices = generate_sequential_indices(NUM_ACCESSES, NUM_SEQUENCES);
    let total_elements = (NUM_ACCESSES * seq_len) as u64;

    let codecs = [
        ("Gamma", VariableCodecSpec::Gamma),
        ("Delta", VariableCodecSpec::Delta),
        ("Zeta3", VariableCodecSpec::Zeta { k: Some(3) }),
    ];

    let mut group = c.benchmark_group("SeqCodecRead");
    group.throughput(Throughput::Elements(total_elements));

    for (codec_name, codec_spec) in codecs {
        let seqvec: LESeqVec<u32> = SeqVec::builder()
            .codec(codec_spec)
            .build(&sequences)
            .expect("Failed to build SeqVec");

        group.bench_function(codec_name, |b| {
            b.iter(|| {
                let mut buffer = Vec::with_capacity(seq_len);
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    seqvec.decode_into(idx, &mut buffer).unwrap();
                    for &val in &buffer {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });
    }

    group.finish();
}

/// Compares decode performance with and without stored sequence lengths.
///
/// This benchmark measures the impact of the `store_lengths(true)` builder option.
///
/// ## Low-Level Differences
///
/// **Without stored lengths** (default):
/// - Decode loop terminates via `while bit_pos() < end_bit`
/// - Each iteration calls `bit_pos()` which reads from the reader struct
/// - Buffer grows dynamically during decode (potential reallocations)
///
/// **With stored lengths**:
/// - Decode loop terminates via `for _ in 0..count`
/// - Simple register decrement and zero-compare
/// - `buf.reserve(count)` called before loop (single allocation check)
///
/// The benefit is most pronounced for:
/// - Short sequences (loop overhead is relatively larger)
/// - Variable-length sequences (reserve prevents unexpected reallocations)
fn benchmark_store_lengths(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    // Test with fixed-length sequences (warmup eliminates reallocation differences)
    {
        let seq_len = 20;
        let sequences = generate_fixed_length_sequences(&mut rng, NUM_SEQUENCES, seq_len);
        let indices = generate_sequential_indices(NUM_ACCESSES, NUM_SEQUENCES);
        let total_elements = (NUM_ACCESSES * seq_len) as u64;

        let seqvec_without: LESeqVec<u32> = SeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .store_lengths(false)
            .build(&sequences)
            .expect("Failed to build SeqVec");

        let seqvec_with: LESeqVec<u32> = SeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .store_lengths(true)
            .build(&sequences)
            .expect("Failed to build SeqVec");

        let mut group = c.benchmark_group("SeqStoreLengths/FixedLen20");
        group.throughput(Throughput::Elements(total_elements));

        // decode_into without stored lengths
        group.bench_function("decode_into_NoLengths", |b| {
            b.iter(|| {
                let mut buffer = Vec::with_capacity(seq_len);
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    seqvec_without.decode_into(idx, &mut buffer).unwrap();
                    for &val in &buffer {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // decode_into with stored lengths
        group.bench_function("decode_into_WithLengths", |b| {
            b.iter(|| {
                let mut buffer = Vec::with_capacity(seq_len);
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    seqvec_with.decode_into(idx, &mut buffer).unwrap();
                    for &val in &buffer {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // get() iterator without stored lengths
        group.bench_function("get_iter_NoLengths", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    for val in seqvec_without.get(idx).unwrap() {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // get() iterator with stored lengths
        // (iterator still uses end-bit check, but size_hint is exact)
        group.bench_function("get_iter_WithLengths", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    for val in seqvec_with.get(idx).unwrap() {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        group.finish();
    }

    // Test with power-law length distribution (reserve benefit is more visible)
    {
        let sequences = generate_power_law_sequences(&mut rng, NUM_SEQUENCES);
        let indices = generate_sequential_indices(NUM_ACCESSES, NUM_SEQUENCES);
        let total_elements: u64 = indices.iter().map(|&i| sequences[i].len() as u64).sum();

        let seqvec_without: LESeqVec<u32> = SeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .store_lengths(false)
            .build(&sequences)
            .expect("Failed to build SeqVec");

        let seqvec_with: LESeqVec<u32> = SeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .store_lengths(true)
            .build(&sequences)
            .expect("Failed to build SeqVec");

        let mut group = c.benchmark_group("SeqStoreLengths/PowerLaw");
        group.throughput(Throughput::Elements(total_elements));

        // decode_into without stored lengths
        group.bench_function("decode_into_NoLengths", |b| {
            b.iter(|| {
                let mut buffer = Vec::with_capacity(64);
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    seqvec_without.decode_into(idx, &mut buffer).unwrap();
                    for &val in &buffer {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // decode_into with stored lengths
        group.bench_function("decode_into_WithLengths", |b| {
            b.iter(|| {
                let mut buffer = Vec::with_capacity(64);
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    seqvec_with.decode_into(idx, &mut buffer).unwrap();
                    for &val in &buffer {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // get().collect() without stored lengths
        // (collect uses size_hint for allocation)
        group.bench_function("get_collect_NoLengths", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    let seq: Vec<u32> = seqvec_without.get(idx).unwrap().collect();
                    for val in seq {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // get().collect() with stored lengths
        // (exact size_hint enables perfect allocation)
        group.bench_function("get_collect_WithLengths", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    let seq: Vec<u32> = seqvec_with.get(idx).unwrap().collect();
                    for val in seq {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets =
        benchmark_api_methods,
        benchmark_buffer_reuse,
        benchmark_codec_read,
        benchmark_store_lengths
}

criterion_main!(benches);
