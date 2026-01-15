// benches/seq/bench_seq_readers.rs
//
// Benchmarks for SeqVec reader methods and API variants.
//
// Measures:
// 1. API methods: get() vs get_vec() vs get_into()
// 2. Buffer reuse benefit
// 3. Sequence length impact on method overhead

use compressed_intvec::seq::{LESeqVec, SeqVec, VariableCodecSpec};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;

const NUM_SEQUENCES: usize = 50_000;
const NUM_ACCESSES: usize = 20_000;

/// Generates sequences with a fixed length.
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

/// Generates sequential access indices.
fn generate_sequential_indices(num_accesses: usize, num_sequences: usize) -> Vec<usize> {
    (0..num_accesses).map(|i| i % num_sequences).collect()
}

/// Compares API methods with different sequence lengths.
fn benchmark_api_methods(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    // Short sequences: high relative overhead. Long sequences: decode dominates.
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

        // Baseline: Vec<Vec<u32>>
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

        // get_vec() (allocates Vec per call)
        group.bench_function("get_vec", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    let seq = seqvec.get_vec(idx).unwrap();
                    for val in seq {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // get_into() with buffer reuse
        group.bench_function("get_into_reuse", |b| {
            b.iter(|| {
                let mut buffer = Vec::with_capacity(seq_len);
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    seqvec.get_into(idx, &mut buffer).unwrap();
                    for &val in &buffer {
                        sum += val as u64;
                    }
                }
                black_box(sum)
            })
        });

        // SeqVecSeqReader with get_into (stateful + buffer reuse)
        group.bench_function("seq_reader_get_into", |b| {
            b.iter(|| {
                let mut seq_reader = seqvec.seq_reader();
                let mut buffer = Vec::with_capacity(seq_len);
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    seq_reader.get_into(idx, &mut buffer).unwrap();
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

/// Measures buffer reuse benefit.
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

    // Allocate new Vec each time
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

    // Reuse buffer with get_into
    group.bench_function("Reuse_Buffer", |b| {
        b.iter(|| {
            let mut buffer = Vec::with_capacity(seq_len);
            let mut sum = 0u64;
            for &idx in black_box(&indices) {
                seqvec.get_into(idx, &mut buffer).unwrap();
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
                let mut seq_reader = seqvec.seq_reader();
                let mut buffer = Vec::with_capacity(seq_len);
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    seq_reader.get_into(idx, &mut buffer).unwrap();
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

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(5));
    targets = benchmark_api_methods, benchmark_buffer_reuse, benchmark_codec_read
}

criterion_main!(benches);
