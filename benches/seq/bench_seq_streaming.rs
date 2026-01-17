// benches/seq/bench_seq_streaming.rs
//
// Dedicated benchmarks for SeqVec streaming access APIs.
//
// Measures:
// 1. get() iteration vs for_each() vs fold()
// 2. Baseline comparison against decode_into() with buffer reuse

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

/// Compares streaming APIs against iterator and buffer-based access.
fn benchmark_streaming_apis(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    // Short sequences emphasize overhead. Longer sequences emphasize decode.
    let sequence_lengths = [5, 50];

    for &seq_len in &sequence_lengths {
        let sequences = generate_fixed_length_sequences(&mut rng, NUM_SEQUENCES, seq_len);

        let seqvec: LESeqVec<u32> = SeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .build(&sequences)
            .expect("Failed to build SeqVec");

        let indices = generate_sequential_indices(NUM_ACCESSES, NUM_SEQUENCES);
        let total_elements = (NUM_ACCESSES * seq_len) as u64;

        let mut group = c.benchmark_group(format!("SeqStreaming/len_{}", seq_len));
        group.throughput(Throughput::Elements(total_elements));

        // get() + inline iteration
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

        // for_each() streaming callback
        group.bench_function("for_each", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    seqvec
                        .for_each(idx, |value| {
                            sum += value as u64;
                        })
                        .unwrap();
                }
                black_box(sum)
            })
        });

        // fold() streaming fold
        group.bench_function("fold", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in black_box(&indices) {
                    let local_sum = seqvec
                        .fold(idx, 0u64, |acc, value| acc + value as u64)
                        .unwrap();
                    sum += local_sum;
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

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(5));
    targets = benchmark_streaming_apis
}

criterion_main!(benches);
