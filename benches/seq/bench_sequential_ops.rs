use compressed_intvec::{prelude::*, seq::LESeqVec};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Duration;

fn generate_sequences(num_sequences: usize, avg_len: usize, rng_seed: u64) -> Vec<Vec<u64>> {
    let mut rng = SmallRng::seed_from_u64(rng_seed);
    (0..num_sequences)
        .map(|_| {
            let seq_len = std::cmp::max(1, rng.random_range((avg_len / 2)..=(avg_len * 2)));
            (0..seq_len).map(|_| rng.random_range(0..100_000)).collect()
        })
        .collect()
}

fn benchmark_sequential_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("seq_sequential_ops");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(2));

    let num_sequences = 5_000;
    let sequences = generate_sequences(num_sequences, 100, 42);

    // Count total elements for throughput
    let total_elements: u64 = sequences.iter().map(|s| s.len() as u64).sum();
    group.throughput(Throughput::Elements(total_elements));

    let seq_vec = {
        let slice_refs: Vec<&[u64]> = sequences.iter().map(|s| s.as_slice()).collect();
        LESeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .build(&slice_refs)
            .expect("Failed to build SeqVec")
    };

    // 1. Full scan: iterate all sequences and sum all elements
    group.bench_function("iter_all_sum", |b| {
        b.iter(|| {
            let sum: u64 = black_box(&seq_vec)
                .iter()
                .flatten()
                .sum();
            black_box(sum);
        })
    });

    // 2. Full scan: iterate and count all sequences and elements
    group.bench_function("iter_all_count", |b| {
        b.iter(|| {
            let mut seq_count = 0usize;
            let mut elem_count = 0usize;
            for seq_iter in black_box(&seq_vec).iter() {
                seq_count += 1;
                elem_count += seq_iter.count();
            }
            black_box((seq_count, elem_count));
        })
    });

    // 3. Metadata-only scan: iterate without decoding elements
    group.bench_function("iter_sequence_count", |b| {
        b.iter(|| {
            let count = black_box(&seq_vec).iter().count();
            black_box(count);
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_secs(2));
    targets = benchmark_sequential_ops
}
criterion_main!(benches);
