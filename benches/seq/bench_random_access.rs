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

fn benchmark_random_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("seq_random_access");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(2));

    // Build test dataset: 1000 sequences with varying average lengths
    let num_sequences = 1_000;
    let seq_vec = {
        let sequences = generate_sequences(num_sequences, 50, 42);
        let slice_refs: Vec<&[u64]> = sequences.iter().map(|s| s.as_slice()).collect();
        LESeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .build(&slice_refs)
            .expect("Failed to build SeqVec")
    };

    // Generate random access indices
    let access_indices = {
        let mut rng = SmallRng::seed_from_u64(123);
        (0..100)
            .map(|_| rng.random_range(0..num_sequences))
            .collect::<Vec<_>>()
    };

    let num_accesses = access_indices.len() as u64;
    group.throughput(Throughput::Elements(num_accesses));

    // 1. Fresh reader for each access (baseline)
    group.bench_function("fresh_reader_per_access", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for &idx in black_box(access_indices.iter()) {
                if let Some(seq_iter) = black_box(&seq_vec).get(idx) {
                    sum = sum.wrapping_add(seq_iter.count() as u64);
                }
            }
            black_box(sum);
        })
    });

    // 2. Reused stateless reader (still allocates on each get)
    group.bench_function("reused_reader_get", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            let reader = black_box(&seq_vec).reader();
            for &idx in black_box(access_indices.iter()) {
                if let Some(seq_iter) = reader.get(idx) {
                    sum = sum.wrapping_add(seq_iter.count() as u64);
                }
            }
            black_box(sum);
        })
    });

    // 3. Reused stateful seq_reader (optimized for sequential-like access)
    group.bench_function("reused_seq_reader_get", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            let mut seq_reader = black_box(&seq_vec).seq_reader();
            for &idx in black_box(access_indices.iter()) {
                if let Some(seq_iter) = seq_reader.get(idx) {
                    sum = sum.wrapping_add(seq_iter.count() as u64);
                }
            }
            black_box(sum);
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
    targets = benchmark_random_access
}
criterion_main!(benches);
