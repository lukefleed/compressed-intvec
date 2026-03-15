// benches/bench_serde.rs
//
// Serde round-trip benchmarks for FixedVec, VarVec, and SeqVec.
// Run with: cargo bench --bench bench_serde --features serde

use std::time::Duration;

use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, RngExt, SeedableRng};

/// Generates a vector with uniformly random values up to a given maximum.
fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

fn benchmark_serde(c: &mut Criterion) {
    const FIXED_SIZE: usize = 1_000_000;
    const VAR_SIZE: usize = 1_000_000;
    const SEQ_COUNT: usize = 50_000;
    const SEQ_LEN: usize = 20;

    // --- FixedVec serde benchmarks ---
    for &bit_width in &[16u32, 32] {
        let data = generate_random_vec(FIXED_SIZE, 1u64 << bit_width);
        let vec = LEFixedVec::builder()
            .bit_width(BitWidth::Explicit(bit_width as usize))
            .build(&data)
            .unwrap();

        let serialized = bincode::serialize(&vec).unwrap();
        let serialized_size = serialized.len() as u64;

        let mut group = c.benchmark_group(format!("Serde/FixedVec/{}bit", bit_width));
        group.throughput(Throughput::Bytes(serialized_size));

        group.bench_function("serialize", |b| {
            b.iter(|| {
                black_box(bincode::serialize(black_box(&vec)).unwrap());
            })
        });

        group.bench_function("deserialize", |b| {
            b.iter(|| {
                black_box(bincode::deserialize::<LEFixedVec>(black_box(&serialized)).unwrap());
            })
        });

        group.finish();
    }

    // --- VarVec serde benchmarks ---
    for (codec_name, codec) in &[("Gamma", Codec::Gamma), ("Delta", Codec::Delta)] {
        let data = generate_random_vec(VAR_SIZE, 1_000);
        let vec = LEVarVec::builder()
            .k(32)
            .codec(*codec)
            .build(&data)
            .unwrap();

        let serialized = bincode::serialize(&vec).unwrap();
        let serialized_size = serialized.len() as u64;

        let mut group = c.benchmark_group(format!("Serde/VarVec/{}", codec_name));
        group.throughput(Throughput::Bytes(serialized_size));

        group.bench_function("serialize", |b| {
            b.iter(|| {
                black_box(bincode::serialize(black_box(&vec)).unwrap());
            })
        });

        group.bench_function("deserialize", |b| {
            b.iter(|| {
                black_box(bincode::deserialize::<LEVarVec>(black_box(&serialized)).unwrap());
            })
        });

        group.finish();
    }

    // --- SeqVec serde benchmarks ---
    {
        let mut rng = SmallRng::seed_from_u64(42);
        let sequences: Vec<Vec<u64>> = (0..SEQ_COUNT)
            .map(|_| {
                (0..SEQ_LEN)
                    .map(|_| rng.random_range(0..1_000u64))
                    .collect()
            })
            .collect();

        let vec: LESeqVec<u64> = SeqVec::builder()
            .codec(Codec::Delta)
            .build(&sequences)
            .unwrap();

        let serialized = bincode::serialize(&vec).unwrap();
        let serialized_size = serialized.len() as u64;

        let mut group = c.benchmark_group("Serde/SeqVec/Delta");
        group.throughput(Throughput::Bytes(serialized_size));

        group.bench_function("serialize", |b| {
            b.iter(|| {
                black_box(bincode::serialize(black_box(&vec)).unwrap());
            })
        });

        group.bench_function("deserialize", |b| {
            b.iter(|| {
                black_box(
                    bincode::deserialize::<LESeqVec<u64>>(black_box(&serialized)).unwrap(),
                );
            })
        });

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(5));
    targets = benchmark_serde
}
criterion_main!(benches);
