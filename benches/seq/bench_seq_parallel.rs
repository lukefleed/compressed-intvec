// benches/seq/bench_seq_parallel.rs
//
// Benchmarks for parallel operations on SeqVec.
//
// Measures:
// 1. Sequential vs parallel iteration crossover point by dataset size
// 2. par_decode_many throughput at varying batch sizes
// 3. par_into_vecs vs sequential into_vecs
// 4. Effect of sequence length on parallel efficiency
//
// These benchmarks help users determine when parallel APIs provide benefit
// over sequential alternatives, accounting for thread spawn overhead and
// cache locality trade-offs.

#![cfg(feature = "parallel")]

use compressed_intvec::seq::{LESeqVec, SeqVec, VariableCodecSpec};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Duration;

/// Generates sequences with power-law length distribution.
///
/// This distribution models realistic graph adjacency lists where most nodes
/// have few neighbors and few nodes have many neighbors. The average sequence
/// length is approximately 15 elements.
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

/// Generates sequences with fixed length for controlled experiments.
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

/// Counts total elements across all sequences.
fn count_total_elements(sequences: &[Vec<u32>]) -> u64 {
    sequences.iter().map(|s| s.len() as u64).sum()
}

/// Benchmark: iter() vs par_iter() crossover by dataset size.
///
/// This benchmark identifies the dataset size at which parallel iteration
/// becomes beneficial. For small datasets, thread spawn overhead and reduced
/// cache locality make sequential iteration faster.
///
/// The parallel pattern uses sequential inner iteration to avoid per-element
/// scheduling overhead. Parallelism is at the sequence level, not element level.
fn benchmark_iter_vs_par_iter(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    // Test various dataset sizes to find crossover point.
    // Sizes chosen to span from "definitely sequential wins" to "definitely parallel wins".
    let sequence_counts = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000];

    let mut group = c.benchmark_group("SeqParallel/iter_vs_par_iter");

    for &num_sequences in &sequence_counts {
        let sequences = generate_power_law_sequences(&mut rng, num_sequences);
        let total_elements = count_total_elements(&sequences);

        let seqvec: LESeqVec<u32> = SeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .build(&sequences)
            .expect("Failed to build SeqVec");

        group.throughput(Throughput::Elements(total_elements));

        // Baseline: uncompressed Vec<Vec<u32>> sequential
        group.bench_with_input(
            BenchmarkId::new("Baseline_seq", num_sequences),
            &sequences,
            |b, seqs| {
                b.iter(|| {
                    let sum: u64 = seqs
                        .iter()
                        .map(|s| s.iter().map(|&v| v as u64).sum::<u64>())
                        .sum();
                    black_box(sum)
                })
            },
        );

        // Baseline: uncompressed Vec<Vec<u32>> parallel
        // Parallelism at sequence level only; inner iteration is sequential.
        group.bench_with_input(
            BenchmarkId::new("Baseline_par", num_sequences),
            &sequences,
            |b, seqs| {
                b.iter(|| {
                    let sum: u64 = seqs
                        .par_iter()
                        .map(|s| s.iter().map(|&v| v as u64).sum::<u64>())
                        .sum();
                    black_box(sum)
                })
            },
        );

        // SeqVec sequential iteration
        group.bench_with_input(
            BenchmarkId::new("SeqVec_iter", num_sequences),
            &seqvec,
            |b, vec| {
                b.iter(|| {
                    let sum: u64 = vec.iter().map(|s| s.map(|v| v as u64).sum::<u64>()).sum();
                    black_box(sum)
                })
            },
        );

        // SeqVec parallel iteration
        // par_iter() returns Vec<T> per sequence; inner iteration is sequential.
        group.bench_with_input(
            BenchmarkId::new("SeqVec_par_iter", num_sequences),
            &seqvec,
            |b, vec| {
                b.iter(|| {
                    let sum: u64 = vec
                        .par_iter()
                        .map(|s| s.iter().map(|&v| v as u64).sum::<u64>())
                        .sum();
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: par_decode_many at varying batch sizes.
///
/// Measures how par_decode_many scales with batch size. For small batches,
/// the overhead of parallel dispatch may outweigh benefits. This helps users
/// choose between decode_many (sequential with sorting) and par_decode_many.
fn benchmark_par_decode_many_scaling(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    const NUM_SEQUENCES: usize = 100_000;
    let sequences = generate_power_law_sequences(&mut rng, NUM_SEQUENCES);

    let seqvec: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences)
        .expect("Failed to build SeqVec");

    // Batch sizes from small (overhead-dominated) to large (throughput-dominated).
    let batch_sizes = [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000];

    let mut group = c.benchmark_group("SeqParallel/decode_many_scaling");

    for &batch_size in &batch_sizes {
        // Generate random indices for this batch size.
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..NUM_SEQUENCES))
            .collect();

        let total_elements: u64 = indices.iter().map(|&i| sequences[i].len() as u64).sum();
        group.throughput(Throughput::Elements(total_elements));

        // Sequential decode_many (internally sorts for cache locality)
        group.bench_with_input(
            BenchmarkId::new("decode_many", batch_size),
            &indices,
            |b, idx| {
                b.iter(|| {
                    let results = seqvec
                        .decode_many(black_box(idx))
                        .expect("decode_many failed");
                    let sum: u64 = results
                        .iter()
                        .map(|s| s.iter().map(|&v| v as u64).sum::<u64>())
                        .sum();
                    black_box(sum)
                })
            },
        );

        // Parallel decode_many
        group.bench_with_input(
            BenchmarkId::new("par_decode_many", batch_size),
            &indices,
            |b, idx| {
                b.iter(|| {
                    let results = seqvec
                        .par_decode_many(black_box(idx))
                        .expect("par_decode_many failed");
                    let sum: u64 = results
                        .iter()
                        .map(|s| s.iter().map(|&v| v as u64).sum::<u64>())
                        .sum();
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: into_vecs vs par_into_vecs for full materialization.
///
/// When a user needs all sequences materialized as Vec<Vec<T>>, this benchmark
/// shows whether parallel materialization provides speedup.
fn benchmark_into_vecs(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    let sequence_counts = [1_000, 5_000, 10_000, 50_000];

    let mut group = c.benchmark_group("SeqParallel/into_vecs");

    for &num_sequences in &sequence_counts {
        let sequences = generate_power_law_sequences(&mut rng, num_sequences);
        let total_elements = count_total_elements(&sequences);

        group.throughput(Throughput::Elements(total_elements));

        // Sequential into_vecs
        group.bench_function(BenchmarkId::new("into_vecs", num_sequences), |b| {
            b.iter_with_setup(
                || {
                    SeqVec::builder()
                        .codec(VariableCodecSpec::Delta)
                        .build(&sequences)
                        .expect("Failed to build SeqVec")
                },
                |seqvec: LESeqVec<u32>| {
                    let vecs = seqvec.into_vecs();
                    black_box(vecs)
                },
            )
        });

        // Parallel into_vecs
        group.bench_function(BenchmarkId::new("par_into_vecs", num_sequences), |b| {
            b.iter_with_setup(
                || {
                    SeqVec::builder()
                        .codec(VariableCodecSpec::Delta)
                        .build(&sequences)
                        .expect("Failed to build SeqVec")
                },
                |seqvec: LESeqVec<u32>| {
                    let vecs = seqvec.par_into_vecs();
                    black_box(vecs)
                },
            )
        });
    }

    group.finish();
}

/// Benchmark: Effect of sequence length on parallel efficiency.
///
/// Parallel iteration is more efficient when individual sequences are longer,
/// as each thread has more work to amortize dispatch overhead. This benchmark
/// uses fixed-length sequences to isolate the effect of sequence length.
fn benchmark_sequence_length_effect(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    const NUM_SEQUENCES: usize = 10_000;
    let sequence_lengths = [5, 20, 50, 100, 500];

    let mut group = c.benchmark_group("SeqParallel/sequence_length_effect");

    for &seq_len in &sequence_lengths {
        let sequences = generate_fixed_length_sequences(&mut rng, NUM_SEQUENCES, seq_len);
        let total_elements = (NUM_SEQUENCES * seq_len) as u64;

        let seqvec: LESeqVec<u32> = SeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .build(&sequences)
            .expect("Failed to build SeqVec");

        group.throughput(Throughput::Elements(total_elements));

        // Sequential
        group.bench_function(BenchmarkId::new("iter", seq_len), |b| {
            b.iter(|| {
                let sum: u64 = seqvec
                    .iter()
                    .map(|s| s.map(|v| v as u64).sum::<u64>())
                    .sum();
                black_box(sum)
            })
        });

        // Parallel
        group.bench_function(BenchmarkId::new("par_iter", seq_len), |b| {
            b.iter(|| {
                let sum: u64 = seqvec
                    .par_iter()
                    .map(|s| s.iter().map(|&v| v as u64).sum::<u64>())
                    .sum();
                black_box(sum)
            })
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    targets =
        benchmark_iter_vs_par_iter,
        benchmark_par_decode_many_scaling,
        benchmark_into_vecs,
        benchmark_sequence_length_effect
}

criterion_main!(benches);
