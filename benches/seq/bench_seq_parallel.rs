// benches/seq/bench_seq_parallel.rs
//
// Benchmarks for parallel operations on SeqVec.
//
// Measures:
// 1. Sequential vs parallel iteration crossover point by dataset size
// 2. par_for_each vs par_iter for consumptive operations
// 3. par_decode_many throughput at varying batch sizes
// 4. par_into_vecs vs sequential into_vecs
// 5. Effect of sequence length on parallel efficiency
//
// These benchmarks help users determine when parallel APIs provide benefit
// over sequential alternatives, accounting for thread spawn overhead,
// allocation costs, and cache locality trade-offs.

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

        // SeqVec parallel iteration (allocates Vec<T> per sequence)
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

/// Benchmark: par_for_each vs par_iter for consumptive operations.
///
/// This benchmark compares the zero-allocation par_for_each against par_iter
/// for operations that don't need to retain the decoded data. par_for_each
/// should be faster due to avoiding Vec allocation and double traversal.
fn benchmark_par_for_each_vs_par_iter(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    let sequence_counts = [1_000, 5_000, 10_000, 50_000, 100_000];

    let mut group = c.benchmark_group("SeqParallel/par_for_each_vs_par_iter");

    for &num_sequences in &sequence_counts {
        let sequences = generate_power_law_sequences(&mut rng, num_sequences);
        let total_elements = count_total_elements(&sequences);

        let seqvec: LESeqVec<u32> = SeqVec::builder()
            .codec(VariableCodecSpec::Delta)
            .build(&sequences)
            .expect("Failed to build SeqVec");

        group.throughput(Throughput::Elements(total_elements));

        // par_iter: allocates Vec<T> per sequence, then sums
        group.bench_with_input(
            BenchmarkId::new("par_iter_sum", num_sequences),
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

        // par_for_each: zero allocation, sums directly from iterator
        group.bench_with_input(
            BenchmarkId::new("par_for_each_sum", num_sequences),
            &seqvec,
            |b, vec| {
                b.iter(|| {
                    let sums: Vec<u64> = vec.par_for_each(|seq| seq.map(|v| v as u64).sum());
                    let total: u64 = sums.iter().sum();
                    black_box(total)
                })
            },
        );

        // par_for_each_reduce: zero allocation with parallel reduction
        group.bench_with_input(
            BenchmarkId::new("par_for_each_reduce_sum", num_sequences),
            &seqvec,
            |b, vec| {
                b.iter(|| {
                    let total: u64 = vec.par_for_each_reduce(
                        |seq| seq.map(|v| v as u64).sum::<u64>(),
                        || 0u64,
                        |a, b| a + b,
                    );
                    black_box(total)
                })
            },
        );

        // Sequential iter for comparison
        group.bench_with_input(
            BenchmarkId::new("iter_sum", num_sequences),
            &seqvec,
            |b, vec| {
                b.iter(|| {
                    let sum: u64 = vec.iter().map(|s| s.map(|v| v as u64).sum::<u64>()).sum();
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: par_for_each with different operations.
///
/// Tests how par_for_each performs with various consumptive operations:
/// sum, count, max, and any() predicate.
fn benchmark_par_for_each_operations(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    const NUM_SEQUENCES: usize = 50_000;
    let sequences = generate_power_law_sequences(&mut rng, NUM_SEQUENCES);
    let total_elements = count_total_elements(&sequences);

    let seqvec: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences)
        .expect("Failed to build SeqVec");

    let mut group = c.benchmark_group("SeqParallel/par_for_each_operations");
    group.throughput(Throughput::Elements(total_elements));

    // Sum operation
    group.bench_function("sum", |b| {
        b.iter(|| {
            let sums: Vec<u64> = seqvec.par_for_each(|seq| seq.map(|v| v as u64).sum());
            black_box(sums)
        })
    });

    // Count operation
    group.bench_function("count", |b| {
        b.iter(|| {
            let counts: Vec<usize> = seqvec.par_for_each(|seq| seq.count());
            black_box(counts)
        })
    });

    // Max operation
    group.bench_function("max", |b| {
        b.iter(|| {
            let maxes: Vec<Option<u32>> = seqvec.par_for_each(|seq| seq.max());
            black_box(maxes)
        })
    });

    // // Any predicate (early termination possible)
    // group.bench_function("any_gt_5000", |b| {
    //     b.iter(|| {
    //         let results: Vec<bool> = seqvec.par_for_each(|seq| seq.any(|v| v > 5000));
    //         black_box(results)
    //     })
    // });

    // Collect first element only (minimal work per sequence)
    group.bench_function("first_element", |b| {
        b.iter(|| {
            let firsts: Vec<Option<u32>> = seqvec.par_for_each(|seq| seq.into_iter().next());
            black_box(firsts)
        })
    });

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

        // Calculate throughput based on actual elements in selected sequences.
        let total_elements: u64 = indices.iter().map(|&i| sequences[i].len() as u64).sum();
        group.throughput(Throughput::Elements(total_elements));

        // Sequential decode_many (sorts indices for linear scan)
        group.bench_with_input(
            BenchmarkId::new("decode_many", batch_size),
            &indices,
            |b, idx| {
                b.iter(|| {
                    let results = seqvec.decode_many(idx).unwrap();
                    black_box(results)
                })
            },
        );

        // Parallel decode_many
        group.bench_with_input(
            BenchmarkId::new("par_decode_many", batch_size),
            &indices,
            |b, idx| {
                b.iter(|| {
                    let results = seqvec.par_decode_many(idx).unwrap();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: par_for_each_many for sparse consumptive operations.
///
/// Compares par_for_each_many against par_decode_many for operations that
/// don't need to retain the decoded data.
fn benchmark_par_for_each_many_scaling(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    const NUM_SEQUENCES: usize = 100_000;
    let sequences = generate_power_law_sequences(&mut rng, NUM_SEQUENCES);

    let seqvec: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences)
        .expect("Failed to build SeqVec");

    let batch_sizes = [100, 1_000, 5_000, 10_000, 50_000];

    let mut group = c.benchmark_group("SeqParallel/par_for_each_many_scaling");

    for &batch_size in &batch_sizes {
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..NUM_SEQUENCES))
            .collect();

        let total_elements: u64 = indices.iter().map(|&i| sequences[i].len() as u64).sum();
        group.throughput(Throughput::Elements(total_elements));

        // par_decode_many then sum (allocates Vec<T> per sequence)
        group.bench_with_input(
            BenchmarkId::new("par_decode_many_sum", batch_size),
            &indices,
            |b, idx| {
                b.iter(|| {
                    let results = seqvec.par_decode_many(idx).unwrap();
                    let sum: u64 = results
                        .iter()
                        .map(|s| s.iter().map(|&v| v as u64).sum::<u64>())
                        .sum();
                    black_box(sum)
                })
            },
        );

        // par_for_each_many (zero allocation)
        group.bench_with_input(
            BenchmarkId::new("par_for_each_many_sum", batch_size),
            &indices,
            |b, idx| {
                b.iter(|| {
                    let sums = seqvec
                        .par_for_each_many(idx, |seq| seq.map(|v| v as u64).sum::<u64>())
                        .unwrap();
                    let total: u64 = sums.iter().sum();
                    black_box(total)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: into_vecs vs par_into_vecs.
///
/// Compares sequential and parallel bulk decoding of all sequences.
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
            b.iter_batched(
                || {
                    SeqVec::builder()
                        .codec(VariableCodecSpec::Delta)
                        .build(&sequences)
                        .expect("Failed to build SeqVec")
                },
                |vec: LESeqVec<u32>| {
                    let results = vec.into_vecs();
                    black_box(results)
                },
                criterion::BatchSize::SmallInput,
            )
        });

        // Parallel into_vecs
        group.bench_function(BenchmarkId::new("par_into_vecs", num_sequences), |b| {
            b.iter_batched(
                || {
                    SeqVec::builder()
                        .codec(VariableCodecSpec::Delta)
                        .build(&sequences)
                        .expect("Failed to build SeqVec")
                },
                |vec: LESeqVec<u32>| {
                    let results = vec.par_into_vecs();
                    black_box(results)
                },
                criterion::BatchSize::SmallInput,
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

        // Parallel (allocating)
        group.bench_function(BenchmarkId::new("par_iter", seq_len), |b| {
            b.iter(|| {
                let sum: u64 = seqvec
                    .par_iter()
                    .map(|s| s.iter().map(|&v| v as u64).sum::<u64>())
                    .sum();
                black_box(sum)
            })
        });

        // Parallel (zero allocation)
        group.bench_function(BenchmarkId::new("par_for_each", seq_len), |b| {
            b.iter(|| {
                let total: u64 = seqvec.par_for_each_reduce(
                    |seq| seq.map(|v| v as u64).sum::<u64>(),
                    || 0u64,
                    |a, b| a + b,
                );
                black_box(total)
            })
        });
    }

    group.finish();
}

/// Benchmark: Effect of stored lengths on parallel performance.
///
/// When sequence lengths are stored, par_iter and par_for_each can pre-allocate
/// buffers and avoid the bit_pos() check in the inner loop. This benchmark
/// measures the impact of stored lengths on parallel performance.
fn benchmark_stored_lengths_effect(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    const NUM_SEQUENCES: usize = 50_000;
    let sequences = generate_power_law_sequences(&mut rng, NUM_SEQUENCES);
    let total_elements = count_total_elements(&sequences);

    // Build SeqVec without stored lengths
    let seqvec_no_lengths: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .store_lengths(false)
        .build(&sequences)
        .expect("Failed to build SeqVec without lengths");

    // Build SeqVec with stored lengths
    let seqvec_with_lengths: LESeqVec<u32> = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .store_lengths(true)
        .build(&sequences)
        .expect("Failed to build SeqVec with lengths");

    let mut group = c.benchmark_group("SeqParallel/stored_lengths_effect");
    group.throughput(Throughput::Elements(total_elements));

    // par_iter without stored lengths
    group.bench_function("par_iter_no_lengths", |b| {
        b.iter(|| {
            let sum: u64 = seqvec_no_lengths
                .par_iter()
                .map(|s| s.iter().map(|&v| v as u64).sum::<u64>())
                .sum();
            black_box(sum)
        })
    });

    // par_iter with stored lengths
    group.bench_function("par_iter_with_lengths", |b| {
        b.iter(|| {
            let sum: u64 = seqvec_with_lengths
                .par_iter()
                .map(|s| s.iter().map(|&v| v as u64).sum::<u64>())
                .sum();
            black_box(sum)
        })
    });

    // par_for_each without stored lengths
    group.bench_function("par_for_each_no_lengths", |b| {
        b.iter(|| {
            let total: u64 = seqvec_no_lengths.par_for_each_reduce(
                |seq| seq.map(|v| v as u64).sum::<u64>(),
                || 0u64,
                |a, b| a + b,
            );
            black_box(total)
        })
    });

    // par_for_each with stored lengths
    group.bench_function("par_for_each_with_lengths", |b| {
        b.iter(|| {
            let total: u64 = seqvec_with_lengths.par_for_each_reduce(
                |seq| seq.map(|v| v as u64).sum::<u64>(),
                || 0u64,
                |a, b| a + b,
            );
            black_box(total)
        })
    });

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
        benchmark_par_for_each_vs_par_iter,
        benchmark_par_for_each_operations,
        benchmark_par_decode_many_scaling,
        benchmark_par_for_each_many_scaling,
        benchmark_into_vecs,
        benchmark_sequence_length_effect,
        benchmark_stored_lengths_effect
}

criterion_main!(benches);
