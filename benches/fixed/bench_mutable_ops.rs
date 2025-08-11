use std::time::Duration;
use compressed_intvec::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use succinct::int_vec::{IntVecMut, IntVector};
use sux::prelude::{BitFieldSliceMut, BitFieldVec};

/// Generates a vector with uniformly random values up to a given maximum.
///
/// # Arguments
/// * `size` - The number of elements to generate.
/// * `max_val_exclusive` - The exclusive upper bound for the random values. A value of 0
///   indicates that the full range of `u64` should be used.
fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    if max_val_exclusive == 0 {
        return (0..size).map(|_| rng.random::<u64>()).collect();
    }
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

/// The main benchmark function for mutable and construction operations.
fn benchmark_mutable_ops(c: &mut Criterion) {
    const VECTOR_SIZE: usize = 1_000_000;
    const BIT_WIDTH: u32 = 16; // A power-of-two width.

    let mut group = c.benchmark_group(format!("MutableOps/{}bit", BIT_WIDTH));
    group.sample_size(10);

    let data = generate_random_vec(VECTOR_SIZE, 1u64 << BIT_WIDTH);

    // --- 1. Construction from Slice ---
    group.bench_function("LEFixedVec/build_from_slice", |b| {
        b.iter(|| {
            black_box(
                LEFixedVec::builder()
                    .bit_width(BitWidth::Explicit(BIT_WIDTH as usize))
                    .build(black_box(&data))
                    .unwrap(),
            );
        })
    });

    group.bench_function("sux::BitFieldVec/from_slice", |b| {
        b.iter(|| {
            black_box(BitFieldVec::<u64>::from_slice(black_box(&data)).unwrap());
        })
    });

    // --- 2. Build via `push` loop ---
    group.bench_function("Baseline_Vec<u64>/push_loop", |b| {
        b.iter(|| {
            let mut vec = Vec::with_capacity(VECTOR_SIZE);
            for &val in black_box(&data) {
                vec.push(val);
            }
            black_box(vec);
        })
    });

    group.bench_function("LEFixedVec/push_loop", |b| {
        b.iter(|| {
            let mut vec = LEFixedVec::with_capacity(BIT_WIDTH as usize, VECTOR_SIZE).unwrap();
            for &val in black_box(&data) {
                vec.push(val);
            }
            black_box(vec);
        })
    });

    group.bench_function("sux::BitFieldVec/push_loop", |b| {
        b.iter(|| {
            let mut vec = BitFieldVec::<u64>::with_capacity(BIT_WIDTH as usize, VECTOR_SIZE);
            for &val in black_box(&data) {
                vec.push(val);
            }
            black_box(vec);
        })
    });

    group.bench_function("succinct::IntVector/push_loop", |b| {
        b.iter(|| {
            let mut vec = IntVector::<u64>::with_capacity(BIT_WIDTH as usize, VECTOR_SIZE as u64);
            for &val in black_box(&data) {
                vec.push(val);
            }
            black_box(vec);
        })
    });

    // --- 3. Resize ---
    const RESIZE_FROM: usize = VECTOR_SIZE / 2;
    const RESIZE_TO: usize = VECTOR_SIZE;
    let resize_value = data[0]; // A value that fits the bit width.

    group.bench_function("Baseline_Vec<u64>/resize", |b| {
        b.iter_with_setup(
            || data[..RESIZE_FROM].to_vec(),
            |mut vec| {
                vec.resize(RESIZE_TO, resize_value);
                black_box(vec);
            },
        );
    });

    group.bench_function("LEFixedVec/resize", |b| {
        b.iter_with_setup(
            || {
                LEFixedVec::builder()
                    .bit_width(BitWidth::Explicit(BIT_WIDTH as usize))
                    .build(&data[..RESIZE_FROM])
                    .unwrap()
            },
            |mut vec| {
                vec.resize(RESIZE_TO, resize_value);
                black_box(vec);
            },
        );
    });

    group.bench_function("sux::BitFieldVec/resize", |b| {
        b.iter_with_setup(
            || {
                let mut vec = BitFieldVec::<u64>::with_capacity(BIT_WIDTH as usize, RESIZE_FROM);
                for &val in &data[..RESIZE_FROM] {
                    vec.push(val);
                }
                vec
            },
            |mut vec| {
                vec.resize(RESIZE_TO, resize_value);
                black_box(vec);
            },
        );
    });

    group.bench_function("succinct::IntVector/resize", |b| {
        b.iter_with_setup(
            || {
                let mut vec =
                    IntVector::<u64>::with_capacity(BIT_WIDTH as usize, RESIZE_FROM as u64);
                for &val in &data[..RESIZE_FROM] {
                    vec.push(val);
                }
                vec
            },
            |mut vec| {
                vec.resize(RESIZE_TO as u64, resize_value);
                black_box(vec);
            },
        );
    });

    // --- 4. Set at Random Index ---
    const NUM_SETS: usize = 100_000;
    let mut rng = SmallRng::seed_from_u64(1338);
    let set_indices: Vec<usize> = (0..NUM_SETS)
        .map(|_| rng.random_range(0..VECTOR_SIZE))
        .collect();
    let set_values: Vec<u64> = generate_random_vec(NUM_SETS, 1u64 << BIT_WIDTH);

    let base_le_fixed_vec = LEFixedVec::builder()
        .bit_width(BitWidth::Explicit(BIT_WIDTH as usize))
        .build(&data)
        .unwrap();
    let base_sux_bfv = BitFieldVec::<u64>::from_slice(&data).unwrap();
    let mut base_succinct_iv =
        IntVector::<u64>::with_capacity(BIT_WIDTH as usize, VECTOR_SIZE as u64);
    for &val in &data {
        base_succinct_iv.push(val);
    }

    group.bench_function("LEFixedVec/set_unchecked_random", |b| {
        b.iter_with_setup(
            || base_le_fixed_vec.clone(),
            |mut vec| {
                for i in 0..NUM_SETS {
                    unsafe { vec.set_unchecked(set_indices[i], set_values[i]) };
                }
                black_box(vec);
            },
        );
    });

    group.bench_function("sux::BitFieldVec/set_unchecked_random", |b| {
        b.iter_with_setup(
            || base_sux_bfv.clone(),
            |mut vec| {
                for i in 0..NUM_SETS {
                    unsafe { vec.set_unchecked(set_indices[i], set_values[i]) };
                }
                black_box(vec);
            },
        );
    });

    group.bench_function("succinct::IntVector/set_random", |b| {
        b.iter_with_setup(
            || base_succinct_iv.clone(),
            |mut vec| {
                for i in 0..NUM_SETS {
                    // succinct::IntVector::set performs bounds checks
                    vec.set(set_indices[i] as u64, set_values[i]);
                }
                black_box(vec);
            },
        );
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_secs(2));
    targets = benchmark_mutable_ops
}
criterion_main!(benches);