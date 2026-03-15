//! Comprehensive and robust test suite for `AtomicFixedVec`.
//!
//! This suite validates all aspects of the atomic fixed-width vector, including:
//! - Correctness of the single-word lock-free strategy for non-spanning values.
//! - Correctness of the lock-based strategy for values that span word boundaries.
//! - Behavior with both signed (ZigZag encoded) and unsigned integer types.
//! - All atomic operations: load, store, swap, and compare_exchange.
//! - All RMW (Read-Modify-Write) operations like `fetch_add`, `fetch_and`, etc.
//! - Edge cases such as zero bit width, max bit width, and boundary indices.
//! - Robustness under various multi-threaded concurrency patterns.
//! - Ergonomics and correctness of the new builder, `TryFrom`, `From`, and macro APIs.

use compressed_intvec::atomic_fixed_vec;
use compressed_intvec::fixed::atomic::{SAtomicFixedVec, UAtomicFixedVec};
use compressed_intvec::fixed::{BitWidth, Error, FixedVec, UFixedVec};
use rand::{RngExt, rngs::SmallRng, SeedableRng};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// --- Test Data Generation Helpers ---

fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    if max_val_exclusive == 0 {
        // This case is for u64 full range
        return (0..size).map(|_| rng.random::<u64>()).collect();
    }
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

fn generate_random_signed_vec(size: usize, max_abs_val: i64) -> Vec<i64> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size)
        .map(|_| rng.random_range(-max_abs_val..max_abs_val))
        .collect()
}

// --- Macro for Systematic Testing Across Types ---

macro_rules! test_atomic_api_for_type {
    ($test_name:ident, $T:ty, $is_signed:ident, $max_val:expr) => {
        #[test]
        fn $test_name() {
            let data: Vec<$T> = if $is_signed {
                generate_random_signed_vec(256, $max_val as i64)
                    .into_iter()
                    .map(|x| x as $T)
                    .collect()
            } else {
                generate_random_vec(256, $max_val)
                    .into_iter()
                    .map(|x| x as $T)
                    .collect()
            };

            // 1. Test Builder API
            let vec_builder = UAtomicFixedVec::<$T>::builder().build(&data).unwrap();
            assert_eq!(vec_builder.len(), data.len());
            if !data.is_empty() {
                assert_eq!(
                    vec_builder.load(10, Ordering::Relaxed),
                    data[10],
                    "Builder load failed for {}",
                    stringify!($T)
                );
            }

            // 2. Test TryFrom API
            let vec_tryfrom = UAtomicFixedVec::<$T>::try_from(data.as_slice()).unwrap();
            assert_eq!(vec_builder.bit_width(), vec_tryfrom.bit_width());
            assert_eq!(vec_tryfrom.len(), data.len());
            if !data.is_empty() {
                assert_eq!(
                    vec_tryfrom.load(20, Ordering::Relaxed),
                    data[20],
                    "TryFrom load failed for {}",
                    stringify!($T)
                );
            }

            // 3. Test Core Atomic Operations
            let vec = vec_tryfrom; // Use one of the created vectors for further tests
            if data.len() < 3 {
                return;
            }
            let val0 = data[0];
            let val1 = data[1];
            let val2 = data[2];

            vec.store(0, val0, Ordering::SeqCst);
            assert_eq!(vec.load(0, Ordering::SeqCst), val0);

            let old = vec.swap(0, val1, Ordering::SeqCst);
            assert_eq!(old, val0);
            assert_eq!(vec.load(0, Ordering::SeqCst), val1);

            let result = vec.compare_exchange(0, val1, val2, Ordering::SeqCst, Ordering::Relaxed);
            assert_eq!(result, Ok(val1));
            assert_eq!(vec.load(0, Ordering::SeqCst), val2);

            let result_fail =
                vec.compare_exchange(0, val0, val1, Ordering::SeqCst, Ordering::Relaxed);
            assert_eq!(result_fail, Err(val2));
            assert_eq!(vec.load(0, Ordering::SeqCst), val2);
        }
    };
}

macro_rules! test_atomic_rmw_for_type {
    ($test_name:ident, $T:ty, $v1_lit:expr, $v2_lit:expr, $v3_lit:expr) => {
        #[test]
        fn $test_name() {
            let v1 = $v1_lit as $T;
            let v2 = $v2_lit as $T;
            let v3 = $v3_lit as $T;

            let builder = UAtomicFixedVec::<$T>::builder().bit_width(BitWidth::Explicit(32));

            // Test fetch_add
            let vec = builder.clone().build(&[v1]).unwrap();
            assert_eq!(
                vec.fetch_add(0, v2, Ordering::SeqCst),
                v1,
                "fetch_add: wrong return value"
            );
            assert_eq!(
                vec.load(0, Ordering::Relaxed),
                v1.wrapping_add(v2),
                "fetch_add: wrong final value"
            );

            // Test fetch_sub
            let vec = builder.clone().build(&[v1]).unwrap();
            assert_eq!(
                vec.fetch_sub(0, v2, Ordering::SeqCst),
                v1,
                "fetch_sub: wrong return value"
            );
            assert_eq!(
                vec.load(0, Ordering::Relaxed),
                v1.wrapping_sub(v2),
                "fetch_sub: wrong final value"
            );

            // Test fetch_and
            let vec = builder.clone().build(&[v1]).unwrap();
            assert_eq!(
                vec.fetch_and(0, v3, Ordering::SeqCst),
                v1,
                "fetch_and: wrong return value"
            );
            assert_eq!(
                vec.load(0, Ordering::Relaxed),
                v1 & v3,
                "fetch_and: wrong final value"
            );

            // Test fetch_or
            let vec = builder.clone().build(&[v1]).unwrap();
            assert_eq!(
                vec.fetch_or(0, v3, Ordering::SeqCst),
                v1,
                "fetch_or: wrong return value"
            );
            assert_eq!(
                vec.load(0, Ordering::Relaxed),
                v1 | v3,
                "fetch_or: wrong final value"
            );

            // Test fetch_xor
            let vec = builder.clone().build(&[v1]).unwrap();
            assert_eq!(
                vec.fetch_xor(0, v3, Ordering::SeqCst),
                v1,
                "fetch_xor: wrong return value"
            );
            assert_eq!(
                vec.load(0, Ordering::Relaxed),
                v1 ^ v3,
                "fetch_xor: wrong final value"
            );

            // Test fetch_max
            let vec = builder.clone().build(&[v1]).unwrap();
            assert_eq!(
                vec.fetch_max(0, v2, Ordering::SeqCst),
                v1,
                "fetch_max: wrong return value"
            );
            assert_eq!(
                vec.load(0, Ordering::Relaxed),
                v1.max(v2),
                "fetch_max: wrong final value"
            );

            // Test fetch_min
            let vec = builder.build(&[v1]).unwrap();
            assert_eq!(
                vec.fetch_min(0, v2, Ordering::SeqCst),
                v1,
                "fetch_min: wrong return value"
            );
            assert_eq!(
                vec.load(0, Ordering::Relaxed),
                v1.min(v2),
                "fetch_min: wrong final value"
            );
        }
    };
}

// --- Test Suite Execution ---

// Unsigned types
test_atomic_api_for_type!(test_api_u8, u8, false, u8::MAX as u64);
test_atomic_api_for_type!(test_api_u16, u16, false, u16::MAX as u64);
test_atomic_api_for_type!(test_api_u32, u32, false, u32::MAX as u64);
test_atomic_api_for_type!(test_api_u64, u64, false, 0); // 0 indicates full range
test_atomic_rmw_for_type!(test_rmw_u32, u32, 100, 50, 0xF0);

// Signed types
test_atomic_api_for_type!(test_api_i8, i8, true, i8::MAX as u64);
test_atomic_api_for_type!(test_api_i16, i16, true, i16::MAX as u64);
test_atomic_api_for_type!(test_api_i32, i32, true, i32::MAX as u64);
test_atomic_api_for_type!(test_api_i64, i64, true, i64::MAX as u64);
test_atomic_rmw_for_type!(test_rmw_i32, i32, -100, 50, 0xF0);

// --- Standalone Tests for Macros and Edge Cases ---

#[test]
fn test_atomic_fixed_vec_macro() {
    // From list
    let vec = atomic_fixed_vec![-10i32, 20, -30];
    let _: SAtomicFixedVec<i32> = vec; // Type assertion
    assert_eq!(vec.len(), 3);
    assert_eq!(vec.load(0, Ordering::Relaxed), -10);
    assert_eq!(vec.load(2, Ordering::Relaxed), -30);

    // From repetition
    let vec_rep = atomic_fixed_vec![42u64; 100];
    let _: UAtomicFixedVec<u64> = vec_rep; // Type assertion
    assert_eq!(vec_rep.len(), 100);
    assert_eq!(vec_rep.load(99, Ordering::Relaxed), 42);

    // Empty
    let empty_vec: UAtomicFixedVec<u8> = atomic_fixed_vec![];
    assert!(empty_vec.is_empty());
}

#[test]
fn test_builder_failures() {
    // --- Test Case 1: ValueTooLarge error ---
    // Create data where one element (50) cannot fit into the specified bit width.
    let data: &[u32] = &[10, 20, 50];
    // The builder should fail because 50 requires 6 bits, but we specified only 4.
    let result = UAtomicFixedVec::<u32>::builder()
        .bit_width(BitWidth::Explicit(4))
        .build(data);
    assert!(matches!(result, Err(Error::ValueTooLarge { .. })));

    // --- Test Case 2: InvalidParameters error ---
    // The bit width (65) is larger than the storage word size (u64 = 64 bits).
    // This should fail regardless of the input data (even if empty).
    let result_bw = UAtomicFixedVec::<u64>::builder()
        .bit_width(BitWidth::Explicit(65))
        .build(&[]); // Using an empty slice is sufficient to test this parameter validation.
    assert!(matches!(result_bw, Err(Error::InvalidParameters(_))));
}

#[test]
fn test_edge_case_zero_bit_width() {
    // Building a non-empty vec with bit_width=0 should fail.
    let data = vec![0u32; 100];
    let result = UAtomicFixedVec::<u32>::builder()
        .bit_width(BitWidth::Explicit(0))
        .build(&data);
    assert!(matches!(result, Err(Error::InvalidParameters(_))));

    // Building an empty vec with bit_width=0 is allowed.
    let empty_data: Vec<u32> = vec![];
    let vec = UAtomicFixedVec::<u32>::builder()
        .bit_width(BitWidth::Explicit(0))
        .build(&empty_data)
        .unwrap();
    assert!(vec.is_empty());
}

#[test]
fn test_from_conversions() {
    let data: Vec<u32> = (0..100).collect();
    let fixed_vec: UFixedVec<u32> = FixedVec::builder().build(&data).unwrap();

    let bit_width = fixed_vec.bit_width();
    let len = fixed_vec.len();

    // Convert FixedVec -> AtomicFixedVec
    let atomic_vec = UAtomicFixedVec::<u32>::from(fixed_vec);
    assert_eq!(atomic_vec.bit_width(), bit_width);
    assert_eq!(atomic_vec.len(), len);
    assert_eq!(atomic_vec.load(50, Ordering::Relaxed), 50);

    // Convert AtomicFixedVec -> FixedVec
    let new_fixed_vec = FixedVec::from(atomic_vec);
    assert_eq!(new_fixed_vec.bit_width(), bit_width);
    assert_eq!(new_fixed_vec.len(), len);
    assert_eq!(new_fixed_vec.get(50), Some(50));
}

#[test]
fn test_ergonomic_read_apis() {
    let data: Vec<u32> = (0..100).collect();
    let vec = UAtomicFixedVec::<u32>::builder().build(&data).unwrap();

    // Test get()
    assert_eq!(vec.get(50), Some(50));
    // assert_eq!(vec.get(100), None);

    // Test iter()
    let collected: Vec<u32> = vec.iter().collect();
    assert_eq!(collected, data);

    // Test IntoIterator for a reference
    let mut collected_ref = vec![];
    for val in &vec {
        collected_ref.push(val);
    }
    assert_eq!(collected_ref, data);

    // Test on empty vector
    let empty_vec: UAtomicFixedVec<u8> = atomic_fixed_vec![];
    assert!(empty_vec.get(0).is_none());
    assert_eq!(empty_vec.iter().next(), None);
}

// --- Concurrency Tests (kept separate for clarity) ---

#[test]
fn test_concurrent_disjoint_stores() {
    const NUM_THREADS: usize = 4;
    const LEN: usize = 1000;
    // The vector must be created with enough bit width for the values that will be stored.
    // Max value is approx 3 * 1000 + 999 = 3999, which needs 12 bits.
    let vec = Arc::new(
        UAtomicFixedVec::<u16>::builder()
            .bit_width(BitWidth::Explicit(12))
            .build(&vec![0; LEN])
            .unwrap(),
    );

    thread::scope(|s| {
        for thread_id in 0..NUM_THREADS {
            let vec_clone = Arc::clone(&vec);
            s.spawn(move || {
                let chunk_size = LEN / NUM_THREADS;
                let start = thread_id * chunk_size;
                let end = start + chunk_size;
                for i in start..end {
                    vec_clone.store(i, (thread_id * 1000 + i) as u16, Ordering::SeqCst);
                }
            });
        }
    });

    // Verify that all writes are correctly visible.
    for thread_id in 0..NUM_THREADS {
        let chunk_size = LEN / NUM_THREADS;
        let start = thread_id * chunk_size;
        let end = start + chunk_size;
        for i in start..end {
            assert_eq!(vec.load(i, Ordering::SeqCst), (thread_id * 1000 + i) as u16);
        }
    }
}

#[test]
fn test_concurrent_cas_contention() {
    // Multiple threads incrementing the same counter using CAS.
    // Max value is 10 * 1000 = 10000, which needs 14 bits. We use 16.
    let vec = Arc::new(
        UAtomicFixedVec::<u32>::builder()
            .bit_width(BitWidth::Explicit(16))
            .build(&[0; 1])
            .unwrap(),
    );

    const NUM_THREADS: usize = 10;
    const INCREMENTS_PER_THREAD: u32 = 1000;

    thread::scope(|s| {
        for _ in 0..NUM_THREADS {
            let vec_clone = Arc::clone(&vec);
            s.spawn(move || {
                for _ in 0..INCREMENTS_PER_THREAD {
                    let mut current = vec_clone.load(0, Ordering::Relaxed);
                    loop {
                        match vec_clone.compare_exchange(
                            0,
                            current,
                            current.wrapping_add(1),
                            Ordering::SeqCst,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(actual) => current = actual,
                        }
                    }
                }
            });
        }
    });

    assert_eq!(
        vec.load(0, Ordering::SeqCst),
        NUM_THREADS as u32 * INCREMENTS_PER_THREAD
    );
}

#[test]
fn test_concurrent_fetch_add_contention() {
    let vec = Arc::new(
        UAtomicFixedVec::<u32>::builder()
            .bit_width(BitWidth::Explicit(16))
            .build(&[0; 1])
            .unwrap(),
    );

    const NUM_THREADS: usize = 10;
    const INCREMENTS_PER_THREAD: u32 = 1000;

    thread::scope(|s| {
        for _ in 0..NUM_THREADS {
            let vec_clone = Arc::clone(&vec);
            s.spawn(move || {
                for _ in 0..INCREMENTS_PER_THREAD {
                    vec_clone.fetch_add(0, 1, Ordering::SeqCst);
                }
            });
        }
    });

    assert_eq!(
        vec.load(0, Ordering::SeqCst),
        NUM_THREADS as u32 * INCREMENTS_PER_THREAD
    );
}

#[test]
#[cfg(feature = "parallel")]
fn test_par_iter_mut_contention() {
    // This test replaces the unsound `test_chunks_mut_parallel`.
    // It verifies the new, safe parallel mutation API on AtomicFixedVec.
    let data: Vec<u32> = (0..10_000).collect();
    // Max value will be 9999 * 2 = 19998. This needs 15 bits.
    let vec: UAtomicFixedVec<u32> = UAtomicFixedVec::builder()
        .bit_width(BitWidth::Explicit(15))
        .build(&data)
        .unwrap();

    // Use the new parallel iterator to modify the vector in place.
    vec.par_iter_mut().for_each(|mut proxy| {
        *proxy *= 2;
    });

    // Verify the results sequentially.
    for (i, &expected) in data.iter().enumerate().take(vec.len()) {
        assert_eq!(
            vec.load(i, Ordering::Relaxed),
            expected * 2,
            "Mismatch at index {}",
            i
        );
    }
}
