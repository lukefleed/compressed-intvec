//! Comprehensive and robust test suite for `AtomicFixedVec`.
//!
//! This suite validates all aspects of the atomic fixed-width vector, including:
//! - Correctness of the single-word lock-free strategy for non-spanning values.
//! - Correctness of the lock-based strategy for values that span word boundaries.
//! - Behavior with both signed (ZigZag encoded) and unsigned integer types.
//! - All atomic operations: load, store, swap, and compare_exchange.
//! - Edge cases such as zero bit width, max bit width, and boundary indices.
//! - Robustness under various multi-threaded concurrency patterns.
//! - Ergonomics and correctness of the new builder, `TryFrom`, and macro APIs.

use compressed_intvec::atomic_fixed_vec;
use compressed_intvec::fixed::atomic::{SAtomicFixedVec, UAtomicFixedVec};
use compressed_intvec::fixed::{BitWidth, Error};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;

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

// --- Test Suite Execution ---

// Unsigned types
test_atomic_api_for_type!(test_api_u8, u8, false, u8::MAX as u64);
test_atomic_api_for_type!(test_api_u16, u16, false, u16::MAX as u64);
test_atomic_api_for_type!(test_api_u32, u32, false, u32::MAX as u64);
test_atomic_api_for_type!(test_api_u64, u64, false, 0); // 0 indicates full range

// Signed types
test_atomic_api_for_type!(test_api_i8, i8, true, i8::MAX as u64);
test_atomic_api_for_type!(test_api_i16, i16, true, i16::MAX as u64);
test_atomic_api_for_type!(test_api_i32, i32, true, i32::MAX as u64);
test_atomic_api_for_type!(test_api_i64, i64, true, i64::MAX as u64);

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
            assert_eq!(
                vec.load(i, Ordering::SeqCst),
                (thread_id * 1000 + i) as u16
            );
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