//! Comprehensive and robust test suite for `AtomicFixedVec`.
//!
//! This suite validates all aspects of the atomic fixed-width vector, including:
//! - Correctness of the lock-free strategy (power-of-two bit widths).
//! - Correctness of the hybrid seqlock/mutex strategy (non-power-of-two bit widths).
//! - Behavior with both signed (ZigZag encoded) and unsigned integer types.
//! - All atomic operations: load, store, swap, and compare_exchange.
//! - Edge cases such as zero bit width, max bit width, and boundary indices.
//! - Robustness under various multi-threaded concurrency patterns.

#![cfg(feature = "atomic")]

use compressed_intvec::fixed::atomic::AtomicFixedVec;
use compressed_intvec::fixed::traits::{Storable, Word};
use num_traits::{Bounded, FromPrimitive, One, ToPrimitive, Zero};
use std::fmt::Debug;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Barrier};
use std::thread;

// --- Test: Construction and Basic Properties ---

#[test]
fn test_construction_and_properties() {
    // Test valid construction
    let vec = AtomicFixedVec::<u8, u64>::new(8, 100).unwrap();
    assert_eq!(vec.len(), 100);
    assert_eq!(vec.bit_width(), 8);
    assert!(!vec.is_empty());

    // Test empty vector
    let empty_vec = AtomicFixedVec::<u16, u64>::new(10, 0).unwrap();
    assert_eq!(empty_vec.len(), 0);
    assert!(empty_vec.is_empty());

    // Test invalid bit width (greater than word size)
    let result = AtomicFixedVec::<u32, u64>::new(65, 10);
    assert!(result.is_err());
}

// --- Generic Test Runner for Core API ---

/// A generic test runner that validates the core atomic operations for a given configuration.
fn run_core_api_tests<T, W>(bit_width: usize, len: usize)
where
    T: Storable<W>
        + Bounded
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + Debug
        + Copy
        + PartialEq,
    W: Word + Bounded + Zero + One,
    W::AtomicType: Debug,
{
    let vec = AtomicFixedVec::<T, W>::new(bit_width, len).unwrap();
    let max_val_word = if bit_width == <W as Word>::BITS {
        W::max_value()
    } else {
        (W::one() << bit_width).wrapping_sub(W::one())
    };
    let max_val = <T as Storable<W>>::from_word(max_val_word);
    let mid_val = <T as Storable<W>>::from_word(max_val_word >> 1);

    // --- Test `store` and `load` ---
    vec.store(0, T::from_u8(0).unwrap(), Ordering::SeqCst);
    assert_eq!(vec.load(0, Ordering::SeqCst), T::from_u8(0).unwrap());

    vec.store(len - 1, max_val, Ordering::SeqCst);
    assert_eq!(vec.load(len - 1, Ordering::SeqCst), max_val);

    let mid_idx = len / 2;
    vec.store(mid_idx, mid_val, Ordering::SeqCst);
    assert_eq!(vec.load(mid_idx, Ordering::SeqCst), mid_val);

    // --- Test `swap` ---
    let old = vec.swap(0, max_val, Ordering::SeqCst);
    assert_eq!(old, T::from_u8(0).unwrap());
    assert_eq!(vec.load(0, Ordering::SeqCst), max_val);

    // --- Test `compare_exchange` (success) ---
    let result =
        vec.compare_exchange(mid_idx, mid_val, max_val, Ordering::SeqCst, Ordering::Relaxed);
    assert_eq!(result, Ok(mid_val));
    assert_eq!(vec.load(mid_idx, Ordering::SeqCst), max_val);

    // --- Test `compare_exchange` (failure) ---
    let wrong_current = T::from_u8(1).unwrap();
    let result_fail = vec.compare_exchange(
        mid_idx,
        wrong_current,
        T::from_u8(0).unwrap(),
        Ordering::SeqCst,
        Ordering::Relaxed,
    );
    assert_eq!(result_fail, Err(max_val));
    assert_eq!(vec.load(mid_idx, Ordering::SeqCst), max_val); // Unchanged
}

// --- Section: Lock-Free Strategy (Power-of-Two bit_width) ---
#[test]
fn test_lock_free_u16() {
    run_core_api_tests::<u16, u64>(16, 256);
}
#[test]
fn test_lock_free_i32() {
    run_core_api_tests::<i32, u64>(32, 256);
}
#[test]
fn test_lock_free_u8() {
    run_core_api_tests::<u8, u64>(8, 512);
}

// --- Section: Locked Strategy (Non-Power-of-Two bit_width) ---
#[test]
fn test_locked_u32() {
    run_core_api_tests::<u32, u64>(21, 256);
}
#[test]
fn test_locked_i8() {
    run_core_api_tests::<i8, u64>(7, 256);
}
#[test]
fn test_locked_u64() {
    run_core_api_tests::<u64, u64>(40, 128);
}

// --- Section: Edge Case Tests ---
#[test]
fn test_edge_case_zero_bit_width() {
    let vec = AtomicFixedVec::<u32, u64>::new(0, 100).unwrap();
    for i in 0..100 {
        // With 0 bits, the only valid value is 0.
        assert_eq!(vec.load(i, Ordering::SeqCst), 0);
        vec.store(i, 0, Ordering::SeqCst);
        assert_eq!(vec.load(i, Ordering::SeqCst), 0);
        assert_eq!(vec.swap(i, 0, Ordering::SeqCst), 0);
        assert_eq!(
            vec.compare_exchange(i, 0, 0, Ordering::SeqCst, Ordering::Relaxed),
            Ok(0)
        );
    }
}

#[test]
fn test_edge_case_max_bit_width() {
    run_core_api_tests::<u64, u64>(64, 128);
    run_core_api_tests::<i64, u64>(64, 128);
}

// --- Section: Concurrency Tests ---

#[test]
fn test_concurrent_disjoint_stores() {
    const NUM_THREADS: usize = 4;
    const LEN: usize = 1000;
    let vec = Arc::new(AtomicFixedVec::<u16, u64>::new(12, LEN).unwrap());
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let vec_clone = Arc::clone(&vec);
            let barrier_clone = Arc::clone(&barrier);
            thread::spawn(move || {
                let chunk_size = LEN / NUM_THREADS;
                let start = thread_id * chunk_size;
                let end = start + chunk_size;

                barrier_clone.wait();
                for i in start..end {
                    // Each thread writes a unique value based on its ID and index.
                    vec_clone.store(i, (thread_id * 1000 + i) as u16, Ordering::SeqCst);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

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
fn test_concurrent_reads() {
    let vec = Arc::new(AtomicFixedVec::<u32, u64>::new(20, 500).unwrap());
    for i in 0..500 {
        vec.store(i, i as u32, Ordering::SeqCst);
    }

    const NUM_THREADS: usize = 8;
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::new();

    for _ in 0..NUM_THREADS {
        let vec_clone = Arc::clone(&vec);
        let barrier_clone = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier_clone.wait();
            for _ in 0..100 {
                // Each thread repeatedly reads the entire vector.
                for i in 0..500 {
                    assert_eq!(vec_clone.load(i, Ordering::SeqCst), i as u32);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_cas_contention() {
    // Multiple threads incrementing the same counter using CAS.
    let vec = Arc::new(AtomicFixedVec::<u32, u64>::new(16, 1).unwrap());
    vec.store(0, 0, Ordering::SeqCst);

    const NUM_THREADS: usize = 10;
    const INCREMENTS_PER_THREAD: u32 = 1000;
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|_| {
            let vec_clone = Arc::clone(&vec);
            let barrier_clone = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier_clone.wait();
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
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // The final value must be the total number of increments.
    assert_eq!(
        vec.load(0, Ordering::SeqCst),
        NUM_THREADS as u32 * INCREMENTS_PER_THREAD
    );
}

#[test]
fn test_concurrent_mixed_ops_stress() {
    const NUM_THREADS: usize = 8;
    const LEN: usize = 128;
    // Use a non-power-of-two bit_width to stress the locked path.
    let vec = Arc::new(AtomicFixedVec::<i16, u64>::new(11, LEN).unwrap());
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let vec_clone = Arc::clone(&vec);
            let barrier_clone = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier_clone.wait();
                for i in 0..500 {
                    let index = (i * 3 + thread_id) % LEN;
                    let val = (i * 7 + thread_id) as i16;
                    match (i + thread_id) % 4 {
                        0 => vec_clone.store(index, val, Ordering::Relaxed),
                        1 => {
                            let _ = vec_clone.load(index, Ordering::Relaxed);
                        }
                        2 => {
                            let _ = vec_clone.swap(index, val, Ordering::Relaxed);
                        }
                        3 => {
                            let current = vec_clone.load(index, Ordering::Relaxed);
                            let _ = vec_clone.compare_exchange(
                                index,
                                current,
                                val,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            );
                        }
                        _ => unreachable!(),
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // The test passes if it completes without deadlocking or panicking.
    // The final state is non-deterministic.
}