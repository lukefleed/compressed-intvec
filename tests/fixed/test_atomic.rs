//! Comprehensive and robust test suite for `AtomicFixedVec`.
//!
//! This suite validates all aspects of the atomic fixed-width vector, including:
//! - Correctness of the single-word lock-free strategy (power-of-two bit widths).
//! - Correctness of the multi-word spanning strategy using `atomic::Atomic<u128>`.
//! - Behavior with both signed (ZigZag encoded) and unsigned integer types.
//! - All atomic operations: load, store, swap, and compare_exchange.
//! - Edge cases such as zero bit width, max bit width, and boundary indices.
//! - Robustness under various multi-threaded concurrency patterns, including
//!   a specific test to detect and prevent torn writes.

#![cfg(feature = "atomic")]

use compressed_intvec::fixed::atomic::AtomicFixedVec;
use compressed_intvec::fixed::traits::{Storable, Word};
use num_traits::{Bounded, FromPrimitive, One, ToPrimitive, Zero};
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use sux::prelude::AtomicBitFieldSlice;

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

    // Test invalid configuration: spanning logic is only supported for u64 words.
    let result_spanning_on_u32 = AtomicFixedVec::<u16, u32>::new(17, 10);
    assert!(result_spanning_on_u32.is_err());
    // This should be ok as it does not require spanning logic
    assert!(AtomicFixedVec::<u8, u32>::new(8, 10).is_ok());
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
    W: Word + Bounded + Zero + One + FromPrimitive,
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
    let zero_val = <T as Storable<W>>::from_word(W::zero());

    // --- Test `store` and `load` ---
    vec.store(0, zero_val, Ordering::SeqCst);
    assert_eq!(vec.load(0, Ordering::SeqCst), zero_val);

    vec.store(len - 1, max_val, Ordering::SeqCst);
    assert_eq!(vec.load(len - 1, Ordering::SeqCst), max_val);

    let mid_idx = len / 2;
    vec.store(mid_idx, mid_val, Ordering::SeqCst);
    assert_eq!(vec.load(mid_idx, Ordering::SeqCst), mid_val);

    // --- Test `swap` ---
    let old = vec.swap(0, max_val, Ordering::SeqCst);
    assert_eq!(old, zero_val);
    assert_eq!(vec.load(0, Ordering::SeqCst), max_val);

    // --- Test `compare_exchange` (success) ---
    let result =
        vec.compare_exchange(mid_idx, mid_val, max_val, Ordering::SeqCst, Ordering::Relaxed);
    assert_eq!(result, Ok(mid_val));
    assert_eq!(vec.load(mid_idx, Ordering::SeqCst), max_val);

    // --- Test `compare_exchange` (failure) ---
    let wrong_current = T::from_u8(1).unwrap();
    let result_fail =
        vec.compare_exchange(mid_idx, wrong_current, zero_val, Ordering::SeqCst, Ordering::Relaxed);
    assert_eq!(result_fail, Err(max_val));
    assert_eq!(vec.load(mid_idx, Ordering::SeqCst), max_val); // Unchanged
}

// --- Section: Single-Word Lock-Free Strategy ---
#[test]
fn test_single_word_u16_on_u64() {
    run_core_api_tests::<u16, u64>(16, 256);
}
#[test]
fn test_single_word_i32_on_u64() {
    run_core_api_tests::<i32, u64>(32, 256);
}
#[test]
fn test_single_word_u8_on_u64() {
    run_core_api_tests::<u8, u64>(8, 512);
}

// --- Section: Multi-Word (Spanning) Strategy ---
#[test]
fn test_spanning_u32_on_u64() {
    run_core_api_tests::<u32, u64>(21, 256);
}
#[test]
fn test_spanning_i8_on_u64() {
    run_core_api_tests::<i8, u64>(7, 256);
}
#[test]
fn test_spanning_u64_on_u64() {
    run_core_api_tests::<u64, u64>(40, 128);
}

// --- Section: Edge Case Tests ---
#[test]
fn test_edge_case_zero_bit_width() {
    let vec = AtomicFixedVec::<u32, u64>::new(0, 100).unwrap();
    for i in 0..100 {
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
    let vec = Arc::new(AtomicFixedVec::<u32, u64>::new(16, 1).unwrap());
    vec.store(0, 0, Ordering::SeqCst);

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
fn test_concurrent_spanning_writes_correctness() {
    const BIT_WIDTH: usize = 21;
    const LEN: usize = 256;
    let vec = Arc::new(AtomicFixedVec::<u64, u64>::new(BIT_WIDTH, LEN).unwrap());

    let pattern_a = 0b010101010101010101010;
    let pattern_b = 0b101010101010101010101;

    let indices_to_test: Vec<usize> = (0..LEN)
        .filter(|&i| {
            let bit_pos = i * BIT_WIDTH;
            let bit_offset = bit_pos % 64;
            bit_offset + BIT_WIDTH > 64
        })
        .collect();
    assert!(!indices_to_test.is_empty());

    thread::scope(|s| {
        for &test_index in &indices_to_test {
            let vec_a = Arc::clone(&vec);
            s.spawn(move || {
                for _ in 0..100 {
                    vec_a.store(test_index, pattern_a, Ordering::SeqCst);
                }
            });

            let vec_b = Arc::clone(&vec);
            s.spawn(move || {
                for _ in 0..100 {
                    vec_b.store(test_index, pattern_b, Ordering::SeqCst);
                }
            });
        }
    });

    for &test_index in &indices_to_test {
        let final_value = vec.load(test_index, Ordering::SeqCst);
        assert!(
            final_value == pattern_a || final_value == pattern_b,
            "Torn write detected at index {}! Final value was {:b}, expected {:b} or {:b}",
            test_index,
            final_value,
            pattern_a,
            pattern_b
        );
    }
}

#[test]
fn test_sux_torn_write_scenario() {
    const BIT_WIDTH: usize = 21;
    const LEN: usize = 256;
    const NUM_ITERATIONS: usize = 50; // How many times to retry the race.

    let pattern_a = 0b010101010101010101010;
    let pattern_b = 0b101010101010101010101;

    let test_index = (0..LEN)
        .find(|&i| (i * BIT_WIDTH) % 64 + BIT_WIDTH > 64)
        .expect("Test setup failed: no spanning index found");

    for i in 0..NUM_ITERATIONS {
        let sux_vec_storage: Arc<Vec<AtomicU64>> = Arc::new(
            (0..(LEN * BIT_WIDTH).div_ceil(u64::BITS as usize) + 2)
                .map(|_| AtomicU64::new(0))
                .collect(),
        );

        let sux_vec = unsafe {
            sux::bits::AtomicBitFieldVec::<u64, _>::from_raw_parts(
                sux_vec_storage.as_slice(),
                BIT_WIDTH,
                LEN,
            )
        };
        let sux_vec = Arc::new(sux_vec);
        let stop_signal = Arc::new(AtomicBool::new(false));

        thread::scope(|s| {
            let sux_a = Arc::clone(&sux_vec);
            let stop_a = Arc::clone(&stop_signal);
            s.spawn(move || {
                while !stop_a.load(Ordering::Relaxed) {
                    unsafe {
                        sux_a.set_atomic_unchecked(test_index, pattern_a, Ordering::SeqCst);
                    }
                }
            });

            let sux_b = Arc::clone(&sux_vec);
            let stop_b = Arc::clone(&stop_signal);
            s.spawn(move || {
                while !stop_b.load(Ordering::Relaxed) {
                    unsafe {
                        sux_b.set_atomic_unchecked(test_index, pattern_b, Ordering::SeqCst);
                    }
                }
            });
            
            // Let the threads race for a short period.
            thread::sleep(Duration::from_millis(10));
            stop_signal.store(true, Ordering::Relaxed);
        });

        let final_value = unsafe { sux_vec.get_atomic_unchecked(test_index, Ordering::SeqCst) };

        if final_value != pattern_a && final_value != pattern_b {
            println!(
                "SUCCESS: Detected torn write in sux on iteration {}. Final value: {:b}",
                i + 1,
                final_value
            );
            return; // Test successfully demonstrated the race condition.
        }
    }

    // If the loop finishes, the race condition was not triggered.
    // This is not a failure of our test, but a success for the scheduler.
    // We print a warning instead of panicking.
    println!("Warning: Torn write in sux was not detected after {} iterations.", NUM_ITERATIONS);
}