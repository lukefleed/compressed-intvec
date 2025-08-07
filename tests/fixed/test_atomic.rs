//! Comprehensive test suite for AtomicFixedVec
//!
//! This test suite covers all edge cases and scenarios for the atomic
//! fixed-width vector implementation, including:
//! - Single-word operations (lock-free strategy)
//! - Multi-word operations (striped-locking strategy)
//! - Concurrent access patterns
//! - Memory ordering semantics
//! - Boundary conditions
//! - Error handling

#![cfg(feature = "atomic")]

use compressed_intvec::fixed::atomic::AtomicFixedVec;
use dsi_bitstream::prelude::BE;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Barrier};
use std::thread;

// Type aliases for different configurations
type AtomicVecU8BE = AtomicFixedVec<u8, u64, BE>;
type AtomicVecU16BE = AtomicFixedVec<u16, u64, BE>;
type AtomicVecU32BE = AtomicFixedVec<u32, u64, BE>;
type AtomicVecU64BE = AtomicFixedVec<u64, u64, BE>;

#[test]
fn test_basic_construction() {
    // Test valid constructions
    let vec = AtomicVecU8BE::new(8, 10).unwrap();
    assert_eq!(vec.len(), 10);
    assert_eq!(vec.bit_width(), 8);
    assert!(!vec.is_empty());

    // Test empty vector
    let empty_vec = AtomicVecU8BE::new(8, 0).unwrap();
    assert_eq!(empty_vec.len(), 0);
    assert!(empty_vec.is_empty());

    // Test invalid bit width
    let result = AtomicVecU8BE::new(65, 10); // bit_width > word size
    assert!(result.is_err());

    // Test zero bit width - should be valid
    let zero_width = AtomicVecU8BE::new(0, 10).unwrap();
    assert_eq!(zero_width.bit_width(), 0);
}

#[test]
fn test_single_word_operations_power_of_two() {
    // Test bit widths that are powers of 2 (lock-free strategy)
    for bit_width in [1, 2, 4, 8, 16, 32] {
        let vec = AtomicVecU32BE::new(bit_width, 100).unwrap();
        let max_val = (1u32 << bit_width) - 1;
        
        // Test store and load
        vec.store(0, max_val, Ordering::SeqCst);
        assert_eq!(vec.load(0, Ordering::SeqCst), max_val);
        
        vec.store(50, max_val / 2, Ordering::SeqCst);
        assert_eq!(vec.load(50, Ordering::SeqCst), max_val / 2);
        
        // Test boundary values
        vec.store(99, 0, Ordering::SeqCst);
        assert_eq!(vec.load(99, Ordering::SeqCst), 0);
        
        vec.store(99, max_val, Ordering::SeqCst);
        assert_eq!(vec.load(99, Ordering::SeqCst), max_val);
    }
}

#[test]
fn test_multi_word_operations_non_power_of_two() {
    // Test bit widths that are NOT powers of 2 (striped-locking strategy)
    for bit_width in [3, 5, 6, 7, 9, 10, 12, 15, 17, 20, 24, 31] {
        let vec = AtomicVecU32BE::new(bit_width, 50).unwrap();
        let max_val = (1u32 << bit_width) - 1;
        
        // Test store and load
        vec.store(0, max_val, Ordering::SeqCst);
        assert_eq!(vec.load(0, Ordering::SeqCst), max_val);
        
        vec.store(25, max_val / 3, Ordering::SeqCst);
        assert_eq!(vec.load(25, Ordering::SeqCst), max_val / 3);
        
        // Test boundary values
        vec.store(49, 0, Ordering::SeqCst);
        assert_eq!(vec.load(49, Ordering::SeqCst), 0);
        
        vec.store(49, max_val, Ordering::SeqCst);
        assert_eq!(vec.load(49, Ordering::SeqCst), max_val);
    }
}

#[test]
fn test_atomic_swap() {
    let vec = AtomicVecU16BE::new(12, 20).unwrap();
    let max_val = (1u16 << 12) - 1;
    
    // Initialize with some values
    vec.store(0, 100, Ordering::SeqCst);
    vec.store(10, 200, Ordering::SeqCst);
    vec.store(19, max_val, Ordering::SeqCst);
    
    // Test swap operations
    let old_val = vec.swap(0, 500, Ordering::SeqCst);
    assert_eq!(old_val, 100);
    assert_eq!(vec.load(0, Ordering::SeqCst), 500);
    
    let old_val = vec.swap(10, 0, Ordering::SeqCst);
    assert_eq!(old_val, 200);
    assert_eq!(vec.load(10, Ordering::SeqCst), 0);
    
    let old_val = vec.swap(19, 1, Ordering::SeqCst);
    assert_eq!(old_val, max_val);
    assert_eq!(vec.load(19, Ordering::SeqCst), 1);
}

#[test]
fn test_compare_exchange_success() {
    let vec = AtomicVecU8BE::new(6, 30).unwrap();
    let max_val = (1u8 << 6) - 1;
    
    // Initialize values
    vec.store(0, 42, Ordering::SeqCst);
    vec.store(15, max_val, Ordering::SeqCst);
    vec.store(29, 0, Ordering::SeqCst);
    
    // Test successful compare_exchange
    let result = vec.compare_exchange(
        0, 42, 55, 
        Ordering::SeqCst, Ordering::SeqCst
    );
    assert_eq!(result, Ok(42));
    assert_eq!(vec.load(0, Ordering::SeqCst), 55);
    
    let result = vec.compare_exchange(
        15, max_val, 1, 
        Ordering::SeqCst, Ordering::SeqCst
    );
    assert_eq!(result, Ok(max_val));
    assert_eq!(vec.load(15, Ordering::SeqCst), 1);
    
    let result = vec.compare_exchange(
        29, 0, max_val, 
        Ordering::SeqCst, Ordering::SeqCst
    );
    assert_eq!(result, Ok(0));
    assert_eq!(vec.load(29, Ordering::SeqCst), max_val);
}

#[test]
fn test_compare_exchange_failure() {
    let vec = AtomicVecU8BE::new(7, 10).unwrap();
    
    // Initialize values
    vec.store(0, 100, Ordering::SeqCst);
    vec.store(5, 50, Ordering::SeqCst);
    vec.store(9, 127, Ordering::SeqCst); // max for 7 bits
    
    // Test failed compare_exchange (wrong current value)
    let result = vec.compare_exchange(
        0, 99, 200, // current is 100, not 99
        Ordering::SeqCst, Ordering::SeqCst
    );
    assert_eq!(result, Err(100));
    assert_eq!(vec.load(0, Ordering::SeqCst), 100); // unchanged
    
    let result = vec.compare_exchange(
        5, 51, 75, // current is 50, not 51
        Ordering::SeqCst, Ordering::SeqCst
    );
    assert_eq!(result, Err(50));
    assert_eq!(vec.load(5, Ordering::SeqCst), 50); // unchanged
    
    let result = vec.compare_exchange(
        9, 126, 1, // current is 127, not 126
        Ordering::SeqCst, Ordering::SeqCst
    );
    assert_eq!(result, Err(127));
    assert_eq!(vec.load(9, Ordering::SeqCst), 127); // unchanged
}

#[test]
fn test_memory_ordering_variants() {
    let vec = AtomicVecU8BE::new(4, 5).unwrap();
    
    // Test all memory ordering variants
    let orderings = [
        Ordering::Relaxed,
        Ordering::Acquire,
        Ordering::Release,
        Ordering::AcqRel,
        Ordering::SeqCst,
    ];
    
    for (i, &ordering) in orderings.iter().enumerate() {
        let value = (i as u8) % 16; // 4-bit values
        
        vec.store(i % 5, value, ordering);
        let loaded = vec.load(i % 5, ordering);
        assert_eq!(loaded, value);
        
        let swapped = vec.swap(i % 5, value + 1, ordering);
        assert_eq!(swapped, value);
        
        // Test compare_exchange with different orderings
        let result = vec.compare_exchange(
            i % 5, value + 1, value + 2,
            ordering, ordering
        );
        assert_eq!(result, Ok(value + 1));
    }
}

#[test]
fn test_boundary_indices() {
    let vec = AtomicVecU16BE::new(10, 100).unwrap();
    let max_val = (1u16 << 10) - 1;
    
    // Test first index
    vec.store(0, max_val, Ordering::SeqCst);
    assert_eq!(vec.load(0, Ordering::SeqCst), max_val);
    
    // Test last index
    vec.store(99, max_val / 2, Ordering::SeqCst);
    assert_eq!(vec.load(99, Ordering::SeqCst), max_val / 2);
    
    // Test some middle indices
    for i in [1, 33, 50, 66, 98] {
        let val = (i as u16) % max_val;
        vec.store(i, val, Ordering::SeqCst);
        assert_eq!(vec.load(i, Ordering::SeqCst), val);
    }
}

#[test]
fn test_word_boundary_crossing() {
    // Use a bit width that will cause elements to cross word boundaries
    let vec = AtomicVecU64BE::new(33, 50).unwrap(); // 33 bits per element
    let max_val = (1u64 << 33) - 1;
    
    // Test elements that span word boundaries
    for i in 0..50 {
        let val = ((i as u64) * 12345) % max_val;
        vec.store(i, val, Ordering::SeqCst);
        assert_eq!(vec.load(i, Ordering::SeqCst), val);
    }
    
    // Test specific patterns that stress word boundary crossing
    let test_values = [
        0,
        1,
        max_val,
        max_val / 2,
        0x155555555, // Alternating bits
        0xAAAAAAAAA, // Alternating bits (inverse)
    ];
    
    for (i, &val) in test_values.iter().enumerate() {
        if i < 50 {
            vec.store(i, val, Ordering::SeqCst);
            assert_eq!(vec.load(i, Ordering::SeqCst), val);
        }
    }
}

#[test]
fn test_concurrent_single_threaded_operations() {
    let vec = AtomicVecU32BE::new(16, 1000).unwrap();
    
    // Perform many operations in sequence to test consistency
    for i in 0..1000 {
        let val = (i * 7) % 65536; // 16-bit values
        vec.store(i, val as u32, Ordering::SeqCst);
    }
    
    // Verify all values
    for i in 0..1000 {
        let expected = (i * 7) % 65536;
        assert_eq!(vec.load(i, Ordering::SeqCst), expected as u32);
    }
    
    // Test interleaved operations
    for i in 0..500 {
        let old_val = vec.swap(i, (i * 13) as u32, Ordering::SeqCst);
        let expected_old = (i * 7) % 65536;
        assert_eq!(old_val, expected_old as u32);
    }
}

#[test]
fn test_concurrent_multi_threaded_loads() {
    let vec = Arc::new(AtomicVecU8BE::new(8, 100).unwrap());
    
    // Initialize with known values
    for i in 0..100 {
        vec.store(i, (i * 3) as u8, Ordering::SeqCst);
    }
    
    const NUM_THREADS: usize = 8;
    const READS_PER_THREAD: usize = 1000;
    
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::new();
    
    for _thread_id in 0..NUM_THREADS {
        let vec = Arc::clone(&vec);
        let barrier = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier.wait();
            
            for _ in 0..READS_PER_THREAD {
                for i in 0..100 {
                    let val = vec.load(i, Ordering::SeqCst);
                    assert_eq!(val, (i * 3) as u8);
                }
            }
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_multi_threaded_stores() {
    let vec = Arc::new(AtomicVecU16BE::new(12, 100).unwrap());
    const NUM_THREADS: usize = 4;
    
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::new();
    
    for thread_id in 0..NUM_THREADS {
        let vec = Arc::clone(&vec);
        let barrier = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier.wait();
            
            // Each thread writes to its own range of indices
            let start = thread_id * 25;
            let end = start + 25;
            
            for i in start..end {
                let val = ((thread_id * 1000 + i) % 4096) as u16; // 12-bit values
                vec.store(i, val, Ordering::SeqCst);
            }
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify all values were written correctly
    for thread_id in 0..NUM_THREADS {
        let start = thread_id * 25;
        let end = start + 25;
        
        for i in start..end {
            let expected = ((thread_id * 1000 + i) % 4096) as u16;
            assert_eq!(vec.load(i, Ordering::SeqCst), expected);
        }
    }
}

#[test]
fn test_concurrent_compare_exchange_contention() {
    let vec = Arc::new(AtomicVecU8BE::new(8, 1).unwrap());
    vec.store(0, 0, Ordering::SeqCst);
    
    const NUM_THREADS: usize = 10;
    const INCREMENTS_PER_THREAD: usize = 100;
    
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::new();
    
    for _ in 0..NUM_THREADS {
        let vec = Arc::clone(&vec);
        let barrier = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier.wait();
            
            for _ in 0..INCREMENTS_PER_THREAD {
                loop {
                    let current = vec.load(0, Ordering::SeqCst);
                    if current >= 255 { break; } // Prevent overflow
                    
                    match vec.compare_exchange(
                        0, current, current + 1,
                        Ordering::SeqCst, Ordering::SeqCst
                    ) {
                        Ok(_) => break, // Successfully incremented
                        Err(_) => continue, // Retry
                    }
                }
            }
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Final value should be the number of successful increments
    let final_val = vec.load(0, Ordering::SeqCst);
    assert!(final_val <= (NUM_THREADS * INCREMENTS_PER_THREAD) as u8);
    assert!(final_val > 0); // At least some increments should have succeeded
}

#[test]
fn test_concurrent_mixed_operations() {
    let vec = Arc::new(AtomicVecU32BE::new(20, 50).unwrap());
    const NUM_THREADS: usize = 6;
    
    // Initialize with some values
    for i in 0..50 {
        vec.store(i, i as u32, Ordering::SeqCst);
    }
    
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::new();
    
    for thread_id in 0..NUM_THREADS {
        let vec = Arc::clone(&vec);
        let barrier = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier.wait();
            
            for iteration in 0..100 {
                let index = (thread_id * 7 + iteration) % 50;
                
                match thread_id % 3 {
                    0 => {
                        // Store operations
                        let val = ((thread_id * 1000 + iteration) % 1048576) as u32; // 20-bit
                        vec.store(index, val, Ordering::SeqCst);
                    }
                    1 => {
                        // Load operations
                        let _val = vec.load(index, Ordering::SeqCst);
                    }
                    2 => {
                        // Swap operations
                        let new_val = ((thread_id * 500 + iteration) % 1048576) as u32;
                        let _old_val = vec.swap(index, new_val, Ordering::SeqCst);
                    }
                    _ => unreachable!(),
                }
            }
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify the vector is still in a consistent state
    for i in 0..50 {
        let val = vec.load(i, Ordering::SeqCst);
        assert!(val < 1048576); // All values should be within 20-bit range
    }
}

#[test]
fn test_stress_test_large_vector() {
    let vec = AtomicVecU64BE::new(40, 10000).unwrap();
    let max_val = (1u64 << 40) - 1;
    
    // Fill with pseudo-random pattern
    for i in 0..10000 {
        let val = ((i as u64).wrapping_mul(1664525).wrapping_add(1013904223)) % max_val;
        vec.store(i, val, Ordering::SeqCst);
    }
    
    // Verify all values
    for i in 0..10000 {
        let expected = ((i as u64).wrapping_mul(1664525).wrapping_add(1013904223)) % max_val;
        assert_eq!(vec.load(i, Ordering::SeqCst), expected);
    }
    
    // Test random access pattern
    let indices = [0, 1, 100, 999, 1000, 5000, 9998, 9999];
    for &i in &indices {
        let old_val = vec.load(i, Ordering::SeqCst);
        let new_val = old_val.wrapping_add(1) % max_val;
        
        let swapped = vec.swap(i, new_val, Ordering::SeqCst);
        assert_eq!(swapped, old_val);
        assert_eq!(vec.load(i, Ordering::SeqCst), new_val);
    }
}

#[test]
fn test_edge_case_bit_widths() {
    // Test minimum bit width
    let vec1 = AtomicVecU8BE::new(1, 10).unwrap();
    for i in 0..10 {
        vec1.store(i, (i % 2) as u8, Ordering::SeqCst);
        assert_eq!(vec1.load(i, Ordering::SeqCst), (i % 2) as u8);
    }
    
    // Test maximum bit width for each word type
    let vec8 = AtomicVecU8BE::new(8, 10).unwrap();
    for i in 0..10 {
        vec8.store(i, 255, Ordering::SeqCst);
        assert_eq!(vec8.load(i, Ordering::SeqCst), 255);
    }
    
    let vec16 = AtomicVecU16BE::new(16, 10).unwrap();
    for i in 0..10 {
        vec16.store(i, 65535, Ordering::SeqCst);
        assert_eq!(vec16.load(i, Ordering::SeqCst), 65535);
    }
    
    let vec32 = AtomicVecU32BE::new(32, 10).unwrap();
    for i in 0..10 {
        vec32.store(i, 0xFFFFFFFF, Ordering::SeqCst);
        assert_eq!(vec32.load(i, Ordering::SeqCst), 0xFFFFFFFF);
    }
    
    let vec64 = AtomicVecU64BE::new(64, 10).unwrap();
    for i in 0..10 {
        vec64.store(i, 0xFFFFFFFFFFFFFFFF, Ordering::SeqCst);
        assert_eq!(vec64.load(i, Ordering::SeqCst), 0xFFFFFFFFFFFFFFFF);
    }
}

#[test]
fn test_alignment_and_packing() {
    // Test that values are correctly packed and aligned
    let vec = AtomicVecU32BE::new(12, 100).unwrap();
    
    // Store alternating pattern
    for i in 0..100 {
        let val = if i % 2 == 0 { 0xFFF } else { 0x000 }; // 12-bit max or min
        vec.store(i, val, Ordering::SeqCst);
    }
    
    // Verify pattern
    for i in 0..100 {
        let expected = if i % 2 == 0 { 0xFFF } else { 0x000 };
        assert_eq!(vec.load(i, Ordering::SeqCst), expected);
    }
    
    // Test specific bit patterns
    let bit_patterns = [
        0x000, // All zeros
        0xFFF, // All ones
        0x555, // Alternating 0101...
        0xAAA, // Alternating 1010...
        0x123, // Random pattern
        0x789, // Another pattern
    ];
    
    for (i, &pattern) in bit_patterns.iter().enumerate() {
        if i < 100 {
            vec.store(i, pattern, Ordering::SeqCst);
            assert_eq!(vec.load(i, Ordering::SeqCst), pattern);
        }
    }
}

#[test]
fn test_zero_bit_width_edge_case() {
    let vec = AtomicVecU8BE::new(0, 100).unwrap();
    
    // All operations should work with 0-bit width (all values are 0)
    for i in 0..100 {
        vec.store(i, 0, Ordering::SeqCst); // Only 0 is valid for 0-bit width
        assert_eq!(vec.load(i, Ordering::SeqCst), 0);
        
        let swapped = vec.swap(i, 0, Ordering::SeqCst);
        assert_eq!(swapped, 0);
        
        let result = vec.compare_exchange(
            i, 0, 0,
            Ordering::SeqCst, Ordering::SeqCst
        );
        assert_eq!(result, Ok(0));
    }
}

#[test]
fn test_performance_characteristics() {
    // This test is more about ensuring operations complete in reasonable time
    // rather than measuring exact performance
    
    let start = std::time::Instant::now();
    
    // Test lock-free operations (should be very fast)
    let vec_lockfree = AtomicVecU32BE::new(16, 10000).unwrap(); // Power of 2
    for i in 0..10000 {
        vec_lockfree.store(i, (i % 65536) as u32, Ordering::SeqCst);
    }
    for i in 0..10000 {
        let _val = vec_lockfree.load(i, Ordering::SeqCst);
    }
    
    let lockfree_time = start.elapsed();
    
    let start = std::time::Instant::now();
    
    // Test locked operations (should still be reasonably fast)
    let vec_locked = AtomicVecU32BE::new(17, 10000).unwrap(); // Not power of 2
    for i in 0..10000 {
        vec_locked.store(i, (i % 131072) as u32, Ordering::SeqCst);
    }
    for i in 0..10000 {
        let _val = vec_locked.load(i, Ordering::SeqCst);
    }
    
    let locked_time = start.elapsed();
    
    // Both should complete in reasonable time (less than 1 second each)
    assert!(lockfree_time.as_secs() < 1);
    assert!(locked_time.as_secs() < 1);
    
    println!("Lock-free operations took: {:?}", lockfree_time);
    println!("Locked operations took: {:?}", locked_time);
}

#[test]
fn test_deterministic_concurrent_behavior() {
    // Test that concurrent operations produce deterministic results
    // when the access patterns don't conflict
    
    let vec = Arc::new(AtomicVecU32BE::new(16, 1000).unwrap());
    const NUM_RUNS: usize = 5;
    
    for _run in 0..NUM_RUNS {
        // Reset vector
        for i in 0..1000 {
            vec.store(i, 0, Ordering::SeqCst);
        }
        
        const NUM_THREADS: usize = 4;
        let barrier = Arc::new(Barrier::new(NUM_THREADS));
        let mut handles = Vec::new();
        
        for thread_id in 0..NUM_THREADS {
            let vec = Arc::clone(&vec);
            let barrier = Arc::clone(&barrier);
            
            let handle = thread::spawn(move || {
                barrier.wait();
                
                // Each thread works on its own disjoint range
                let start = thread_id * 250;
                let end = start + 250;
                
                for i in start..end {
                    vec.store(i, (i * 17) as u32, Ordering::SeqCst);
                }
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify results are consistent across runs
        for i in 0..1000 {
            let expected = (i * 17) as u32;
            assert_eq!(vec.load(i, Ordering::SeqCst), expected);
        }
    }
}
