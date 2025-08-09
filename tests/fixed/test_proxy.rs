//! Integration tests for the MutProxy object.

use compressed_intvec::fixed::{FixedVec, UFixedVec};

#[test]
fn test_at_mut_proxy() {
    // Explicitly create a vector with a bit_width large enough for all test values.
    // 999 requires 10 bits. Let's use 10.
    let mut vec: UFixedVec<u32> = FixedVec::new(10).unwrap();
    for i in 0..100u32 {
        vec.push(i);
    }

    // Read and write through the proxy
    assert_eq!(vec.get(10), Some(10));
    *vec.at_mut(10).unwrap() = 999;
    assert_eq!(vec.get(10), Some(999));

    // Perform a read-modify-write operation
    // Initial value is 20. 20 + 50 = 70. 70 fits in 10 bits.
    *vec.at_mut(20).unwrap() += 50;
    assert_eq!(vec.get(20), Some(70));

    // Test out of bounds
    assert!(vec.at_mut(100).is_none());
}

#[test]
#[should_panic]
fn test_at_mut_value_too_large_panics() {
    // This vec is configured with a bit_width of 4, so it can only hold values up to 15.
    let mut vec: UFixedVec<u8> = FixedVec::new(4).unwrap();
    vec.push(10);

    // The value 20 is a valid u8 literal, so this compiles.
    // However, it is larger than the maximum value (15) that can be stored
    // with a bit_width of 4.
    // The panic should occur inside the `Drop` impl of the proxy, when it calls `set`,
    // which in turn panics because the value is too large for the configured bit_width.
    *vec.at_mut(0).unwrap() = 20;
}
