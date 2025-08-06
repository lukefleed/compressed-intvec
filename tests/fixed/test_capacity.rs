//! Integration tests for capacity management in `FixedVec`.

use compressed_intvec::fixed::{FixedVec, SFixedVec, UFixedVec};

#[test]
fn test_with_capacity() {
    // Unsigned
    let vec_u: UFixedVec<u32> = FixedVec::with_capacity(10, 1000).unwrap();
    assert_eq!(vec_u.len(), 0);
    assert!(vec_u.capacity() >= 1000);
    let initial_word_cap = vec_u.word_capacity();
    assert!(initial_word_cap > 0);

    // Signed
    let vec_s: SFixedVec<i16> = FixedVec::with_capacity(9, 500).unwrap();
    assert_eq!(vec_s.len(), 0);
    assert!(vec_s.capacity() >= 500);

    // Zero capacity
    let vec_zero: UFixedVec<u8> = FixedVec::with_capacity(8, 0).unwrap();
    assert_eq!(vec_zero.capacity(), 0);
    assert_eq!(vec_zero.word_capacity(), 0);
}

#[test]
fn test_reserve() {
    let mut vec: UFixedVec<u64> = FixedVec::with_capacity(20, 10).unwrap();
    assert_eq!(vec.len(), 0);
    assert!(vec.capacity() >= 10);
    let initial_word_cap = vec.word_capacity();

    // Reserve less than current capacity, should not reallocate.
    vec.reserve(5);
    assert_eq!(vec.word_capacity(), initial_word_cap);

    // Reserve more, should reallocate.
    let current_len = vec.len();
    vec.reserve(100);
    assert!(vec.capacity() >= current_len + 100);
    assert!(vec.word_capacity() > initial_word_cap);
}

#[test]
fn test_push_triggers_reserve() {
    // We create a vector with bit_width=17 and a u32 word size (32 bits).
    // This configuration forces elements to span word boundaries frequently.
    type TestVec = FixedVec<u32, u32, dsi_bitstream::prelude::LE>;
    let mut vec: TestVec = TestVec::with_capacity(17, 1).unwrap();
    
    // The first element fits.
    // required_bits = 1 * 17 = 17.
    // required_words = (17 + 31) / 32 = 1.
    // required_vec_len = 1 + 1 = 2.
    // The initial capacity of `bits` should be 2.
    vec.push(1);
    assert_eq!(vec.len(), 1);
    let cap_before = vec.word_capacity();

    // The second element.
    // required_bits = 2 * 17 = 34.
    // required_words = (34 + 31) / 32 = 2.
    // required_vec_len = 2 + 1 = 3.
    // The `bits` vector must now grow to length 3, which will very likely
    // trigger a reallocation if the initial capacity was small (e.g., 2).
    vec.push(2);
    let cap_after = vec.word_capacity();

    assert_eq!(vec.len(), 2);
    assert_eq!(vec.get(1), Some(2));
    assert!(cap_after > cap_before, "Capacity should have grown. Before: {}, After: {}", cap_before, cap_after);
}

#[test]
fn test_shrink_to_fit() {
    let mut vec: UFixedVec<u32> = FixedVec::with_capacity(10, 1000).unwrap();
    for i in 0..100 {
        vec.push(i);
    }

    assert_eq!(vec.len(), 100);
    let cap_before_shrink = vec.capacity();
    assert!(cap_before_shrink >= 1000);

    vec.shrink_to_fit();

    let cap_after_shrink = vec.capacity();
    assert_eq!(vec.len(), 100);
    assert!(
        cap_after_shrink < cap_before_shrink,
        "Capacity should have been reduced"
    );
    // Capacity might be slightly larger than len due to word alignment, but not by much.
    assert!(
        cap_after_shrink >= 100 && cap_after_shrink < 110,
        "Capacity after shrink is unexpected: {}",
        cap_after_shrink
    );

    // Verify content is preserved.
    for i in 0..100 {
        assert_eq!(vec.get(i), Some(i as u32));
    }
}

#[test]
fn test_shrink_to_fit_on_empty() {
    let mut vec: UFixedVec<u8> = FixedVec::with_capacity(8, 100).unwrap();
    assert!(vec.word_capacity() > 0);

    vec.shrink_to_fit();
    assert_eq!(vec.word_capacity(), 0);

    vec.push(10);
    vec.clear();
    assert!(vec.word_capacity() > 0);
    vec.shrink_to_fit();
    assert_eq!(vec.word_capacity(), 0);
}

#[test]
fn test_resize() {
    // 1. Test extending the vector
    let mut vec: UFixedVec<u32> = FixedVec::new(8).unwrap();
    vec.push(10);
    vec.push(20);
    vec.push(30);

    vec.resize(5, 99);
    assert_eq!(vec.len(), 5);
    assert_eq!(vec.get(0), Some(10));
    assert_eq!(vec.get(1), Some(20));
    assert_eq!(vec.get(2), Some(30));
    assert_eq!(vec.get(3), Some(99));
    assert_eq!(vec.get(4), Some(99));

    // 2. Test truncating the vector
    vec.resize(2, 0); // The value '0' should be ignored
    assert_eq!(vec.len(), 2);
    assert_eq!(vec.get(0), Some(10));
    assert_eq!(vec.get(1), Some(20));
    assert_eq!(vec.get(2), None);

    // 3. Test resizing to the same length
    vec.resize(2, 111);
    assert_eq!(vec.len(), 2);
    assert_eq!(vec.get(1), Some(20));

    // 4. Test resizing an empty vector
    let mut empty_vec: SFixedVec<i16> = FixedVec::new(10).unwrap();
    empty_vec.resize(3, -1);
    assert_eq!(empty_vec.len(), 3);
    assert_eq!(empty_vec.get(0), Some(-1));
    assert_eq!(empty_vec.get(1), Some(-1));
    assert_eq!(empty_vec.get(2), Some(-1));

    // 5. Test resizing to zero
    empty_vec.resize(0, 0);
    assert!(empty_vec.is_empty());
}