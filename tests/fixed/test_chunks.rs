//! Integration tests for chunking functionality.

use compressed_intvec::fixed::{FixedVec, UFixedVec};

#[test]
fn test_chunks_exact() {
    let vec: UFixedVec<u32> = (0..100u32).collect();
    let mut iter = vec.chunks(20);

    let chunk1 = iter.next().unwrap();
    assert_eq!(chunk1.len(), 20);
    assert_eq!(chunk1.get(0), Some(0));
    assert_eq!(chunk1.get(19), Some(19));

    let chunk5 = iter.nth(3).unwrap(); // Skip 3, get the 5th chunk (0-indexed 4th)
    assert_eq!(chunk5.len(), 20);
    assert_eq!(chunk5.get(0), Some(80));
    assert_eq!(chunk5.get(19), Some(99));

    assert!(iter.next().is_none());
}

#[test]
fn test_chunks_with_remainder() {
    let vec: UFixedVec<u8> = (0..10u8).collect();
    let mut iter = vec.chunks(3);

    let chunk1 = iter.next().unwrap();
    assert_eq!(chunk1, &[0, 1, 2][..]);

    let chunk2 = iter.next().unwrap();
    assert_eq!(chunk2, &[3, 4, 5][..]);

    let chunk3 = iter.next().unwrap();
    assert_eq!(chunk3, &[6, 7, 8][..]);

    let chunk4 = iter.next().unwrap(); // Remainder
    assert_eq!(chunk4.len(), 1);
    assert_eq!(chunk4, &[9][..]);

    assert!(iter.next().is_none());
}

#[test]
fn test_chunks_larger_than_vec() {
    let vec: UFixedVec<u16> = (0..10u16).collect();
    let mut iter = vec.chunks(100);

    let chunk = iter.next().unwrap();
    assert_eq!(chunk.len(), 10);
    assert_eq!(chunk, &vec);

    assert!(iter.next().is_none());
}

#[test]
fn test_chunks_on_empty_vec() {
    let vec: UFixedVec<u64> = FixedVec::new(8).unwrap();
    let mut iter = vec.chunks(10);
    assert!(iter.next().is_none());
}

#[test]
#[should_panic]
fn test_chunks_with_zero_size() {
    let vec: UFixedVec<u32> = (0..10u32).collect();
    // This should panic
    let _ = vec.chunks(0);
}

#[test]
fn test_chunks_mut() {
    // Max value to be written is 90 + 9*100 = 990. This needs 10 bits.
    let mut vec: UFixedVec<u32> = FixedVec::with_capacity(10, 100).unwrap();
    for i in 0..100u32 {
        vec.push(i);
    }

    // Modify the first element of each chunk.
    for (chunk_idx, mut chunk) in vec.chunks_mut(10).enumerate() {
        let first_val = chunk.get(0).unwrap();
        *chunk.at_mut(0).unwrap() = first_val + (chunk_idx as u32 * 100);
    }

    assert_eq!(vec.get(0), Some(0)); // 0 + 0*100
    assert_eq!(vec.get(9), Some(9)); // Unchanged
    assert_eq!(vec.get(10), Some(110)); // 10 + 1*100
    assert_eq!(vec.get(19), Some(19)); // Unchanged
    assert_eq!(vec.get(90), Some(990)); // 90 + 9*100
}

#[test]
fn test_chunks_mut_with_remainder() {
    // The value 99 needs 7 bits.
    let mut vec: UFixedVec<u8> = FixedVec::with_capacity(7, 10).unwrap();
    for i in 0..10u8 {
        vec.push(i);
    }
    
    for mut chunk in vec.chunks_mut(3) {
        if !chunk.is_empty() {
            *chunk.at_mut(0).unwrap() = 99;
        }
    }
    
    assert_eq!(vec.get(0), Some(99));
    assert_eq!(vec.get(1), Some(1));
    assert_eq!(vec.get(3), Some(99));
    assert_eq!(vec.get(6), Some(99));
    assert_eq!(vec.get(9), Some(99));
}