// tests/variable/test_deprecated.rs

//! Tests for deprecated type aliases.
//!
//! This test suite verifies that all deprecated type aliases (IntVec*, UIntVec, etc.)
//! remain functional and correctly map to their new counterparts (VarVec*, UVarVec, etc.).
//! These tests ensure backward compatibility for external users during the deprecation period.

#![allow(deprecated)]

use compressed_intvec::variable::{
    BEIntVec, BESIntVec, IntVec, IntVecBuilder, IntVecError, IntVecFromIterBuilder, IntVecIntoIter,
    IntVecIter, IntVecReader, IntVecSeqReader, IntVecSlice, IntVecSliceIter, LEIntVec, LESIntVec,
    SIntVec, UIntVec, VarVec, VariableCodecSpec,
};
use dsi_bitstream::prelude::{BE, LE};

/// Tests the deprecated `IntVec` type alias for basic functionality.
#[test]
fn test_deprecated_intvec() {
    let data: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 200];
    let vec: IntVec<u32, LE> = VarVec::from_slice(&data).unwrap();

    assert_eq!(vec.len(), 7);
    assert!(!vec.is_empty());
    assert_eq!(vec.get(0), Some(1));
    assert_eq!(vec.get(5), Some(100));
    assert_eq!(vec.get(6), Some(200));
    assert_eq!(vec.get(7), None);
}

/// Tests the deprecated `UIntVec` convenience alias.
#[test]
fn test_deprecated_uintvec() {
    let data: Vec<u32> = vec![10, 20, 30, 40, 50];
    let vec: UIntVec<u32> = VarVec::from_slice(&data).unwrap();

    assert_eq!(vec.len(), 5);
    assert_eq!(vec.get(2), Some(30));

    let sum: u32 = vec.iter().sum();
    assert_eq!(sum, 150);
}

/// Tests the deprecated `SIntVec` convenience alias with signed integers.
#[test]
fn test_deprecated_sintvec() {
    let data: Vec<i32> = vec![-10, 20, -30, 40, -50];
    let vec: SIntVec<i32> = VarVec::from_slice(&data).unwrap();

    assert_eq!(vec.len(), 5);
    assert_eq!(vec.get(0), Some(-10));
    assert_eq!(vec.get(2), Some(-30));
    assert_eq!(vec.get(4), Some(-50));
}

/// Tests the deprecated `LEIntVec` and `BEIntVec` aliases.
#[test]
fn test_deprecated_endianness_aliases() {
    let data: Vec<u64> = vec![1, 2, 3, 4];

    let le_vec: LEIntVec = VarVec::from_slice(&data).unwrap();
    assert_eq!(le_vec.len(), 4);
    assert_eq!(le_vec.get(1), Some(2));

    let be_vec: BEIntVec = VarVec::from_slice(&data).unwrap();
    assert_eq!(be_vec.len(), 4);
    assert_eq!(be_vec.get(1), Some(2));
}

/// Tests the deprecated signed endianness aliases.
#[test]
fn test_deprecated_signed_endianness_aliases() {
    let data: Vec<i64> = vec![-1, 2, -3, 4];

    let le_vec: LESIntVec = VarVec::from_slice(&data).unwrap();
    assert_eq!(le_vec.len(), 4);
    assert_eq!(le_vec.get(0), Some(-1));

    let be_vec: BESIntVec = VarVec::from_slice(&data).unwrap();
    assert_eq!(be_vec.len(), 4);
    assert_eq!(be_vec.get(2), Some(-3));
}

/// Tests the deprecated `IntVecBuilder` type alias.
#[test]
fn test_deprecated_intvec_builder() {
    let data: Vec<u32> = (0..100).map(|i| i * i).collect();

    let builder: IntVecBuilder<u32, LE> = VarVec::builder();
    let vec = builder
        .k(16)
        .codec(VariableCodecSpec::Gamma)
        .build(&data)
        .unwrap();

    assert_eq!(vec.len(), 100);
    assert_eq!(vec.get_sampling_rate(), 16);
    assert_eq!(vec.get(10), Some(100));
}

/// Tests the deprecated `IntVecFromIterBuilder` type alias.
#[test]
fn test_deprecated_intvec_from_iter_builder() {
    let data: Vec<u64> = vec![5, 10, 15, 20, 25];

    let builder: IntVecFromIterBuilder<u64, LE, _> = VarVec::builder().k(8).from_iter(data.clone());
    let vec = builder.unwrap();

    assert_eq!(vec.len(), 5);
    assert_eq!(vec.get(2), Some(15));
}

/// Tests the deprecated `IntVecReader` type alias.
#[test]
fn test_deprecated_intvec_reader() {
    let data: Vec<u32> = vec![1, 2, 3, 4, 5];
    let vec: IntVec<u32, LE> = VarVec::from_slice(&data).unwrap();

    let reader: IntVecReader<u32, LE, Vec<u64>> = vec.reader();
    assert_eq!(reader.get(0), Some(1));
    assert_eq!(reader.get(4), Some(5));
    assert_eq!(reader.get(5), None);
}

/// Tests the deprecated `IntVecSeqReader` type alias.
#[test]
fn test_deprecated_intvec_seq_reader() {
    let data: Vec<u32> = (0..50).collect();
    let vec: UIntVec<u32> = VarVec::from_slice(&data).unwrap();

    let mut seq_reader: IntVecSeqReader<u32, LE, Vec<u64>> = vec.seq_reader();

    // Sequential access pattern
    assert_eq!(seq_reader.get(0), Some(0));
    assert_eq!(seq_reader.get(1), Some(1));
    assert_eq!(seq_reader.get(2), Some(2));

    // Jump backward
    assert_eq!(seq_reader.get(0), Some(0));

    // Jump forward
    assert_eq!(seq_reader.get(49), Some(49));
}

/// Tests the deprecated `IntVecSlice` type alias.
#[test]
fn test_deprecated_intvec_slice() {
    let data: Vec<u32> = (0..100).collect();
    let vec: UIntVec<u32> = VarVec::from_slice(&data).unwrap();

    let slice: IntVecSlice<u32, LE, Vec<u64>> = vec.slice(20, 30).unwrap();

    assert_eq!(slice.len(), 30);
    assert_eq!(slice.get(0), Some(20));
    assert_eq!(slice.get(29), Some(49));
    assert_eq!(slice.get(30), None);
}

/// Tests the deprecated `IntVecIter` type alias.
#[test]
fn test_deprecated_intvec_iter() {
    let data: Vec<u32> = vec![1, 2, 3, 4, 5];
    let vec: UIntVec<u32> = VarVec::from_slice(&data).unwrap();

    let iter: IntVecIter<u32, LE, Vec<u64>> = vec.iter();
    let collected: Vec<u32> = iter.collect();

    assert_eq!(collected, data);
}

/// Tests the deprecated `IntVecSliceIter` type alias.
#[test]
fn test_deprecated_intvec_slice_iter() {
    let data: Vec<u32> = (0..50).collect();
    let vec: UIntVec<u32> = VarVec::from_slice(&data).unwrap();

    let slice: IntVecSlice<u32, LE, Vec<u64>> = vec.slice(10, 20).unwrap();
    let iter: IntVecSliceIter<u32, LE, Vec<u64>> = slice.iter();

    let collected: Vec<u32> = iter.collect();
    assert_eq!(collected.len(), 20);
    assert_eq!(collected[0], 10);
    assert_eq!(collected[19], 29);
}

/// Tests the deprecated `IntVecIntoIter` type alias.
#[test]
fn test_deprecated_intvec_into_iter() {
    let data: Vec<u32> = vec![10, 20, 30, 40, 50];
    let vec: UIntVec<u32> = VarVec::from_slice(&data).unwrap();

    let into_iter: IntVecIntoIter<u32, LE> = vec.into_iter();
    let collected: Vec<u32> = into_iter.collect();

    assert_eq!(collected, data);
}

/// Tests the deprecated `IntVecError` type alias.
#[test]
fn test_deprecated_intvec_error() {
    // Create an error condition by trying to slice beyond bounds
    let data: Vec<u32> = vec![1, 2, 3];
    let vec: UIntVec<u32> = VarVec::from_slice(&data).unwrap();

    let result = vec.slice(0, 10);
    assert!(result.is_err());

    if let Err(e) = result {
        // Verify we can match the error type
        let _error: IntVecError = e;
    }
}

/// Tests interoperability: building with deprecated aliases, using with new types.
#[test]
fn test_deprecated_interoperability() {
    let data: Vec<u64> = vec![1, 2, 3, 4, 5];

    // Build with deprecated alias
    let builder: IntVecBuilder<u64, LE> = VarVec::builder();
    let vec = builder.build(&data).unwrap();

    // Use with new types
    let reader = vec.reader();
    assert_eq!(reader.get(2), Some(3));

    let slice = vec.slice(1, 3).unwrap();
    assert_eq!(slice.len(), 3);
}

/// Tests that deprecated and new types are truly the same.
#[test]
fn test_deprecated_type_equivalence() {
    let data: Vec<u32> = vec![1, 2, 3, 4, 5];

    // Create using deprecated alias
    let vec_old: IntVec<u32, LE> = VarVec::from_slice(&data).unwrap();

    // Create using new name
    let vec_new = VarVec::from_slice(&data).unwrap();

    // They should be identical
    assert_eq!(vec_old.len(), vec_new.len());
    assert_eq!(vec_old.get_sampling_rate(), vec_new.get_sampling_rate());

    for i in 0..data.len() {
        assert_eq!(vec_old.get(i), vec_new.get(i));
    }
}
