//! Integration tests for Serde functionality.

use crate::common::helpers::{generate_random_signed_vec, generate_random_vec};
use compressed_intvec::fixed::{
    atomic::{AtomicFixedVec, SAtomicFixedVec, UAtomicFixedVec},
    traits::{Storable, Word},
    BitWidth, FixedVec,
};
use dsi_bitstream::prelude::{BE, LE};
use dsi_bitstream::traits::Endianness;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Helper to run a comprehensive serde round-trip test for `FixedVec`.
fn run_fixedvec_serde_test<T, W, E>(data: &[T], bit_width_strategy: BitWidth)
where
    T: Storable<W> + Debug + PartialEq + Default + Copy + 'static,
    W: Word + Serialize + for<'de> Deserialize<'de>,
    E: Endianness + Debug,
    for<'a> dsi_bitstream::impls::BufBitWriter<
        E,
        dsi_bitstream::impls::MemWordWriterVec<W, Vec<W>>,
    >: dsi_bitstream::prelude::BitWrite<E, Error = std::convert::Infallible>,
    FixedVec<T, W, E, Vec<W>>: for<'de> Deserialize<'de>,
{
    let original_vec = FixedVec::<T, W, E>::builder()
        .bit_width(bit_width_strategy)
        .build(data)
        .unwrap();

    // 1. Test with bincode (binary format)
    let encoded_bincode = bincode::serialize(&original_vec).unwrap();
    let decoded_bincode: FixedVec<T, W, E> = bincode::deserialize(&encoded_bincode).unwrap();
    assert_eq!(original_vec, decoded_bincode, "Bincode round-trip failed");

    // 2. Test with serde_json (text format)
    let encoded_json = serde_json::to_string(&original_vec).unwrap();
    let decoded_json: FixedVec<T, W, E> = serde_json::from_str(&encoded_json).unwrap();
    assert_eq!(original_vec, decoded_json, "JSON round-trip failed");
}

/// Helper to run a comprehensive serde round-trip test for `AtomicFixedVec`.
fn run_atomic_fixedvec_serde_test<T>(data: &[T], bit_width_strategy: BitWidth)
where
    T: Storable<u64>
        + Debug
        + PartialEq
        + Eq
        + Default
        + Copy
        + num_traits::ToPrimitive,
    UAtomicFixedVec<T>: for<'de> Deserialize<'de>,
{
    let original_vec: UAtomicFixedVec<T> = AtomicFixedVec::builder()
        .bit_width(bit_width_strategy)
        .build(data)
        .unwrap();

    // 1. Bincode
    let encoded_bincode = bincode::serialize(&original_vec).unwrap();
    let decoded_bincode: UAtomicFixedVec<T> = bincode::deserialize(&encoded_bincode).unwrap();
    assert_eq!(
        original_vec, decoded_bincode,
        "Bincode round-trip failed for AtomicFixedVec"
    );

    // 2. JSON
    let encoded_json = serde_json::to_string(&original_vec).unwrap();
    let decoded_json: UAtomicFixedVec<T> = serde_json::from_str(&encoded_json).unwrap();
    assert_eq!(
        original_vec, decoded_json,
        "JSON round-trip failed for AtomicFixedVec"
    );
}

#[test]
fn test_fixedvec_serde_unsigned() {
    let data_u32 = generate_random_vec(100, 1_000_000)
        .into_iter()
        .map(|x| x as u32)
        .collect::<Vec<_>>();
    run_fixedvec_serde_test::<u32, usize, LE>(&data_u32, BitWidth::Minimal);
    run_fixedvec_serde_test::<u32, u64, BE>(&data_u32, BitWidth::PowerOfTwo);
}

#[test]
fn test_fixedvec_serde_signed() {
    let data_i16 = generate_random_signed_vec(100, 30_000)
        .into_iter()
        .map(|x| x as i16)
        .collect::<Vec<_>>();
    run_fixedvec_serde_test::<i16, usize, LE>(&data_i16, BitWidth::Explicit(16));
    run_fixedvec_serde_test::<i16, u32, BE>(&data_i16, BitWidth::Minimal);
}

#[test]
fn test_fixedvec_serde_edge_cases() {
    // Empty vector
    run_fixedvec_serde_test::<u64, u64, LE>(&[], BitWidth::Explicit(8));
    // Single element vector
    run_fixedvec_serde_test::<i8, u16, BE>(&[-42], BitWidth::Minimal);
}

#[test]
fn test_atomic_fixedvec_serde_unsigned() {
    let data_u32 = generate_random_vec(100, 100_000)
        .into_iter()
        .map(|x| x as u32)
        .collect::<Vec<_>>();
    run_atomic_fixedvec_serde_test::<u32>(&data_u32, BitWidth::Minimal);
}

#[test]
fn test_atomic_fixedvec_serde_signed() {
    let data_i16 = generate_random_signed_vec(100, 30_000)
        .into_iter()
        .map(|x| x as i16)
        .collect::<Vec<_>>();
    // Signed atomic vec is just an alias
    let original_vec: SAtomicFixedVec<i16> = AtomicFixedVec::builder()
        .bit_width(BitWidth::Minimal)
        .build(&data_i16)
        .unwrap();

    let encoded = bincode::serialize(&original_vec).unwrap();
    let decoded: SAtomicFixedVec<i16> = bincode::deserialize(&encoded).unwrap();
    assert_eq!(original_vec, decoded);
}

#[test]
fn test_atomic_fixedvec_serde_edge_cases() {
    // Empty vector
    run_atomic_fixedvec_serde_test::<u64>(&[], BitWidth::Explicit(8));
    // Single element vector
    run_atomic_fixedvec_serde_test::<i32>(&[-1], BitWidth::Minimal);
}