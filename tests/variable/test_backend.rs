use compressed_intvec::{
    // FIX: Rimosso l'import non utilizzato di FixedVecError
    prelude::*,
    variable::{BEVarVec, LEVarVec, VarVec, codec::Codec},
};
use dsi_bitstream::prelude::{BE, LE};

use crate::common::helpers::{generate_random_signed_vec, generate_random_vec};

#[test]
fn test_intvec_owned_to_borrowed_conversion() {
    let data = generate_random_vec(1000, 10000);

    // 1. Create an owned vector, which will be our source of truth.
    let owned_vec = LEVarVec::builder()
        .k(16)
        .codec(Codec::Delta)
        .build(&data)
        .unwrap();

    // 2. Extract references to the raw components from the owned vector.
    let data_limbs = owned_vec.as_limbs();
    let samples_vec = owned_vec.samples_ref();
    let samples_limbs = samples_vec.as_limbs();
    let samples_len = samples_vec.len();
    let samples_num_bits = samples_vec.bit_width();
    let k = owned_vec.sampling_rate();
    let len = owned_vec.len();
    let encoding = owned_vec.encoding();

    // 3. Create a borrowed VarVec from the extracted parts.
    let borrowed_vec = VarVec::<u64, LE, &[u64]>::from_parts(
        data_limbs,
        samples_limbs,
        samples_len,
        samples_num_bits,
        k,
        len,
        encoding,
    )
    .unwrap();

    // 4. Assert that the borrowed view is functionally identical to the owned one.
    assert_eq!(borrowed_vec.len(), owned_vec.len());
    assert_eq!(
        borrowed_vec.sampling_rate(),
        owned_vec.sampling_rate()
    );
    assert_eq!(
        borrowed_vec.iter().collect::<Vec<_>>(),
        owned_vec.iter().collect::<Vec<_>>()
    );

    // Spot check some values.
    assert_eq!(borrowed_vec.get(100), Some(data[100]));
    assert_eq!(borrowed_vec.get(500), Some(data[500]));
}

#[test]
fn test_from_parts_validation() {
    let data = generate_random_vec(100, 1000);
    let owned_vec = BEVarVec::builder().k(8).build(&data).unwrap();

    let data_limbs = owned_vec.as_limbs();
    let samples_vec = owned_vec.samples_ref();
    let samples_limbs = samples_vec.as_limbs();
    let samples_len = samples_vec.len();
    let samples_num_bits = samples_vec.bit_width();
    let len = owned_vec.len();
    let encoding = owned_vec.encoding();

    // Fail: k = 0
    let result = VarVec::<u64, BE, &[u64]>::from_parts(
        data_limbs,
        samples_limbs,
        samples_len,
        samples_num_bits,
        0, // Invalid k
        len,
        encoding,
    );
    // FIX: L'errore restituito è VarVecError, non FixedVecError.
    assert!(matches!(result, Err(VarVecError::InvalidParameters(_))));

    // Fail: Inconsistent number of samples
    let result = VarVec::<u64, BE, &[u64]>::from_parts(
        data_limbs,
        samples_limbs,
        samples_len + 1, // Mismatch
        samples_num_bits,
        8,
        len,
        encoding,
    );
    assert!(matches!(result, Err(VarVecError::InvalidParameters(_))));
}

#[test]
fn test_sintvec_owned_to_borrowed_conversion() {
    let data = generate_random_signed_vec(1000, 10000);
    let owned_svec: VarVec<i64, LE> = VarVec::builder()
        .k(16)
        .codec(Codec::Auto)
        .build(&data)
        .unwrap();

    let data_limbs = owned_svec.as_limbs();
    let samples_vec = owned_svec.samples_ref();
    let samples_limbs = samples_vec.as_limbs();
    let samples_len = samples_vec.len();
    let samples_num_bits = samples_vec.bit_width();
    let k = owned_svec.sampling_rate();
    let len = owned_svec.len();
    let encoding = owned_svec.encoding();

    let borrowed_svec = VarVec::<i64, LE, &[u64]>::from_parts(
        data_limbs,
        samples_limbs,
        samples_len,
        samples_num_bits,
        k,
        len,
        encoding,
    )
    .unwrap();

    assert_eq!(borrowed_svec.len(), owned_svec.len());
    assert_eq!(
        borrowed_svec.iter().collect::<Vec<_>>(),
        owned_svec.iter().collect::<Vec<_>>()
    );
    assert_eq!(borrowed_svec.get(123), Some(data[123]));
}
