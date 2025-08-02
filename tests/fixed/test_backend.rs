//! Integration tests for the generic backend functionality of `FixedVec`.

use compressed_intvec::prelude::*;
use dsi_bitstream::prelude::{BE, LE};

#[test]
fn test_owned_to_borrowed_conversion() {
    let data = vec![10u32, 20, 30, 40, 50];
    // 1. Create an owned vector (backend is Vec<u64>)
    let owned_vec = LEFixedVec::builder(&data)
        .bit_width(BitWidth::Minimal)
        .build()
        .unwrap();

    // 2. Create a borrowed vector (view) from the owned vector's data
    let borrowed_vec: FixedVec<LE, &[u64]> =
        FixedVec::from_parts(owned_vec.as_limbs(), owned_vec.len(), owned_vec.num_bits()).unwrap();

    // Verify they are identical
    assert_eq!(owned_vec.len(), borrowed_vec.len());
    assert_eq!(owned_vec.num_bits(), borrowed_vec.num_bits());
    assert_eq!(
        owned_vec.iter().collect::<Vec<_>>(),
        borrowed_vec.iter().collect::<Vec<_>>()
    );

    // Verify content
    assert_eq!(borrowed_vec.get(2), Some(30));
    assert_eq!(borrowed_vec, &data[..]);
}

#[test]
fn test_from_parts_validation() {
    let data = vec![100u64, 200, 300];
    let owned_vec = BEFixedVec::builder(&data).build().unwrap();
    let limbs = owned_vec.as_limbs();
    let len = owned_vec.len();
    let num_bits = owned_vec.num_bits();

    // Success case is tested above.

    // Fail: num_bits > 64
    let result = FixedVec::<BE, _>::from_parts(limbs, len, 65);
    assert!(matches!(result, Err(FixedVecError::InvalidParameters(_))));

    // Fail: buffer too small (no space for padding word)
    let total_bits = len * num_bits;
    let data_words = (total_bits + 63) / 64;
    let insufficient_limbs = &limbs[..data_words]; // Exactly enough for data, but not padding
    let result = FixedVec::<BE, _>::from_parts(insufficient_limbs, len, num_bits);
    assert!(matches!(result, Err(FixedVecError::InvalidParameters(_))));
}

#[test]
fn test_s_fixed_vec_from_parts() {
    let data = vec![-10i16, 20, -30, 40, 50];
    // 1. Create an owned signed vector
    let owned_s_vec = LESFixedVec::builder(&data).build().unwrap();

    // 2. Create an inner `FixedVec` view
    let inner_view: FixedVec<LE, &[u64]> = FixedVec::from_parts(
        owned_s_vec.as_limbs(),
        owned_s_vec.len(),
        owned_s_vec.num_bits(),
    )
    .unwrap();

    // 3. Create the borrowed `SFixedVec` from the inner view
    let borrowed_s_vec = SFixedVec::from_parts(inner_view);

    assert_eq!(owned_s_vec, borrowed_s_vec);
    assert_eq!(borrowed_s_vec.get(2), Some(-30));
    assert_eq!(borrowed_s_vec, &data[..]);
}

#[test]
fn test_slice_on_borrowed_backend() {
    let data = (0..100).collect::<Vec<u64>>();
    let owned_vec = LEFixedVec::builder(&data).build().unwrap();
    let borrowed_vec: FixedVec<LE, &[u64]> =
        FixedVec::from_parts(owned_vec.as_limbs(), owned_vec.len(), owned_vec.num_bits()).unwrap();

    let slice = borrowed_vec.slice(20, 30).unwrap();
    assert_eq!(slice.len(), 30);
    assert_eq!(slice.get(0), Some(20));
    assert_eq!(slice.get(29), Some(49));
    assert_eq!(slice, &data[20..50]);
}
