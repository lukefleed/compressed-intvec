//! Integration tests for the generic backend functionality of `FixedVec`.

use compressed_intvec::fixed::{BitWidth, Error as FixedVecError, FixedVec, SFixedVec, UFixedVec};
use dsi_bitstream::prelude::BE;

#[test]
fn test_owned_to_borrowed_conversion() {
    // Use a non-default configuration to test genericity
    type TestVecOwned = FixedVec<u32, u64, BE, Vec<u64>>;
    type TestVecBorrowed<'a> = FixedVec<u32, u64, BE, &'a [u64]>;

    let data = vec![10u32, 20, 30, 40, 50];

    // 1. Create an owned vector.
    let owned_vec: TestVecOwned = FixedVec::builder()
        .bit_width(BitWidth::Minimal)
        .build(&data)
        .unwrap();

    // 2. Create a borrowed vector (view) from the owned vector's data.
    let borrowed_vec: TestVecBorrowed =
        FixedVec::from_parts(owned_vec.as_limbs(), owned_vec.len(), owned_vec.bit_width()).unwrap();

    // 3. Verify they are functionally identical.
    assert_eq!(owned_vec.len(), borrowed_vec.len());
    assert_eq!(owned_vec.bit_width(), borrowed_vec.bit_width());
    assert_eq!(
        owned_vec.iter().collect::<Vec<_>>(),
        borrowed_vec.iter().collect::<Vec<_>>()
    );

    // Test PartialEq between different backend types.
    assert_eq!(owned_vec, borrowed_vec);

    // Verify content of the borrowed vec.
    assert_eq!(borrowed_vec.get(2), Some(30));
    assert_eq!(borrowed_vec, &data[..]);
}

#[test]
fn test_from_parts_validation() {
    type TestVec = UFixedVec<u64>; // LE, usize word
    let data = vec![100u64, 200, 300];
    let owned_vec: TestVec = FixedVec::builder().build(&data).unwrap();

    let limbs = owned_vec.as_limbs();
    let len = owned_vec.len();
    let bit_width = owned_vec.bit_width();
    let bits_per_word = <usize as compressed_intvec::fixed::traits::Word>::BITS;

    // Success case is implicitly tested in test_owned_to_borrowed_conversion.

    // Fail: bit_width > word bits
    let result = FixedVec::<u64, usize, dsi_bitstream::prelude::LE, _>::from_parts(
        limbs,
        len,
        bits_per_word + 1,
    );
    assert!(matches!(result, Err(FixedVecError::InvalidParameters(_))));

    // Fail: buffer too small (no space for padding word)
    let total_bits = len * bit_width;
    let data_words = total_bits.div_ceil(bits_per_word);
    let insufficient_limbs = &limbs[..data_words]; // Exactly enough for data, but not padding.
    let result = FixedVec::<u64, usize, dsi_bitstream::prelude::LE, _>::from_parts(
        insufficient_limbs,
        len,
        bit_width,
    );
    assert!(matches!(result, Err(FixedVecError::InvalidParameters(_))));
}

#[test]
fn test_signed_vec_from_borrowed_backend() {
    type TestVecOwned = SFixedVec<i16>; // LE, usize word
    type TestVecBorrowed<'a> = FixedVec<i16, usize, dsi_bitstream::prelude::LE, &'a [usize]>;

    let data = vec![-10i16, 20, -30, 40, 50];

    // 1. Create an owned signed vector.
    let owned_s_vec: TestVecOwned = FixedVec::builder().build(&data).unwrap();

    // 2. Create a borrowed version from its parts.
    let borrowed_s_vec: TestVecBorrowed = FixedVec::from_parts(
        owned_s_vec.as_limbs(),
        owned_s_vec.len(),
        owned_s_vec.bit_width(),
    )
    .unwrap();

    // 3. Verify equality and content.
    assert_eq!(owned_s_vec, borrowed_s_vec);
    assert_eq!(borrowed_s_vec.get(2), Some(-30));
    assert_eq!(borrowed_s_vec, &data[..]);
}

#[test]
fn test_slice_on_borrowed_backend() {
    type TestVecOwned = UFixedVec<u64>;
    type TestVecBorrowed<'a> = FixedVec<u64, usize, dsi_bitstream::prelude::LE, &'a [usize]>;

    let data: Vec<u64> = (0..100).collect();
    let owned_vec: TestVecOwned = FixedVec::builder().build(&data).unwrap();

    // Create a borrowed vec from the owned vec's limbs.
    let borrowed_vec: TestVecBorrowed =
        FixedVec::from_parts(owned_vec.as_limbs(), owned_vec.len(), owned_vec.bit_width()).unwrap();

    // Now, create a slice (view) of the *borrowed* vec.
    let slice = borrowed_vec.slice(20, 30).unwrap();

    assert_eq!(slice.len(), 30);
    assert_eq!(slice.get(0), Some(20));
    assert_eq!(slice.get(29), Some(49));
    assert_eq!(slice, &data[20..50]);
}
