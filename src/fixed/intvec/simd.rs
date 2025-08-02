//! # SIMD-accelerated batch operations for `FixedVec`.
//!
//! This module is conditionally compiled and provides `unsafe`, high-performance
//! functions for accessing `FixedVec` data in batches using SIMD intrinsics.
//!
//! The core function, `gather_simd`, dispatches to specialized routines based
//! on the bit-width of the stored integers. The primary optimization targets
//! byte-aligned bit-widths (8, 16, 32, 64) on Little-Endian systems, where
//! direct memory reinterpretation and SIMD vector instructions can be used
//! for maximum throughput.

use super::{Endianness, FixedVec, BE, LE};
use core::simd::{LaneCount, SimdElement, SupportedLaneCount};
use std::any::TypeId;

/// A generic helper function that uses `slice::as_simd` to process aligned chunks.
///
/// This function takes a slice of a primitive type `T` (like `u8`, `u16`, `u32`),
/// processes it in SIMD vectors of `LANES` elements, casts each vector to
/// `Simd<u64, LANES>`, and writes the result to the destination buffer. Unaligned
/// prefixes and suffixes are handled with a scalar loop.
///
/// # Safety
/// The caller must ensure that `source_slice.len() == results.len()`.
#[inline]
unsafe fn gather_and_widen_simd<T, const LANES: usize>(source_slice: &[T], results: &mut [u64])
where
    T: SimdElement + bytemuck::Pod + Into<u64>,
    LaneCount<LANES>: SupportedLaneCount,
{
    debug_assert_eq!(source_slice.len(), results.len());

    // Use `as_simd` to get aligned chunks, which is the core of this optimization.
    let (prefix, middle, suffix) = source_slice.as_simd::<LANES>();

    // Handle the unaligned prefix scalarly.
    for (i, &val) in prefix.iter().enumerate() {
        results[i] = val.into();
    }

    // Handle the aligned middle part with SIMD instructions.
    let middle_start = prefix.len();
    for (chunk_idx, simd_val) in middle.iter().enumerate() {
        // Convert each element individually since direct casting isn't available
        let widened_array: [u64; LANES] = std::array::from_fn(|i| simd_val[i].into());
        let chunk_start = middle_start + chunk_idx * LANES;
        results[chunk_start..chunk_start + LANES].copy_from_slice(&widened_array);
    }

    // Handle the unaligned suffix scalarly.
    let suffix_start = middle_start + middle.len() * LANES;
    for (i, &val) in suffix.iter().enumerate() {
        results[suffix_start + i] = val.into();
    }
}

/// Gathers elements from a `FixedVec` into a results slice using SIMD where possible.
///
/// This is the main entry point for SIMD-accelerated batch reads. It dispatches
/// to the optimal implementation based on `vec.num_bits()`.
///
/// # Safety
/// The caller must ensure that the range `start_index..start_index + results.len()`
/// is within the bounds of the `FixedVec`.
#[inline]
pub(super) unsafe fn gather_simd<E: Endianness, B: AsRef<[u64]>>(
    vec: &FixedVec<E, B>,
    start_index: usize,
    results: &mut [u64],
) {
    match vec.num_bits() {
        64 => {
            let len = results.len();
            // Direct slice copy for 64-bit data.
            let source_slice = &vec.as_limbs()[start_index..start_index + len];
            results.copy_from_slice(source_slice);

            // For Big Endian, the u64 values in memory are byte-swapped.
            // We need to convert them back to the native representation.
            if TypeId::of::<E>() == TypeId::of::<BE>() {
                results.iter_mut().for_each(|v| *v = u64::from_be(*v));
            }
        }
        32 => {
            if TypeId::of::<E>() == TypeId::of::<LE>() {
                const LANES: usize = 8; // Target 256-bit registers (8 * 32-bit)
                let all_bytes: &[u8] = bytemuck::cast_slice(vec.as_limbs());
                let start_byte = start_index * 4;
                let end_byte = start_byte + results.len() * 4;
                let data_bytes = &all_bytes[start_byte..end_byte];
                let source_slice: &[u32] = bytemuck::cast_slice(data_bytes);
                gather_and_widen_simd::<u32, LANES>(source_slice, results);
            } else {
                // Fallback for Big Endian.
                for i in 0..results.len() {
                    results[i] = vec.get_unchecked(start_index + i);
                }
            }
        }
        16 => {
            if TypeId::of::<E>() == TypeId::of::<LE>() {
                const LANES: usize = 16; // Target 256-bit registers (16 * 16-bit)
                let all_bytes: &[u8] = bytemuck::cast_slice(vec.as_limbs());
                let start_byte = start_index * 2;
                let end_byte = start_byte + results.len() * 2;
                let data_bytes = &all_bytes[start_byte..end_byte];
                let source_slice: &[u16] = bytemuck::cast_slice(data_bytes);
                gather_and_widen_simd::<u16, LANES>(source_slice, results);
            } else {
                // Fallback for Big Endian.
                for i in 0..results.len() {
                    results[i] = vec.get_unchecked(start_index + i);
                }
            }
        }
        8 => {
            if TypeId::of::<E>() == TypeId::of::<LE>() {
                const LANES: usize = 32; // Target 256-bit registers (32 * 8-bit)
                let all_bytes: &[u8] = bytemuck::cast_slice(vec.as_limbs());
                let start_byte = start_index;
                let end_byte = start_byte + results.len();
                let source_slice = &all_bytes[start_byte..end_byte];
                gather_and_widen_simd::<u8, LANES>(source_slice, results);
            } else {
                // Fallback for Big Endian.
                for i in 0..results.len() {
                    results[i] = vec.get_unchecked(start_index + i);
                }
            }
        }
        _ => core::hint::unreachable_unchecked(),
    }
}
