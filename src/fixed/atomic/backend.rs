//! # Atomic Storage Backend for `AtomicFixedVec`
//!
//! This module defines the underlying storage architecture for the thread-safe
//! `AtomicFixedVec`. It encapsulates the raw data buffer, ensuring it has the
//! necessary 16-byte memory alignment for high-performance atomic operations,
//! including 128-bit atomics on supported platforms.

#![cfg(feature = "atomic")]

use crate::fixed::traits::Word;
use common_traits::{Atomic, IntoAtomic};
use num_traits::Zero;
use std::fmt::Debug;

/// A wrapper struct that enforces 16-byte memory alignment for a word `W`.
///
/// This is critical for ensuring that pointers to adjacent 64-bit words can be
/// safely and efficiently cast to a 128-bit atomic type (`atomic::Atomic<u128>`)
/// for lock-free operations that span word boundaries.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(16))]
pub(super) struct AlignedWord<W>(pub W);

/// The owned storage backend for an `AtomicFixedVec`.
///
/// This struct holds the actual data buffer in a `Vec` of `AlignedWord`,
/// ensuring correct memory alignment for all atomic operations.
#[derive(Debug)]
pub(super) struct OwnedAtomicBackend<W: Word + IntoAtomic>
where
    W::AtomicType: Debug,
{
    pub(super) storage: Vec<AlignedWord<W::AtomicType>>,
}

impl<W: Word + IntoAtomic + Zero> OwnedAtomicBackend<W>
where
    W::AtomicType: Atomic + Debug,
{
    /// Creates a new `OwnedAtomicBackend` with a zero-initialized, 16-byte
    /// aligned buffer.
    pub(super) fn new(bit_width: usize, capacity: usize) -> Self {
        let total_bits = capacity.saturating_mul(bit_width);
        let num_words = total_bits.div_ceil(<W as Word>::BITS);
        // The buffer must have at least 2 extra words of padding for safe
        // unaligned/spanning reads.
        let buffer_len = if capacity == 0 { 0 } else { num_words + 2 };

        let storage = (0..buffer_len)
            .map(|_| AlignedWord(<W::AtomicType as Atomic>::new(W::zero())))
            .collect();

        Self { storage }
    }
}