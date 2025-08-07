//! # Atomic Storage Backend for `AtomicFixedVec`
//!
//! This module defines the underlying storage architecture for the thread-safe
//! `AtomicFixedVec`. It encapsulates the raw atomic data buffer and the
//! synchronization primitives required for safe concurrent access.
//!
//! The central component is `OwnedAtomicBackend`, which implements the "striped
//! locking" strategy. This involves partitioning the lock space into a fixed
//! number of "stripes," each guarded by a lightweight mutex. When a multi-word
//! atomic operation is required, only the locks corresponding to the affected
//! words are acquired, minimizing contention and allowing unrelated concurrent
//! operations to proceed in parallel.

#![cfg(feature = "atomic")]

use crate::fixed::traits::Word;
use common_traits::Atomic;
use mem_dbg::{MemDbg, MemSize};
use parking_lot::Mutex;
use std::fmt::Debug;

/// The number of concurrent locks to use for the striped locking strategy.
///
/// A power of two is chosen to allow for efficient calculation of a stripe index
/// from a word index using a bitwise AND operation instead of a slower modulo.
/// 64 stripes provide a good balance, offering high granularity to reduce
/// lock contention without excessive memory overhead for the lock array itself.
const NUM_STRIPES: usize = 64;

/// A trait that defines the interface for an atomic storage backend.
///
/// This abstraction allows the `AtomicFixedVec` to be generic over its
/// storage, facilitating testing and potential future extensions (e.g., a
/// borrowed, non-owning backend).
pub(super) trait AtomicBackend<W: Word + Atomic>:
    Send + Sync + Debug + MemDbg + MemSize
{
    /// Returns a slice of the underlying atomic words.
    fn as_atomic_slice(&self) -> &[W::AtomicType];

    /// Returns a slice of the mutexes used for striped locking.
    fn locks(&self) -> &[Mutex<()>];
}


#[derive(Debug, MemDbg, MemSize)]
pub(super) struct OwnedAtomicBackend<W: Word + Atomic> {
    /// The bit-packed data, stored in atomic words.
    bits: Vec<W::AtomicType>,
    /// A fixed-size array of lightweight mutexes for striped locking.
    locks: Box<[Mutex<()>; NUM_STRIPES]>,
}

impl<W: Word + Atomic> OwnedAtomicBackend<W> {
    /// Creates a new `OwnedAtomicBackend` with a specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity`: The number of elements the backend should be able to hold
    ///   without reallocating.
    /// * `bit_width`: The number of bits used for each element.
    pub(super) fn new(capacity: usize, bit_width: usize) -> Self {
        // Calculate the number of words needed for the data buffer.
        // We add padding to ensure that even boundary-crossing reads/writes
        // near the end of the vector do not go out of bounds.
        let bits_per_word = <W as Word>::BITS;
        let total_bits = capacity.saturating_mul(bit_width);
        let num_words = total_bits.div_ceil(bits_per_word);
        let buffer_len = if capacity == 0 { 0 } else { num_words + 2 };

        // Initialize the atomic data buffer with zeros.
        let bits = (0..buffer_len)
            .map(|_| W::AtomicType::new(W::ZERO))
            .collect();

        // Initialize the array of mutexes for striped locking.
        // `std::array::from_fn` is the idiomatic way to create an array of
        // non-Copy types like Mutex.
        let locks = Box::new(std::array::from_fn(|_| Mutex::new(())));

        Self { bits, locks }
    }
}

impl<W: Word + Atomic> AtomicBackend<W> for OwnedAtomicBackend<W> {
    /// Returns a slice of the underlying atomic words.
    #[inline]
    fn as_atomic_slice(&self) -> &[W::AtomicType] {
        &self.bits
    }

    /// Returns a slice of the mutexes used for striped locking.
    #[inline]
    fn locks(&self) -> &[Mutex<()>] {
        &self.locks
    }
}