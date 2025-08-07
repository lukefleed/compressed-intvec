//! # `AtomicAccess` Trait for Strategy Dispatch
//!
//! This module defines the `AtomicAccess` trait, which provides the core
//! abstraction for performing atomic operations on the backend storage of an
//! `AtomicFixedVec`.
//!
//! By using this trait as a bound, `AtomicFixedVec` can delegate its atomic
//! operations (`load`, `store`, `compare_exchange`, etc.) to a specific
//! underlying strategy. The compiler then selects the correct implementation
//! at compile time—either the high-performance lock-free path for single-word
//! operations or the robust striped-locking path for multi-word operations.
//!
//! This "strategy pattern" via traits is central to providing both maximum
//! performance and correctness. The trait is sealed to prevent external
//! implementations, ensuring that only the provided, correct strategies can be
//! used.

#![cfg(feature = "atomic")]

use super::traits::Word;
use super::backend::AtomicBackend;
use common_traits::Atomic;
use std::sync::atomic::Ordering;

/// A private module to seal the `AtomicAccess` trait.
mod private {
    use super::*;

    /// The sealed trait that defines the contract for atomic access strategies.
    ///
    /// This trait is implemented by `AtomicBackend` in the `strategy.rs` file.
    pub trait SealedAtomicAccess<W: Word + Atomic> {
        /// Atomically loads a value from the specified index.
        fn atomic_load(&self, index: usize, bit_width: usize, mask: W, order: Ordering) -> W;

        /// Atomically stores a value at the specified index.
        fn atomic_store(&self, index: usize, value: W, bit_width: usize, mask: W, order: Ordering);

        /// Atomically swaps a value at the specified index, returning the old value.
        fn atomic_swap(&self, index: usize, value: W, bit_width: usize, mask: W, order: Ordering) -> W;

        /// Atomically compares the value at `index` with `current`. If they are
        /// equal, it is replaced with `new`.
        ///
        /// Returns `Ok(old_value)` on success (where `old_value` is guaranteed
        /// to be the same as `current`) or `Err(old_value)` on failure.
        fn atomic_compare_exchange(
            &self,
            index: usize,
            current: W,
            new: W,
            bit_width: usize,
            mask: W,
            success: Ordering,
            failure: Ordering,
        ) -> Result<W, W>;
    }
}

/// A trait that abstracts the specific strategy for atomic operations.
///
/// This trait is implemented by the storage backend (`AtomicBackend`) and allows
/// `AtomicFixedVec` to transparently use the most efficient and correct atomic
/// strategy (lock-free or striped-locking) based on its configuration.
pub(super) trait AtomicAccess<W: Word + Atomic>:
    AtomicBackend<W> + private::SealedAtomicAccess<W>
{
}

/// Blanket implementation of the public trait for any type that implements the sealed trait.
impl<B, W> AtomicAccess<W> for B
where
    B: AtomicBackend<W> + private::SealedAtomicAccess<W>,
    W: Word + Atomic,
{
}