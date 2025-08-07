//! # Atomic Storage Backend for `AtomicFixedVec`
//!
//! This module defines the underlying storage architecture for the thread-safe
//! `AtomicFixedVec`. It encapsulates the raw data buffer and the
//! synchronization primitives required for safe concurrent access.
//!
//! The central component is `OwnedAtomicBackend`, which holds an `AtomicStorage`
//! enum. This enum allows the backend to transparently switch between two highly
//! optimized strategies based on the vector's configuration:
//!
//! - **`AtomicStorage::LockFree`**: For configurations where elements fit within
//!   a single word (i.e., power-of-two bit widths), this variant uses a simple
//!   `Vec` of atomic words. All operations are dispatched to a high-performance,
//!   lock-free implementation.
//!
//! - **`AtomicStorage::SeqLock`**: For configurations where elements may span
//!   word boundaries, this variant uses a hybrid mechanism. Reads are performed
//!   optimistically using non-blocking seqlocks for maximum concurrency. All
//!   write operations (`store`, `swap`, `compare_exchange`) use fine-grained
//!   striped mutexes to ensure correctness and prevent torn reads/writes.

#![cfg(feature = "atomic")]

use crate::fixed::traits::Word;
use common_traits::{Atomic, IntoAtomic};
use num_traits::Zero;
use parking_lot::Mutex;
use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The number of concurrent writer locks for the striped locking strategy.
pub(super) const NUM_STRIPES: usize = 64;

/// A single word of data protected by a seqlock for optimistic reads.
#[derive(Debug)]
pub(super) struct SeqLockWord<W: Word> {
    pub(super) version: AtomicUsize,
    pub(super) data: UnsafeCell<W>,
}

impl<W: Word + Zero> SeqLockWord<W> {
    /// Creates a new `SeqLockWord` initialized to zero.
    fn new() -> Self {
        Self {
            version: AtomicUsize::new(0),
            data: UnsafeCell::new(W::zero()),
        }
    }
}

// SAFETY: `SeqLockWord` is safe to send/sync if `W` is `Send`.
// The `UnsafeCell` is only accessed via the seqlock protocol.
unsafe impl<W: Word + Send> Send for SeqLockWord<W> {}
unsafe impl<W: Word + Send> Sync for SeqLockWord<W> {}

/// An enum representing the two possible storage strategies for the atomic vector.
#[derive(Debug)]
pub(super) enum AtomicStorage<W: Word + IntoAtomic>
where
    W::AtomicType: Debug,
{
    /// Storage for lock-free operations (power-of-two bit widths).
    LockFree(Vec<W::AtomicType>),
    /// Storage for seqlock/mutex-based operations (other bit widths).
    SeqLock {
        seqlocks: Vec<SeqLockWord<W>>,
        writer_locks: Box<[Mutex<()>; NUM_STRIPES]>,
    },
}

/// The owned storage backend for an `AtomicFixedVec`.
///
/// This struct holds the actual storage and decides which strategy to use
/// upon construction based on the provided `bit_width`.
#[derive(Debug)]
pub(super) struct OwnedAtomicBackend<W: Word + IntoAtomic>
where
    W::AtomicType: Debug,
{
    pub(super) storage: AtomicStorage<W>,
}

impl<W: Word + IntoAtomic + Zero> OwnedAtomicBackend<W>
where
    W::AtomicType: Debug,
{
    /// Creates a new `OwnedAtomicBackend`, automatically selecting the optimal
    /// storage strategy.
    pub(super) fn new(bit_width: usize, capacity: usize) -> Self {
        let total_bits = capacity.saturating_mul(bit_width);
        let num_words = total_bits.div_ceil(<W as Word>::BITS);
        let buffer_len = if capacity == 0 { 0 } else { num_words + 2 }; // +2 for padding

        let storage = if bit_width.is_power_of_two() {
            // --- Lock-Free Strategy ---
            // The backend is a simple vector of atomic words.
            let atomic_limbs = (0..buffer_len)
                .map(|_| <W::AtomicType as Atomic>::new(W::zero()))
                .collect();
            AtomicStorage::LockFree(atomic_limbs)
        } else {
            // --- SeqLock/Mutex Strategy ---
            // The backend holds seqlocked words for reading and mutexes for writing.
            let seqlocks = (0..buffer_len).map(|_| SeqLockWord::new()).collect();
            let writer_locks = Box::new(std::array::from_fn(|_| Mutex::new(())));
            AtomicStorage::SeqLock {
                seqlocks,
                writer_locks,
            }
        };

        Self { storage }
    }
}

impl<W: Word> SeqLockWord<W> {
    /// Performs a lock-free, optimistic read of the data. Spins if a write is in progress.
    #[inline(always)]
    pub(super) fn optimistic_read(&self) -> W {
        loop {
            let version1 = self.version.load(Ordering::Acquire);
            if version1 % 2 != 0 {
                std::hint::spin_loop();
                continue;
            }
            // SAFETY: Version check ensures no writer is active.
            let data = unsafe { *self.data.get() };
            let version2 = self.version.load(Ordering::Acquire);
            if version1 == version2 {
                return data;
            }
        }
    }
}