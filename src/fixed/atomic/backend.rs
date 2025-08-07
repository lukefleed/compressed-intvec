//! # Atomic Storage Backend for `AtomicFixedVec`
//!
//! This module defines the underlying storage architecture for the thread-safe
//! `AtomicFixedVec`. It encapsulates the raw data buffer and the
//! synchronization primitives required for safe concurrent access.
//!
//! The central component is `OwnedAtomicBackend`, which implements a hybrid
//! synchronization strategy:
//! - **SeqLock**: For reads and simple stores, it uses a per-word seqlock,
//!   allowing for highly concurrent, non-blocking reads. Readers retry if a
//!   write occurs, but are never blocked.
//! - **Striped Mutexes**: For complex read-modify-write operations (like
//!   `compare_exchange`), it falls back to a striped mutex locking strategy.
//!   This guarantees correctness and atomicity for these operations while
//!   minimizing contention.

#![cfg(feature = "atomic")]

use crate::fixed::traits::Word;
use common_traits::IntoAtomic;
use num_traits::Zero;
use parking_lot::Mutex;
use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The number of concurrent locks to use for the striped locking strategy,
/// which is reserved for complex read-modify-write operations.
///
/// A power of two is chosen to allow for efficient calculation of a stripe index
/// from a word index using a bitwise AND operation instead of a slower modulo.
/// 64 stripes provide a good balance, offering high granularity to reduce
/// lock contention without excessive memory overhead for the lock array itself.
pub(super) const NUM_STRIPES: usize = 64;

/// A single word of data protected by a seqlock.
///
/// This allows for highly concurrent reads. A read operation checks the version
/// counter before and after reading the data. If the counter has changed, the
/// read is retried. Writes increment the counter to an odd number, perform the
/// write, and then increment it back to an even number.
#[derive(Debug)]
pub(super) struct SeqLockWord<W: Word> {
    /// The version counter for the seqlock. Even = unlocked, Odd = write in progress.
    pub(super) version: AtomicUsize,
    /// The actual data, wrapped in UnsafeCell for interior mutability.
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

// SAFETY: `SeqLockWord` is safe to send across threads if `W` is `Send`.
// The internal `UnsafeCell` is only accessed according to the seqlock protocol,
// which ensures thread safety.
unsafe impl<W: Word + Send> Send for SeqLockWord<W> {}
unsafe impl<W: Word + Send> Sync for SeqLockWord<W> {}

/// A trait that defines the interface for an atomic storage backend.
///
/// This abstraction allows the `AtomicFixedVec` to be generic over its
/// storage, facilitating testing and potential future extensions.
pub trait AtomicBackend<W: Word + IntoAtomic>: Send + Sync + Debug {
    /// Returns a slice of the underlying seqlocked words.
    fn seqlocks(&self) -> &[SeqLockWord<W>];

    /// Returns a slice of the mutexes used for striped locking complex RMW operations.
    fn writer_locks(&self) -> &[Mutex<()>];
}

/// An owned, growable atomic storage backend for `AtomicFixedVec`.
///
/// It contains:
/// * `seqlocks`: A vector of `SeqLockWord`s that holds the packed data,
///   enabling highly concurrent reads.
/// * `writer_locks`: A fixed-size array of `parking_lot::Mutex`es for write operations.
#[derive(Debug)]
pub struct OwnedAtomicBackend<W: Word + IntoAtomic> {
    /// The bit-packed data, stored in seqlocked words.
    seqlocks: Vec<SeqLockWord<W>>,
    /// A fixed-size array of lightweight mutexes for complex write operations.
    writer_locks: Box<[Mutex<()>; NUM_STRIPES]>,
}

impl<W: Word + IntoAtomic + Zero> OwnedAtomicBackend<W> {
    /// Creates a new `OwnedAtomicBackend` with a specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity`: The number of elements the backend should be able to hold
    ///   without reallocating.
    /// * `bit_width`: The number of bits used for each element.
    pub fn new(capacity: usize, bit_width: usize) -> Self {
        // Calculate the number of words needed for the data buffer.
        // We add padding to ensure that even boundary-crossing reads/writes
        // near the end of the vector do not go out of bounds.
        let total_bits = capacity.saturating_mul(bit_width);
        let num_words = total_bits.div_ceil(<W as Word>::BITS);
        let buffer_len = if capacity == 0 { 0 } else { num_words + 2 };

        // Initialize the seqlocked data buffer with zeros.
        let seqlocks = (0..buffer_len).map(|_| SeqLockWord::new()).collect();

        // Initialize the array of mutexes for striped locking.
        // `std::array::from_fn` is the idiomatic way to create an array of
        // non-Copy types like Mutex.
        let writer_locks = Box::new(std::array::from_fn(|_| Mutex::new(())));

        Self {
            seqlocks,
            writer_locks,
        }
    }
}

impl<W: Word + IntoAtomic + Zero> AtomicBackend<W> for OwnedAtomicBackend<W> {
    /// Returns a slice of the underlying seqlocked words.
    #[inline]
    fn seqlocks(&self) -> &[SeqLockWord<W>] {
        &self.seqlocks
    }

    /// Returns a slice of the mutexes used for striped locking.
    #[inline]
    fn writer_locks(&self) -> &[Mutex<()>] {
        &*self.writer_locks
    }
}

// The following implementations for SeqLockWord are not strictly required by the
// backend itself but are the core of the seqlock mechanism that will be used
// in the `strategy.rs` module. They are included here for completeness.

impl<W: Word> SeqLockWord<W> {
    /// Performs a lock-free, optimistic read of the data.
    ///
    /// This function will spin if a write is in progress.
    #[inline(always)]
    pub(super) fn optimistic_read(&self) -> W {
        loop {
            let version1 = self.version.load(Ordering::Acquire);
            if version1 % 2 != 0 {
                // A write is in progress, spin.
                std::hint::spin_loop();
                continue;
            }

            // SAFETY: The version check ensures no writer is active.
            let data = unsafe { *self.data.get() };

            let version2 = self.version.load(Ordering::Acquire);

            if version1 == version2 {
                // The version is unchanged, so the read was consistent.
                return data;
            }
            // A write occurred during our read, so we retry.
        }
    }

    /// Writes a value, locking optimistically.
    ///
    /// This is used for simple `store` operations.
    #[inline(always)]
    pub(super) fn write(&self, f: impl FnOnce(W) -> W) {
        // Increment version to odd, signaling a write is starting.
        self.version.fetch_add(1, Ordering::Release);

        // SAFETY: We have signaled that we are writing, so we can safely mutate.
        let current_data = unsafe { *self.data.get() };
        let new_data = f(current_data);
        unsafe {
            *self.data.get() = new_data;
        }

        // Increment version to even, signaling the write is complete.
        self.version.fetch_add(1, Ordering::Release);
    }
}