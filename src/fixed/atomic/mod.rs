//! # `AtomicFixedVec`: A Thread-Safe Fixed-Width Integer Vector
//!
//! This module provides `AtomicFixedVec`, a variant of `FixedVec` designed for
//! safe, highly-performant concurrent access from multiple threads. It guarantees
//! atomicity for all operations by transparently selecting the optimal strategy.
//!
//! Unlike `FixedVec`, this structure does not support dynamic resizing operations
//! like `push` or `pop`, as these cannot be implemented efficiently and safely
//! in a lock-free manner for a shared vector. The intended use case is to
//! create the vector from an initial set of data and then share it across
//! threads for concurrent reads and updates of existing elements.
//!
//! ## Concurrency Strategy
//!
//! This implementation uses a hybrid strategy to balance performance, space efficiency,
//! and correctness, with a storage backend fixed to `u64` words:
//!
//! - **Single-Word Operations (Lock-Free)**: For elements that are guaranteed to be
//!   contained within a single `AtomicU64`, all operations are fully lock-free
//!   using standard compare-and-swap (CAS) loops. This is the common path for
//!   bit-widths that are powers of two (e.g., 8, 16, 32).
//!
//! - **Spanning-Word Operations (Locked)**: For elements that may span the
//!   boundary between two `AtomicU64` words, operations are protected by a lock.
//!   This ensures that multi-word updates are fully atomic and prevents "torn writes".
//!   To minimize contention, this implementation uses "lock striping", where a pool
//!   of locks is used to protect different regions of the vector.
//!
//! This design ensures that `AtomicFixedVec` is always as space-efficient as
//! `FixedVec` while providing robust atomic guarantees for all possible bit-widths.
//!
//! # Examples
//!
//! ```
//! use compressed_intvec::prelude::*;
//! use std::sync::Arc;
//! use std::thread;
//! use std::sync::atomic::Ordering;
//!
//! // Create from a slice using the builder.
//! let initial_data: Vec<u32> = vec!;
//! let atomic_vec: Arc<UAtomicFixedVec<u32>> = Arc::new(
//!     AtomicFixedVec::builder()
//!         .build(&initial_data)
//!         .unwrap()
//! );
//!
//! // Share the vector across threads.
//! let mut handles = vec![];
//! for i in 0..4 {
//!     let vec_clone = Arc::clone(&atomic_vec);
//!     handles.push(thread::spawn(move || {
//!         // Each thread atomically updates its own slot.
//!         vec_clone.store(i, 99, Ordering::SeqCst);
//!     }));
//! }
//! for handle in handles {
//!     handle.join().unwrap();
//! }
//! assert_eq!(atomic_vec.load(3, Ordering::SeqCst), 99);
//! ```

#[macro_use]
pub mod macros;
pub mod builder;

use crate::fixed::traits::Storable;
use crate::fixed::{BitWidth, Error};
use num_traits::{ToPrimitive};
use parking_lot::Mutex;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// A thread-safe `FixedVec` for unsigned integers.
pub type UAtomicFixedVec<T> = AtomicFixedVec<T>;
/// A thread-safe `FixedVec` for signed integers.
pub type SAtomicFixedVec<T> = AtomicFixedVec<T>;

/// The number of locks used for lock striping. A power of two is chosen
/// to allow fast bitwise AND instead of a slower modulo operation.
const NUM_LOCKS: usize = 256;

/// A thread-safe, compressed, randomly accessible vector of integers with
/// fixed-width encoding, backed by `u64` atomic words.
#[derive(Debug)]
pub struct AtomicFixedVec<T>
where
    T: Storable<u64>,
{
    /// The underlying storage for the bit-packed data.
    pub(crate) storage: Vec<AtomicU64>,
    /// A pool of locks to protect spanning-word operations.
    locks: Vec<Mutex<()>>,
    bit_width: usize,
    mask: u64,
    len: usize,
    _phantom: PhantomData<T>,
}

// Public API implementation
impl<T> AtomicFixedVec<T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    /// Creates a builder for constructing an `AtomicFixedVec` from a slice.
    pub fn builder() -> builder::AtomicFixedVecBuilder<T> {
        builder::AtomicFixedVecBuilder::new()
    }

    /// Returns the number of elements in the vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of bits used to encode each element.
    #[inline]
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// Atomically loads the value at `index`.
    pub fn load(&self, index: usize, order: Ordering) -> T {
        assert!(index < self.len, "load index out of bounds");
        let loaded_word = self.atomic_load(index, order);
        T::from_word(loaded_word)
    }

    /// Atomically stores `value` at `index`.
    pub fn store(&self, index: usize, value: T, order: Ordering) {
        assert!(index < self.len, "store index out of bounds");
        let value_w = T::into_word(value);
        self.atomic_store(index, value_w, order);
    }

    /// Atomically swaps the value at `index` with `value`.
    pub fn swap(&self, index: usize, value: T, order: Ordering) -> T {
        assert!(index < self.len, "swap index out of bounds");
        let value_w = T::into_word(value);
        let old_word = self.atomic_swap(index, value_w, order);
        T::from_word(old_word)
    }

    /// Atomically compares the value at `index` with `current` and replaces it with `new`.
    pub fn compare_exchange(
        &self,
        index: usize,
        current: T,
        new: T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<T, T> {
        assert!(index < self.len, "compare_exchange index out of bounds");
        let current_w = T::into_word(current);
        let new_w = T::into_word(new);
        match self.atomic_compare_exchange(index, current_w, new_w, success, failure) {
            Ok(w) => Ok(T::from_word(w)),
            Err(w) => Err(T::from_word(w)),
        }
    }
}

// `TryFrom` implementation.
impl<T> TryFrom<&[T]> for AtomicFixedVec<T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    type Error = Error;

    /// Creates an `AtomicFixedVec<T>` from a slice using `BitWidth::Minimal`.
    fn try_from(slice: &[T]) -> Result<Self, Self::Error> {
        AtomicFixedVec::builder()
            .bit_width(BitWidth::Minimal)
            .build(slice)
    }
}

// Constructor (internal to the crate, used by the builder).
impl<T> AtomicFixedVec<T>
where
    T: Storable<u64>,
{
    /// Creates a new, zero-initialized `AtomicFixedVec`.
    pub(crate) fn new(bit_width: usize, len: usize) -> Result<Self, Error> {
        if bit_width > u64::BITS as usize {
            return Err(Error::InvalidParameters(format!(
                "bit_width ({}) cannot be greater than the word size ({})",
                bit_width,
                u64::BITS
            )));
        }

        let mask = if bit_width == u64::BITS as usize {
            u64::MAX
        } else {
            (1u64 << bit_width).wrapping_sub(1)
        };

        let total_bits = len.saturating_mul(bit_width);
        let num_words = total_bits.div_ceil(u64::BITS as usize);
        let buffer_len = if len == 0 { 0 } else { num_words + 1 }; // +1 for padding
        let storage = (0..buffer_len).map(|_| AtomicU64::new(0)).collect();

        let locks = (0..NUM_LOCKS).map(|_| Mutex::new(())).collect();

        Ok(Self {
            storage,
            locks,
            bit_width,
            mask,
            len,
            _phantom: PhantomData,
        })
    }
}

// --- Private Implementation of Atomic Operations ---
impl<T> AtomicFixedVec<T>
where
    T: Storable<u64>,
{
    fn atomic_load(&self, index: usize, order: Ordering) -> u64 {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / u64::BITS as usize;
        let bit_offset = bit_pos % u64::BITS as usize;

        if bit_offset + self.bit_width <= u64::BITS as usize {
            // Lock-free path for single-word values.
            let word = self.storage[word_index].load(order);
            (word >> bit_offset) & self.mask
        } else {
            // Locked path for spanning values.
            let lock_index = word_index & (NUM_LOCKS - 1);
            let _guard = self.locks[lock_index].lock();
            let low_word = self.storage[word_index].load(Ordering::Relaxed);
            let high_word = self.storage[word_index + 1].load(Ordering::Relaxed);
            let combined = (low_word >> bit_offset) | (high_word << (u64::BITS as usize - bit_offset));
            combined & self.mask
        }
    }

    fn atomic_store(&self, index: usize, value: u64, order: Ordering) {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / u64::BITS as usize;
        let bit_offset = bit_pos % u64::BITS as usize;

        if bit_offset + self.bit_width <= u64::BITS as usize {
            // Lock-free path.
            let atomic_word_ref = &self.storage[word_index];
            let store_mask = self.mask << bit_offset;
            let store_value = value << bit_offset;
            atomic_word_ref
                .fetch_update(order, order, |old_word| {
                    Some((old_word & !store_mask) | store_value)
                })
                .unwrap();
        } else {
            // Locked path.
            let lock_index = word_index & (NUM_LOCKS - 1);
            let _guard = self.locks[lock_index].lock();
            // SAFETY: The lock guarantees exclusive access.
            unsafe {
                let low_word_ptr = self.storage.as_ptr().add(word_index) as *mut u64;
                let high_word_ptr = self.storage.as_ptr().add(word_index + 1) as *mut u64;

                *low_word_ptr &= !(u64::MAX << bit_offset);
                *low_word_ptr |= value << bit_offset;

                let bits_in_high = (bit_offset + self.bit_width) - u64::BITS as usize;
                let high_mask = (1u64 << bits_in_high).wrapping_sub(1);
                *high_word_ptr &= !high_mask;
                *high_word_ptr |= value >> (u64::BITS as usize - bit_offset);
            }
        }
    }

    fn atomic_swap(&self, index: usize, value: u64, order: Ordering) -> u64 {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / u64::BITS as usize;
        let bit_offset = bit_pos % u64::BITS as usize;

        if bit_offset + self.bit_width <= u64::BITS as usize {
            // Lock-free path.
            let atomic_word_ref = &self.storage[word_index];
            let store_mask = self.mask << bit_offset;
            let store_value = value << bit_offset;
            let mut old_word = atomic_word_ref.load(Ordering::Relaxed);
            loop {
                let new_word = (old_word & !store_mask) | store_value;
                match atomic_word_ref.compare_exchange_weak(old_word, new_word, order, Ordering::Relaxed) {
                    Ok(_) => return (old_word >> bit_offset) & self.mask,
                    Err(x) => old_word = x,
                }
            }
        } else {
            // Locked path.
            let lock_index = word_index & (NUM_LOCKS - 1);
            let _guard = self.locks[lock_index].lock();
            // SAFETY: The lock guarantees exclusive access.
            unsafe {
                let low_word_ptr = self.storage.as_ptr().add(word_index) as *mut u64;
                let high_word_ptr = self.storage.as_ptr().add(word_index + 1) as *mut u64;

                let combined_old = (*low_word_ptr >> bit_offset) | (*high_word_ptr << (u64::BITS as usize - bit_offset));
                let old_val = combined_old & self.mask;

                *low_word_ptr &= !(u64::MAX << bit_offset);
                *low_word_ptr |= value << bit_offset;

                let bits_in_high = (bit_offset + self.bit_width) - u64::BITS as usize;
                let high_mask = (1u64 << bits_in_high).wrapping_sub(1);
                *high_word_ptr &= !high_mask;
                *high_word_ptr |= value >> (u64::BITS as usize - bit_offset);

                old_val
            }
        }
    }

    fn atomic_compare_exchange(
        &self,
        index: usize,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / u64::BITS as usize;
        let bit_offset = bit_pos % u64::BITS as usize;

        if bit_offset + self.bit_width <= u64::BITS as usize {
            // Lock-free path.
            let atomic_word_ref = &self.storage[word_index];
            let store_mask = self.mask << bit_offset;
            let new_value_shifted = new << bit_offset;
            let mut old_word = atomic_word_ref.load(failure);
            loop {
                let old_val = (old_word >> bit_offset) & self.mask;
                if old_val != current {
                    return Err(old_val);
                }
                let new_word = (old_word & !store_mask) | new_value_shifted;
                match atomic_word_ref.compare_exchange_weak(old_word, new_word, success, failure) {
                    Ok(_) => return Ok(current),
                    Err(x) => old_word = x,
                }
            }
        } else {
            // Locked path.
            let lock_index = word_index & (NUM_LOCKS - 1);
            let _guard = self.locks[lock_index].lock();
            // SAFETY: The lock guarantees exclusive access.
            unsafe {
                let low_word_ptr = self.storage.as_ptr().add(word_index) as *mut u64;
                let high_word_ptr = self.storage.as_ptr().add(word_index + 1) as *mut u64;

                let combined_old = (*low_word_ptr >> bit_offset) | (*high_word_ptr << (u64::BITS as usize - bit_offset));
                let old_val = combined_old & self.mask;

                if old_val != current {
                    return Err(old_val);
                }

                *low_word_ptr &= !(u64::MAX << bit_offset);
                *low_word_ptr |= new << bit_offset;

                let bits_in_high = (bit_offset + self.bit_width) - u64::BITS as usize;
                let high_mask = (1u64 << bits_in_high).wrapping_sub(1);
                *high_word_ptr &= !high_mask;
                *high_word_ptr |= new >> (u64::BITS as usize - bit_offset);

                Ok(current)
            }
        }
    }
}