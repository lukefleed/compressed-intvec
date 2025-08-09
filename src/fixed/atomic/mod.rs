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
use crate::fixed::{BitWidth, Error, FixedVec};
use mem_dbg::{DbgFlags, MemDbgImpl, MemSize, SizeFlags};
use num_traits::{Bounded, ToPrimitive};
use parking_lot::Mutex;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// A thread-safe `FixedVec` for unsigned integers.
pub type UAtomicFixedVec<T> = AtomicFixedVec<T>;
/// A thread-safe `FixedVec` for signed integers.
pub type SAtomicFixedVec<T> = AtomicFixedVec<T>;

/// The upper bound on the number of locks to prevent excessive memory usage.
const MAX_LOCKS: usize = 1024;
/// The minimum number of locks to create, ensuring some striping even for small vectors.
const MIN_LOCKS: usize = 2;
/// A heuristic to determine the stripe size: one lock per this many data words.
const WORDS_PER_LOCK: usize = 64;

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
    #[inline(always)]
    pub fn builder() -> builder::AtomicFixedVecBuilder<T> {
        builder::AtomicFixedVecBuilder::new()
    }

    /// Returns the number of elements in the vector.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of bits used to encode each element.
    #[inline(always)]
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// Returns a read-only slice of the underlying atomic storage words.
    #[inline(always)]
    pub fn as_slice(&self) -> &[AtomicU64] {
        &self.storage
    }

    /// Atomically loads the value at `index`.
    #[inline(always)]
    pub fn load(&self, index: usize, order: Ordering) -> T {
        assert!(index < self.len, "load index out of bounds");
        let loaded_word = self.atomic_load(index, order);
        T::from_word(loaded_word)
    }

    /// Atomically stores `value` at `index`.
    #[inline(always)]
    pub fn store(&self, index: usize, value: T, order: Ordering) {
        assert!(index < self.len, "store index out of bounds");
        let value_w = T::into_word(value);
        self.atomic_store(index, value_w, order);
    }

    /// Atomically swaps the value at `index` with `value`.
    #[inline(always)]
    pub fn swap(&self, index: usize, value: T, order: Ordering) -> T {
        assert!(index < self.len, "swap index out of bounds");
        let value_w = T::into_word(value);
        let old_word = self.atomic_swap(index, value_w, order);
        T::from_word(old_word)
    }

    /// Atomically compares the value at `index` with `current` and replaces it with `new`.
    #[inline(always)]
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

// Extended atomic RMW operations
impl<T> AtomicFixedVec<T>
where
    T: Storable<u64> + Bounded + Copy + ToPrimitive,
{
    /// Atomically adds to the value at `index`, returning the previous value.
    #[inline(always)]
    pub fn fetch_add(&self, index: usize, val: T, order: Ordering) -> T {
        self.atomic_rmw(index, val, order, |a, b| {
            T::from_word(T::into_word(a).wrapping_add(T::into_word(b)))
        })
    }

    /// Atomically subtracts from the value at `index`, returning the previous value.
    #[inline(always)]
    pub fn fetch_sub(&self, index: usize, val: T, order: Ordering) -> T {
        self.atomic_rmw(index, val, order, |a, b| {
            T::from_word(T::into_word(a).wrapping_sub(T::into_word(b)))
        })
    }

    /// Atomically performs a bitwise AND on the value at `index`, returning the previous value.
    #[inline(always)]
    pub fn fetch_and(&self, index: usize, val: T, order: Ordering) -> T {
        self.atomic_rmw(index, val, order, |a, b| {
            T::from_word(T::into_word(a) & T::into_word(b))
        })
    }

    /// Atomically performs a bitwise OR on the value at `index`, returning the previous value.
    #[inline(always)]
    pub fn fetch_or(&self, index: usize, val: T, order: Ordering) -> T {
        self.atomic_rmw(index, val, order, |a, b| {
            T::from_word(T::into_word(a) | T::into_word(b))
        })
    }

    /// Atomically performs a bitwise XOR on the value at `index`, returning the previous value.
    #[inline(always)]
    pub fn fetch_xor(&self, index: usize, val: T, order: Ordering) -> T {
        self.atomic_rmw(index, val, order, |a, b| {
            T::from_word(T::into_word(a) ^ T::into_word(b))
        })
    }

    /// Atomically computes the maximum of the value at `index` and `val`, returning the previous value.
    #[inline(always)]
    pub fn fetch_max(&self, index: usize, val: T, order: Ordering) -> T
    where
        T: Ord,
    {
        self.atomic_rmw(index, val, order, |a, b| a.max(b))
    }

    /// Atomically computes the minimum of the value at `index` and `val`, returning the previous value.
    #[inline(always)]
    pub fn fetch_min(&self, index: usize, val: T, order: Ordering) -> T
    where
        T: Ord,
    {
        self.atomic_rmw(index, val, order, |a, b| a.min(b))
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

        // Heuristic to determine the number of locks for striping.
        let num_locks = if len == 0 {
            MIN_LOCKS
        } else {
            let num_cores = std::thread::available_parallelism().map_or(MIN_LOCKS, |n| n.get());
            let target_locks = (num_words / WORDS_PER_LOCK).max(1);
            (target_locks.max(num_cores) * 2)
                .next_power_of_two()
                .min(MAX_LOCKS)
        };

        let locks = (0..num_locks).map(|_| Mutex::new(())).collect();

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
    #[inline(always)]
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
            let lock_index = word_index & (self.locks.len() - 1);
            let _guard = self.locks[lock_index].lock();
            let low_word = self.storage[word_index].load(Ordering::Relaxed);
            let high_word = self.storage[word_index + 1].load(Ordering::Relaxed);
            let combined =
                (low_word >> bit_offset) | (high_word << (u64::BITS as usize - bit_offset));
            combined & self.mask
        }
    }

    #[inline(always)]
    fn atomic_store(&self, index: usize, value: u64, order: Ordering) {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / u64::BITS as usize;
        let bit_offset = bit_pos % u64::BITS as usize;

        if bit_offset + self.bit_width <= u64::BITS as usize {
            // Lock-free path for single-word values.
            // This uses a manual compare-and-swap (CAS) loop, which can be more
            // performant than `fetch_update` by avoiding closure overhead and
            // allowing for more fine-grained control over memory orderings.
            let atomic_word_ref = &self.storage[word_index];
            let store_mask = self.mask << bit_offset;
            let store_value = value << bit_offset;

            // Start with a relaxed load, as we only need atomicity for the final write.
            let mut old_word = atomic_word_ref.load(Ordering::Relaxed);
            loop {
                // Calculate the new word value by clearing the target bits and ORing the new value.
                let new_word = (old_word & !store_mask) | store_value;

                // Attempt to swap the old word with the new one.
                // `compare_exchange_weak` is used as it can be faster inside a loop,
                // even if it fails spuriously. The failure ordering can be relaxed.
                match atomic_word_ref.compare_exchange_weak(
                    old_word,
                    new_word,
                    order, // Use the user-specified ordering on success.
                    Ordering::Relaxed, // Relaxed ordering is sufficient on failure.
                ) {
                    Ok(_) => break, // The store was successful.
                    Err(x) => old_word = x, // The word was modified by another thread; retry.
                }
            }
        } else {
            // Locked path for values spanning two words.
            let lock_index = word_index & (self.locks.len() - 1);
            let _guard = self.locks[lock_index].lock();
            // SAFETY: The lock guarantees exclusive access, so we can perform non-atomic writes.
            // The pointers are obtained from a valid, owned Vec.
            unsafe {
                let low_word_ptr = self.storage.as_ptr().add(word_index) as *mut u64;
                let high_word_ptr = self.storage.as_ptr().add(word_index + 1) as *mut u64;

                // Modify the lower word.
                *low_word_ptr &= !(u64::MAX << bit_offset);
                *low_word_ptr |= value << bit_offset;

                // Modify the higher word.
                let bits_in_high = (bit_offset + self.bit_width) - u64::BITS as usize;
                let high_mask = (1u64 << bits_in_high).wrapping_sub(1);
                *high_word_ptr &= !high_mask;
                *high_word_ptr |= value >> (u64::BITS as usize - bit_offset);
            }
        }
    }

    #[inline(always)]
    fn atomic_swap(&self, index: usize, value: u64, order: Ordering) -> u64 {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / u64::BITS as usize;
        let bit_offset = bit_pos % u64::BITS as usize;

        if bit_offset + self.bit_width <= u64::BITS as usize {
            // Lock-free path for single-word values.
            let atomic_word_ref = &self.storage[word_index];
            let store_mask = self.mask << bit_offset;
            let store_value = value << bit_offset;
            let mut old_word = atomic_word_ref.load(Ordering::Relaxed);
            loop {
                let new_word = (old_word & !store_mask) | store_value;
                match atomic_word_ref.compare_exchange_weak(
                    old_word,
                    new_word,
                    order,
                    Ordering::Relaxed,
                ) {
                    // If the CAS succeeds, extract and return the old value.
                    Ok(_) => return (old_word >> bit_offset) & self.mask,
                    // Otherwise, retry with the updated value.
                    Err(x) => old_word = x,
                }
            }
        } else {
            // Locked path for spanning values.
            let lock_index = word_index & (self.locks.len() - 1);
            let _guard = self.locks[lock_index].lock();
            // SAFETY: The lock guarantees exclusive access.
            unsafe {
                let low_word_ptr = self.storage.as_ptr().add(word_index) as *mut u64;
                let high_word_ptr = self.storage.as_ptr().add(word_index + 1) as *mut u64;

                let combined_old = (*low_word_ptr >> bit_offset)
                    | (*high_word_ptr << (u64::BITS as usize - bit_offset));
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

    #[inline(always)]
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
            // Lock-free path for single-word values.
            let atomic_word_ref = &self.storage[word_index];
            let store_mask = self.mask << bit_offset;
            let new_value_shifted = new << bit_offset;

            // The `failure` ordering is important here for the initial load.
            let mut old_word = atomic_word_ref.load(failure);
            loop {
                // Extract the current value from the bitfield.
                let old_val_extracted = (old_word >> bit_offset) & self.mask;

                // If the value in the bitfield does not match the expected one, fail immediately.
                if old_val_extracted != current {
                    return Err(old_val_extracted);
                }

                // Calculate the new word to be written.
                let new_word = (old_word & !store_mask) | new_value_shifted;

                // Attempt the atomic exchange.
                match atomic_word_ref.compare_exchange_weak(
                    old_word,
                    new_word,
                    success,
                    failure, // Use the specified failure ordering.
                ) {
                    Ok(_) => return Ok(current), // Success, return the expected value.
                    Err(x) => old_word = x, // Failure, the word was updated, retry the loop.
                }
            }
        } else {
            // Locked path for spanning values.
            let lock_index = word_index & (self.locks.len() - 1);
            let _guard = self.locks[lock_index].lock();
            // SAFETY: The lock guarantees exclusive access.
            unsafe {
                let low_word_ptr = self.storage.as_ptr().add(word_index) as *mut u64;
                let high_word_ptr = self.storage.as_ptr().add(word_index + 1) as *mut u64;

                let combined_old = (*low_word_ptr >> bit_offset)
                    | (*high_word_ptr << (u64::BITS as usize - bit_offset));
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

        /// Generic implementation for all Read-Modify-Write (RMW) operations.
    #[inline(always)]
    fn atomic_rmw(&self, index: usize, val: T, order: Ordering, op: impl Fn(T, T) -> T) -> T {
        assert!(index < self.len, "RMW index out of bounds");
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / u64::BITS as usize;
        let bit_offset = bit_pos % u64::BITS as usize;

        if bit_offset + self.bit_width <= u64::BITS as usize {
            // Lock-free path
            let atomic_word_ref = &self.storage[word_index];
            let store_mask = self.mask << bit_offset;
            let mut old_word = atomic_word_ref.load(Ordering::Relaxed);
            loop {
                // 1. Extract the current ENCODED value from the word.
                let old_val_encoded = (old_word >> bit_offset) & self.mask;

                // 2. DECODE the value to perform the operation on the actual numbers.
                let old_val_decoded = T::from_word(old_val_encoded);

                // 3. Perform the user-provided operation on the DECODED values.
                let new_val_decoded = op(old_val_decoded, val);

                // 4. RE-ENCODE the result before storing it.
                let new_val_encoded = T::into_word(new_val_decoded) & self.mask;

                // 5. Prepare the new word for the CAS operation.
                let new_word = (old_word & !store_mask) | (new_val_encoded << bit_offset);

                // 6. Attempt the compare-and-swap.
                match atomic_word_ref.compare_exchange_weak(
                    old_word,
                    new_word,
                    order,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return old_val_decoded, // Return the *old* DECODED value on success.
                    Err(x) => old_word = x,
                }
            }
        } else {
            // Locked path
            let lock_index = word_index & (self.locks.len() - 1);
            let _guard = self.locks[lock_index].lock();

            // The logic inside the lock is simpler as it's serialized.
            let old_val_encoded = self.atomic_load(index, Ordering::Relaxed);
            let old_val_decoded = T::from_word(old_val_encoded);

            let new_val_decoded = op(old_val_decoded, val);
            let new_val_encoded = T::into_word(new_val_decoded) & self.mask;

            self.atomic_store(index, new_val_encoded, Ordering::Relaxed);
            old_val_decoded
        }
    }

}

// --- Conversions between AtomicFixedVec and FixedVec ---

impl<T, W, E> From<FixedVec<T, W, E, Vec<W>>> for AtomicFixedVec<T>
where
    T: Storable<W> + Storable<u64>,
    W: crate::fixed::traits::Word,
    E: dsi_bitstream::prelude::Endianness,
{
    /// Creates an `AtomicFixedVec` from an owned `FixedVec`.
    /// This is a zero-copy operation that re-uses the allocated buffer.
    fn from(fixed_vec: FixedVec<T, W, E, Vec<W>>) -> Self {
        // SAFETY: This transmutation is safe because `AtomicU64` and `u64` have
        // the same in-memory representation. We are taking ownership of the Vec,
        // ensuring no other references to the non-atomic data exist.
        let storage = unsafe {
            let mut md = std::mem::ManuallyDrop::new(fixed_vec.bits);
            Vec::from_raw_parts(md.as_mut_ptr() as *mut AtomicU64, md.len(), md.capacity())
        };

        let num_words = (fixed_vec.len * fixed_vec.bit_width).div_ceil(u64::BITS as usize);
        let num_locks = if fixed_vec.len == 0 {
            MIN_LOCKS
        } else {
            let num_cores = std::thread::available_parallelism().map_or(MIN_LOCKS, |n| n.get());
            let target_locks = (num_words / WORDS_PER_LOCK).max(1);
            (target_locks.max(num_cores) * 2)
                .next_power_of_two()
                .min(MAX_LOCKS)
        };
        let locks = (0..num_locks).map(|_| Mutex::new(())).collect();

        Self {
            storage,
            locks,
            bit_width: fixed_vec.bit_width,
            mask: fixed_vec.mask.to_u64().unwrap(),
            len: fixed_vec.len,
            _phantom: PhantomData,
        }
    }
}

impl<T> From<AtomicFixedVec<T>> for FixedVec<T, u64, dsi_bitstream::prelude::LE, Vec<u64>>
where
    T: Storable<u64>,
{
    /// Creates a `FixedVec` from an owned `AtomicFixedVec`.
    /// This is a zero-copy operation that re-uses the allocated buffer.
    fn from(atomic_vec: AtomicFixedVec<T>) -> Self {
        // SAFETY: This transmutation is safe because `u64` and `AtomicU64` have
        // the same in-memory representation. We are taking ownership of the Vec,
        // ensuring no other references to the atomic data exist.
        let bits = unsafe {
            let mut md = std::mem::ManuallyDrop::new(atomic_vec.storage);
            Vec::from_raw_parts(md.as_mut_ptr() as *mut u64, md.len(), md.capacity())
        };

        unsafe { FixedVec::new_unchecked(bits, atomic_vec.len, atomic_vec.bit_width) }
    }
}

// --- MemDbg and MemSize Implementations ---

impl<T> MemSize for AtomicFixedVec<T>
where
    T: Storable<u64>,
{
    fn mem_size(&self, flags: SizeFlags) -> usize {
        // Since `parking_lot::Mutex` does not implement `CopyType`, we must calculate
        // the size of the `locks` vector manually.
        let locks_size = if flags.contains(SizeFlags::CAPACITY) {
            self.locks.capacity() * core::mem::size_of::<Mutex<()>>()
        } else {
            self.locks.len() * core::mem::size_of::<Mutex<()>>()
        };

        core::mem::size_of::<Self>()
            + self.storage.mem_size(flags)
            + core::mem::size_of::<Vec<Mutex<()>>>()
            + locks_size
    }
}

impl<T: Storable<u64>> MemDbgImpl for AtomicFixedVec<T> {
    fn _mem_dbg_rec_on(
        &self,
        writer: &mut impl core::fmt::Write,
        total_size: usize,
        max_depth: usize,
        prefix: &mut String,
        _is_last: bool,
        flags: DbgFlags,
    ) -> core::fmt::Result {
        // Manual implementation to avoid trying to lock and inspect mutexes.
        self.bit_width._mem_dbg_rec_on(
            writer,
            total_size,
            max_depth,
            prefix,
            false,
            flags,
        )?;
        self.len
            ._mem_dbg_rec_on(writer, total_size, max_depth, prefix, false, flags)?;
        self.mask
            ._mem_dbg_rec_on(writer, total_size, max_depth, prefix, false, flags)?;

        // Display the size of the lock vector, but do not recurse into it.
        let locks_size = core::mem::size_of::<Vec<Mutex<()>>>()
            + self.locks.capacity() * core::mem::size_of::<Mutex<()>>();
        locks_size._mem_dbg_rec_on(writer, total_size, max_depth, prefix, false, flags)?;

        self.storage
            ._mem_dbg_rec_on(writer, total_size, max_depth, prefix, true, flags)?;
        Ok(())
    }
}