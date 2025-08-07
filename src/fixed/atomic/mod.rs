//! # `AtomicFixedVec`: A Thread-Safe Fixed-Width Integer Vector
//!
//! This module provides `AtomicFixedVec`, a variant of `FixedVec` designed for
//! safe, concurrent access from multiple threads. It guarantees atomicity for
//! all operations, including those that span word boundaries, by transparently
//! selecting the optimal synchronization strategy at construction time.
//!
//! ## Strategies
//!
//! - **Lock-Free (Single-Word)**: For configurations where elements are guaranteed
//!   to be contained within a single atomic word (i.e., when `bit_width` is a
//!   power of two), all operations are implemented using highly-efficient,
//!   lock-free compare-and-swap (CAS) loops on a backend of atomic words.
//!
//! - **Hybrid SeqLock/Mutex (Multi-Word)**: For configurations where elements may
//!   span word boundaries, atomicity is guaranteed using a hybrid mechanism.
//!   Reads are performed optimistically using non-blocking seqlocks for maximum
//!   concurrency. All write operations (`store`, `swap`, `compare_exchange`) use
//!   fine-grained striped mutexes to ensure correctness and prevent torn writes.
//!
//! The choice of strategy is a compile-time decision, ensuring zero runtime
//! overhead for the dispatch mechanism.

// Declare submodules.
mod backend;

use crate::fixed::atomic::backend::{AtomicStorage, OwnedAtomicBackend, SeqLockWord, NUM_STRIPES};
use crate::fixed::traits::{Storable, Word};
use crate::fixed::Error;
use common_traits::{Atomic, IntoAtomic};
use num_traits::{Bounded, One, Zero};
use std::fmt::Debug;
use std::hint;
use std::marker::PhantomData;
use std::sync::atomic::{self, Ordering};

/// A thread-safe, compressed, randomly accessible vector of integers with
/// fixed-width encoding.
///
/// `AtomicFixedVec` provides an API similar to `std::sync::atomic` types, with
/// methods like `load`, `store`, `swap`, and `compare_exchange`. It ensures
/// that all operations are atomic, even for elements that span across the
/// boundaries of the underlying storage words. This implementation uses a
/// native-endian bit layout for performance.
#[derive(Debug)]
pub struct AtomicFixedVec<T, W>
where
    T: Storable<W>,
    W: Word + IntoAtomic,
    W::AtomicType: Debug,
{
    /// The underlying atomic storage backend.
    backend: OwnedAtomicBackend<W>,
    /// The number of bits used to encode each element.
    bit_width: usize,
    /// A mask with the lowest `bit_width` bits set to one.
    mask: W,
    /// The number of elements in the vector.
    len: usize,
    /// Zero-sized markers for the generic type parameters.
    _phantom: PhantomData<T>,
}

// Public API implementation
impl<T, W> AtomicFixedVec<T, W>
where
    T: Storable<W>,
    W: Word + IntoAtomic + Zero + One + Bounded,
    W::AtomicType: Debug,
{
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
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn load(&self, index: usize, order: Ordering) -> T {
        assert!(index < self.len, "load index out of bounds");
        let loaded_word = self.atomic_load(index, order);
        <T as Storable<W>>::from_word(loaded_word)
    }

    /// Atomically stores `value` at `index`.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds. The provided `value` will be
    /// automatically truncated to fit the configured `bit_width`.
    #[inline]
    pub fn store(&self, index: usize, value: T, order: Ordering) {
        assert!(index < self.len, "store index out of bounds");
        let value_w = <T as Storable<W>>::into_word(value);
        self.atomic_store(index, value_w, order);
    }

    /// Atomically swaps the value at `index` with `value`, returning the previous value.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds. The provided `value` will be
    /// automatically truncated to fit the configured `bit_width`.
    #[inline]
    pub fn swap(&self, index: usize, value: T, order: Ordering) -> T {
        assert!(index < self.len, "swap index out of bounds");
        let value_w = <T as Storable<W>>::into_word(value);
        let old_word = self.atomic_swap(index, value_w, order);
        <T as Storable<W>>::from_word(old_word)
    }

    /// Atomically compares the value at `index` with `current`. If they are
    /// equal, it is replaced with `new`.
    ///
    /// See `std::sync::atomic::AtomicU64::compare_exchange` for details on memory ordering.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds. The provided `new` value will be
    /// automatically truncated to fit the configured `bit_width`.
    #[inline]
    pub fn compare_exchange(
        &self,
        index: usize,
        current: T,
        new: T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<T, T> {
        assert!(index < self.len, "compare_exchange index out of bounds");
        let current_w = <T as Storable<W>>::into_word(current);
        let new_w = <T as Storable<W>>::into_word(new);

        match self.atomic_compare_exchange(index, current_w, new_w, success, failure) {
            Ok(w) => Ok(<T as Storable<W>>::from_word(w)),
            Err(w) => Err(<T as Storable<W>>::from_word(w)),
        }
    }
}

// Constructor implementation for the owned backend.
impl<T, W> AtomicFixedVec<T, W>
where
    T: Storable<W>,
    W: Word + IntoAtomic + Zero + One + Bounded,
    W::AtomicType: Debug,
{
    /// Creates a new, zero-initialized `AtomicFixedVec`.
    ///
    /// # Arguments
    /// * `bit_width`: The number of bits for each element.
    /// * `len`: The number of elements in the vector.
    pub fn new(bit_width: usize, len: usize) -> Result<Self, Error> {
        if bit_width > <W as Word>::BITS {
            return Err(Error::InvalidParameters(format!(
                "bit_width ({}) cannot be greater than the word size ({})",
                bit_width,
                <W as Word>::BITS
            )));
        }

        let mask = if bit_width == <W as Word>::BITS {
            <W as Bounded>::max_value()
        } else {
            (<W as One>::one() << bit_width).wrapping_sub(<W as One>::one())
        };

        let backend = OwnedAtomicBackend::new(bit_width, len);

        Ok(Self {
            backend,
            bit_width,
            mask,
            len,
            _phantom: PhantomData,
        })
    }
}

// --- Private Implementation of Atomic Operations ---
// This block contains the core logic that dispatches to the correct backend strategy.
impl<T, W> AtomicFixedVec<T, W>
where
    T: Storable<W>,
    W: Word + IntoAtomic + Bounded + One,
    W::AtomicType: Debug,
{
    fn atomic_load(&self, index: usize, order: Ordering) -> W {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        match &self.backend.storage {
            AtomicStorage::LockFree(atomic_limbs) => {
                let word_ref = unsafe { atomic_limbs.get_unchecked(word_index) };
                let word = word_ref.load(order);
                if self.bit_width == <W as Word>::BITS {
                    word
                } else {
                    (word >> bit_offset) & self.mask
                }
            }
            AtomicStorage::SeqLock { seqlocks, .. } => {
                if bit_offset + self.bit_width <= <W as Word>::BITS {
                    let seqlock = unsafe { seqlocks.get_unchecked(word_index) };
                    let word = seqlock.optimistic_read();
                    (word >> bit_offset) & self.mask
                } else {
                    let word_index2 = word_index + 1;
                    let seqlock1 = unsafe { seqlocks.get_unchecked(word_index) };
                    let seqlock2 = unsafe { seqlocks.get_unchecked(word_index2) };

                    loop {
                        let ver1_start = seqlock1.version.load(Ordering::Acquire);
                        let ver2_start = seqlock2.version.load(Ordering::Acquire);
                        if ver1_start % 2 != 0 || ver2_start % 2 != 0 {
                            hint::spin_loop();
                            continue;
                        }
                        let val1 = unsafe { *seqlock1.data.get() };
                        let val2 = unsafe { *seqlock2.data.get() };
                        atomic::fence(Ordering::Acquire);
                        let ver1_end = seqlock1.version.load(Ordering::Relaxed);
                        let ver2_end = seqlock2.version.load(Ordering::Relaxed);
                        if ver1_start == ver1_end && ver2_start == ver2_end {
                            return ((val1 >> bit_offset)
                                | (val2 << (<W as Word>::BITS - bit_offset)))
                                & self.mask;
                        }
                    }
                }
            }
        }
    }

    fn atomic_store(&self, index: usize, value: W, order: Ordering) {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        match &self.backend.storage {
            AtomicStorage::LockFree(atomic_limbs) => {
                let word_ref = unsafe { atomic_limbs.get_unchecked(word_index) };
                if self.bit_width == <W as Word>::BITS {
                    word_ref.store(value, order);
                    return;
                }

                let store_mask = ((W::one() << self.bit_width) - W::one()) << bit_offset;
                let store_value = (value & self.mask) << bit_offset;
                let mut old_word = word_ref.load(Ordering::Relaxed);
                loop {
                    let new_word = (old_word & !store_mask) | store_value;
                    match word_ref.compare_exchange_weak(
                        old_word,
                        new_word,
                        order,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(x) => old_word = x,
                    }
                }
            }
            AtomicStorage::SeqLock {
                seqlocks,
                writer_locks,
            } => {
                if bit_offset + self.bit_width <= <W as Word>::BITS {
                    let _guard = writer_locks[word_index % NUM_STRIPES].lock();
                    unsafe {
                        let seqlock = seqlocks.get_unchecked(word_index);
                        let mut current = *seqlock.data.get();
                        current &= !(self.mask << bit_offset);
                        current |= (value & self.mask) << bit_offset;
                        versioned_write(seqlock, current);
                    }
                } else {
                    let word_index2 = word_index + 1;
                    with_two_locks(&**writer_locks, word_index, word_index2, || unsafe {
                        let seqlock1 = seqlocks.get_unchecked(word_index);
                        let seqlock2 = seqlocks.get_unchecked(word_index2);

                        let mut word1 = *seqlock1.data.get();
                        word1 &= (W::one() << bit_offset) - W::one();
                        word1 |= (value & self.mask) << bit_offset;
                        versioned_write(seqlock1, word1);

                        let mut word2 = *seqlock2.data.get();
                        let bits_in_second = (bit_offset + self.bit_width) - <W as Word>::BITS;
                        let mask_in_second = W::max_value() >> (<W as Word>::BITS - bits_in_second);
                        word2 &= !mask_in_second;
                        word2 |= (value & self.mask) >> (<W as Word>::BITS - bit_offset);
                        versioned_write(seqlock2, word2);
                    });
                }
            }
        }
    }

    fn atomic_swap(&self, index: usize, value: W, order: Ordering) -> W {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        match &self.backend.storage {
            AtomicStorage::LockFree(atomic_limbs) => {
                let word_ref = unsafe { atomic_limbs.get_unchecked(word_index) };
                if self.bit_width == <W as Word>::BITS {
                    return word_ref.swap(value, order);
                }
                let store_mask = ((W::one() << self.bit_width) - W::one()) << bit_offset;
                let store_value = (value & self.mask) << bit_offset;
                let mut old_word = word_ref.load(Ordering::Relaxed);
                loop {
                    let new_word = (old_word & !store_mask) | store_value;
                    match word_ref.compare_exchange_weak(
                        old_word,
                        new_word,
                        order,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => return (old_word >> bit_offset) & self.mask,
                        Err(x) => old_word = x,
                    }
                }
            }
            AtomicStorage::SeqLock {
                seqlocks,
                writer_locks,
            } => {
                if bit_offset + self.bit_width <= <W as Word>::BITS {
                    let _guard = writer_locks[word_index % NUM_STRIPES].lock();
                    unsafe {
                        let seqlock = seqlocks.get_unchecked(word_index);
                        let old_word = *seqlock.data.get();
                        let old_val = (old_word >> bit_offset) & self.mask;
                        let mut new_word = old_word;
                        new_word &= !(self.mask << bit_offset);
                        new_word |= (value & self.mask) << bit_offset;
                        versioned_write(seqlock, new_word);
                        old_val
                    }
                } else {
                    let word_index2 = word_index + 1;
                    with_two_locks(&**writer_locks, word_index, word_index2, || unsafe {
                        let seqlock1 = seqlocks.get_unchecked(word_index);
                        let seqlock2 = seqlocks.get_unchecked(word_index2);
                        let word1 = *seqlock1.data.get();
                        let word2 = *seqlock2.data.get();
                        let old_val = ((word1 >> bit_offset)
                            | (word2 << (<W as Word>::BITS - bit_offset)))
                            & self.mask;

                        let mut new_word1 = word1;
                        new_word1 &= (W::one() << bit_offset) - W::one();
                        new_word1 |= (value & self.mask) << bit_offset;
                        versioned_write(seqlock1, new_word1);

                        let bits_in_second = (bit_offset + self.bit_width) - <W as Word>::BITS;
                        let mask_in_second =
                            W::max_value() >> (<W as Word>::BITS - bits_in_second);
                        let mut new_word2 = word2;
                        new_word2 &= !mask_in_second;
                        new_word2 |= (value & self.mask) >> (<W as Word>::BITS - bit_offset);
                        versioned_write(seqlock2, new_word2);

                        old_val
                    })
                }
            }
        }
    }

    fn atomic_compare_exchange(
        &self,
        index: usize,
        current: W,
        new: W,
        success: Ordering,
        failure: Ordering,
    ) -> Result<W, W> {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        match &self.backend.storage {
            AtomicStorage::LockFree(atomic_limbs) => {
                let word_ref = unsafe { atomic_limbs.get_unchecked(word_index) };
                if self.bit_width == <W as Word>::BITS {
                    return word_ref.compare_exchange(current, new, success, failure);
                }
                let store_mask = ((W::one() << self.bit_width) - W::one()) << bit_offset;
                let mut old_word = word_ref.load(failure);
                loop {
                    let old_val = (old_word >> bit_offset) & self.mask;
                    if old_val != current {
                        return Err(old_val);
                    }
                    let new_word = (old_word & !store_mask) | ((new & self.mask) << bit_offset);
                    match word_ref.compare_exchange_weak(old_word, new_word, success, failure) {
                        Ok(_) => return Ok(current),
                        Err(x) => old_word = x,
                    }
                }
            }
            AtomicStorage::SeqLock {
                seqlocks,
                writer_locks,
            } => {
                if bit_offset + self.bit_width <= <W as Word>::BITS {
                    let _guard = writer_locks[word_index % NUM_STRIPES].lock();
                    unsafe {
                        let seqlock = seqlocks.get_unchecked(word_index);
                        let fetched_word = *seqlock.data.get();
                        let fetched_val = (fetched_word >> bit_offset) & self.mask;
                        if fetched_val != current {
                            return Err(fetched_val);
                        }
                        let mut new_word = fetched_word;
                        new_word &= !(self.mask << bit_offset);
                        new_word |= (new & self.mask) << bit_offset;
                        versioned_write(seqlock, new_word);
                        Ok(current)
                    }
                } else {
                    let word_index2 = word_index + 1;
                    with_two_locks(&**writer_locks, word_index, word_index2, || unsafe {
                        let seqlock1 = seqlocks.get_unchecked(word_index);
                        let seqlock2 = seqlocks.get_unchecked(word_index2);
                        let word1 = *seqlock1.data.get();
                        let word2 = *seqlock2.data.get();
                        let fetched_val = ((word1 >> bit_offset)
                            | (word2 << (<W as Word>::BITS - bit_offset)))
                            & self.mask;
                        if fetched_val != current {
                            return Err(fetched_val);
                        }

                        let mut new_word1 = word1;
                        new_word1 &= (W::one() << bit_offset) - W::one();
                        new_word1 |= (new & self.mask) << bit_offset;
                        versioned_write(seqlock1, new_word1);

                        let bits_in_second = (bit_offset + self.bit_width) - <W as Word>::BITS;
                        let mask_in_second =
                            W::max_value() >> (<W as Word>::BITS - bits_in_second);
                        let mut new_word2 = word2;
                        new_word2 &= !mask_in_second;
                        new_word2 |= (new & self.mask) >> (<W as Word>::BITS - bit_offset);
                        versioned_write(seqlock2, new_word2);

                        Ok(current)
                    })
                }
            }
        }
    }
}

/// Helper to acquire one or two locks in a globally consistent order to prevent deadlock.
fn with_two_locks<F, R>(
    locks: &[parking_lot::Mutex<()>],
    idx1: usize,
    idx2: usize,
    f: F,
) -> R
where
    F: FnOnce() -> R,
{
    let lock_idx1 = idx1 % NUM_STRIPES;
    let lock_idx2 = idx2 % NUM_STRIPES;

    if lock_idx1 == lock_idx2 {
        let _guard = locks[lock_idx1].lock();
        f()
    } else {
        let (first_idx, second_idx) = if lock_idx1 < lock_idx2 {
            (lock_idx1, lock_idx2)
        } else {
            (lock_idx2, lock_idx1)
        };
        let _guard1 = locks[first_idx].lock();
        let _guard2 = locks[second_idx].lock();
        f()
    }
}

/// Helper to perform a versioned write on a seqlock word.
/// This must be called while holding the appropriate writer lock.
#[inline(always)]
unsafe fn versioned_write<W: Word>(seqlock: &SeqLockWord<W>, new_val: W) {
    seqlock.version.fetch_add(1, Ordering::Release);
    *seqlock.data.get() = new_val;
    seqlock.version.fetch_add(1, Ordering::Release);
}