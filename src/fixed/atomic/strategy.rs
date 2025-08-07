//! # Implementations of Atomic Access Strategies
//!
//! This module provides the concrete implementations for the `SealedAtomicAccess`
//! trait, defining the logic for both the lock-free (single-word) and the
//! hybrid seqlock/mutex (multi-word) strategies.
//!
//! A key feature of this module is the compile-time dispatch mechanism. We use
//! a "const generic" approach on a helper struct, `AccessDispatch`, to allow
//! the Rust compiler to select the correct implementation based on whether an
//! element can span multiple words. This avoids runtime checks and ensures that
//! the most performant strategy is chosen automatically.
//!
//! - `AccessDispatch<..., true>`: Implements the lock-free strategy using CAS loops.
//! - `AccessDispatch<..., false>`: Implements a hybrid strategy. It uses
//!   non-blocking seqlocks for reads and striped mutexes for all write operations
//!   to ensure correctness and atomicity for elements spanning word boundaries.

#![cfg(feature = "atomic")]

use super::access::private::SealedAtomicAccess;
use super::access::CompareExchangeParams;
use super::backend::{AtomicBackend, SeqLockWord, NUM_STRIPES};
use crate::fixed::traits::Word;
use common_traits::{Atomic, IntoAtomic};
use num_traits::{Bounded, One};
use std::hint;
use std::marker::PhantomData;
use std::sync::atomic::{self, Ordering};

/// A helper struct to dispatch to the correct atomic strategy implementation
/// using const generics.
///
/// - `IS_SINGLE_WORD`: A boolean constant. If `true`, the lock-free implementation
///   is used. If `false`, the hybrid seqlock/mutex implementation is used.
struct AccessDispatch<'a, W: Word + IntoAtomic, B: AtomicBackend<W>, const IS_SINGLE_WORD: bool> {
    backend: &'a B,
    _phantom: PhantomData<W>,
}

/// A helper function to determine at compile time if the lock-free strategy can be used.
const fn is_single_word(bit_width: usize) -> bool {
    // The check is simple: the bit width must be a power of two. This guarantees
    // that an element never crosses a word boundary, because the total number of
    // bits in a word (e.g., 64) is always a power of two.
    bit_width.is_power_of_two()
}

/// Blanket implementation of the sealed trait for `AtomicBackend`.
///
/// This implementation uses the `AccessDispatch` helper to delegate to the
/// appropriate strategy based on the compile-time constant `IS_SINGLE_WORD`.
impl<W, B> SealedAtomicAccess<W> for B
where
    W: Word + IntoAtomic + One + Bounded,
    B: AtomicBackend<W>,
{
    #[inline(always)]
    fn atomic_load(&self, index: usize, bit_width: usize, mask: W, order: Ordering) -> W {
        if is_single_word(bit_width) {
            AccessDispatch::<W, B, true> { backend: self, _phantom: PhantomData }
                .atomic_load(index, bit_width, mask, order)
        } else {
            AccessDispatch::<W, B, false> { backend: self, _phantom: PhantomData }
                .atomic_load(index, bit_width, mask, order)
        }
    }

    #[inline(always)]
    fn atomic_store(&self, index: usize, value: W, bit_width: usize, mask: W, order: Ordering) {
        if is_single_word(bit_width) {
            AccessDispatch::<W, B, true> { backend: self, _phantom: PhantomData }
                .atomic_store(index, value, bit_width, mask, order)
        } else {
            AccessDispatch::<W, B, false> { backend: self, _phantom: PhantomData }
                .atomic_store(index, value, bit_width, mask, order)
        }
    }

    #[inline(always)]
    fn atomic_swap(&self, index: usize, value: W, bit_width: usize, mask: W, order: Ordering) -> W {
        if is_single_word(bit_width) {
            AccessDispatch::<W, B, true> { backend: self, _phantom: PhantomData }
                .atomic_swap(index, value, bit_width, mask, order)
        } else {
            AccessDispatch::<W, B, false> { backend: self, _phantom: PhantomData }
                .atomic_swap(index, value, bit_width, mask, order)
        }
    }

    #[inline(always)]
    fn atomic_compare_exchange(
        &self,
        index: usize,
        bit_width: usize,
        mask: W,
        params: CompareExchangeParams<W>,
    ) -> Result<W, W> {
        if is_single_word(bit_width) {
            AccessDispatch::<W, B, true> { backend: self, _phantom: PhantomData }
                .atomic_compare_exchange(index, bit_width, mask, params)
        } else {
            AccessDispatch::<W, B, false> { backend: self, _phantom: PhantomData }
                .atomic_compare_exchange(index, bit_width, mask, params)
        }
    }
}

// --- Implementation for Lock-Free (Single-Word) Strategy ---

impl<W, B> AccessDispatch<'_, W, B, true>
where
    W: Word + IntoAtomic + One + Bounded,
    B: AtomicBackend<W>,
{
    fn atomic_load(&self, index: usize, bit_width: usize, mask: W, order: Ordering) -> W {
        let bit_pos = index * bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;
        // The data is stored in native endian, so we can read it directly.
        // SAFETY: Backend is guaranteed to be large enough by constructor.
        // We use the atomic equivalent of the word for the load.
        let word = unsafe {
            (*self.backend.seqlocks().get_unchecked(word_index).data.get()).to_atomic()
        }
        .load(order);
        (word >> bit_offset) & mask
    }

    fn atomic_store(&self, index: usize, value: W, bit_width: usize, _mask: W, order: Ordering) {
        let bit_pos = index * bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        let store_mask = if bit_width == <W as Word>::BITS {
            <W as Bounded>::max_value()
        } else {
            ((<W as One>::one() << bit_width) - <W as One>::one()) << bit_offset
        };

        let store_value = value << bit_offset;

        // SAFETY: Backend is guaranteed to be large enough by constructor.
        let word_ref = unsafe {
            &(*self.backend.seqlocks().get_unchecked(word_index).data.get()).to_atomic()
        };

        // Use a CAS loop for the general case.
        let mut old_word = word_ref.load(Ordering::Relaxed);
        loop {
            let mut new_word = old_word;
            new_word &= !store_mask;
            new_word |= store_value;

            match word_ref.compare_exchange_weak(old_word, new_word, order, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => old_word = x,
            }
        }
    }

    fn atomic_swap(&self, index: usize, value: W, bit_width: usize, mask: W, order: Ordering) -> W {
        let bit_pos = index * bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        let store_mask = if bit_width == <W as Word>::BITS {
            <W as Bounded>::max_value()
        } else {
            ((<W as One>::one() << bit_width) - <W as One>::one()) << bit_offset
        };
        let store_value = value << bit_offset;

        // SAFETY: Backend is guaranteed to be large enough by constructor.
        let word_ref = unsafe {
            &(*self.backend.seqlocks().get_unchecked(word_index).data.get()).to_atomic()
        };

        let mut old_word = word_ref.load(Ordering::Relaxed);
        loop {
            let mut new_word = old_word;
            new_word &= !store_mask;
            new_word |= store_value;

            match word_ref.compare_exchange_weak(old_word, new_word, order, Ordering::Relaxed) {
                Ok(_) => return (old_word >> bit_offset) & mask,
                Err(x) => old_word = x,
            }
        }
    }

    fn atomic_compare_exchange(
        &self,
        index: usize,
        bit_width: usize,
        mask: W,
        params: CompareExchangeParams<W>,
    ) -> Result<W, W> {
        let bit_pos = index * bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        let store_mask = if bit_width == <W as Word>::BITS {
            <W as Bounded>::max_value()
        } else {
            ((<W as One>::one() << bit_width) - <W as One>::one()) << bit_offset
        };

        // SAFETY: Backend is guaranteed to be large enough by constructor.
        let word_ref = unsafe {
            &(*self.backend.seqlocks().get_unchecked(word_index).data.get()).to_atomic()
        };

        let mut old_word = word_ref.load(params.failure);
        loop {
            let old_val = (old_word >> bit_offset) & mask;
            if old_val != params.current {
                return Err(old_val);
            }

            let mut new_word = old_word;
            new_word &= !store_mask;
            new_word |= params.new << bit_offset;

            match word_ref.compare_exchange_weak(old_word, new_word, params.success, params.failure)
            {
                Ok(_) => return Ok(params.current),
                Err(previous_word) => old_word = previous_word,
            }
        }
    }
}

// --- Implementation for Hybrid SeqLock/Mutex (Multi-Word) Strategy ---

impl<'a, W, B> AccessDispatch<'a, W, B, false>
where
    W: Word + IntoAtomic + One,
    B: AtomicBackend<W>,
{
    fn atomic_load(&self, index: usize, bit_width: usize, mask: W, _order: Ordering) -> W {
        let bit_pos = index * bit_width;
        let word_idx1 = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        let seqlocks = self.backend.seqlocks();

        if bit_offset + bit_width <= <W as Word>::BITS {
            // Fast path: value is contained in a single word.
            // We still use the seqlock protocol for a consistent read.
            // SAFETY: Constructor ensures seqlocks has enough capacity.
            let seqlock1 = unsafe { seqlocks.get_unchecked(word_idx1) };
            let val = seqlock1.optimistic_read();
            (val >> bit_offset) & mask
        } else {
            // Slow path: value spans two words. We need to read both consistently.
            let word_idx2 = word_idx1 + 1;
            // SAFETY: Constructor ensures seqlocks has enough capacity.
            let seqlock1 = unsafe { seqlocks.get_unchecked(word_idx1) };
            let seqlock2 = unsafe { seqlocks.get_unchecked(word_idx2) };

            loop {
                let ver1_start = seqlock1.version.load(Ordering::Acquire);
                let ver2_start = seqlock2.version.load(Ordering::Acquire);

                // Ensure we are not in the middle of a write on either word.
                if ver1_start % 2 != 0 || ver2_start % 2 != 0 {
                    hint::spin_loop();
                    continue;
                }

                // SAFETY: Version check ensures no writer is active on these words.
                let val1 = unsafe { *seqlock1.data.get() };
                let val2 = unsafe { *seqlock2.data.get() };

                atomic::fence(Ordering::Acquire);

                let ver1_end = seqlock1.version.load(Ordering::Relaxed);
                let ver2_end = seqlock2.version.load(Ordering::Relaxed);

                if ver1_start == ver1_end && ver2_start == ver2_end {
                    // Consistent read, combine and return the value.
                    return ((val1 >> bit_offset)
                        | (val2 << (<W as Word>::BITS - bit_offset)))
                        & mask;
                }
                // A write occurred, retry.
            }
        }
    }

    /// Helper to acquire one or two locks in a globally consistent order to prevent deadlock.
    fn with_two_locks<F, R>(&self, idx1: usize, idx2: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let locks = self.backend.writer_locks();
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
    unsafe fn versioned_write(seqlock: &SeqLockWord<W>, new_val: W) {
        seqlock.version.fetch_add(1, Ordering::Release);
        *seqlock.data.get() = new_val;
        seqlock.version.fetch_add(1, Ordering::Release);
    }

    fn atomic_store(&self, index: usize, value: W, bit_width: usize, mask: W, _order: Ordering) {
        let bit_pos = index * bit_width;
        let word_idx1 = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        let seqlocks = self.backend.seqlocks();

        if bit_offset + bit_width <= <W as Word>::BITS {
            let _guard = self.backend.writer_locks()[word_idx1 % NUM_STRIPES].lock();
            // SAFETY: We hold the lock, so we can access the data.
            unsafe {
                let seqlock = seqlocks.get_unchecked(word_idx1);
                let mut current_word = *seqlock.data.get();
                current_word &= !(mask << bit_offset);
                current_word |= value << bit_offset;
                Self::versioned_write(seqlock, current_word);
            }
        } else {
            let word_idx2 = word_idx1 + 1;
            self.with_two_locks(word_idx1, word_idx2, || {
                // SAFETY: We hold the locks for both words.
                unsafe {
                    let seqlock1 = seqlocks.get_unchecked(word_idx1);
                    let seqlock2 = seqlocks.get_unchecked(word_idx2);

                    let mut word1 = *seqlock1.data.get();
                    word1 &= (<W as One>::one() << bit_offset) - <W as One>::one();
                    word1 |= value << bit_offset;
                    Self::versioned_write(seqlock1, word1);

                    let mut word2 = *seqlock2.data.get();
                    word2 &= !(mask >> (<W as Word>::BITS - bit_offset));
                    word2 |= value >> (<W as Word>::BITS - bit_offset);
                    Self::versioned_write(seqlock2, word2);
                }
            });
        }
    }

    fn atomic_swap(&self, index: usize, value: W, bit_width: usize, mask: W, _order: Ordering) -> W {
        let bit_pos = index * bit_width;
        let word_idx1 = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        let seqlocks = self.backend.seqlocks();

        if bit_offset + bit_width <= <W as Word>::BITS {
            let _guard = self.backend.writer_locks()[word_idx1 % NUM_STRIPES].lock();
            // SAFETY: We hold the lock.
            unsafe {
                let seqlock = seqlocks.get_unchecked(word_idx1);
                let old_word = *seqlock.data.get();
                let old_val = (old_word >> bit_offset) & mask;

                let mut new_word = old_word;
                new_word &= !(mask << bit_offset);
                new_word |= value << bit_offset;

                Self::versioned_write(seqlock, new_word);
                old_val
            }
        } else {
            let word_idx2 = word_idx1 + 1;
            self.with_two_locks(word_idx1, word_idx2, || {
                // SAFETY: We hold the locks for both words.
                unsafe {
                    let seqlock1 = seqlocks.get_unchecked(word_idx1);
                    let seqlock2 = seqlocks.get_unchecked(word_idx2);

                    let word1 = *seqlock1.data.get();
                    let word2 = *seqlock2.data.get();
                    let old_val = ((word1 >> bit_offset)
                        | (word2 << (<W as Word>::BITS - bit_offset)))
                        & mask;

                    let mut new_word1 = word1;
                    new_word1 &= (<W as One>::one() << bit_offset) - <W as One>::one();
                    new_word1 |= value << bit_offset;
                    Self::versioned_write(seqlock1, new_word1);

                    let mut new_word2 = word2;
                    new_word2 &= !(mask >> (<W as Word>::BITS - bit_offset));
                    new_word2 |= value >> (<W as Word>::BITS - bit_offset);
                    Self::versioned_write(seqlock2, new_word2);

                    old_val
                }
            })
        }
    }

    fn atomic_compare_exchange(
        &self,
        index: usize,
        bit_width: usize,
        mask: W,
        params: CompareExchangeParams<W>,
    ) -> Result<W, W> {
        let bit_pos = index * bit_width;
        let word_idx1 = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        let seqlocks = self.backend.seqlocks();

        if bit_offset + bit_width <= <W as Word>::BITS {
            let _guard = self.backend.writer_locks()[word_idx1 % NUM_STRIPES].lock();
            // SAFETY: We hold the lock.
            unsafe {
                let seqlock = seqlocks.get_unchecked(word_idx1);
                let fetched_word = *seqlock.data.get();
                let fetched_val = (fetched_word >> bit_offset) & mask;

                if fetched_val != params.current {
                    return Err(fetched_val);
                }

                let mut new_word = fetched_word;
                new_word &= !(mask << bit_offset);
                new_word |= params.new << bit_offset;
                Self::versioned_write(seqlock, new_word);

                Ok(params.current)
            }
        } else {
            let word_idx2 = word_idx1 + 1;
            self.with_two_locks(word_idx1, word_idx2, || {
                // SAFETY: We hold the locks.
                unsafe {
                    let seqlock1 = seqlocks.get_unchecked(word_idx1);
                    let seqlock2 = seqlocks.get_unchecked(word_idx2);

                    let word1 = *seqlock1.data.get();
                    let word2 = *seqlock2.data.get();
                    let fetched_val = ((word1 >> bit_offset)
                        | (word2 << (<W as Word>::BITS - bit_offset)))
                        & mask;

                    if fetched_val != params.current {
                        return Err(fetched_val);
                    }

                    let mut new_word1 = word1;
                    new_word1 &= (<W as One>::one() << bit_offset) - <W as One>::one();
                    new_word1 |= params.new << bit_offset;
                    Self::versioned_write(seqlock1, new_word1);

                    let mut new_word2 = word2;
                    new_word2 &= !(mask >> (<W as Word>::BITS - bit_offset));
                    new_word2 |= params.new >> (<W as Word>::BITS - bit_offset);
                    Self::versioned_write(seqlock2, new_word2);

                    Ok(params.current)
                }
            })
        }
    }
}