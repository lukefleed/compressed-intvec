//! # Implementations of Atomic Access Strategies
//!
//! This module provides the concrete implementations for the `SealedAtomicAccess`
//! trait, defining the logic for both the lock-free (single-word) and the
//! striped-locking (multi-word) strategies.
//!
//! A key feature of this module is the compile-time dispatch mechanism. We use
//! a "const generic" approach on a helper struct, `AccessDispatch`, to allow
//! the Rust compiler to select the correct implementation based on whether an
//! element can span multiple words. This avoids runtime checks and ensures that
//! the most performant strategy is chosen automatically.
//!
//! - `AccessDispatch<..., true>`: Implements the lock-free strategy.
//! - `AccessDispatch<..., false>`: Implements the striped-locking strategy.

#![cfg(feature = "atomic")]

use super::access::private::SealedAtomicAccess;
use super::access::CompareExchangeParams;
use super::backend::{AtomicBackend, NUM_STRIPES};
use crate::fixed::traits::Word;
use common_traits::{Atomic, IntoAtomic};
use num_traits::One;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;

/// A helper struct to dispatch to the correct atomic strategy implementation
/// using const generics.
///
/// - `IS_SINGLE_WORD`: A boolean constant. If `true`, the lock-free implementation
///   is used. If `false`, the striped-locking implementation is used.
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
/// The constant is calculated from the vector's `bit_width`.
impl<W, B> SealedAtomicAccess<W> for B
where
    W: Word + IntoAtomic + One,
    B: AtomicBackend<W>,
{
    #[inline(always)]
    fn atomic_load(&self, index: usize, bit_width: usize, mask: W, order: Ordering) -> W {
        if is_single_word(bit_width) {
            let dispatch = AccessDispatch::<W, B, true> { backend: self, _phantom: PhantomData };
            dispatch.atomic_load(index, bit_width, mask, order)
        } else {
            let dispatch = AccessDispatch::<W, B, false> { backend: self, _phantom: PhantomData };
            dispatch.atomic_load(index, bit_width, mask, order)
        }
    }

    #[inline(always)]
    fn atomic_store(&self, index: usize, value: W, bit_width: usize, mask: W, order: Ordering) {
        if is_single_word(bit_width) {
            let dispatch = AccessDispatch::<W, B, true> { backend: self, _phantom: PhantomData };
            dispatch.atomic_store(index, value, bit_width, mask, order)
        } else {
            let dispatch = AccessDispatch::<W, B, false> { backend: self, _phantom: PhantomData };
            dispatch.atomic_store(index, value, bit_width, mask, order)
        }
    }

    #[inline(always)]
    fn atomic_swap(&self, index: usize, value: W, bit_width: usize, mask: W, order: Ordering) -> W {
        if is_single_word(bit_width) {
            let dispatch = AccessDispatch::<W, B, true> { backend: self, _phantom: PhantomData };
            dispatch.atomic_swap(index, value, bit_width, mask, order)
        } else {
            let dispatch = AccessDispatch::<W, B, false> { backend: self, _phantom: PhantomData };
            dispatch.atomic_swap(index, value, bit_width, mask, order)
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
            let dispatch = AccessDispatch::<W, B, true> { backend: self, _phantom: PhantomData };
            dispatch.atomic_compare_exchange(index, bit_width, mask, params)
        } else {
            let dispatch = AccessDispatch::<W, B, false> { backend: self, _phantom: PhantomData };
            dispatch.atomic_compare_exchange(index, bit_width, mask, params)
        }
    }
}

// --- Implementation for Lock-Free (Single-Word) Strategy ---

impl<W, B> AccessDispatch<'_, W, B, true>
where
    W: Word + IntoAtomic + One,
    B: AtomicBackend<W>,
{
    fn atomic_load(&self, index: usize, bit_width: usize, mask: W, order: Ordering) -> W {
        let bit_pos = index * bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;
        let word = self.backend.as_atomic_slice()[word_index].load(order);
        (word >> bit_offset) & mask
    }

    fn atomic_store(&self, index: usize, value: W, bit_width: usize, _mask: W, order: Ordering) {
        let bit_pos = index * bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;
        let store_mask = (<W as One>::one() << bit_width) - <W as One>::one();
        let store_value = value << bit_offset;

        let word_ref = &self.backend.as_atomic_slice()[word_index];
        // fetch_update is often compiled more efficiently than a manual CAS loop for simple stores.
        word_ref
            .fetch_update(order, order, |mut current_word| {
                current_word &= !(store_mask << bit_offset);
                current_word |= store_value;
                Some(current_word)
            })
            .unwrap();
    }

    fn atomic_swap(&self, index: usize, value: W, bit_width: usize, mask: W, order: Ordering) -> W {
        let bit_pos = index * bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;
        let store_mask = (<W as One>::one() << bit_width) - <W as One>::one();
        let store_value = value << bit_offset;

        let word_ref = &self.backend.as_atomic_slice()[word_index];
        // fetch_update on success returns Err(previous_value), which is what we need.
        let old_word =
            word_ref
                .fetch_update(order, order, |mut current_word| {
                    current_word &= !(store_mask << bit_offset);
                    current_word |= store_value;
                    Some(current_word)
                })
                .unwrap_err();

        (old_word >> bit_offset) & mask
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
        let store_mask = ((<W as One>::one() << bit_width) - <W as One>::one()) << bit_offset;

        let word_ref = &self.backend.as_atomic_slice()[word_index];

        // Loop until the CAS succeeds or fails definitively.
        let mut old_word = word_ref.load(params.failure);
        loop {
            let old_val = (old_word >> bit_offset) & mask;

            if old_val != params.current {
                return Err(old_val);
            }

            let mut new_word = old_word;
            new_word &= !store_mask;
            new_word |= params.new << bit_offset;

            match word_ref.compare_exchange_weak(old_word, new_word, params.success, params.failure) {
                Ok(_) => return Ok(params.current),
                Err(previous_word) => old_word = previous_word,
            }
        }
    }
}

// --- Implementation for Striped-Locking (Multi-Word) Strategy ---

impl<W, B> AccessDispatch<'_, W, B, false>
where
    W: Word + IntoAtomic + One,
    B: AtomicBackend<W>,
{
    fn atomic_load(&self, index: usize, bit_width: usize, mask: W, _order: Ordering) -> W {
        let bit_pos = index * bit_width;
        let word_idx1 = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        let lock_idx1 = word_idx1 % NUM_STRIPES;
        let _guard1 = self.backend.locks()[lock_idx1].lock();

        let limbs = self.backend.as_atomic_slice();
        let val_part1 = limbs[word_idx1].load(Ordering::Relaxed);

        if bit_offset + bit_width <= <W as Word>::BITS {
            (val_part1 >> bit_offset) & mask
        } else {
            let word_idx2 = word_idx1 + 1;
            let lock_idx2 = word_idx2 % NUM_STRIPES;

            if lock_idx1 == lock_idx2 {
                // Both words are protected by the same lock.
                let val_part2 = limbs[word_idx2].load(Ordering::Relaxed);
                ((val_part1 >> bit_offset) | (val_part2 << (<W as Word>::BITS - bit_offset))) & mask
            } else {
                // Different locks; acquire them in a globally consistent order to prevent deadlock.
                drop(_guard1);

                let (first_lock_idx, second_lock_idx) =
                    if lock_idx1 < lock_idx2 { (lock_idx1, lock_idx2) } else { (lock_idx2, lock_idx1) };

                let _guard_a = self.backend.locks()[first_lock_idx].lock();
                let _guard_b = self.backend.locks()[second_lock_idx].lock();

                let val1 = limbs[word_idx1].load(Ordering::Relaxed);
                let val2 = limbs[word_idx2].load(Ordering::Relaxed);
                ((val1 >> bit_offset) | (val2 << (<W as Word>::BITS - bit_offset))) & mask
            }
        }
    }

    fn atomic_store(&self, index: usize, value: W, bit_width: usize, mask: W, _order: Ordering) {
        let bit_pos = index * bit_width;
        let word_idx1 = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;
        let limbs = self.backend.as_atomic_slice();

        if bit_offset + bit_width <= <W as Word>::BITS {
            let lock_idx1 = word_idx1 % NUM_STRIPES;
            let _guard1 = self.backend.locks()[lock_idx1].lock();
            let mut current_word = limbs[word_idx1].load(Ordering::Relaxed);
            current_word &= !(mask << bit_offset);
            current_word |= value << bit_offset;
            limbs[word_idx1].store(current_word, Ordering::Relaxed);
        } else {
            let word_idx2 = word_idx1 + 1;
            let lock_idx1 = word_idx1 % NUM_STRIPES;
            let lock_idx2 = word_idx2 % NUM_STRIPES;

            let (first_lock_idx, second_lock_idx) = if lock_idx1 < lock_idx2 { (lock_idx1, lock_idx2) } else { (lock_idx2, lock_idx1) };
            let _guard_a = self.backend.locks()[first_lock_idx].lock();
            let _guard_b = if first_lock_idx != second_lock_idx { Some(self.backend.locks()[second_lock_idx].lock()) } else { None };

            // Part 1: First word
            let mut word1 = limbs[word_idx1].load(Ordering::Relaxed);
            word1 &= (<W as One>::one() << bit_offset) - <W as One>::one();
            word1 |= value << bit_offset;
            limbs[word_idx1].store(word1, Ordering::Relaxed);

            // Part 2: Second word
            let mut word2 = limbs[word_idx2].load(Ordering::Relaxed);
            word2 &= !(mask >> (<W as Word>::BITS - bit_offset));
            word2 |= value >> (<W as Word>::BITS - bit_offset);
            limbs[word_idx2].store(word2, Ordering::Relaxed);
        }
    }

    fn atomic_swap(&self, index: usize, value: W, bit_width: usize, mask: W, _order: Ordering) -> W {
        let bit_pos = index * bit_width;
        let word_idx1 = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;
        let limbs = self.backend.as_atomic_slice();

        if bit_offset + bit_width <= <W as Word>::BITS {
            let lock_idx1 = word_idx1 % NUM_STRIPES;
            let _guard1 = self.backend.locks()[lock_idx1].lock();

            let old_word = limbs[word_idx1].load(Ordering::Relaxed);
            let old_val = (old_word >> bit_offset) & mask;

            let mut new_word = old_word;
            new_word &= !(mask << bit_offset);
            new_word |= value << bit_offset;

            limbs[word_idx1].store(new_word, Ordering::Relaxed);
            old_val
        } else {
            let word_idx2 = word_idx1 + 1;
            let lock_idx1 = word_idx1 % NUM_STRIPES;
            let lock_idx2 = word_idx2 % NUM_STRIPES;

            let (first_lock_idx, second_lock_idx) = if lock_idx1 < lock_idx2 { (lock_idx1, lock_idx2) } else { (lock_idx2, lock_idx1) };
            let _guard_a = self.backend.locks()[first_lock_idx].lock();
            let _guard_b = if first_lock_idx != second_lock_idx { Some(self.backend.locks()[second_lock_idx].lock()) } else { None };

            let word1 = limbs[word_idx1].load(Ordering::Relaxed);
            let word2 = limbs[word_idx2].load(Ordering::Relaxed);
            let old_val = ((word1 >> bit_offset) | (word2 << (<W as Word>::BITS - bit_offset))) & mask;

            let mut new_word1 = word1;
            new_word1 &= (<W as One>::one() << bit_offset) - <W as One>::one();
            new_word1 |= value << bit_offset;
            limbs[word_idx1].store(new_word1, Ordering::Relaxed);

            let mut new_word2 = word2;
            new_word2 &= !(mask >> (<W as Word>::BITS - bit_offset));
            new_word2 |= value >> (<W as Word>::BITS - bit_offset);
            limbs[word_idx2].store(new_word2, Ordering::Relaxed);

            old_val
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
        let limbs = self.backend.as_atomic_slice();

        if bit_offset + bit_width <= <W as Word>::BITS {
            let lock_idx1 = word_idx1 % NUM_STRIPES;
            let _guard1 = self.backend.locks()[lock_idx1].lock();

            let fetched_word = limbs[word_idx1].load(Ordering::Relaxed);
            let fetched_val = (fetched_word >> bit_offset) & mask;

            if fetched_val != params.current {
                return Err(fetched_val);
            }

            let mut new_word = fetched_word;
            new_word &= !(mask << bit_offset);
            new_word |= params.new << bit_offset;
            limbs[word_idx1].store(new_word, Ordering::Relaxed);

            Ok(params.current)
        } else {
            let word_idx2 = word_idx1 + 1;
            let lock_idx1 = word_idx1 % NUM_STRIPES;
            let lock_idx2 = word_idx2 % NUM_STRIPES;

            let (first_lock_idx, second_lock_idx) = if lock_idx1 < lock_idx2 { (lock_idx1, lock_idx2) } else { (lock_idx2, lock_idx1) };
            let _guard_a = self.backend.locks()[first_lock_idx].lock();
            let _guard_b = if first_lock_idx != second_lock_idx { Some(self.backend.locks()[second_lock_idx].lock()) } else { None };

            let word1 = limbs[word_idx1].load(Ordering::Relaxed);
            let word2 = limbs[word_idx2].load(Ordering::Relaxed);
            let fetched_val = ((word1 >> bit_offset) | (word2 << (<W as Word>::BITS - bit_offset))) & mask;

            if fetched_val != params.current {
                return Err(fetched_val);
            }

            let mut new_word1 = word1;
            new_word1 &= (<W as One>::one() << bit_offset) - <W as One>::one();
            new_word1 |= params.new << bit_offset;
            limbs[word_idx1].store(new_word1, Ordering::Relaxed);

            let mut new_word2 = word2;
            new_word2 &= !(mask >> (<W as Word>::BITS - bit_offset));
            new_word2 |= params.new >> (<W as Word>::BITS - bit_offset);
            limbs[word_idx2].store(new_word2, Ordering::Relaxed);

            Ok(params.current)
        }
    }
}