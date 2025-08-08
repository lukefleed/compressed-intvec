//! # `AtomicFixedVec`: A Thread-Safe Fixed-Width Integer Vector
//!
//! This module provides `AtomicFixedVec`, a variant of `FixedVec` designed for
//! safe, highly-performant concurrent access from multiple threads. It guarantees
//! atomicity for all operations by transparently selecting the optimal strategy.
//!
//! ## Strategies
//!
//! - **Single-Word Operations**: For elements guaranteed to be contained within
//!   a single atomic word, operations are fully lock-free using standard
//!   compare-and-swap (CAS) loops.
//!
//! - **Multi-Word Operations**: For elements that may span word boundaries,
//!   this implementation uses `atomic::Atomic<u128>` from the `atomic` crate.
//!   This provides a portable and correct lock-free implementation on modern
//!   64-bit platforms, and falls back to a lock-based implementation on other
//!   targets, offering the best available performance without sacrificing correctness.

// Declare submodules.
mod backend;

use crate::fixed::atomic::backend::OwnedAtomicBackend;
use crate::fixed::traits::{Storable, Word};
use crate::fixed::Error;
use atomic::Atomic;
use common_traits::{Atomic as CommonAtomic, IntoAtomic};
use num_traits::{Bounded, FromPrimitive, One, Zero};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;

// Type alias for clarity, pointing to the external crate's type.
type AtomicU128 = Atomic<u128>;

/// A thread-safe, compressed, randomly accessible vector of integers with
/// fixed-width encoding.
#[derive(Debug)]
pub struct AtomicFixedVec<T, W>
where
    T: Storable<W>,
    W: Word + IntoAtomic,
    W::AtomicType: Debug,
{
    backend: OwnedAtomicBackend<W>,
    bit_width: usize,
    mask: W,
    len: usize,
    _phantom: PhantomData<T>,
}

// Public API implementation
impl<T, W> AtomicFixedVec<T, W>
where
    T: Storable<W>,
    W: Word + IntoAtomic + Zero + One + Bounded + FromPrimitive,
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
    #[inline]
    pub fn load(&self, index: usize, order: Ordering) -> T {
        assert!(index < self.len, "load index out of bounds");
        let loaded_word = self.atomic_load(index, order);
        <T as Storable<W>>::from_word(loaded_word)
    }

    /// Atomically stores `value` at `index`.
    #[inline]
    pub fn store(&self, index: usize, value: T, order: Ordering) {
        assert!(index < self.len, "store index out of bounds");
        let value_w = <T as Storable<W>>::into_word(value);
        self.atomic_store(index, value_w, order);
    }

    /// Atomically swaps the value at `index` with `value`.
    #[inline]
    pub fn swap(&self, index: usize, value: T, order: Ordering) -> T {
        assert!(index < self.len, "swap index out of bounds");
        let value_w = <T as Storable<W>>::into_word(value);
        let old_word = self.atomic_swap(index, value_w, order);
        <T as Storable<W>>::from_word(old_word)
    }

    /// Atomically compares the value at `index` with `current` and replaces it with `new`.
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

// Constructor
impl<T, W> AtomicFixedVec<T, W>
where
    T: Storable<W>,
    W: Word + IntoAtomic + Zero + One + Bounded,
    W::AtomicType: Debug,
{
    /// Creates a new, zero-initialized `AtomicFixedVec`.
    pub fn new(bit_width: usize, len: usize) -> Result<Self, Error> {
        if bit_width > <W as Word>::BITS {
            return Err(Error::InvalidParameters(format!(
                "bit_width ({}) cannot be greater than the word size ({})",
                bit_width,
                <W as Word>::BITS
            )));
        }

        let is_power_of_two = bit_width.is_power_of_two();
        let requires_spanning_logic =
            bit_width > 0 && (!is_power_of_two || (<W as Word>::BITS % bit_width != 0));

        if requires_spanning_logic && <W as Word>::BITS != 64 {
            return Err(Error::InvalidParameters(
                "Atomic operations on values that span word boundaries are only supported for 64-bit words (u64).".to_string()
            ));
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
impl<T, W> AtomicFixedVec<T, W>
where
    T: Storable<W>,
    W: Word + IntoAtomic + Bounded + One + FromPrimitive,
    W::AtomicType: Debug,
{
    fn atomic_load(&self, index: usize, order: Ordering) -> W {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        let atomic_limbs = &self.backend.storage;

        if bit_offset + self.bit_width <= <W as Word>::BITS {
            let word = atomic_limbs[word_index].0.load(order);
            (word >> bit_offset) & self.mask
        } else {
            let ptr_128 =
                unsafe { atomic_limbs.as_ptr().add(word_index) as *const AtomicU128 };
            let word_128 = unsafe { (*ptr_128).load(order) };
            W::from_u128(word_128 >> bit_offset).unwrap() & self.mask
        }
    }

    fn atomic_store(&self, index: usize, value: W, order: Ordering) {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        let atomic_limbs = &self.backend.storage;

        if bit_offset + self.bit_width <= <W as Word>::BITS {
            let atomic_word_ref = &atomic_limbs[word_index].0;
            let store_mask = self.mask << bit_offset;
            let store_value = (value & self.mask) << bit_offset;
            atomic_word_ref
                .fetch_update(order, order, |old_word| {
                    Some((old_word & !store_mask) | store_value)
                })
                .unwrap();
        } else {
            let ptr_128 =
                unsafe { atomic_limbs.as_ptr().add(word_index) as *const AtomicU128 };
            let store_mask_128 = self.mask.to_u128().unwrap() << bit_offset;
            let store_value_128 = (value & self.mask).to_u128().unwrap() << bit_offset;

            unsafe {
                (*ptr_128)
                    .fetch_update(order, order, |old_val_128| {
                        Some((old_val_128 & !store_mask_128) | store_value_128)
                    })
                    .unwrap()
            };
        }
    }

    fn atomic_swap(&self, index: usize, value: W, order: Ordering) -> W {
        let bit_pos = index * self.bit_width;
        let word_index = bit_pos / <W as Word>::BITS;
        let bit_offset = bit_pos % <W as Word>::BITS;

        let atomic_limbs = &self.backend.storage;

        if bit_offset + self.bit_width <= <W as Word>::BITS {
            let atomic_word_ref = &atomic_limbs[word_index].0;
            let store_mask = self.mask << bit_offset;
            let store_value = (value & self.mask) << bit_offset;
            let mut old_word = atomic_word_ref.load(Ordering::Relaxed);
            loop {
                let new_word = (old_word & !store_mask) | store_value;
                match atomic_word_ref
                    .compare_exchange_weak(old_word, new_word, order, Ordering::Relaxed)
                {
                    Ok(_) => return (old_word >> bit_offset) & self.mask,
                    Err(x) => old_word = x,
                }
            }
        } else {
            let ptr_128 =
                unsafe { atomic_limbs.as_ptr().add(word_index) as *const AtomicU128 };
            let store_mask_128: u128 = self.mask.to_u128().unwrap() << bit_offset;
            let store_value_128: u128 = (value & self.mask).to_u128().unwrap() << bit_offset;
            let mut old_val_128 = unsafe { (*ptr_128).load(Ordering::Relaxed) };
            loop {
                let new_val_128 = (old_val_128 & !store_mask_128) | store_value_128;
                match unsafe {
                    (*ptr_128).compare_exchange_weak(
                        old_val_128,
                        new_val_128,
                        order,
                        Ordering::Relaxed,
                    )
                } {
                    Ok(returned_old) => {
                        return W::from_u128(returned_old >> bit_offset).unwrap() & self.mask
                    }
                    Err(x) => old_val_128 = x,
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

        let atomic_limbs = &self.backend.storage;

        if bit_offset + self.bit_width <= <W as Word>::BITS {
            let atomic_word_ref = &atomic_limbs[word_index].0;
            let store_mask = self.mask << bit_offset;
            let mut old_word = atomic_word_ref.load(failure);
            loop {
                let old_val = (old_word >> bit_offset) & self.mask;
                if old_val != current {
                    return Err(old_val);
                }
                let new_word = (old_word & !store_mask) | ((new & self.mask) << bit_offset);
                match atomic_word_ref.compare_exchange_weak(old_word, new_word, success, failure) {
                    Ok(_) => return Ok(current),
                    Err(x) => old_word = x,
                }
            }
        } else {
            let ptr_128 =
                unsafe { atomic_limbs.as_ptr().add(word_index) as *const AtomicU128 };
            let store_mask_128: u128 = self.mask.to_u128().unwrap() << bit_offset;
            let new_value_128: u128 = (new & self.mask).to_u128().unwrap() << bit_offset;
            let mut old_val_128 = unsafe { (*ptr_128).load(failure) };
            loop {
                let old_val = W::from_u128(old_val_128 >> bit_offset).unwrap() & self.mask;
                if old_val != current {
                    return Err(old_val);
                }
                let new_val_128 = (old_val_128 & !store_mask_128) | new_value_128;
                match unsafe {
                    (*ptr_128).compare_exchange_weak(old_val_128, new_val_128, success, failure)
                } {
                    Ok(_) => return Ok(current),
                    Err(x) => old_val_128 = x,
                }
            }
        }
    }
}