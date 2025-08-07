//! # `AtomicFixedVec`: A Thread-Safe Fixed-Width Integer Vector
//!
//! This module provides `AtomicFixedVec`, a variant of `FixedVec` designed for
//! safe, concurrent access from multiple threads. It guarantees atomicity for
//! all operations, including those that span word boundaries, by transparently
//! selecting the optimal synchronization strategy at compile time.
//!
//! ## Strategies
//!
//! - **Lock-Free (Single-Word)**: For configurations where elements are guaranteed
//!   to be contained within a single atomic word (i.e., when `bit_width` is a
//!   power of two), all operations are implemented using highly-efficient,
//!   lock-free compare-and-swap (CAS) loops.
//!
//! - **Striped Locking (Multi-Word)**: For configurations where elements may
//!   span word boundaries, atomicity is guaranteed using a fine-grained striped
//!   locking mechanism. This avoids torn reads/writes and provides excellent
//!   scalability by minimizing lock contention.
//!
//! The choice of strategy is a compile-time decision, ensuring zero runtime
//! overhead for the dispatch mechanism.

// Declare submodules.
mod access;
mod backend;
mod strategy;

use super::traits::{Storable, Word};
use crate::fixed::atomic::access::private::SealedAtomicAccess;
use crate::fixed::atomic::access::CompareExchangeParams;
use crate::fixed::atomic::backend::{AtomicBackend, OwnedAtomicBackend};
use crate::fixed::Error;
use common_traits::IntoAtomic;
use dsi_bitstream::prelude::Endianness;
use num_traits::{Bounded, One, Zero};
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use crate::fixed::fmt::Debug;

/// A thread-safe, compressed, randomly accessible vector of integers with
/// fixed-width encoding.
///
/// `AtomicFixedVec` provides an API similar to `std::sync::atomic` types, with
/// methods like `load`, `store`, `swap`, and `compare_exchange`. It ensures
/// that all operations are atomic, even for elements that span across the
/// boundaries of the underlying storage words.
#[derive(Debug)]
pub struct AtomicFixedVec<T, W, E, B = OwnedAtomicBackend<W>>
where
    T: Storable<W>,
    W: Word + IntoAtomic,
    W::AtomicType: Debug,
    E: Endianness,
    B: AtomicBackend<W>,
{
    /// The underlying atomic storage backend, containing data and locks.
    backend: B,
    /// The number of bits used to encode each element.
    bit_width: usize,
    /// A mask with the lowest `bit_width` bits set to one.
    mask: W,
    /// The number of elements in the vector.
    len: usize,
    /// Zero-sized markers for the generic type parameters.
    _phantom: PhantomData<(T, W, E)>,
}

// Public API implementation
impl<T, W, E, B> AtomicFixedVec<T, W, E, B>
where
    T: Storable<W>,
    W: Word + IntoAtomic + Zero + One,
    W::AtomicType: Debug,
    E: Endianness,
    B: AtomicBackend<W> + SealedAtomicAccess<W>,
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
        let loaded_word = self
            .backend
            .atomic_load(index, self.bit_width, self.mask, order);
        <T as Storable<W>>::from_word(loaded_word)
    }

    /// Atomically stores `value` at `index`.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds or if `value` does not fit in the
    /// configured `bit_width`.
    #[inline]
    pub fn store(&self, index: usize, value: T, order: Ordering) {
        assert!(index < self.len, "store index out of bounds");
        let value_w = <T as Storable<W>>::into_word(value);
        if self.bit_width < <W as Word>::BITS && (value_w & !self.mask) != <W as Zero>::zero() {
            panic!(
                "Value {:?} does not fit in the configured bit_width of {}",
                value_w, self.bit_width
            );
        }
        self.backend
            .atomic_store(index, value_w, self.bit_width, self.mask, order);
    }

    /// Atomically swaps the value at `index` with `value`, returning the previous value.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds or if `value` does not fit in the
    /// configured `bit_width`.
    #[inline]
    pub fn swap(&self, index: usize, value: T, order: Ordering) -> T {
        assert!(index < self.len, "swap index out of bounds");
        let value_w = <T as Storable<W>>::into_word(value);
        if self.bit_width < <W as Word>::BITS && (value_w & !self.mask) != <W as Zero>::zero() {
            panic!(
                "Value {:?} does not fit in the configured bit_width of {}",
                value_w, self.bit_width
            );
        }
        let old_word = self
            .backend
            .atomic_swap(index, value_w, self.bit_width, self.mask, order);
        <T as Storable<W>>::from_word(old_word)
    }

    /// Atomically compares the value at `index` with `current`. If they are
    /// equal, it is replaced with `new`.
    ///
    /// See `std::sync::atomic::AtomicU64::compare_exchange` for details on memory ordering.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds or if `new` does not fit in the
    /// configured `bit_width`.
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
        if self.bit_width < <W as Word>::BITS && (new_w & !self.mask) != <W as Zero>::zero() {
            panic!(
                "New value {:?} does not fit in the configured bit_width of {}",
                new_w, self.bit_width
            );
        }

        match self.backend.atomic_compare_exchange(
            index,
            self.bit_width,
            self.mask,
            CompareExchangeParams {
                current: current_w,
                new: new_w,
                success,
                failure,
            },
        ) {
            Ok(w) => Ok(<T as Storable<W>>::from_word(w)),
            Err(w) => Err(<T as Storable<W>>::from_word(w)),
        }
    }
}

// Constructor implementation for the owned backend.
impl<T, W, E> AtomicFixedVec<T, W, E, OwnedAtomicBackend<W>>
where
    T: Storable<W>,
    W: Word + IntoAtomic + Zero + One + Bounded,
    W::AtomicType: Debug,
    E: Endianness,
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
            (<W as One>::one() << bit_width) - <W as One>::one()
        };

        let backend = OwnedAtomicBackend::new(len, bit_width);

        Ok(Self {
            backend,
            bit_width,
            mask,
            len,
            _phantom: PhantomData,
        })
    }
}