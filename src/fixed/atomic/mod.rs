//! A thread-safe, compressed vector of integers with fixed-width encoding.
//!
//! This module provides [`AtomicFixedVec`], a data structure that behaves like
//! [`FixedVec`] but allows for concurrent access and
//! modification from multiple threads. It is designed for scenarios where a
//! large collection of integers must be shared and mutated in a parallel
//! environment.
//!
//! All operations that modify the vector's contents are implemented using
//! atomic instructions (e.g., compare-and-swap loops), ensuring thread safety
//! without requiring a global lock.
//!
//! # Atomicity Guarantees and Locking
//!
//! The atomicity of operations depends on the configured `bit_width`.
//!
//! - **Power-of-Two `bit_width`**: When the `bit_width` is a power of two
//!   (e.g., 2, 4, 8, 16, 32), and it evenly divides the 64-bit word size,
//!   most operations can be performed with lock-free atomic instructions.
//!   This is because each element is guaranteed to be fully contained within a
//!   single [`AtomicU64`] word.
//!
//! - **Non-Power-of-Two `bit_width`**: When the `bit_width` is not a power of
//!   two, an element's value may span across the boundary of two [`AtomicU64`]
//!   words. Modifying such an element requires updating two words simultaneously,
//!   which cannot be done in a single atomic hardware instruction.
//!
//! To handle this case, [`AtomicFixedVec`] uses a technique called _lock striping_.
//! It maintains a pool of [`parking_lot::Mutex`] locks. When an operation needs
//! to modify a value that spans two words, it acquires a lock for that specific
//! memory region. This ensures that the two-word update is itself atomic with
//! respect to other threads, while still allowing concurrent operations on
//! different parts of the vector. This approach avoids a single global lock,
//! preserving a high degree of parallelism.
//!
//! > Future version may introduce a more sophisticated locking strategy
//!
//! ### Performance Considerations
//!
//! The trade-off is between memory compactness and performance. While a
//! non-power-of-two `bit_width` provides the most space-efficient storage,
//! it may incur a performance overhead for write operations that span word
//! boundaries due to locking.
//!
//! For write-heavy, performance-critical workloads, choosing a power-of-two
//! `bit_width` (e.g., by using [`BitWidth::PowerOfTwo`]) is recommended to
//! ensure all operations remain lock-free.
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::prelude::*;
//! use compressed_intvec::fixed::{AtomicFixedVec, UAtomicFixedVec};
//! use std::sync::Arc;
//! use std::thread;
//! use std::sync::atomic::Ordering;
//!
//! // Create from a slice using the builder.
//! let initial_data: Vec<u32> = vec![10, 20, 30, 40];
//! let atomic_vec: Arc<UAtomicFixedVec<u32>> = Arc::new(
//!     AtomicFixedVec::builder()
//!         .build(&initial_data)?
//! );
//!
//! // Share the vector across threads.
//! let mut handles = vec![];
//! for i in 0..4 {
//!     let vec_clone = Arc::clone(&atomic_vec);
//!     handles.push(thread::spawn(move || {
//!         // Each thread atomically updates its own slot.
//!         vec_clone.store(i, 63, Ordering::SeqCst);
//!     }));
//! }
//! for handle in handles {
//!     handle.join().unwrap();
//! }
//! assert_eq!(atomic_vec.load(3, Ordering::SeqCst), 63);
//! # Ok(())
//! # }
//! ```
//!
//! ## Storing Signed Integers
//!
//! [`AtomicFixedVec`] can also store signed integers. The underlying [`Storable`]
//! trait uses zig-zag encoding to store signed values efficiently, so that
//! small negative numbers require few bits, just like small positive numbers.
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use compressed_intvec::prelude::*;
//! use compressed_intvec::fixed::{AtomicFixedVec, SAtomicFixedVec};
//! use std::sync::Arc;
//! use std::sync::atomic::Ordering;
//!
//! // The values range from -2 to 1. To also store -3 later, we need 3 bits.
//! let initial_data: Vec<i16> = vec![-2, -1, 0, 1];
//! let atomic_vec: Arc<SAtomicFixedVec<i16>> = Arc::new(
//!     AtomicFixedVec::builder()
//!         .bit_width(BitWidth::Explicit(3)) // Explicitly set bit width
//!         .build(&initial_data)?
//! );
//!
//! assert_eq!(atomic_vec.bit_width(), 3);
//! assert_eq!(atomic_vec.load(0, Ordering::SeqCst), -2);
//!
//! // Atomically update a value.
//! atomic_vec.store(0, -3, Ordering::SeqCst);
//! assert_eq!(atomic_vec.load(0, Ordering::SeqCst), -3);
//! # Ok(())
//! # }
//! ```
//!
//! //! ## Parallel Iteration
//!
//! When the `parallel` feature is enabled (by default is enabled), you can use [`par_iter`](AtomicFixedVec::par_iter) to process
//! the vector's elements concurrently.
//!
//! ```
//! # #[cfg(feature = "parallel")] {
//! use compressed_intvec::prelude::*;
//! use compressed_intvec::fixed::{AtomicFixedVec, UAtomicFixedVec};
//! use rayon::prelude::*;
//!
//! let data: Vec<u32> = (0..10_000).collect();
//! let vec: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
//!     .build(&data)
//!     .unwrap();
//!
//! // Use the parallel iterator to find the sum of all even numbers.
//! let sum_of_evens: u32 = vec.par_iter()
//!     .filter(|&x| x % 2 == 0)
//!     .sum();
//!
//! let expected_sum: u32 = (0..10_000).filter(|&x| x % 2 == 0).sum();
//! assert_eq!(sum_of_evens, expected_sum);
//! # }
//! ```
//!
//! ### Memory Ordering and Locking
//!
//! The memory [`Ordering`] specified in methods like [`load`](AtomicFixedVec::load), [`store`](AtomicFixedVec::store), or
//! [`fetch_add`](AtomicFixedVec::fetch_add) is always respected, but its interaction with the internal locking
//! mechanism is important to understand.
//!
//! -   **Lock-Free Path**: When an element is fully contained within a single
//!     [`u64`] word, the specified [`Ordering`] is applied directly to the underlying
//!     atomic instructions, providing the standard guarantees described in the
//!     Rust documentation.
//!
//! -   **Locked Path**: When an element spans two [`u64`] words, a fine-grained
//!     mutex is acquired. This lock ensures that the two-word operation is
//!     atomic with respect to other locked operations on the same memory region.
//!     The specified [`Ordering`] is then applied to the atomic writes performed
//!     within the locked critical section. This guarantees that the effects of
//!     the operation become visible to other threads according to the chosen
//!     ordering, but the visibility is still mediated by the mutual exclusion
//!     provided by the lock.
//!

#[macro_use]
pub mod macros;
pub mod builder;

use crate::fixed::traits::Storable;
use crate::fixed::{BitWidth, Error, FixedVec};
use mem_dbg::{DbgFlags, MemDbgImpl, MemSize, SizeFlags};
use num_traits::{Bounded, ToPrimitive, WrappingAdd, WrappingSub};
use parking_lot::Mutex;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{BitAnd, BitOr, BitXor, Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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

/// A proxy object for mutable access to an element within an [`AtomicFixedVec`]
/// during parallel iteration.
///
/// This struct is returned by the [`par_iter_mut`](AtomicFixedVec::par_iter_mut)
/// parallel iterator. It holds a temporary copy of an element's value. When the
/// proxy is dropped, its `Drop` implementation atomically writes the (potentially
/// modified) value back into the parent vector.
#[cfg(feature = "parallel")]
pub struct AtomicMutProxy<'a, T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    vec: &'a AtomicFixedVec<T>,
    index: usize,
    value: T,
}

#[cfg(feature = "parallel")]
impl<T> fmt::Debug for AtomicMutProxy<'_, T>
where
    T: Storable<u64> + Copy + ToPrimitive + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AtomicMutProxy")
            .field("value", &self.value)
            .finish()
    }
}

#[cfg(feature = "parallel")]
impl<'a, T> AtomicMutProxy<'a, T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    /// Creates a new `AtomicMutProxy`.
    ///
    /// This is called by `par_iter_mut`. It reads the initial value
    /// from the vector.
    fn new(vec: &'a AtomicFixedVec<T>, index: usize) -> Self {
        let value = vec.load(index, Ordering::Relaxed);
        Self { vec, index, value }
    }

    /// Consumes the proxy, returning the current value without writing it back.
    ///
    /// This can be used to avoid the overhead of a write operation if the value
    /// was read but not modified.
    pub fn into_inner(self) -> T {
        use std::mem;

        let value = self.value;
        mem::forget(self); // Prevent the proxy from writing back
        value
    }
}

#[cfg(feature = "parallel")]
impl<T> Deref for AtomicMutProxy<'_, T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

#[cfg(feature = "parallel")]
impl<T> DerefMut for AtomicMutProxy<'_, T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

#[cfg(feature = "parallel")]
impl<T> Drop for AtomicMutProxy<'_, T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    /// Writes the potentially modified value back to the [`AtomicFixedVec`] when the
    /// proxy goes out of scope.
    fn drop(&mut self) {
        // The value is copied before being passed to store.
        // Relaxed ordering is sufficient here because the synchronization is
        // handled by Rayon's fork-join model. The writes will be visible after
        // the parallel block completes.
        self.vec.store(self.index, self.value, Ordering::Relaxed);
    }
}

/// A thread-safe, compressed, randomly accessible vector of integers with
/// fixed-width encoding, backed by [`u64`] atomic words.
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
    /// Creates a builder for constructing an [`AtomicFixedVec`] from a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::prelude::*;
    /// use compressed_intvec::fixed::{AtomicFixedVec, UAtomicFixedVec, BitWidth};
    ///
    /// let data: &[i16] = &[-100, 0, 100, 200];
    /// let vec: UAtomicFixedVec<i16> = AtomicFixedVec::builder()
    ///     .bit_width(BitWidth::PowerOfTwo) // Force 16 bits for signed values
    ///     .build(data)?;
    ///
    /// assert_eq!(vec.len(), 4);
    /// assert_eq!(vec.bit_width(), 16);
    /// # Ok(())
    /// # }
    /// ```
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
    ///
    /// [`load`](AtomicFixedVec::load) takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. For more information, see the [Rust documentation on
    /// memory ordering](https://doc.rust-lang.org/std/sync/atomic/enum.Ordering.html).
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline(always)]
    pub fn load(&self, index: usize, order: Ordering) -> T {
        assert!(index < self.len, "load index out of bounds");
        let loaded_word = self.atomic_load(index, order);
        T::from_word(loaded_word)
    }

    /// Atomically loads the value at `index` without bounds checking.
    ///
    /// [`load_unchecked`](AtomicFixedVec::load_unchecked) takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. For more information, see the [Rust documentation on
    /// memory ordering](https://doc.rust-lang.org/std/sync/atomic/enum.Ordering.html).
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    #[inline(always)]
    pub unsafe fn load_unchecked(&self, index: usize, order: Ordering) -> T {
        debug_assert!(index < self.len, "load_unchecked index out of bounds");
        let loaded_word = self.atomic_load(index, order);
        T::from_word(loaded_word)
    }

    /// Atomically stores `value` at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds. Note that the stored value is not
    /// checked for whether it fits in the configured `bit_width` and will be
    /// truncated if it is too large.
    #[inline(always)]
    pub fn store(&self, index: usize, value: T, order: Ordering) {
        assert!(index < self.len, "store index out of bounds");
        let value_w = T::into_word(value);
        self.atomic_store(index, value_w, order);
    }

    /// Atomically stores `value` at `index` without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    /// Note that the stored value is not checked for whether it fits in the
    /// configured `bit_width` and will be truncated if it is too large.
    #[inline(always)]
    pub unsafe fn store_unchecked(&self, index: usize, value: T, order: Ordering) {
        debug_assert!(index < self.len, "store_unchecked index out of bounds");
        let value_w = T::into_word(value);
        self.atomic_store(index, value_w, order);
    }

    /// Atomically swaps the value at `index` with `value`, returning the
    /// previous value.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline(always)]
    pub fn swap(&self, index: usize, value: T, order: Ordering) -> T {
        assert!(index < self.len, "swap index out of bounds");
        let value_w = T::into_word(value);
        let old_word = self.atomic_swap(index, value_w, order);
        T::from_word(old_word)
    }

    /// Atomically swaps the value at `index` with `value` without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    #[inline(always)]
    pub unsafe fn swap_unchecked(&self, index: usize, value: T, order: Ordering) -> T {
        debug_assert!(index < self.len, "swap_unchecked index out of bounds");
        let value_w = T::into_word(value);
        let old_word = self.atomic_swap(index, value_w, order);
        T::from_word(old_word)
    }

    /// Atomically compares the value at `index` with `current` and, if they are
    /// equal, replaces it with `new`.
    ///
    /// Returns `Ok` with the previous value on success, or `Err` with the
    /// actual value if the comparison fails. This is also known as a
    /// "compare-and-set" (CAS) operation.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
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

    /// Atomically compares the value at `index` with `current` and, if they are
    /// equal, replaces it with `new`, without bounds checking.
    ///
    /// Returns `Ok` with the previous value on success, or `Err` with the
    /// actual value if the comparison fails. This is also known as a
    /// "compare-and-set" (CAS) operation.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    #[inline(always)]
    pub unsafe fn compare_exchange_unchecked(
        &self,
        index: usize,
        current: T,
        new: T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<T, T> {
        debug_assert!(
            index < self.len,
            "compare_exchange_unchecked index out of bounds"
        );
        let current_w = T::into_word(current);
        let new_w = T::into_word(new);
        match self.atomic_compare_exchange(index, current_w, new_w, success, failure) {
            Ok(w) => Ok(T::from_word(w)),
            Err(w) => Err(T::from_word(w)),
        }
    }

    /// Returns the element at `index`, or `None` if out of bounds.
    ///
    /// This is an ergonomic wrapper around [`load`](AtomicFixedVec::load) that uses [`Ordering::SeqCst`].
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        Some(self.load(index, Ordering::SeqCst))
    }

    /// Returns the element at `index` without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds `index` is undefined behavior.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        unsafe { self.load_unchecked(index, Ordering::SeqCst) }
    }

    /// Returns an iterator over the elements of the vector.
    ///
    /// The iterator atomically loads each element using [`Ordering::SeqCst`].
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        AtomicFixedVecIter {
            vec: self,
            current_index: 0,
        }
    }

    /// Returns a parallel iterator over the elements of the vector.
    ///
    /// The iterator atomically loads each element using [`Ordering::Relaxed`].
    /// This operation is highly parallelizable as each element can be loaded
    /// independently.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::prelude::*;
    /// use compressed_intvec::fixed::{AtomicFixedVec, UAtomicFixedVec, BitWidth};
    /// use rayon::prelude::*;
    /// use std::sync::atomic::Ordering;
    ///
    /// let data: Vec<u32> = (0..1000).collect();
    /// let vec: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
    ///     .build(&data)
    ///     .unwrap();
    ///
    /// // Sum the elements in parallel.
    /// let sum: u32 = vec.par_iter().sum();
    /// assert_eq!(sum, (0..1000).sum());
    /// # }
    /// ```
    #[cfg(feature = "parallel")]
    pub fn par_iter(&self) -> impl ParallelIterator<Item = T> + '_
    where
        T: Send + Sync,
    {
        (0..self.len())
            .into_par_iter()
            .map(move |i| self.load(i, Ordering::Relaxed))
    }

    /// Returns a parallel iterator that allows modifying elements of the vector in place.
    ///
    /// Each element is accessed via an [`AtomicMutProxy`], which ensures that
    /// all modifications are written back atomically.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")] {
    /// use compressed_intvec::prelude::*;
    /// use compressed_intvec::fixed::{AtomicFixedVec, UAtomicFixedVec, BitWidth};
    /// use rayon::prelude::*;
    /// use std::sync::atomic::Ordering;
    ///
    /// let data: Vec<u32> = (0..100).collect();
    /// let vec: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
    ///     .bit_width(BitWidth::Explicit(8)) // 2*99 = 198, needs 8 bits
    ///     .build(&data)
    ///     .unwrap();
    ///
    /// vec.par_iter_mut().for_each(|mut proxy| {
    ///     *proxy *= 2;
    /// });
    ///
    /// assert_eq!(vec.load(50, Ordering::Relaxed), 100);
    /// # }
    /// ```
    #[cfg(feature = "parallel")]
    pub fn par_iter_mut(&self) -> impl ParallelIterator<Item = AtomicMutProxy<'_, T>>
    where
        T: Send + Sync,
    {
        (0..self.len())
            .into_par_iter()
            .map(move |i| AtomicMutProxy::new(self, i))
    }
}

// Extended atomic RMW operations
impl<T> AtomicFixedVec<T>
where
    T: Storable<u64> + Bounded + Copy + ToPrimitive,
{
    /// Atomically adds to the value at `index`, returning the previous value.
    ///
    /// This operation is a "read-modify-write" (RMW) operation. It atomically
    /// reads the value at `index`, adds `val` to it (with wrapping on overflow),
    /// and writes the result back.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::prelude::*;
    /// use std::sync::atomic::Ordering;
    ///
    /// // The initial value is 10. The result will be 15, which needs 4 bits.
    /// let data = vec![10u32, 20];
    /// let vec: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
    ///     .bit_width(BitWidth::Explicit(5))
    ///     .build(&data)?;
    ///
    /// let previous = vec.fetch_add(0, 5, Ordering::SeqCst);
    ///
    /// assert_eq!(previous, 10);
    /// assert_eq!(vec.load(0, Ordering::SeqCst), 15);
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn fetch_add(&self, index: usize, val: T, order: Ordering) -> T
    where
        T: WrappingAdd,
    {
        self.atomic_rmw(index, val, order, |a, b| a.wrapping_add(&b))
    }

    /// Atomically subtracts from the value at `index`, returning the previous value.
    ///
    /// This is an atomic "read-modify-write" (RMW) operation.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::prelude::*;
    /// use std::sync::atomic::Ordering;
    ///
    /// // The initial value is 10. The result will be 5, which fits.
    /// let data = vec![10u32, 20];
    /// let vec: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
    ///     .bit_width(BitWidth::Explicit(5))
    ///     .build(&data)?;
    ///
    /// let previous = vec.fetch_sub(0, 5, Ordering::SeqCst);
    ///
    /// assert_eq!(previous, 10);
    /// assert_eq!(vec.load(0, Ordering::SeqCst), 5);
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn fetch_sub(&self, index: usize, val: T, order: Ordering) -> T
    where
        T: WrappingSub,
    {
        self.atomic_rmw(index, val, order, |a, b| a.wrapping_sub(&b))
    }

    /// Atomically performs a bitwise AND on the value at `index`, returning the previous value.
    ///
    /// This is an atomic "read-modify-write" (RMW) operation.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::prelude::*;
    /// use std::sync::atomic::Ordering;
    ///
    /// // 0b1100 = 12. Needs 4 bits.
    /// let data = vec![12u32];
    /// let vec: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
    ///     .bit_width(BitWidth::Explicit(4))
    ///     .build(&data)?;
    ///
    /// // 0b1010 = 10
    /// let previous = vec.fetch_and(0, 10, Ordering::SeqCst);
    ///
    /// assert_eq!(previous, 12);
    /// // 0b1100 & 0b1010 = 0b1000 = 8
    /// assert_eq!(vec.load(0, Ordering::SeqCst), 8);
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn fetch_and(&self, index: usize, val: T, order: Ordering) -> T
    where
        T: BitAnd<Output = T>,
    {
        self.atomic_rmw(index, val, order, |a, b| a & b)
    }

    /// Atomically performs a bitwise OR on the value at `index`, returning the previous value.
    ///
    /// This is an atomic "read-modify-write" (RMW) operation.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::prelude::*;
    /// use std::sync::atomic::Ordering;
    ///
    /// // 0b1100 = 12. Needs 4 bits.
    /// let data = vec![12u32];
    /// let vec: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
    ///     .bit_width(BitWidth::Explicit(4))
    ///     .build(&data)?;
    ///
    /// // 0b1010 = 10
    /// let previous = vec.fetch_or(0, 10, Ordering::SeqCst);
    ///
    /// assert_eq!(previous, 12);
    /// // 0b1100 | 0b1010 = 0b1110 = 14
    /// assert_eq!(vec.load(0, Ordering::SeqCst), 14);
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn fetch_or(&self, index: usize, val: T, order: Ordering) -> T
    where
        T: BitOr<Output = T>,
    {
        self.atomic_rmw(index, val, order, |a, b| a | b)
    }

    /// Atomically performs a bitwise XOR on the value at `index`, returning the previous value.
    ///
    /// This is an atomic "read-modify-write" (RMW) operation.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::prelude::*;
    /// use std::sync::atomic::Ordering;
    ///
    /// // 0b1100 = 12. Needs 4 bits.
    /// let data = vec![12u32];
    /// let vec: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
    ///     .bit_width(BitWidth::Explicit(4))
    ///     .build(&data)?;
    ///
    /// // 0b1010 = 10
    /// let previous = vec.fetch_xor(0, 10, Ordering::SeqCst);
    ///
    /// assert_eq!(previous, 12);
    /// // 0b1100 ^ 0b1010 = 0b0110 = 6
    /// assert_eq!(vec.load(0, Ordering::SeqCst), 6);
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn fetch_xor(&self, index: usize, val: T, order: Ordering) -> T
    where
        T: BitXor<Output = T>,
    {
        self.atomic_rmw(index, val, order, |a, b| a ^ b)
    }

    /// Atomically computes the maximum of the value at `index` and `val`, returning the previous value.
    ///
    /// This is an atomic "read-modify-write" (RMW) operation.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::prelude::*;
    /// use std::sync::atomic::Ordering;
    ///
    /// // Value 20 needs 6 bits with zig-zag encoding.
    /// let data = vec![10i32];
    /// let vec: SAtomicFixedVec<i32> = AtomicFixedVec::builder()
    ///     .bit_width(BitWidth::Explicit(6))
    ///     .build(&data)?;
    ///
    /// // Attempt to store a larger value
    /// let previous = vec.fetch_max(0, 20, Ordering::SeqCst);
    /// assert_eq!(previous, 10);
    /// assert_eq!(vec.load(0, Ordering::SeqCst), 20);
    ///
    /// // Attempt to store a smaller value
    /// let previous2 = vec.fetch_max(0, 5, Ordering::SeqCst);
    /// assert_eq!(previous2, 20);
    /// assert_eq!(vec.load(0, Ordering::SeqCst), 20); // Value is unchanged
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn fetch_max(&self, index: usize, val: T, order: Ordering) -> T
    where
        T: Ord,
    {
        self.atomic_rmw(index, val, order, |a, b| a.max(b))
    }

    /// Atomically computes the minimum of the value at `index` and `val`, returning the previous value.
    ///
    /// This is an atomic "read-modify-write" (RMW) operation.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::prelude::*;
    /// use std::sync::atomic::Ordering;
    ///
    /// // Value 10 needs 5 bits with zig-zag encoding.
    /// let data = vec![10i32];
    /// let vec: SAtomicFixedVec<i32> = AtomicFixedVec::builder()
    ///     .bit_width(BitWidth::Explicit(5))
    ///     .build(&data)?;
    ///
    /// // Attempt to store a smaller value
    /// let previous = vec.fetch_min(0, 5, Ordering::SeqCst);
    /// assert_eq!(previous, 10);
    /// assert_eq!(vec.load(0, Ordering::SeqCst), 5);
    ///
    /// // Attempt to store a larger value
    /// let previous2 = vec.fetch_min(0, 20, Ordering::SeqCst);
    /// assert_eq!(previous2, 5);
    /// assert_eq!(vec.load(0, Ordering::SeqCst), 5); // Value is unchanged
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn fetch_min(&self, index: usize, val: T, order: Ordering) -> T
    where
        T: Ord,
    {
        self.atomic_rmw(index, val, order, |a, b| a.min(b))
    }

    /// Atomically modifies the value at `index` using a closure.
    ///
    /// Reads the value, applies the function `f`, and attempts to write the
    /// new value back. If the value has been changed by another thread in the
    /// meantime, the function is re-evaluated with the new current value.
    ///
    /// The closure `f` can return `None` to abort the update.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use compressed_intvec::prelude::*;
    /// use std::sync::atomic::Ordering;
    ///
    /// // Value 20 needs 5 bits.
    /// let data = vec![10u32];
    /// let vec: UAtomicFixedVec<u32> = AtomicFixedVec::builder()
    ///     .bit_width(BitWidth::Explicit(5))
    ///     .build(&data)?;
    ///
    /// // Successfully update the value
    /// let result = vec.fetch_update(0, Ordering::SeqCst, Ordering::Relaxed, |val| {
    ///     Some(val * 2)
    /// });
    /// assert_eq!(result, Ok(10));
    /// assert_eq!(vec.load(0, Ordering::SeqCst), 20);
    ///
    /// // Abort the update
    /// let result_aborted = vec.fetch_update(0, Ordering::SeqCst, Ordering::Relaxed, |val| {
    ///     if val > 15 {
    ///         None // Abort if value is > 15
    ///     } else {
    ///         Some(val + 1)
    ///     }
    /// });
    /// assert_eq!(result_aborted, Err(20));
    /// assert_eq!(vec.load(0, Ordering::SeqCst), 20); // Value remains unchanged
    /// # Ok(())
    /// # }
    /// ```
    pub fn fetch_update<F>(
        &self,
        index: usize,
        success: Ordering,
        failure: Ordering,
        mut f: F,
    ) -> Result<T, T>
    where
        F: FnMut(T) -> Option<T>,
    {
        let mut current = self.load(index, Ordering::Relaxed);
        loop {
            match f(current) {
                Some(new) => match self.compare_exchange(index, current, new, success, failure) {
                    Ok(old) => return Ok(old),
                    Err(actual) => current = actual,
                },
                None => return Err(current),
            }
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
    T: Storable<u64> + Copy + ToPrimitive,
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
                    Ok(_) => break,
                    Err(x) => old_word = x,
                }
            }
        } else {
            // Locked path for values spanning two words.
            let lock_index = word_index & (self.locks.len() - 1);
            let _guard = self.locks[lock_index].lock();
            // The lock guarantees exclusive access to this multi-word operation.
            // We still use atomic operations inside to prevent races with the
            // lock-free path, which might be concurrently accessing one of these words.
            let low_word_ref = &self.storage[word_index];
            let high_word_ref = &self.storage[word_index + 1];

            // Modify the lower word.
            low_word_ref
                .fetch_update(order, Ordering::Relaxed, |mut w| {
                    w &= !(u64::MAX << bit_offset);
                    w |= value << bit_offset;
                    Some(w)
                })
                .unwrap(); // Should not fail under lock.

            // Modify the higher word.
            let bits_in_high = (bit_offset + self.bit_width) - u64::BITS as usize;
            let high_mask = (1u64 << bits_in_high).wrapping_sub(1);
            high_word_ref
                .fetch_update(order, Ordering::Relaxed, |mut w| {
                    w &= !high_mask;
                    w |= value >> (u64::BITS as usize - bit_offset);
                    Some(w)
                })
                .unwrap(); // Should not fail under lock.
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
                    Ok(_) => return (old_word >> bit_offset) & self.mask,
                    Err(x) => old_word = x,
                }
            }
        } else {
            // Locked path for spanning values.
            let lock_index = word_index & (self.locks.len() - 1);
            let _guard = self.locks[lock_index].lock();
            let old_val = self.atomic_load(index, Ordering::Relaxed);
            self.atomic_store(index, value, order);
            old_val
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
            let mut old_word = atomic_word_ref.load(failure);
            loop {
                let old_val_extracted = (old_word >> bit_offset) & self.mask;
                if old_val_extracted != current {
                    return Err(old_val_extracted);
                }
                let new_word = (old_word & !store_mask) | new_value_shifted;
                match atomic_word_ref.compare_exchange_weak(old_word, new_word, success, failure) {
                    Ok(_) => return Ok(current),
                    Err(x) => old_word = x,
                }
            }
        } else {
            // Locked path for spanning values.
            let lock_index = word_index & (self.locks.len() - 1);
            let _guard = self.locks[lock_index].lock();
            let old_val = self.atomic_load(index, failure);
            if old_val != current {
                return Err(old_val);
            }
            self.atomic_store(index, new, success);
            Ok(current)
        }
    }

    /// Generic implementation for all Read-Modify-Write (RMW) operations.
    #[inline(always)]
    fn atomic_rmw(&self, index: usize, val: T, order: Ordering, op: impl Fn(T, T) -> T) -> T {
        // This RMW is implemented as a CAS loop on top of `compare_exchange`.
        let mut current = self.load(index, Ordering::Relaxed);
        loop {
            let new = op(current, val);
            match self.compare_exchange(index, current, new, order, Ordering::Relaxed) {
                Ok(old) => return old,
                Err(actual) => current = actual,
            }
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
        // SAFETY: This transmutation is safe because [`AtomicU64`] and [`u64`] have
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
        // SAFETY: This transmutation is safe because [`u64`] and [`AtomicU64`] have
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
        self.bit_width
            ._mem_dbg_rec_on(writer, total_size, max_depth, prefix, false, flags)?;
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

/// An iterator over the elements of a borrowed [`AtomicFixedVec`].
///
/// This struct is created by the [`iter`](AtomicFixedVec::iter) method. It
/// atomically loads each value on the fly.
pub struct AtomicFixedVecIter<'a, T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    vec: &'a AtomicFixedVec<T>,
    current_index: usize,
}

impl<T> Iterator for AtomicFixedVecIter<'_, T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.vec.len() {
            return None;
        }
        // Use the safe get method, which defaults to SeqCst ordering.
        let value = self.vec.get(self.current_index).unwrap();
        self.current_index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len().saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for AtomicFixedVecIter<'_, T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    fn len(&self) -> usize {
        self.vec.len().saturating_sub(self.current_index)
    }
}

impl<'a, T> IntoIterator for &'a AtomicFixedVec<T>
where
    T: Storable<u64> + Copy + ToPrimitive,
{
    type Item = T;
    type IntoIter = AtomicFixedVecIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        AtomicFixedVecIter {
            vec: self,
            current_index: 0,
        }
    }
}

impl<T> PartialEq for AtomicFixedVec<T>
where
    T: Storable<u64> + PartialEq + Copy + ToPrimitive,
{
    /// Checks for equality between two [`AtomicFixedVec`] instances.
    ///
    /// This comparison is performed by iterating over both vectors and comparing
    /// their elements one by one. The reads are done atomically but the overall
    /// comparison is not a single atomic operation.
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() || self.bit_width() != other.bit_width() {
            return false;
        }
        // Use SeqCst for a strong guarantee in tests.
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T> Eq for AtomicFixedVec<T> where T: Storable<u64> + Eq + Copy + ToPrimitive {}
