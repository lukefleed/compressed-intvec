//! A fixed-width vector whose elements threads can read and change concurrently.

#[cfg(feature = "serde")]
mod serde;
mod stripes;

use super::{
    FixedVec, FixedVecError, ValueTooLarge, checked_bit_width, checked_word, index_out_of_bounds,
    layout, value_too_large,
};
use crate::{
    Storable,
    storable::{unzigzag, zigzag},
    sync::{self, AtomicU64, spin_loop},
};
use std::{
    fmt,
    marker::PhantomData,
    ops::{BitAnd, BitOr, BitXor},
    sync::atomic::Ordering::{self, AcqRel, Acquire, Relaxed, Release, SeqCst},
};
use stripes::Stripes;

/// The number of times a read of a straddling element retries while a writer holds its stripe,
/// before it takes the stripe itself. Under loom it takes the stripe after one attempt.
const READ_ATTEMPTS: usize = if cfg!(loom) { 1 } else { 16 };

/// A packed integer vector with atomic access to individual fixed-width elements.
///
/// Use `AtomicFixedVec` for compact counters or other bounded integers that threads update
/// through shared references. Its length and bit width are fixed at construction. Each load
/// or update is atomic with respect to the indexed element.
///
/// The element layout matches [`FixedVec`]. Conversions consume the source vector and reuse
/// its word allocation when `u64` and [`AtomicU64`](std::sync::atomic::AtomicU64) have the same
/// alignment.
/// [`new`](AtomicFixedVec::new) creates a vector with a chosen width and length, while
/// [`from_slice`](AtomicFixedVec::from_slice) chooses the smallest width for existing values.
///
/// [`load`](AtomicFixedVec::load) and [`store`](AtomicFixedVec::store) provide indexed reads
/// and writes. The `fetch_*` methods update an element and return its previous value, and
/// [`compare_exchange`](AtomicFixedVec::compare_exchange) replaces an element conditionally.
///
/// # Synchronization
///
/// Elements contained in one `u64` word use atomic loads and compare-and-swap updates on that
/// word. Updating one element can cause an update to another element in the same word to retry.
///
/// Elements that straddle two words use a pool of stripe locks, selected by the first word.
/// Writers hold the stripe while updating both words. Readers validate both loads against the
/// stripe's version, retry on a concurrent write, and acquire the stripe after repeated retries.
/// Stripes may cover multiple elements. A contending operation may block until the stripe
/// holder releases its lock.
///
/// # Values and wrapping
///
/// Stored values and operands of fetch operations must fit the bit width, as for
/// [`FixedVec::set`]. A value that does not fit panics. [`fetch_add`](AtomicFixedVec::fetch_add) and
/// [`fetch_sub`](AtomicFixedVec::fetch_sub) wrap at the bit width: modulo `2^bit_width` for
/// unsigned elements, and in the two's complement range of `bit_width` bits for signed elements
/// (see [`Storable::SIGNED`]).
///
/// # Orderings
///
/// Each method takes the same orderings as the method of the same name on
/// [`AtomicU64`](std::sync::atomic::AtomicU64), and panics on the same invalid ones. On an
/// element inside one word, the ordering applies to the atomic operations on that word. On an
/// element that straddles two words, it applies to the operations on both words, and the
/// sequence lock adds acquire and release edges, so the operation is ordered at least as an
/// `Acquire` load or an `AcqRel` change with respect to the other operations on that element.
///
/// # Serialization
///
/// With the `serde` feature, serialization uses the packed format of [`FixedVec`]. Each element
/// is loaded atomically with [`Relaxed`] ordering. Concurrent updates may yield a serialized
/// result containing elements observed in different vector states.
///
/// # Examples
///
/// ```
/// use compressed_intvec::AtomicFixedVec;
/// use std::sync::atomic::Ordering;
/// use std::thread;
///
/// // 1000 counters of 21 bits each, many of which straddle two words.
/// let counters = AtomicFixedVec::<u32>::new(21, 1000)?;
/// thread::scope(|scope| {
///     for _ in 0..4 {
///         scope.spawn(|| {
///             for i in 0..1000 {
///                 counters.fetch_add(i, 1, Ordering::Relaxed);
///             }
///         });
///     }
/// });
/// assert!((0..1000).all(|i| counters.load(i, Ordering::Relaxed) == 4));
/// # Ok::<(), compressed_intvec::FixedVecError>(())
/// ```
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize, mem_dbg::MemDbg))]
pub struct AtomicFixedVec<T> {
    // Invariants:
    // - those of `FixedVec` for `words`, `len`, `bit_width` and `mask`;
    // - `stripes` has stripes for the data words of `words` unless the bit width is 0 or divides
    //   64, when no element straddles two words;
    // - an element that straddles words `w` and `w + 1` is changed only by `modify_straddling`,
    //   which holds stripe `stripes.index(w)`; it is read while holding that stripe, or by
    //   `try_read_straddling`, which validates the stripe's version.
    words: Vec<AtomicU64>,
    stripes: Stripes,
    len: usize,
    bit_width: usize,
    mask: u64,
    element: PhantomData<fn() -> T>,
}

#[inline]
#[track_caller]
fn check_load_order(order: Ordering) {
    match order {
        Release => panic!("there is no such thing as a release load"),
        AcqRel => panic!("there is no such thing as an acquire-release load"),
        _ => {}
    }
}

#[inline]
#[track_caller]
fn check_store_order(order: Ordering) {
    match order {
        Acquire => panic!("there is no such thing as an acquire store"),
        AcqRel => panic!("there is no such thing as an acquire-release store"),
        _ => {}
    }
}

/// Wraps `value` into the two's complement range of `bit_width` bits, and returns the zig-zag
/// word that stores it.
#[inline]
fn wrap_signed(value: i64, bit_width: usize) -> u64 {
    if bit_width == 0 {
        return 0;
    }
    let shift = layout::WORD_BITS - bit_width;
    zigzag((value << shift) >> shift)
}

/// Adds the words of two elements of `T`, wrapping at `bit_width` bits.
fn add_words<T: Storable>(a: u64, b: u64, bit_width: usize, mask: u64) -> u64 {
    if T::SIGNED {
        wrap_signed(unzigzag(a).wrapping_add(unzigzag(b)), bit_width)
    } else {
        a.wrapping_add(b) & mask
    }
}

/// Subtracts the words of two elements of `T`, wrapping at `bit_width` bits.
fn sub_words<T: Storable>(a: u64, b: u64, bit_width: usize, mask: u64) -> u64 {
    if T::SIGNED {
        wrap_signed(unzigzag(a).wrapping_sub(unzigzag(b)), bit_width)
    } else {
        a.wrapping_sub(b) & mask
    }
}

impl<T: Storable> AtomicFixedVec<T> {
    /// Creates a vector of `len` elements of `bit_width` bits each, all stored as the word 0.
    ///
    /// # Errors
    ///
    /// - [`FixedVecError::BitWidthTooLarge`] if `bit_width` exceeds [`Storable::BITS`] of `T`
    ///   or 64;
    /// - [`FixedVecError::LengthOverflow`] if `len * bit_width` overflows `usize`.
    pub fn new(bit_width: u32, len: usize) -> Result<Self, FixedVecError> {
        let width = checked_bit_width::<T>(bit_width)?;
        let required = layout::required_words(len, width)
            .ok_or(FixedVecError::LengthOverflow { len, bit_width })?;
        Ok(FixedVec::from_parts(vec![0; required], len, bit_width)?.into())
    }

    /// Creates a vector holding `values`, with the smallest bit width that holds all of them, as
    /// [`FixedVec::from_slice`].
    ///
    /// # Panics
    ///
    /// As [`FixedVec::from_slice`].
    pub fn from_slice(values: &[T]) -> Self {
        FixedVec::from_slice(values).into()
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of bits used to store each encoded element.
    #[inline]
    pub fn bit_width(&self) -> u32 {
        self.bit_width as u32
    }

    #[inline]
    #[track_caller]
    fn check_index(&self, index: usize) {
        if index >= self.len {
            index_out_of_bounds(index, self.len);
        }
    }

    /// Returns the word of `value`, panicking if it does not fit the bit width.
    #[inline]
    #[track_caller]
    fn fitting_word(&self, value: T) -> u64 {
        super::fitting_word(value, self.bit_width, self.mask)
    }

    /// Returns words `word` and `word + 1`, which hold an element that straddles them.
    #[inline]
    fn pair(&self, word: usize) -> &[AtomicU64; 2] {
        // One bounds check: `word` is a bit position divided by 64, so `word + 2` cannot overflow.
        self.words[word..word + 2]
            .try_into()
            .expect("a slice of 2 words")
    }

    /// Returns the element that starts at bit `offset` of `pair[0]` and ends in `pair[1]`. The
    /// two loads are not atomic together: the caller holds the element's stripe, or validates
    /// the result against the stripe's version.
    #[inline(always)]
    fn read_pair(&self, pair: &[AtomicU64; 2], offset: usize, order: Ordering) -> u64 {
        let read = |order| {
            let low = pair[0].load(order) >> offset;
            let high = pair[1].load(order) << (layout::WORD_BITS - offset);
            (low | high) & self.mask
        };
        // One dispatch on a run-time ordering for both loads; it folds when `order` is constant.
        match order {
            Relaxed => read(Relaxed),
            Acquire => read(Acquire),
            SeqCst => read(SeqCst),
            order => read(order),
        }
    }

    /// Returns the word of element `index < len`.
    #[inline]
    fn load_word(&self, index: usize, order: Ordering) -> u64 {
        let (word, offset) = layout::locate(index, self.bit_width);
        if offset + self.bit_width <= layout::WORD_BITS {
            (self.words[word].load(order) >> offset) & self.mask
        } else {
            let stripe = self.stripes.index(word);
            let pair = self.pair(word);
            self.try_read_straddling(stripe, pair, offset, order)
                .unwrap_or_else(|| self.read_straddling_contended(stripe, pair, offset, order))
        }
    }

    /// Loads the element that starts at bit `offset` of `pair[0]` and ends in `pair[1]`, and
    /// returns it if the stripe's version validates the loads.
    #[inline(always)]
    fn try_read_straddling(
        &self,
        stripe: usize,
        pair: &[AtomicU64; 2],
        offset: usize,
        order: Ordering,
    ) -> Option<u64> {
        let version = self.stripes.begin(stripe)?;
        let value = self.read_pair(pair, offset, order);
        self.stripes.validate(stripe, version).then_some(value)
    }

    /// The rest of a load of a straddling element after its first attempt failed: it retries,
    /// and after `READ_ATTEMPTS` attempts in all it acquires the stripe.
    #[cold]
    #[inline(never)]
    fn read_straddling_contended(
        &self,
        stripe: usize,
        pair: &[AtomicU64; 2],
        offset: usize,
        order: Ordering,
    ) -> u64 {
        spin_loop();
        for _ in 1..READ_ATTEMPTS {
            if let Some(value) = self.try_read_straddling(stripe, pair, offset, order) {
                return value;
            }
            spin_loop();
        }
        // The writers of this stripe kept it busy: hold it, as they do.
        let _guard = self.stripes.lock(stripe);
        self.read_pair(pair, offset, order)
    }

    /// Replaces the element at bit `offset` of `word`, which fits in that word, as
    /// `AtomicU64::fetch_update`.
    #[inline]
    fn update_in_word(
        &self,
        word: usize,
        offset: usize,
        set_order: Ordering,
        fetch_order: Ordering,
        mut f: impl FnMut(u64) -> Option<u64>,
    ) -> Result<u64, u64> {
        let atom = &self.words[word];
        let field = self.mask << offset;
        let mut current = atom.load(fetch_order);
        loop {
            let old = (current >> offset) & self.mask;
            let new = f(old).ok_or(old)?;
            // The exchange also fails when only the other elements of the word changed, and
            // spuriously (it is weak): `f` then runs again, possibly with the same value, as in
            // `AtomicU64::fetch_update`.
            match atom.compare_exchange_weak(
                current,
                (current & !field) | (new << offset),
                set_order,
                fetch_order,
            ) {
                Ok(_) => return Ok(old),
                Err(actual) => current = actual,
            }
        }
    }

    /// Replaces the element that starts at bit `offset` of `word` and ends in `word + 1` with
    /// `f` of it, holding its stripe, and returns the previous word, in `Err` if `f` returns
    /// `None`. `f` must use crate-controlled code because it runs while the stripe is held.
    #[inline]
    fn modify_straddling(
        &self,
        word: usize,
        offset: usize,
        set_order: Ordering,
        fetch_order: Ordering,
        f: impl FnOnce(u64) -> Option<u64>,
    ) -> Result<u64, u64> {
        let pair = self.pair(word);
        let _guard = self.stripes.lock(self.stripes.index(word));
        // Ordered by the stripe: the writers of this element hold it too.
        let old = self.read_pair(pair, offset, fetch_order);
        let new = f(old).ok_or(old)?;
        // Every writer of these bits holds the stripe, so they still hold `old`, and XOR changes
        // them to `new` without touching the bits of the neighbors, which other threads may be
        // changing at the same time with atomic operations on the same words.
        let diff = old ^ new;
        pair[0].fetch_xor(diff << offset, set_order);
        pair[1].fetch_xor(diff >> (layout::WORD_BITS - offset), set_order);
        Ok(old)
    }

    /// Replaces element `index < len` with `f` of it, as `AtomicU64::fetch_update`. `f` must use
    /// crate-controlled code because it may run under a lock.
    ///
    /// `fetch_order` applies to the loads whose value a failure returns. `store`, `swap` and the
    /// arithmetic operations pass `Relaxed`: they return the value that the successful exchange,
    /// ordered by `set_order`, replaced, and a failed exchange only retries.
    #[inline]
    fn modify(
        &self,
        index: usize,
        set_order: Ordering,
        fetch_order: Ordering,
        f: impl Fn(u64) -> Option<u64>,
    ) -> Result<u64, u64> {
        let (word, offset) = layout::locate(index, self.bit_width);
        if offset + self.bit_width <= layout::WORD_BITS {
            self.update_in_word(word, offset, set_order, fetch_order, f)
        } else {
            self.modify_straddling(word, offset, set_order, fetch_order, f)
        }
    }

    /// Replaces element `index < len` with `f` of it, as `AtomicU64::fetch_update`. The callback
    /// runs outside the vector's locks and may call user-provided code.
    ///
    /// # Panics
    ///
    /// If `f` returns a value that does not fit the bit width, rejecting the attempted update.
    #[track_caller]
    fn update(
        &self,
        index: usize,
        set_order: Ordering,
        fetch_order: Ordering,
        mut f: impl FnMut(T) -> Option<T>,
    ) -> Result<T, T> {
        let mut too_large: Option<ValueTooLarge<T>> = None;
        let mut word_f =
            |old: u64| match checked_word(f(T::from_word(old))?, self.bit_width, self.mask) {
                Ok(new) => Some(new),
                Err(error) => {
                    too_large = Some(error);
                    None
                }
            };
        let (word, offset) = layout::locate(index, self.bit_width);
        let result = if offset + self.bit_width <= layout::WORD_BITS {
            self.update_in_word(word, offset, set_order, fetch_order, &mut word_f)
        } else {
            let mut old = self.load_word(index, fetch_order);
            loop {
                let Some(new) = word_f(old) else {
                    break Err(old);
                };
                let exchange = |current| (current == old).then_some(new);
                match self.modify_straddling(word, offset, set_order, fetch_order, exchange) {
                    Ok(_) => break Ok(old),
                    Err(current) => old = current,
                }
            }
        };
        if let Some(error) = too_large {
            value_too_large(error.required, self.bit_width);
        }
        result.map(T::from_word).map_err(T::from_word)
    }

    /// Returns the element at `index`.
    ///
    /// # Panics
    ///
    /// If `index >= len`, or if `order` is [`Release`] or [`AcqRel`].
    #[inline]
    #[track_caller]
    pub fn load(&self, index: usize, order: Ordering) -> T {
        check_load_order(order);
        self.check_index(index);
        T::from_word(self.load_word(index, order))
    }

    /// Stores `value` at `index`.
    ///
    /// # Panics
    ///
    /// If `index >= len`, if `value` does not fit the bit width, or if `order` is [`Acquire`] or
    /// [`AcqRel`].
    #[inline]
    #[track_caller]
    pub fn store(&self, index: usize, value: T, order: Ordering) {
        check_store_order(order);
        self.check_index(index);
        let new = self.fitting_word(value);
        // Never `Err`: the closure always returns `Some`.
        let (Ok(_) | Err(_)) = self.modify(index, order, Relaxed, move |_| Some(new));
    }

    /// Stores `value` at `index` and returns the previous element.
    ///
    /// # Panics
    ///
    /// If `index >= len`, or if `value` does not fit the bit width.
    #[inline]
    #[track_caller]
    pub fn swap(&self, index: usize, value: T, order: Ordering) -> T {
        self.check_index(index);
        let new = self.fitting_word(value);
        let (Ok(old) | Err(old)) = self.modify(index, order, Relaxed, move |_| Some(new));
        T::from_word(old)
    }

    /// Atomically compares and conditionally replaces the encoded element at `index`, as
    /// [`AtomicU64::compare_exchange`](std::sync::atomic::AtomicU64::compare_exchange).
    ///
    /// Compares the stored word with `current.to_word()`. Returns the decoded previous element
    /// in `Ok` when those encoded words match, and in `Err` otherwise.
    ///
    /// A custom [`Storable`] implementation may decode several distinct words to the same
    /// element, so passing a loaded value back as `current` can fail without an intervening
    /// writer. Use [`fetch_update`](AtomicFixedVec::fetch_update) for updates expressed through
    /// decoded values.
    ///
    /// # Panics
    ///
    /// If `index >= len`, if `new` does not fit the bit width, or if `failure` is [`Release`] or
    /// [`AcqRel`].
    #[inline]
    #[track_caller]
    pub fn compare_exchange(
        &self,
        index: usize,
        current: T,
        new: T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<T, T> {
        check_load_order(failure);
        self.check_index(index);
        let current = current.to_word();
        let new = self.fitting_word(new);
        self.modify(index, success, failure, move |old| {
            (old == current).then_some(new)
        })
        .map(T::from_word)
        .map_err(T::from_word)
    }

    /// Atomically updates the element at `index` using a closure on its decoded value.
    ///
    /// As with [`AtomicU64::fetch_update`](std::sync::atomic::AtomicU64::fetch_update), returns
    /// `Ok(previous)` if `f` returns `Some(new)` and the update succeeds. Returning `None`
    /// rejects the update and returns `Err(observed)`.
    ///
    /// `f` may run several times, including with the same value: retries can follow changes to
    /// this element, changes to neighboring elements sharing its word, or a spurious failed
    /// compare-and-swap. The closure runs outside the vector's stripe locks and may access the
    /// vector. Side effects in the closure may repeat, and each call completes at most one update.
    ///
    /// # Panics
    ///
    /// If `index >= len`, if `fetch_order` is [`Release`] or [`AcqRel`], or if `f` returns a
    /// value that does not fit the bit width (the attempted update is rejected).
    #[track_caller]
    pub fn fetch_update(
        &self,
        index: usize,
        set_order: Ordering,
        fetch_order: Ordering,
        f: impl FnMut(T) -> Option<T>,
    ) -> Result<T, T> {
        check_load_order(fetch_order);
        self.check_index(index);
        self.update(index, set_order, fetch_order, f)
    }

    /// Adds `value` to the element at `index`, wrapping at the bit width, and returns the
    /// previous element.
    ///
    /// For built-in signed integers, wrapping uses the signed range of `bit_width` bits:
    ///
    /// ```
    /// use compressed_intvec::AtomicFixedVec;
    /// use std::sync::atomic::Ordering::Relaxed;
    ///
    /// let counters = AtomicFixedVec::<i32>::new(3, 1)?;
    /// counters.store(0, 3, Relaxed);
    /// assert_eq!(counters.fetch_add(0, 1, Relaxed), 3);
    /// assert_eq!(counters.load(0, Relaxed), -4);
    /// # Ok::<(), compressed_intvec::FixedVecError>(())
    /// ```
    ///
    /// # Panics
    ///
    /// If `index >= len`, or if `value` does not fit the bit width.
    #[inline]
    #[track_caller]
    pub fn fetch_add(&self, index: usize, value: T, order: Ordering) -> T {
        self.check_index(index);
        let delta = self.fitting_word(value);
        let (width, mask) = (self.bit_width, self.mask);
        let (Ok(old) | Err(old)) = self.modify(index, order, Relaxed, move |old| {
            Some(add_words::<T>(old, delta, width, mask))
        });
        T::from_word(old)
    }

    /// Subtracts `value` from the element at `index`, wrapping at the bit width, and returns
    /// the previous element.
    ///
    /// # Panics
    ///
    /// If `index >= len`, or if `value` does not fit the bit width.
    #[inline]
    #[track_caller]
    pub fn fetch_sub(&self, index: usize, value: T, order: Ordering) -> T {
        self.check_index(index);
        let delta = self.fitting_word(value);
        let (width, mask) = (self.bit_width, self.mask);
        let (Ok(old) | Err(old)) = self.modify(index, order, Relaxed, move |old| {
            Some(sub_words::<T>(old, delta, width, mask))
        });
        T::from_word(old)
    }

    /// Replaces the element at `index` with the bitwise AND of it and `value`, and returns the
    /// previous element.
    ///
    /// # Panics
    ///
    /// If `index >= len`, or if `value` or the result does not fit the bit width.
    #[track_caller]
    pub fn fetch_and(&self, index: usize, value: T, order: Ordering) -> T
    where
        T: BitAnd<Output = T>,
    {
        self.check_index(index);
        self.fitting_word(value);
        let (Ok(old) | Err(old)) = self.update(index, order, Relaxed, |old| Some(old & value));
        old
    }

    /// Replaces the element at `index` with the bitwise OR of it and `value`, and returns the
    /// previous element.
    ///
    /// # Panics
    ///
    /// If `index >= len`, or if `value` or the result does not fit the bit width.
    #[track_caller]
    pub fn fetch_or(&self, index: usize, value: T, order: Ordering) -> T
    where
        T: BitOr<Output = T>,
    {
        self.check_index(index);
        self.fitting_word(value);
        let (Ok(old) | Err(old)) = self.update(index, order, Relaxed, |old| Some(old | value));
        old
    }

    /// Replaces the element at `index` with the bitwise XOR of it and `value`, and returns the
    /// previous element.
    ///
    /// # Panics
    ///
    /// If `index >= len`, or if `value` or the result does not fit the bit width.
    #[track_caller]
    pub fn fetch_xor(&self, index: usize, value: T, order: Ordering) -> T
    where
        T: BitXor<Output = T>,
    {
        self.check_index(index);
        self.fitting_word(value);
        let (Ok(old) | Err(old)) = self.update(index, order, Relaxed, |old| Some(old ^ value));
        old
    }

    /// Replaces the element at `index` with the larger of it and `value`, and returns the
    /// previous element.
    ///
    /// # Panics
    ///
    /// If `index >= len`, or if `value` does not fit the bit width.
    #[track_caller]
    pub fn fetch_max(&self, index: usize, value: T, order: Ordering) -> T
    where
        T: Ord,
    {
        self.check_index(index);
        self.fitting_word(value);
        let (Ok(old) | Err(old)) = self.update(index, order, Relaxed, |old| Some(old.max(value)));
        old
    }

    /// Replaces the element at `index` with the smaller of it and `value`, and returns the
    /// previous element.
    ///
    /// # Panics
    ///
    /// If `index >= len`, or if `value` does not fit the bit width.
    #[track_caller]
    pub fn fetch_min(&self, index: usize, value: T, order: Ordering) -> T
    where
        T: Ord,
    {
        self.check_index(index);
        self.fitting_word(value);
        let (Ok(old) | Err(old)) = self.update(index, order, Relaxed, |old| Some(old.min(value)));
        old
    }
}

impl<T: Storable> From<FixedVec<T>> for AtomicFixedVec<T> {
    /// Converts the vector to atomic storage, reusing the word allocation on targets where
    /// `u64` and `AtomicU64` have the same alignment.
    fn from(vec: FixedVec<T>) -> Self {
        let (words, len, bit_width) = vec.into_parts();
        let bit_width = bit_width as usize;
        let straddles = bit_width != 0 && !layout::WORD_BITS.is_multiple_of(bit_width);
        Self {
            stripes: Stripes::new((len * bit_width).div_ceil(layout::WORD_BITS), straddles),
            words: sync::into_atomic(words),
            len,
            bit_width,
            mask: layout::mask(bit_width),
            element: PhantomData,
        }
    }
}

impl<T: Storable> From<AtomicFixedVec<T>> for FixedVec<T> {
    /// Converts the vector to ordinary storage, reusing the word allocation on targets where
    /// `u64` and `AtomicU64` have the same alignment.
    fn from(vec: AtomicFixedVec<T>) -> Self {
        FixedVec::from_parts(sync::from_atomic(vec.words), vec.len, vec.bit_width as u32)
            .expect("an AtomicFixedVec keeps the invariant of FixedVec")
    }
}

impl<T: Storable> From<&[T]> for AtomicFixedVec<T> {
    /// Creates a vector with the smallest bit width, as [`AtomicFixedVec::from_slice`].
    fn from(values: &[T]) -> Self {
        Self::from_slice(values)
    }
}

impl<T: Storable + fmt::Debug> fmt::Debug for AtomicFixedVec<T> {
    /// Shows the elements, each loaded with [`Relaxed`] ordering, as the std atomics do.
    ///
    /// Concurrent updates may produce output containing elements observed in different vector
    /// states.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..self.len).map(|index| T::from_word(self.load_word(index, Relaxed))))
            .finish()
    }
}
