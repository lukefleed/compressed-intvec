//! The locks of the elements that straddle two words: one version word per stripe of words,
//! which writers lock and readers validate, and a few parking slots in which threads that find
//! a stripe locked for long wait.

use crate::sync::{AtomicU64, Condvar, Mutex, fence, spin_loop};
use std::sync::{
    PoisonError,
    atomic::Ordering::{Acquire, Relaxed, Release},
};

/// The number of data words per stripe that `Stripes::new` aims for. Under loom each word has
/// its own stripe, so that a model of a few elements covers several stripes.
const WORDS_PER_STRIPE: usize = if cfg!(loom) { 1 } else { 64 };
/// The largest number of stripes.
const MAX_STRIPES: usize = 1 << 16;
/// The largest number of stripes that the words of one block map to. Under loom it is 2, so that
/// a model of a few elements covers several blocks.
const STRIPES_PER_BLOCK: usize = if cfg!(loom) { 2 } else { 256 };
/// The largest number of parking slots. Under loom it is 1, so that every stripe shares it.
const MAX_SLOTS: usize = if cfg!(loom) { 1 } else { 64 };
/// The number of times `lock` polls a locked stripe before it parks. Under loom it parks after
/// one poll.
const SPINS: usize = if cfg!(loom) { 1 } else { 64 };

/// The bit of a version that is set while a thread holds the stripe.
const LOCKED: u64 = 1;
/// The bit of a version that is set when threads wait for the stripe in its parking slot.
const PARKED: u64 = 1 << 63;
/// The bits of a version that count the releases of the stripe, in steps of 2.
const COUNTER: u64 = !(LOCKED | PARKED);

/// The locks and versions of the straddling elements of a vector.
///
/// Version `i` guards the elements whose first word maps to stripe `i` (see `index`). A thread
/// that changes such an element holds the stripe: `lock` sets `LOCKED` with a compare-and-swap,
/// and the release clears it and advances the counter. A reader calls `begin`, loads the
/// element's words, and keeps them if `validate` confirms the stripe's version remains valid.
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize, mem_dbg::MemDbg))]
pub(super) struct Stripes {
    versions: Box<[AtomicU64]>,
    // `slots.len()` is a power of two that divides `versions.len()`, or 0 with no versions.
    slots: Box<[Slot]>,
    block_shift: u32,
    spread_bits: u32,
}

/// Where the threads that wait for the locked stripes of a slot sleep.
struct Slot {
    lock: Mutex<()>,
    wake: Condvar,
}

/// A held stripe, released when dropped.
#[must_use = "dropping the guard releases the stripe"]
pub(super) struct Guard<'a> {
    stripes: &'a Stripes,
    // `stripes.versions[stripe]`.
    version: &'a AtomicU64,
    stripe: usize,
    held: u64,
}

impl Stripes {
    /// Creates the stripes of a vector whose elements fill `data_words` words. Returns empty
    /// stripe and parking-slot arrays when `straddles` is false.
    ///
    /// The data words form blocks of `2^block_shift` words, and the words of a block map to
    /// `2^spread_bits` stripes of their own, by their low bits. Disjoint blocks use separate
    /// stripes, and the straddling elements within a small range spread over many stripes.
    pub(super) fn new(data_words: usize, straddles: bool) -> Self {
        if !straddles {
            return Self {
                versions: Box::new([]),
                slots: Box::new([]),
                block_shift: 0,
                spread_bits: 0,
            };
        }
        let stripes = (data_words / WORDS_PER_STRIPE)
            .clamp(1, MAX_STRIPES)
            .next_power_of_two();
        let spread = stripes.min(STRIPES_PER_BLOCK);
        // `2^block_shift >= data_words / blocks`, so every data word lies in one of the blocks.
        let blocks = stripes / spread;
        let block_shift = data_words
            .div_ceil(blocks)
            .next_power_of_two()
            .trailing_zeros();
        Self {
            versions: (0..stripes).map(|_| AtomicU64::new(0)).collect(),
            slots: (0..stripes.min(MAX_SLOTS))
                .map(|_| Slot {
                    lock: Mutex::new(()),
                    wake: Condvar::new(),
                })
                .collect(),
            block_shift,
            spread_bits: spread.trailing_zeros(),
        }
    }

    /// Returns the stripe of the element whose first word is `word`, a data word.
    #[inline]
    pub(super) fn index(&self, word: usize) -> usize {
        let low = word & ((1 << self.spread_bits) - 1);
        ((word >> self.block_shift) << self.spread_bits) | low
    }

    /// Returns the version of `stripe` to pass to `validate`, or `None` if a thread holds it.
    #[inline]
    pub(super) fn begin(&self, stripe: usize) -> Option<u64> {
        // Pairs with the Release swap in `Guard::drop`: seeing its version makes the words that the
        // releasing thread changed visible.
        let version = self.versions[stripe].load(Acquire);
        (version & LOCKED == 0).then_some(version & !PARKED)
    }

    /// Returns whether the stripe's version still matches `version`, validating the caller's
    /// intervening loads of the element's words.
    #[inline]
    pub(super) fn validate(&self, stripe: usize, version: u64) -> bool {
        // Orders the caller's loads of the words before this load; pairs with the Release fence
        // in `lock`, so a load that saw a change of a holder also sees the stripe changed.
        fence(Acquire);
        // `PARKED` alone changes no word.
        self.versions[stripe].load(Relaxed) & !PARKED == version
    }

    /// Holds `stripe` until the returned guard is dropped. It polls a locked stripe briefly,
    /// then sleeps in the stripe's parking slot until a release wakes it.
    #[inline]
    pub(super) fn lock(&self, stripe: usize) -> Guard<'_> {
        let version = &self.versions[stripe];
        let held = try_lock(version, version.load(Relaxed))
            .unwrap_or_else(|current| self.lock_contended(stripe, version, current));
        Guard {
            stripes: self,
            version,
            stripe,
            held,
        }
    }

    /// The rest of `lock` after its first attempt found the version of `stripe`, `version`, at
    /// `current`, locked or changed by another thread. Returns the held version.
    #[cold]
    #[inline(never)]
    fn lock_contended(&self, stripe: usize, version: &AtomicU64, mut current: u64) -> u64 {
        let mut spins = 0;
        loop {
            if current & LOCKED == 0 {
                match try_lock(version, current) {
                    Ok(held) => return held,
                    Err(actual) => {
                        current = actual;
                        continue;
                    }
                }
            }
            if spins < SPINS {
                spins += 1;
                spin_loop();
            } else {
                self.park(stripe);
                spins = 0;
            }
            current = version.load(Relaxed);
        }
    }

    /// Sleeps until a release of `stripe` wakes the slot, unless `stripe` is already free.
    #[cold]
    #[inline(never)]
    fn park(&self, stripe: usize) {
        let slot = &self.slots[stripe & (self.slots.len() - 1)];
        // No code runs under the slot's lock that can panic, so a poisoned lock is still usable.
        let guard = slot.lock.lock().unwrap_or_else(PoisonError::into_inner);
        // This `fetch_or` and the swap in `Guard::drop` change the same word, so one of them comes
        // first. If the swap does, the stripe is free and this thread returns to take it.
        // Otherwise the releasing thread sees `PARKED` and takes the slot's lock to wake the
        // slot, which it gets only once this thread sleeps.
        if self.versions[stripe].fetch_or(PARKED, Relaxed) & LOCKED != 0 {
            drop(
                slot.wake
                    .wait(guard)
                    .unwrap_or_else(PoisonError::into_inner),
            );
        }
    }

    /// Wakes every thread that sleeps in the parking slot of `stripe`.
    #[cold]
    #[inline(never)]
    fn wake(&self, stripe: usize) {
        let slot = &self.slots[stripe & (self.slots.len() - 1)];
        let _guard = slot.lock.lock().unwrap_or_else(PoisonError::into_inner);
        slot.wake.notify_all();
    }
}

/// Locks the stripe whose version `version` is `current`, if it is free, and returns the held
/// version, or returns the version found instead.
#[inline(always)]
fn try_lock(version: &AtomicU64, current: u64) -> Result<u64, u64> {
    if current & LOCKED != 0 {
        return Err(current);
    }
    // Pairs with the Release swap in `Guard::drop`: the changes of the previous holder are
    // visible to this one.
    version.compare_exchange_weak(current, current | LOCKED, Acquire, Relaxed)?;
    // Orders the locked version before the holder's changes of the words; pairs with the Acquire
    // fence in `validate`.
    fence(Release);
    Ok(current | LOCKED)
}

impl Drop for Guard<'_> {
    #[inline]
    fn drop(&mut self) {
        let next = (self.held & COUNTER).wrapping_add(2) & COUNTER;
        // Pairs with the Acquire loads in `begin` and `lock`: the holder's changes of the words
        // are visible to the threads that see `next`.
        if self.version.swap(next, Release) & PARKED != 0 {
            self.stripes.wake(self.stripe);
        }
    }
}

// The parking slots hold no heap memory; `Condvar` has no `mem_dbg` implementations.
#[cfg(feature = "mem_dbg")]
impl mem_dbg::FlatType for Slot {
    type Flat = mem_dbg::True;
}

#[cfg(feature = "mem_dbg")]
impl mem_dbg::MemSize for Slot {
    fn mem_size_rec(
        &self,
        _flags: mem_dbg::SizeFlags,
        _refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        size_of::<Self>()
    }
}

#[cfg(feature = "mem_dbg")]
impl mem_dbg::MemDbgImpl for Slot {}

// Under loom, atomics work only inside a model.
#[cfg(all(test, not(loom)))]
mod tests;
