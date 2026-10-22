//! The synchronization primitives of the crate.
//!
//! Every atomic and every lock that [`AtomicFixedVec`](crate::AtomicFixedVec) uses comes from
//! here, so that building with `RUSTFLAGS="--cfg loom"` replaces them with the versions of the
//! loom model checker.

#[cfg(loom)]
pub(crate) use loom::{
    hint::spin_loop,
    sync::{
        Condvar, Mutex,
        atomic::{AtomicU64, fence},
    },
};
#[cfg(not(loom))]
pub(crate) use std::{
    hint::spin_loop,
    sync::{
        Condvar, Mutex,
        atomic::{AtomicU64, fence},
    },
};

/// Turns words into atomic words, reusing the allocation when `AtomicU64` has the alignment of
/// `u64`. Under `cfg(loom)` it always copies.
pub(crate) fn into_atomic(words: Vec<u64>) -> Vec<AtomicU64> {
    #[cfg(not(loom))]
    if align_of::<u64>() == align_of::<AtomicU64>() {
        let mut words = std::mem::ManuallyDrop::new(words);
        // SAFETY: `AtomicU64` has the same size and bit validity as `u64` (std documentation),
        // and the same alignment by the check above, so the global allocator allocated `words`
        // with the layout of `words.capacity()` atomic words, and its `words.len()` initialized
        // values are valid atomic words. `words` is never used or dropped again, so the
        // allocation has one owner.
        return unsafe {
            Vec::from_raw_parts(
                words.as_mut_ptr().cast::<AtomicU64>(),
                words.len(),
                words.capacity(),
            )
        };
    }
    words.into_iter().map(AtomicU64::new).collect()
}

/// Turns atomic words back into words, reusing the allocation when `AtomicU64` has the alignment
/// of `u64`. Under `cfg(loom)` it always copies.
pub(crate) fn from_atomic(words: Vec<AtomicU64>) -> Vec<u64> {
    #[cfg(not(loom))]
    if align_of::<u64>() == align_of::<AtomicU64>() {
        let mut words = std::mem::ManuallyDrop::new(words);
        // SAFETY: as in `into_atomic`, in the other direction. Owning the vector means that no
        // other thread can access the atomics any more.
        return unsafe {
            Vec::from_raw_parts(
                words.as_mut_ptr().cast::<u64>(),
                words.len(),
                words.capacity(),
            )
        };
    }
    words.into_iter().map(AtomicU64::into_inner).collect()
}
