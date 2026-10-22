//! The tests of compressed-intvec, written against the public API: each vector type is compared
//! with a model built from std types, after every operation of random sequences, and each error
//! that the documentation promises has a case that reaches it.
//!
//! The inputs come from fixed seeds, so every run sees the same cases and a failure replays as
//! it is. Expected values are computed in `i128` from the documented meaning of a bit width,
//! never with the crate's own encoding.

// Under `--cfg loom` the vectors use loom's atomics, which work only inside a loom model
// (`tests/loom.rs`).
#![cfg(not(loom))]

mod atomic;
mod codec;
mod fixed;
// Under Miri, rayon's crossbeam-epoch 0.9.18 violates Stacked Borrows.
#[cfg(all(feature = "parallel", not(miri)))]
mod parallel;
mod seq;
#[cfg(feature = "serde")]
mod serialization;
mod storable;
mod variable;

use compressed_intvec::{
    AtomicFixedVec, FixedVec, FixedVecIntoIter, FixedVecIter, FixedVecSlice, SeqIter, SeqVec,
    SeqVecIter, Storable, VarVec, VarVecGetMany, VarVecIter,
};
use std::fmt::{Debug, Display};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::num::NonZeroUsize;
use std::ops::{BitAnd, BitOr, BitXor};
use std::panic::{AssertUnwindSafe, catch_unwind};

// Every vector, view and iterator can move to and be shared with other threads, whatever the
// element type: `*const ()` is neither `Send` nor `Sync`, and the vectors hold their element
// type only as `PhantomData<fn() -> T>`. The checks fail to compile if a change of fields breaks
// one of them.
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    type E = *const ();
    send_sync::<FixedVec<E>>();
    send_sync::<FixedVec<E, Box<[u64]>>>();
    send_sync::<FixedVec<E, std::sync::Arc<[u64]>>>();
    send_sync::<FixedVec<E, &[u64]>>();
    send_sync::<FixedVecSlice<'_, E>>();
    send_sync::<FixedVecIter<'_, E>>();
    send_sync::<FixedVecIntoIter<E>>();
    send_sync::<AtomicFixedVec<E>>();
    send_sync::<VarVec<E>>();
    send_sync::<VarVec<E, Box<[u64]>>>();
    send_sync::<VarVec<E, std::sync::Arc<[u64]>>>();
    send_sync::<VarVec<E, &[u64]>>();
    send_sync::<VarVecIter<'_, E>>();
    send_sync::<VarVecGetMany<'_, E, std::ops::Range<usize>>>();
    send_sync::<SeqVec<E>>();
    send_sync::<SeqVec<E, Box<[u64]>>>();
    send_sync::<SeqVec<E, std::sync::Arc<[u64]>>>();
    send_sync::<SeqVec<E, &[u64]>>();
    send_sync::<SeqIter<'_, E>>();
    send_sync::<SeqVecIter<'_, E>>();
};

/// Divides the numbers of random cases and operations under Miri, which interprets them.
pub(crate) const MIRI_DIVISOR: usize = if cfg!(miri) { 20 } else { 1 };

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum AliasedFlag {
    Off,
    On,
}

impl Storable for AliasedFlag {
    const BITS: u32 = 64;

    fn to_word(self) -> u64 {
        match self {
            Self::Off => 0,
            Self::On => 1,
        }
    }

    fn from_word(word: u64) -> Self {
        if word & 1 == 0 { Self::Off } else { Self::On }
    }
}

/// The SplitMix64 generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a value in `0..n`, which must not be empty.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// Returns a word whose number of significant bits is uniform in `0..=64`, so that small and
    /// large words are equally common.
    pub fn word(&mut self) -> u64 {
        match self.below(65) {
            0 => 0,
            bits => self.next_u64() >> (64 - bits),
        }
    }
}

/// The primitive integer types, with exact conversions to `i128`.
pub trait Elem:
    Storable
    + Debug
    + Ord
    + Hash
    + Send
    + Sync
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
{
    fn to_int(self) -> i128;
    /// Returns `None` if `value` is out of the range of the type.
    fn from_int(value: i128) -> Option<Self>;
}

impl<T> Elem for T
where
    T: Storable
        + Debug
        + Ord
        + Hash
        + Send
        + Sync
        + BitAnd<Output = T>
        + BitOr<Output = T>
        + BitXor<Output = T>
        + TryFrom<i128>,
    i128: TryFrom<T>,
{
    fn to_int(self) -> i128 {
        i128::try_from(self)
            .ok()
            .expect("every primitive integer fits i128")
    }

    fn from_int(value: i128) -> Option<Self> {
        T::try_from(value).ok()
    }
}

/// Calls `$f::<T>()` for every primitive element type.
macro_rules! for_each_type {
    ($f:ident) => {
        $f::<u8>();
        $f::<u16>();
        $f::<u32>();
        $f::<u64>();
        $f::<usize>();
        $f::<i8>();
        $f::<i16>();
        $f::<i32>();
        $f::<i64>();
        $f::<isize>();
    };
}
pub(crate) use for_each_type;

pub fn signed<T: Elem>() -> bool {
    T::from_int(-1).is_some()
}

/// The number of bits of `T`, which is the largest bit width of its vectors.
pub fn max_width<T: Elem>() -> u32 {
    8 * size_of::<T>() as u32
}

/// The smallest and the largest value that `bit_width` bits hold: `0..2^w` for an unsigned type,
/// and the two's complement range of `w` bits for a signed one.
pub fn range<T: Elem>(bit_width: u32) -> (i128, i128) {
    match (signed::<T>(), bit_width) {
        (true, 0) => (0, 0),
        (true, w) => (-(1 << (w - 1)), (1 << (w - 1)) - 1),
        (false, w) => (0, (1 << w) - 1),
    }
}

/// The smallest bit width that holds `value`.
pub fn width_of<T: Elem>(value: T) -> u32 {
    let x = value.to_int();
    (0..=max_width::<T>())
        .find(|&w| {
            let (lo, hi) = range::<T>(w);
            (lo..=hi).contains(&x)
        })
        .expect("the full width holds every value")
}

/// Returns a value that fits `bit_width` bits, often one at an end of the range or next to 0.
pub fn value<T: Elem>(rng: &mut Rng, bit_width: u32) -> T {
    let (lo, hi) = range::<T>(bit_width);
    let x = match rng.below(5) {
        0 => lo,
        1 => hi,
        2 => (rng.below(3) as i128 - 1).clamp(lo, hi),
        _ => lo + i128::from(rng.next_u64()) % (hi - lo + 1),
    };
    T::from_int(x).expect("the range of a bit width is within the type")
}

/// Returns a value of `T` that does not fit `bit_width` bits, if there is one.
pub fn too_large<T: Elem>(rng: &mut Rng, bit_width: u32) -> Option<T> {
    let (lo, hi) = range::<T>(bit_width);
    let below = T::from_int(lo - 1);
    let above = T::from_int(hi + 1);
    if rng.coin() {
        above.or(below)
    } else {
        below.or(above)
    }
}

pub fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("not zero")
}

pub fn hash(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// A named case of a table of calls that must panic or fail.
pub type Case<T> = (&'static str, fn(&mut T));

/// Checks that `f` panics.
#[track_caller]
pub fn panics(label: impl Display, f: impl FnOnce()) {
    assert!(
        catch_unwind(AssertUnwindSafe(f)).is_err(),
        "{label}: expected a panic"
    );
}

/// The history of a random run, printed if the thread panics while the trace is alive, so that
/// a failure shows the operations that led to it.
pub struct Trace(pub Vec<String>);

impl Drop for Trace {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!("{}", self.0.join("\n"));
        }
    }
}
