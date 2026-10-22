#![doc = include_str!("../README.md")]
#![warn(missing_docs, missing_debug_implementations)]

mod common;
mod fixed;
mod seq;
mod storable;
mod sync;
mod variable;

// The casts between `u32`, `usize` and `u64` that the crate calls lossless rely on this range.
const _: () = assert!(
    usize::BITS >= 32 && usize::BITS <= 64,
    "compressed-intvec supports targets whose usize has 32 to 64 bits"
);

pub use common::{Automatic, Codec, CodecError};

pub use fixed::{
    AtomicFixedVec, Backend, BitWidthTooLarge, FixedVec, FixedVecError, FixedVecIntoIter,
    FixedVecIter, FixedVecSlice, ValueTooLarge,
};
pub use seq::{SeqIter, SeqVec, SeqVecBuilder, SeqVecIter, SeqVecParts, SeqVecPartsError};
pub use storable::Storable;
pub use variable::{
    VarVec, VarVecBuilder, VarVecGetMany, VarVecIter, VarVecParts, VarVecPartsError,
};
