// src/prelude.rs

//! A prelude for `compressed-intvec`.
//!
//! This prelude is automatically imported when you use `use compressed-intvec::prelude::*;`.
//! It exports all the most common types and traits.

// --- Fixed-Width Vector Prelude ---
pub use crate::fixed::{
    atomic::{AtomicFixedVec, SAtomicFixedVec, UAtomicFixedVec},
    builder::{FixedVecBuilder, FixedVecFromIterBuilder},
    iter::{FixedVecIntoIter, FixedVecIter},
    traits::{Storable as FixedStorable, Word},
    BEFixedVec, BESFixedVec, BitWidth, Error as FixedVecError, FixedVec, LEFixedVec, LESFixedVec,
    SFixedVec, UFixedVec,
};

// --- Variable-Width Vector Prelude ---
pub use crate::variable::{
    builder::{IntVecBuilder, IntVecFromIterBuilder},
    codec::VariableCodecSpec,
    reader::IntVecReader,
    seq_reader::IntVecSeqReader,
    slice::IntVecSlice,
    traits::Storable as VariableStorable,
    BEIntVec, BESIntVec, IntVec, IntVecError, LEIntVec, LESIntVec, SIntVec, UIntVec,
};

// --- Sequence Vector Prelude ---
pub use crate::seq::{
    BESeqVec, LESeqVec, SSeqVec, SeqIter, SeqVec, SeqVecBuilder, SeqVecError,
    SeqVecFromIterBuilder, SeqVecIntoIter, SeqVecIter, SeqVecReader, SeqVecSlice,
    USeqVec,
};

// --- Macros Prelude ---
pub use crate::fixed_vec;
pub use crate::int_vec;
pub use crate::seq_vec;
pub use crate::sfixed_vec;
pub use crate::sint_vec;
pub use crate::sseq_vec;
