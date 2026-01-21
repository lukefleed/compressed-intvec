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
    builder::{VarVecBuilder, VarVecFromIterBuilder},
    codec::VariableCodecSpec,
    iter::{VarVecIntoIter, VarVecIter},
    reader::VarVecReader,
    seq_reader::VarVecSeqReader,
    slice::{VarVecSlice, VarVecSliceIter},
    traits::Storable as VariableStorable,
    BESVarVec, BEVarVec, LESVarVec, LEVarVec, SVarVec, UVarVec, VarVec, VarVecError,
};

// Deprecated Variable-Width Vector Aliases (backward compatibility)
#[allow(deprecated)]
pub use crate::variable::{
    BEIntVec, BESIntVec, IntVec, IntVecBuilder, IntVecError, IntVecFromIterBuilder, IntVecIntoIter,
    IntVecIter, IntVecReader, IntVecSeqReader, IntVecSlice, IntVecSliceIter, LEIntVec, LESIntVec,
    SIntVec, UIntVec,
};

// --- Sequence Vector Prelude ---
pub use crate::seq::{
    BESEqVec, BESeqVec, LESEqVec, LESeqVec, SSeqVec, SeqIter, SeqVec, SeqVecBuilder, SeqVecError,
    SeqVecFromIterBuilder, SeqVecIntoIter, SeqVecIter, SeqVecReader, SeqVecSlice, USeqVec,
};

// --- Macros Prelude ---
pub use crate::fixed_vec;
pub use crate::int_vec;
pub use crate::seq_vec;
pub use crate::sfixed_vec;
pub use crate::sint_vec;
pub use crate::sseq_vec;
