//! A prelude for `compressed-intvec`.
//!
//! This prelude is automatically imported when you use `use compressed_intvec::prelude::*;`.
//! It exports all the most common types and traits.

// Common error types
pub use crate::fixed::intvec::{BitWidth, FixedVecError};
pub use crate::variable::intvec::IntVecError;

// Types for fixed-width vectors
pub use crate::fixed::intvec::{
    BEFixedVec, FixedVec, FixedVecBuilder, FixedVecFromIterBuilder, FixedVecIter, FixedVecSlice,
    LEFixedVec,
};
pub use crate::fixed::sintvec::{
    BESFixedVec, LESFixedVec, SFixedVec, SFixedVecBuilder, SFixedVecFromIterBuilder, SFixedVecIter,
    SFixedVecSlice,
};

// Types for variable-width vectors
pub use crate::variable::codec::VariableCodecSpec;
pub use crate::variable::intvec::{
    BEIntVec, IntVec, IntVecBuilder, IntVecFromIterBuilder, IntVecIter, IntVecReader,
    IntVecSeqReader, LEIntVec,
};
pub use crate::variable::sintvec::{BESIntVec, LESIntVec, SIntVec, SIntVecBuilder};
