//! A prelude for `compressed-intvec`.
//!
//! This prelude is automatically imported when you use `use compressed-intvec::prelude::*;`.
//! It exports all the most common types and traits.

// --- Fixed-Width Vector Prelude ---
pub use crate::fixed::{
    builder::{FixedVecBuilder, FixedVecFromIterBuilder},
    iter::{FixedVecIntoIter, FixedVecIter},
    traits::{Storable, Word},
    BEFixedVec,
    BESFixedVec,
    BitWidth,
    Error as FixedVecError,
    FixedVec,
    LEFixedVec,
    LESFixedVec,
    SFixedVec,
    // Direct re-export of the most common aliases
    UFixedVec,
};

// TODO: Add prelude exports for `variable` module once it's refactored.
