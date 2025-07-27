//! A prelude for `compressed-intvec`.
//!
//! This prelude is automatically imported when you use `use compressed_intvec::prelude::*;`.
//! It exports all the most common types and traits.

pub use crate::codec_spec::{CodecSpec, Encoding};
pub use crate::intvec::{
    BEIntVec, IntVec, IntVecBuilder, IntVecError, IntVecFromIterBuilder, IntVecIter, IntVecReader,
    LEIntVec,
};
pub use crate::sintvec::{BESIntVec, LESIntVec, SIntVec, SIntVecBuilder, SIntVecIter};
