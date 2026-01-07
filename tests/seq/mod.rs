//! Comprehensive integration tests for the [`SeqVec`] module.
//!
//! This test suite validates all functionality of [`SeqVec`], including:
//! - Core construction and query methods
//! - Iterator behavior (single sequences and all sequences)
//! - Builder patterns and codec selection
//! - Backend conversions (owned/borrowed)
//! - Codec-specific behavior and variations
//! - Edge cases and error conditions

mod test_backend;
mod test_codec;
mod test_construction;
mod test_iter;
mod test_reader;
mod test_seq;
mod test_seq_reader;
mod test_slice;
