//! # `SIntVec` Builder
//!
//! This module provides [`SIntVecBuilder`], a builder for creating a compressed
//! signed integer vector, [`SIntVec`].
//!
//! ## Implementation
//!
//! The builder operates by creating an iterator that applies the ZigZag
//! transformation ([`ToNat`]) to each `i64` element on the fly. This stream of
//! `u64` values is then passed directly to the underlying iterator-based
//! builder of [`IntVec`], which handles the actual compression. This streaming
//! approach is highly memory-efficient as it avoids materializing the
//! intermediate unsigned vector.
//!
//! ## Limitations
//!
//! A key consequence of this design is that the builder **does not support
//! automatic parameter selection**. Because the data is processed as a stream,
//! the builder cannot perform a pre-analysis pass to determine optimal codec
//! parameters. Users must therefore provide a fully specified [`CodecSpec`].
//!
//! [`ToNat`]: dsi_bitstream::prelude::ToNat

use super::{IntVec, IntVecError, SIntVec};
use crate::codec_spec::CodecSpec;
use dsi_bitstream::prelude::{CodesWrite, Endianness, ToNat};
use std::marker::PhantomData;

/// A builder for creating an [`SIntVec`] from a slice of signed integers (`&[i64]`).
///
/// This builder handles the conversion from signed to unsigned integers by applying
/// the ZigZag transformation to each element on the fly. The resulting stream of
/// `u64` values is then passed to the underlying [`IntVec`] iterator-based builder
/// for compression.
///
/// # Limitations
///
/// **This builder does not support automatic parameter selection for codecs.**
/// Because the data is transformed and consumed as a stream, the builder cannot
/// analyze the entire dataset beforehand to determine optimal parameters. You must
/// provide a `CodecSpec` with fixed, pre-determined parameters.
///
/// Attempting to use `CodecSpec::Auto` or variants with `None` parameters (e.g.,
/// `CodecSpec::FixedLength { num_bits: None }`) will result in an
/// [`IntVecError::InvalidParameters`].
///
/// # Example
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[i64] = &[-10, 20, -30, -40, 50];
///
/// // We must specify a codec with fixed parameters.
/// let sintvec = LESIntVec::builder(data)
///     .codec(CodecSpec::Gamma)
///     .k(16)
///     .build()
///     .unwrap();
///
/// assert_eq!(sintvec.len(), 5);
/// assert_eq!(sintvec.get(2), Some(-30));
/// ```
pub struct SIntVecBuilder<'a, E: Endianness> {
    input: &'a [i64],
    k: usize,
    codec_spec: CodecSpec,
    _endian: PhantomData<E>,
}

impl<'a, E: Endianness> SIntVecBuilder<'a, E> {
    /// Creates a new builder from a slice of `i64`.
    ///
    /// By default, it is configured with a sampling rate (`k`) of 32 and uses
    /// Gamma coding as a safe, parameter-free default.
    pub fn new(input: &'a [i64]) -> Self {
        Self {
            input,
            k: 32,
            codec_spec: CodecSpec::Gamma, // A safe default
            _endian: PhantomData,
        }
    }

    /// Sets the sampling rate `k` for the underlying [`IntVec`].
    ///
    /// The sampling rate determines the trade-off between random access speed and
    /// memory overhead for variable-length codes. It must be greater than 0.
    /// This parameter is ignored when using `FixedLength` encoding.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the codec specification for compression.
    ///
    /// The provided `CodecSpec` must have fixed, user-specified parameters.
    /// Automatic parameter selection (e.g., `CodecSpec::Auto`) is not supported
    /// and will cause the `build` method to return an error.
    pub fn codec(mut self, codec_spec: CodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the `SIntVec` by transforming and compressing the input data.
    ///
    /// This method consumes the builder and returns a `Result`. The underlying logic
    /// uses the iterator-based builder of `IntVec` to avoid materializing the
    /// intermediate ZigZag-encoded `u64` vector in memory.
    ///
    /// # Errors
    /// This method will return an [`IntVecError`] if:
    /// - A `CodecSpec` requiring automatic parameter selection is used.
    /// - A value in the input data (after ZigZag encoding) is too large to fit
    ///   within the specified number of bits for `FixedLength` encoding.
    /// - The sampling rate `k` is set to `0` for a variable-length code.
    pub fn build(self) -> Result<SIntVec<E>, IntVecError>
    where
        for<'b> crate::intvec::IntVecBitWriter<E>:
            dsi_bitstream::prelude::BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Transform the signed integers to unsigned integers using ZigZag encoding.
        let unsigned_iter = self.input.iter().map(|&x| x.to_nat());

        // Use the iterator-based builder from IntVec to perform the actual encoding.
        // This avoids materializing the intermediate unsigned vector.
        let inner_intvec = IntVec::<E>::from_iter_builder(unsigned_iter)
            .k(self.k)
            .codec(self.codec_spec)
            .build()?;

        Ok(SIntVec {
            inner: inner_intvec,
        })
    }
}
