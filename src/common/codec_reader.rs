//! Internal hybrid codec dispatcher for variable-length integer codes.
//!
//! This module provides `CodecReader`, a two-stage dispatcher that optimizes
//! reading from compressed bitstreams while maintaining compatibility with all
//! codecs supported by [`dsi-bitstream`](https://crates.io/crates/dsi-bitstream).
//!
//! The design prioritizes performance for common codecs while providing a robust
//! fallback for exotic parameter combinations, ensuring that any validly created
//! compressed vector can be read without panicking.

use dsi_bitstream::{
    codes::params::DefaultReadParams,
    dispatch::{Codes, CodesRead, FuncCodeReader, StaticCodeRead},
    impls::{BufBitReader, MemWordReader},
    prelude::{BitRead, BitSeek, Endianness},
};

/// Type alias for bitstream readers operating on in-memory word buffers.
///
/// This reader type is used throughout the variable-length integer modules
/// for decoding compressed data. It is configured to:
/// - Read from memory ([`MemWordReader`])
/// - Use buffered bit-level access ([`BufBitReader`])
/// - Have infallible read operations (memory reads cannot fail)
pub(crate) type VarVecBitReader<'a, E> =
    BufBitReader<E, MemWordReader<u64, &'a [u64], true>, DefaultReadParams>;

/// A hybrid dispatcher for reading compression codes.
///
/// This enum acts as a two-stage dispatcher to read values from a compressed
/// bitstream. It is designed to handle all valid codecs supported by
/// [`dsi-bitstream`](https://crates.io/crates/dsi-bitstream), ensuring
/// correctness while maximizing performance.
///
/// # Dispatch Strategy
///
/// 1. **Fast Path ([`CodecReader::Fast`])**: For common codecs (e.g., Gamma,
///    Delta, or Zeta with small parameters), [`dsi-bitstream`](https://crates.io/crates/dsi-bitstream)
///    provides pre-compiled function pointers via [`FuncCodeReader`]. This path
///    avoids the overhead of a `match` statement on every read operation, as the
///    correct function is resolved once at creation time.
///
/// 2. **Slow Path ([`CodecReader::Slow`])**: For codecs with parameters outside
///    of the pre-compiled set (e.g., `Golomb { b: 15 }`), the fast path is not
///    available. In this case, the dispatcher falls back to storing the [`Codes`]
///    enum variant directly. Each read operation then uses a `match` statement to
///    call the appropriate decoding function. While slightly slower due to the
///    runtime dispatch, this ensures that any validly created compressed vector
///    can be read.
///
/// This hybrid approach guarantees that the reader will never panic due to an
/// "unsupported code" error, which was a critical issue in previous implementations.
/// The [`new`](Self::new) constructor automatically selects the appropriate path.
///
/// # Type Parameters
///
/// - `E`: The endianness used for reading bits from the bitstream.
pub(crate) enum CodecReader<'a, E: Endianness>
where
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Fast-path reader using a pre-resolved function pointer.
    Fast(FuncCodeReader<E, VarVecBitReader<'a, E>>),
    /// Fallback reader using dynamic dispatch on the [`Codes`] enum.
    Slow(Codes),
}

impl<E: Endianness> CodecReader<'_, E>
where
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new [`CodecReader`], automatically selecting the fastest
    /// available dispatch path for the given codec.
    ///
    /// This constructor attempts to create a high-performance [`FuncCodeReader`].
    /// If the codec is not supported by the fast path (e.g., it has uncommon
    /// parameters), it falls back to the dynamic dispatch mechanism.
    /// This method will not panic.
    ///
    /// # Arguments
    ///
    /// * `code` - The compression codec to use for reading.
    #[inline]
    pub(crate) fn new(code: Codes) -> Self {
        match FuncCodeReader::new(code) {
            Ok(fast_reader) => Self::Fast(fast_reader),
            Err(_) => Self::Slow(code),
        }
    }
}

impl<'a, E: Endianness> StaticCodeRead<E, VarVecBitReader<'a, E>> for CodecReader<'a, E>
where
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Reads a single encoded value from the bitstream.
    ///
    /// This method dispatches to either the fast-path function pointer or the
    /// slow-path match statement, depending on which variant of [`CodecReader`]
    /// was selected during construction.
    ///
    /// # Arguments
    ///
    /// * `reader` - A mutable reference to the bitstream reader.
    ///
    /// # Returns
    ///
    /// The decoded value as a `u64`. This operation is infallible for in-memory
    /// readers.
    #[inline]
    fn read(&self, reader: &mut VarVecBitReader<'a, E>) -> Result<u64, core::convert::Infallible> {
        match self {
            // If we have a function pointer, call it directly. This is the fast path.
            Self::Fast(func_reader) => func_reader.read(reader),
            // Otherwise, use the slower dynamic dispatch. This is the fallback path.
            Self::Slow(code) => code.read(reader),
        }
    }
}
