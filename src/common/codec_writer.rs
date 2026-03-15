//! Internal hybrid codec dispatcher for variable-length integer codes on write path.
//!
//! This module provides [`CodecWriter`], a two-stage dispatcher that optimizes
//! writing to compressed bitstreams while maintaining compatibility with all
//! codecs supported by [`dsi-bitstream`](https://crates.io/crates/dsi-bitstream).
//!
//! The design prioritizes performance for common codecs while providing a robust
//! fallback for exotic parameter combinations, ensuring that any validly resolved
//! codec can be written without panicking.

use dsi_bitstream::{
    dispatch::{Codes, CodesWrite, FuncCodeWriter, StaticCodeWrite},
    prelude::Endianness,
};

/// A hybrid dispatcher for writing compression codes.
///
/// This enum acts as a two-stage dispatcher to write values to a compressed
/// bitstream. It is designed to handle all valid codecs supported by
/// [`dsi-bitstream`](https://crates.io/crates/dsi-bitstream), ensuring
/// correctness while maximizing performance.
///
/// # Dispatch Strategy
///
/// 1. **Fast Path ([`CodecWriter::Fast`])**: For common codecs (e.g., Gamma,
///    Delta, or Zeta with small parameters), [`dsi-bitstream`](https://crates.io/crates/dsi-bitstream)
///    provides pre-compiled function pointers via [`FuncCodeWriter`]. This path
///    avoids the overhead of a `match` statement on every write operation, as the
///    correct function is resolved once at creation time.
///
/// 2. **Slow Path ([`CodecWriter::Slow`])**: For codecs with parameters outside
///    of the pre-compiled set (e.g., `Golomb { b: 15 }`), the fast path is not
///    available. In this case, the dispatcher falls back to storing the [`Codes`]
///    enum variant directly. Each write operation then uses a `match` statement to
///    call the appropriate encoding function. While slightly slower due to the
///    runtime dispatch, this ensures that any validly resolved codec can be written.
///
/// This hybrid approach guarantees that the writer will never panic due to an
/// "unsupported code" error. The [`new`](Self::new) constructor automatically
/// selects the appropriate path.
///
/// # Type Parameters
///
/// - `E`: The endianness used for writing bits to the bitstream.
/// - `W`: The bitstream writer type implementing [`CodesWrite`].
pub(crate) enum CodecWriter<E: Endianness, W: CodesWrite<E>> {
    /// Fast-path writer using a pre-resolved function pointer.
    Fast(FuncCodeWriter<E, W>),
    /// Fallback writer using dynamic dispatch on the [`Codes`] enum.
    Slow(Codes),
}

impl<E: Endianness, W: CodesWrite<E>> CodecWriter<E, W> {
    /// Creates a new [`CodecWriter`], automatically selecting the fastest
    /// available dispatch path for the given codec.
    ///
    /// This constructor attempts to create a high-performance [`FuncCodeWriter`].
    /// If the codec is not supported by the fast path (e.g., it has uncommon
    /// parameters), it falls back to the dynamic dispatch mechanism.
    /// This method will not panic.
    ///
    /// # Arguments
    ///
    /// * `code` - The compression codec to use for writing.
    #[inline]
    pub(crate) fn new(code: Codes) -> Self {
        match FuncCodeWriter::new(code) {
            Ok(fast_writer) => Self::Fast(fast_writer),
            Err(_) => Self::Slow(code),
        }
    }
}

impl<E: Endianness, W: CodesWrite<E>> StaticCodeWrite<E, W> for CodecWriter<E, W> {
    /// Writes a single encoded value to the bitstream.
    ///
    /// This method dispatches to either the fast-path function pointer or the
    /// slow-path match statement, depending on which variant of [`CodecWriter`]
    /// was selected during construction.
    ///
    /// # Arguments
    ///
    /// * `writer` - A mutable reference to the bitstream writer.
    /// * `value` - The value to encode and write.
    ///
    /// # Returns
    ///
    /// The number of bits written, or an error if the write operation fails.
    #[inline]
    fn write(&self, writer: &mut W, value: u64) -> Result<usize, W::Error> {
        match self {
            // If we have a function pointer, call it directly. This is the fast path.
            Self::Fast(func_writer) => func_writer.write(writer, value),
            // Otherwise, use the slower dynamic dispatch. This is the fallback path.
            Self::Slow(code) => code.write(writer, value),
        }
    }
}
