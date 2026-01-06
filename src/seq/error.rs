//! Error types for the [`SeqVec`] module.
//!
//! [`SeqVec`]: crate::seq::SeqVec

use crate::fixed::error::Error as FixedVecError;
use std::error::Error;
use std::fmt;

/// The error type for operations on [`SeqVec`].
///
/// [`SeqVec`]: crate::seq::SeqVec
#[derive(Debug)]
pub enum SeqVecError {
    /// An error occurred during an I/O operation, typically from the underlying
    /// bitstream reader or writer.
    Io(std::io::Error),
    /// A generic error from the [`dsi-bitstream`] library, often related to
    /// decoding malformed data.
    ///
    /// [`dsi-bitstream`]: https://crates.io/crates/dsi-bitstream
    Bitstream(Box<dyn Error + Send + Sync>),
    /// An error indicating that one or more parameters provided to a constructor
    /// or builder are invalid.
    InvalidParameters(String),
    /// An error that occurs during codec function dispatch.
    CodecDispatch(String),
    /// An error indicating that a provided sequence index is outside the valid
    /// bounds of the vector.
    IndexOutOfBounds {
        /// The invalid index that was provided.
        index: usize,
        /// The number of sequences in the vector.
        num_sequences: usize,
    },
}

impl fmt::Display for SeqVecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeqVecError::Io(e) => write!(f, "I/O error: {}", e),
            SeqVecError::Bitstream(e) => write!(f, "Bitstream error: {}", e),
            SeqVecError::InvalidParameters(s) => write!(f, "Invalid parameters: {}", s),
            SeqVecError::CodecDispatch(s) => write!(f, "Codec dispatch error: {}", s),
            SeqVecError::IndexOutOfBounds {
                index,
                num_sequences,
            } => {
                write!(
                    f,
                    "Sequence index {} out of bounds for SeqVec with {} sequences",
                    index, num_sequences
                )
            }
        }
    }
}

impl Error for SeqVecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            SeqVecError::Io(e) => Some(e),
            SeqVecError::Bitstream(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SeqVecError {
    fn from(e: std::io::Error) -> Self {
        SeqVecError::Io(e)
    }
}

impl From<core::convert::Infallible> for SeqVecError {
    fn from(_: core::convert::Infallible) -> Self {
        unreachable!()
    }
}

impl From<FixedVecError> for SeqVecError {
    fn from(e: FixedVecError) -> Self {
        SeqVecError::InvalidParameters(e.to_string())
    }
}
