//! Sequential iterator over the decompressed values of an `IntVec`.
//!
//! This module provides the `IntVecIter` struct, which allows for efficient,
//! forward-only iteration over the elements of a compressed `IntVec`,
//! decompressing them on the fly.

use super::{IntVec, IntVecBitReader};
use crate::codec_spec::Encoding;
use dsi_bitstream::{
    dispatch::{CodesRead, FuncCodeReader, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};

/// Helper enum to manage the decoding logic within the iterator.
///
/// This internal enum abstracts over the two possible decoding strategies:
/// either using a function-dispatched code reader for DSI-based encodings,
/// or reading a fixed number of bits for `FixedLength` encoding. This avoids
/// a `match` statement inside the `next()` method's hot loop for better performance.
enum IterLogic<'a, E: Endianness>
where
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible> + CodesRead<E>,
{
    Dsi(FuncCodeReader<E, IntVecBitReader<'a, E>>),
    Fixed { num_bits: usize },
}

/// An iterator over the decompressed `u64` values of an [`IntVec`].
///
/// This struct is created by the [`iter`](IntVec::iter) method on [`IntVec`].
/// It provides a sequential, forward-only scan over the compressed data,
/// decompressing values on the fly. It also implements [`ExactSizeIterator`],
/// allowing the user to know exactly how many items are remaining.
///
/// # Example
///
/// ```rust
/// use compressed_intvec::prelude::*;
///
/// let data: &[u64] = &[100, 200, 255, 0, 1];
///
/// let intvec = LEIntVec::builder(data)
///     .codec(CodecSpec::Gamma)
///     .build()
///     .unwrap();
///
/// // The iterator decompresses values as it is consumed.
/// for (index, value) in intvec.iter().enumerate() {
///     assert_eq!(value, data[index]);
/// }
/// ```
pub struct IntVecIter<'a, E>
where
    E: Endianness,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// The total number of elements in the vector.
    len: usize,
    /// The underlying bitstream reader used for decoding.
    reader: IntVecBitReader<'a, E>,
    /// The pre-configured decoding logic for this iterator instance.
    logic: IterLogic<'a, E>,
    /// The index of the next element to be returned.
    current_index: usize,
    /// A flag to track if the bitstream is still valid.
    /// It is set to `false` if a read error occurs, stopping the iteration.
    valid: bool,
}

impl<'a, E> IntVecIter<'a, E>
where
    E: Endianness,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new iterator for a given `IntVec`.
    ///
    /// This is `pub(super)` and is called by [`IntVec::iter`].
    pub(super) fn new(intvec: &'a IntVec<E>) -> Self {
        let reader = intvec.reader().reader;
        let logic = match intvec.encoding {
            Encoding::Dsi(code) => {
                // Pre-create the code reader to avoid recreating it in `next()`.
                // The `expect` is safe because all codes supported by `IntVec`
                // are also supported by `FuncCodeReader`.
                let code_reader = FuncCodeReader::new(code)
                    .expect("Failed to create code reader for DSI encoding.");
                IterLogic::Dsi(code_reader)
            }
            Encoding::Fixed { num_bits } => IterLogic::Fixed { num_bits },
        };

        Self {
            len: intvec.len,
            reader,
            logic,
            current_index: 0,
            valid: true,
        }
    }
}

impl<E> Iterator for IntVecIter<'_, E>
where
    E: Endianness,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = u64;

    /// Advances the iterator and returns the next decompressed value.
    ///
    /// Returns `None` when iteration is finished or if a decoding error
    /// occurs in the underlying bitstream.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if !self.valid || self.current_index >= self.len {
            return None;
        }

        // Decode the next value based on the pre-selected logic.
        let result = match &self.logic {
            IterLogic::Dsi(code_reader) => code_reader.read(&mut self.reader),
            IterLogic::Fixed { num_bits } => self.reader.read_bits(*num_bits),
        };

        match result {
            Ok(value) => {
                self.current_index += 1;
                Some(value)
            }
            Err(_) => {
                // The underlying bitstream reads are infallible due to the trait
                // bounds, so this arm should theoretically not be hit.
                // However, it provides robustness by invalidating the iterator
                // to stop further attempts if an error were to occur.
                self.valid = false;
                None
            }
        }
    }

    /// Returns a hint about the number of remaining items in the iterator.
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.valid {
            let remaining = self.len.saturating_sub(self.current_index);
            (remaining, Some(remaining))
        } else {
            (0, Some(0))
        }
    }
}

impl<E> std::iter::ExactSizeIterator for IntVecIter<'_, E>
where
    E: Endianness,
    for<'b> IntVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Returns the exact number of remaining items in the iterator.
    fn len(&self) -> usize {
        if self.valid {
            self.len.saturating_sub(self.current_index)
        } else {
            0
        }
    }
}
