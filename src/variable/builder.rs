//! The builder of [`VarVec`].

use super::VarVec;
use crate::common::{Automatic, Codec, CodecError, Encoder, ValidCodec, select};
use crate::{FixedVec, Storable};
use std::{fmt, marker::PhantomData, num::NonZeroUsize};

/// The default sampling rate.
const DEFAULT_SAMPLING_RATE: NonZeroUsize = NonZeroUsize::new(32).unwrap();

/// Configures the compression and random-access index of a [`VarVec`].
///
/// Returned by [`VarVec::builder`]. By default, `C` is [`Automatic`]: the builder selects a code
/// that can encode every input value, and `build` returns a vector directly. Calling
/// [`codec`](VarVecBuilder::codec) changes `C` to [`Codec`], and `build` then returns a
/// [`Result`] because the chosen code may have an invalid parameter or reject a value.
///
/// The sampling rate defaults to 32. Set a lower rate with
/// [`sampling_rate`](VarVecBuilder::sampling_rate) to speed up random access at the cost of a
/// larger index. See [`Codec`] for the automatic selection policy.
///
/// # Examples
///
/// ```
/// use compressed_intvec::{Codec, CodecError, VarVec};
///
/// // Automatic selection returns the vector directly.
/// let vec = VarVec::builder().build(&[1u64, 2, 3]);
/// assert_eq!(vec.get(2), Some(3));
///
/// // Gamma reports u64::MAX as unencodable.
/// let error = VarVec::builder().codec(Codec::Gamma).build(&[1, u64::MAX]).unwrap_err();
/// assert_eq!(error, CodecError::Unencodable { codec: Codec::Gamma, value: u64::MAX });
/// ```
#[derive(Clone, Copy)]
pub struct VarVecBuilder<T, C = Automatic> {
    codec: C,
    sampling_rate: NonZeroUsize,
    element: PhantomData<fn() -> T>,
}

/// Prints the codec and the sampling rate, whatever the element type.
impl<T, C: fmt::Debug> fmt::Debug for VarVecBuilder<T, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VarVecBuilder")
            .field("codec", &self.codec)
            .field("sampling_rate", &self.sampling_rate)
            .finish()
    }
}

impl<T> VarVecBuilder<T> {
    pub(crate) fn new() -> Self {
        Self {
            codec: Automatic,
            sampling_rate: DEFAULT_SAMPLING_RATE,
            element: PhantomData,
        }
    }
}

impl<T, C> VarVecBuilder<T, C> {
    /// Sets the number of elements between two samples (32 by default).
    ///
    /// A random access decodes at most this many codewords. A higher rate stores fewer bit
    /// positions and uses less index memory, but can require more decoding per access. Forward
    /// iteration decodes consecutive elements from the stream.
    pub fn sampling_rate(self, sampling_rate: NonZeroUsize) -> Self {
        Self {
            sampling_rate,
            ..self
        }
    }

    /// Selects a codec explicitly, making the `build` methods return a [`Result`].
    ///
    /// The codec is checked when the vector is built. See [`Codec`] for parameter ranges and
    /// representable values.
    pub fn codec(self, codec: Codec) -> VarVecBuilder<T, Codec> {
        VarVecBuilder {
            codec,
            sampling_rate: self.sampling_rate,
            element: PhantomData,
        }
    }
}

impl<T: Storable> VarVecBuilder<T> {
    /// Encodes `values` with an automatically selected codec.
    ///
    /// Selection considers only codecs that can encode every value, so this returns the vector
    /// directly. The vector owns the resulting encoded storage.
    pub fn build(self, values: &[T]) -> VarVec<T> {
        let codec = select(values.iter().map(|value| value.to_word()));
        encode(values.iter().copied(), codec, self.sampling_rate).unwrap_or_else(|_| {
            unreachable!("the automatic choice picks a valid code that encodes every value")
        })
    }

    /// Collects and encodes an iterator with an automatically selected codec.
    ///
    /// Buffers the input values, selects a codec that represents every value, then encodes them
    /// and returns the vector directly. With an explicit [`codec`](VarVecBuilder::codec), the
    /// builder encodes the iterator in one pass.
    pub fn build_from_iter(self, values: impl IntoIterator<Item = T>) -> VarVec<T> {
        self.build(&values.into_iter().collect::<Vec<T>>())
    }
}

impl<T: Storable> VarVecBuilder<T, Codec> {
    /// Encodes `values` with the selected codec.
    ///
    /// The vector owns the resulting encoded storage.
    ///
    /// # Errors
    ///
    /// - [`CodecError::InvalidParameter`] if the parameter of the code is out of range;
    /// - [`CodecError::Unencodable`] if the code cannot encode one of the values.
    pub fn build(self, values: &[T]) -> Result<VarVec<T>, CodecError<T>> {
        self.build_from_iter(values.iter().copied())
    }

    /// Encodes an iterator in one pass with the selected codec.
    ///
    /// Each value is written to the encoded stream as it is consumed.
    ///
    /// # Errors
    ///
    /// As [`build`](VarVecBuilder::build).
    pub fn build_from_iter(
        self,
        values: impl IntoIterator<Item = T>,
    ) -> Result<VarVec<T>, CodecError<T>> {
        let codec = self
            .codec
            .validate()
            .ok_or(CodecError::InvalidParameter(self.codec))?;
        encode(values, codec, self.sampling_rate)
    }
}

/// Encodes `values` with `codec`.
fn encode<T: Storable>(
    values: impl IntoIterator<Item = T>,
    codec: ValidCodec,
    sampling_rate: NonZeroUsize,
) -> Result<VarVec<T>, CodecError<T>> {
    let mut encoder = Encoder::new(codec);
    let mut samples = Vec::new();
    let mut len = 0;
    let mut until_sample = 0;
    for value in values {
        if until_sample == 0 {
            samples.push(encoder.bit_len());
            until_sample = sampling_rate.get();
        }
        until_sample -= 1;
        encoder
            .write(value.to_word())
            .map_err(|_| CodecError::Unencodable {
                codec: codec.get(),
                value,
            })?;
        len += 1;
    }
    // The stream and the samples are consistent by construction, which is what `from_parts`
    // would check.
    Ok(VarVec {
        data: encoder.finish(),
        samples: FixedVec::from_slice(&samples),
        sampling_rate,
        len,
        codec,
        element: PhantomData,
    })
}
