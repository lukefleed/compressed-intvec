//! Vectors whose elements are stored with a variable-length code.

#![forbid(unsafe_code)]

mod builder;
mod iter;
#[cfg(feature = "parallel")]
mod parallel;
#[cfg(feature = "serde")]
mod serde;

pub use builder::VarVecBuilder;
pub use iter::{VarVecGetMany, VarVecIter};

use crate::common::{CheckedDecoder, Codec, CodewordError, ValidCodec};
use crate::{Backend, FixedVec, FixedVecSlice, Storable};
use iter::Cursor;
use std::{
    cmp::Ordering,
    error::Error,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    num::NonZeroUsize,
};

/// An immutable integer vector with variable-length compression and indexed access.
///
/// `VarVec` suits collections with many small values and occasional large values. Each integer is
/// stored as a variable-length codeword whose length depends on its value, using the encoders
/// and decoders from [dsi-bitstream](https://docs.rs/dsi-bitstream/). Signed integers use zig-zag
/// encoding, so values close to zero have small encoded words.
///
/// [`from_slice`](VarVec::from_slice), [`collect`](Iterator::collect) and conversion from a
/// slice select a [`Codec`] automatically and use a sampling rate of 32. The
/// [`builder`](VarVec::builder) configures either choice.
///
/// The sampling rate controls the index: the vector records a bit position every
/// [`sampling_rate`](VarVec::sampling_rate) elements. [`get`](VarVec::get) uses this index and
/// decodes at most `sampling_rate` codewords. A higher sampling rate reduces index memory and
/// can increase access time. Forward [`iteration`](VarVec::iter) decodes each element once
/// and uses constant auxiliary space.
///
/// `T` implements [`Storable`]. The storage backend `B` holds both the encoded stream and its
/// samples. The default `Vec<u64>` owns growable allocations. Conversion to
/// `VarVec<T, Box<[u64]>>` removes spare capacity, and `&[u64]` provides a borrowed view built with
/// [`as_borrowed`](VarVec::as_borrowed) for an existing vector or
/// [`from_parts`](VarVec::from_parts) for external buffers.
///
/// With the `serde` feature, the serialized form holds the stream up to its last codeword, and
/// deserialization checks the format version, the element signedness and every codeword, and
/// rebuilds the samples in the same pass. Each decoded word must fit the target element type,
/// so a vector can be loaded into a narrower integer type when all of its words fit. Every 1.x
/// release loads the vectors that it and earlier 1.x releases serialized.
/// Equality and hashing depend on the decoded elements, regardless of the codec, sampling
/// rate or storage backend.
///
/// # Examples
///
/// ```
/// use compressed_intvec::VarVec;
///
/// let values: Vec<u64> = (0..1000).map(|x| x % 10).collect();
/// let vec = VarVec::from_slice(&values);
/// assert_eq!(vec.len(), 1000);
/// assert_eq!(vec.get(13), Some(3));
/// assert_eq!(vec.iter().sum::<u64>(), 4500);
/// assert_eq!(vec, values[..]);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize, mem_dbg::MemDbg))]
pub struct VarVec<T, B = Vec<u64>> {
    // Invariants, established by `from_parts` (and by the builder, which encodes the stream
    // itself) and never changed afterwards:
    // - `samples.len() == len.div_ceil(sampling_rate)`;
    // - decoding `len` codewords from bit 0 of `data` succeeds, and every one of them encodes a
    //   word (so dsi-bitstream's decoders read them without overflow), element
    //   `j * sampling_rate` starts at bit `samples[j]`, and every decoded word is below
    //   `2^T::BITS` (for the last property, the builder relies on the contract of
    //   `Storable::to_word`).
    // No unsafe code relies on them: a broken invariant makes reads panic or return wrong
    // values.
    data: B,
    samples: FixedVec<u64, B>,
    sampling_rate: NonZeroUsize,
    len: usize,
    codec: ValidCodec,
    element: PhantomData<fn() -> T>,
}

/// Encoded storage and metadata for reconstructing a [`VarVec`].
///
/// [`VarVec::into_parts`] transfers ownership of these components.
/// [`VarVec::from_parts`] validates the codec, sample positions and the first `len` codewords
/// before constructing a vector. Borrowed buffers produce a view sharing their storage.
///
/// Element `i` is the `i`-th codeword of `codec` in `data`, and the first codeword starts at bit
/// 0. Bit `k` of the stream is bit `k % 8` of byte `k / 8` of `data` in memory, which on
/// little-endian targets is bit `k % 64` of word `k / 64`. `samples[j]` is the bit position of
/// element `j * sampling_rate`. There must be `len.div_ceil(sampling_rate)` samples. Bits and
/// words after the last element are permitted and retained, including when `len` is zero;
/// serialization, with the `serde` feature, leaves them out.
pub struct VarVecParts<B> {
    /// The encoded stream, starting at bit 0. May include trailing bits or words.
    pub data: B,
    /// The bit position of every `sampling_rate`-th element, starting with element 0.
    pub samples: FixedVec<u64, B>,
    /// The number of elements between two samples.
    pub sampling_rate: NonZeroUsize,
    /// The number of elements.
    pub len: usize,
    /// The codec used for every element.
    pub codec: Codec,
}

impl<B: Clone> Clone for VarVecParts<B> {
    fn clone(&self) -> Self {
        let Self {
            data,
            samples,
            sampling_rate,
            len,
            codec,
        } = self;
        Self {
            data: data.clone(),
            samples: samples.clone(),
            sampling_rate: *sampling_rate,
            len: *len,
            codec: *codec,
        }
    }
}

impl<B: Backend> fmt::Debug for VarVecParts<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            data,
            samples,
            sampling_rate,
            len,
            codec,
        } = self;
        f.debug_struct("VarVecParts")
            .field("data", &data.as_ref())
            .field("samples", samples)
            .field("sampling_rate", sampling_rate)
            .field("len", len)
            .field("codec", codec)
            .finish()
    }
}

/// The parts of a [`VarVec`] are inconsistent.
///
/// Returned by [`VarVec::from_parts`]. Deserialization, with the `serde` feature, reports it in
/// the message of its error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VarVecPartsError {
    /// The parameter of the code is outside the range that its [`Codec`] variant documents.
    InvalidParameter(Codec),
    /// The number of samples is not `len.div_ceil(sampling_rate)`.
    SampleCount {
        /// The number of samples that `len` and `sampling_rate` require.
        expected: usize,
        /// The number of samples in the parts.
        found: usize,
    },
    /// A sample is not the bit position of its element.
    SampleMismatch {
        /// The index of the sample.
        sample: usize,
        /// The bit position of element `sample * sampling_rate` in the stream.
        expected: u64,
        /// The stored sample.
        found: u64,
    },
    /// The stream ends before the codeword of an element.
    Truncated {
        /// The index of the element.
        index: usize,
    },
    /// The codeword of an element encodes no word: its value does not fit in a `u64` (see
    /// [`Codec`] for the codes that cannot encode `u64::MAX`).
    InvalidCodeword {
        /// The index of the element.
        index: usize,
    },
    /// A decoded word does not fit the element type: it is not below `2^T::BITS`.
    WordOutOfRange {
        /// The index of the element.
        index: usize,
        /// The decoded word.
        word: u64,
    },
}

impl fmt::Display for VarVecPartsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidParameter(codec) => {
                write!(f, "the parameter of {codec:?} is out of range")
            }
            Self::SampleCount { expected, found } => {
                write!(
                    f,
                    "{found} samples instead of the {expected} that the length requires"
                )
            }
            Self::SampleMismatch {
                sample,
                expected,
                found,
            } => write!(
                f,
                "sample {sample} is {found}, but its element starts at bit {expected}"
            ),
            Self::Truncated { index } => write!(f, "the stream ends before element {index}"),
            Self::InvalidCodeword { index } => {
                write!(f, "the codeword of element {index} encodes no 64-bit word")
            }
            Self::WordOutOfRange { index, word } => {
                write!(
                    f,
                    "word {word} of element {index} does not fit the element type"
                )
            }
        }
    }
}

impl Error for VarVecPartsError {}

/// A borrowed representation shared by the read operations of all storage backends.
#[derive(Clone, Copy)]
pub(crate) struct View<'a> {
    data: &'a [u64],
    samples: FixedVecSlice<'a, u64>,
    sampling_rate: usize,
    len: usize,
    codec: ValidCodec,
}

impl<T: Storable> VarVec<T> {
    /// Encodes `values` with an automatically selected codec and a sampling rate of 32.
    ///
    /// The vector owns the resulting encoded storage. Use
    /// [`builder`](VarVec::builder) to configure the codec or sampling rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::VarVec;
    ///
    /// let vec = VarVec::from_slice(&[3i32, -1, 4, -1, 5]);
    /// assert_eq!(vec.get(1), Some(-1));
    /// assert_eq!(vec.iter().collect::<Vec<_>>(), [3, -1, 4, -1, 5]);
    /// ```
    pub fn from_slice(values: &[T]) -> Self {
        Self::builder().build(values)
    }

    /// Returns a builder for configuring the codec and sampling rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::{Codec, VarVec};
    /// use std::num::NonZeroUsize;
    ///
    /// let values: Vec<u32> = (0..100).collect();
    /// let vec = VarVec::builder()
    ///     .codec(Codec::Delta)
    ///     .sampling_rate(NonZeroUsize::new(8).unwrap())
    ///     .build(&values)?;
    /// assert_eq!(vec.get(99), Some(99));
    /// # Ok::<(), compressed_intvec::CodecError<u32>>(())
    /// ```
    pub fn builder() -> VarVecBuilder<T> {
        VarVecBuilder::new()
    }
}

impl<T, B> VarVec<T, B> {
    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements between consecutive index samples.
    #[inline]
    pub fn sampling_rate(&self) -> NonZeroUsize {
        self.sampling_rate
    }

    /// Returns the codec used to encode the elements.
    #[inline]
    pub fn codec(&self) -> Codec {
        self.codec.get()
    }

    /// Consumes the vector and returns its encoded storage and metadata.
    ///
    /// Transfers ownership of the stored buffers. [`from_parts`](VarVec::from_parts) can
    /// reconstruct the vector, validating the parts before accepting them.
    pub fn into_parts(self) -> VarVecParts<B> {
        let Self {
            data,
            samples,
            sampling_rate,
            len,
            codec,
            element: _,
        } = self;
        VarVecParts {
            data,
            samples,
            sampling_rate,
            len,
            codec: codec.get(),
        }
    }
}

impl<T, B: Backend> VarVec<T, B> {
    /// Returns a view borrowing both the encoded stream and the samples.
    ///
    /// The view shares `self`'s buffers as slices and provides the same read operations as the
    /// owned vector. Creating it takes constant time and performs no allocation or decoding.
    ///
    /// ```
    /// use compressed_intvec::VarVec;
    ///
    /// let owned = VarVec::from_slice(&[1u64, 2, 3]);
    /// let view = owned.as_borrowed();
    /// assert_eq!(view.get(1), Some(2));
    /// let boxed: VarVec<u64, Box<[u64]>> = owned.into();
    /// assert_eq!(boxed.get(2), Some(3));
    /// ```
    #[inline]
    pub fn as_borrowed(&self) -> VarVec<T, &[u64]> {
        VarVec {
            data: self.data.as_ref(),
            samples: self.samples.as_borrowed(),
            sampling_rate: self.sampling_rate,
            len: self.len,
            codec: self.codec,
            element: PhantomData,
        }
    }
}

/// Converts the owned stream and samples to boxed storage, preserving their words and metadata.
///
/// Removes spare capacity from both allocations. Boxing may reallocate their buffers.
impl<T> From<VarVec<T>> for VarVec<T, Box<[u64]>> {
    fn from(vec: VarVec<T>) -> Self {
        let VarVec {
            data,
            samples,
            sampling_rate,
            len,
            codec,
            element,
        } = vec;
        Self {
            data: data.into_boxed_slice(),
            samples: samples.into(),
            sampling_rate,
            len,
            codec,
            element,
        }
    }
}

impl<T: Storable> VarVec<T> {
    /// Validates the first `len` codewords of `data` as [`from_parts`](VarVec::from_parts)
    /// does, and builds the samples in the same decode. Returns the vector and the bit position
    /// after its last codeword.
    #[cfg(feature = "serde")]
    fn from_stream(
        data: Vec<u64>,
        len: usize,
        sampling_rate: NonZeroUsize,
        codec: Codec,
    ) -> Result<(Self, u64), VarVecPartsError> {
        let codec = codec
            .validate()
            .ok_or(VarVecPartsError::InvalidParameter(codec))?;
        // Grows with the decoded codewords, each of at least one bit, so a large `len` that the
        // stream does not hold allocates nothing.
        let mut samples = Vec::new();
        let end = check_stream(&data, len, sampling_rate, codec, T::BITS, |_, position| {
            samples.push(position);
            Ok(())
        })?;
        let vec = Self {
            data,
            samples: FixedVec::from_slice(&samples),
            sampling_rate,
            len,
            codec,
            element: PhantomData,
        };
        Ok((vec, end))
    }
}

impl<T: Storable, B: Backend> VarVec<T, B> {
    /// Validates encoded storage and metadata, then takes ownership of the parts.
    ///
    /// Checks the codec parameter, the number and positions of the samples, and each of the
    /// first `len` codewords. Every decoded word must fit in [`T::BITS`](Storable::BITS).
    /// Validation takes time linear in the bits consumed, plus `len`, with constant auxiliary
    /// space. The vector retains trailing bits and words after the last declared element.
    ///
    /// # Errors
    ///
    /// Returns [`VarVecPartsError`] if the codec parameter, samples, codewords or decoded
    /// words are invalid, as described in [`VarVecParts`].
    ///
    /// # Examples
    ///
    /// Reconstructing a view that shares external buffers. [`as_borrowed`](VarVec::as_borrowed)
    /// creates a view of an existing vector in constant time.
    ///
    /// ```
    /// use compressed_intvec::{FixedVec, VarVec, VarVecParts};
    ///
    /// let vec = VarVec::from_slice(&[5u32, 8, 13, 21]);
    /// let parts = vec.clone().into_parts();
    /// let (sample_words, sample_count, sample_width) = parts.samples.into_parts();
    /// let view = VarVec::<u32, &[u64]>::from_parts(VarVecParts {
    ///     data: &parts.data[..],
    ///     samples: FixedVec::from_parts(&sample_words[..], sample_count, sample_width)?,
    ///     sampling_rate: parts.sampling_rate,
    ///     len: parts.len,
    ///     codec: parts.codec,
    /// })?;
    /// assert_eq!(view, vec);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_parts(parts: VarVecParts<B>) -> Result<Self, VarVecPartsError> {
        let VarVecParts {
            data,
            samples,
            sampling_rate,
            len,
            codec,
        } = parts;
        let codec = codec
            .validate()
            .ok_or(VarVecPartsError::InvalidParameter(codec))?;
        let expected = len.div_ceil(sampling_rate.get());
        if samples.len() != expected {
            return Err(VarVecPartsError::SampleCount {
                expected,
                found: samples.len(),
            });
        }
        let stored = samples.as_slice();
        check_stream(
            data.as_ref(),
            len,
            sampling_rate,
            codec,
            T::BITS,
            |sample, expected| {
                let found = stored.get(sample).expect("the sample count was checked");
                if found == expected {
                    Ok(())
                } else {
                    Err(VarVecPartsError::SampleMismatch {
                        sample,
                        expected,
                        found,
                    })
                }
            },
        )?;
        Ok(Self {
            data,
            samples,
            sampling_rate,
            len,
            codec,
            element: PhantomData,
        })
    }

    #[inline]
    fn view(&self) -> View<'_> {
        View {
            data: self.data.as_ref(),
            samples: self.samples.as_slice(),
            sampling_rate: self.sampling_rate.get(),
            len: self.len,
            codec: self.codec,
        }
    }

    /// Returns the element at `index`, or `None` if `index >= len`.
    ///
    /// Seeks to the nearest sample at or before the element and decodes at most
    /// [`sampling_rate`](VarVec::sampling_rate) codewords. For consecutive elements, prefer
    /// [`iter`](VarVec::iter). For several indices, [`get_many`](VarVec::get_many) can reuse its
    /// decoding position.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        (index < self.len).then(|| T::from_word(Cursor::new(self.view()).read_at(index)))
    }

    /// Returns an iterator yielding elements by value.
    ///
    /// Forward iteration decodes each element once and uses constant auxiliary space. Skipping
    /// with [`nth`](Iterator::nth) or [`skip`](Iterator::skip) uses the sampled index, requiring at
    /// most `sampling_rate` decodes to reach the target. A range read with
    /// `vec.iter().skip(a).take(n)` needs at most `sampling_rate + n` decodes.
    /// Reverse iteration allocates a cache of up to `sampling_rate` decoded words. See
    /// [`VarVecIter`] for the cost of changing direction.
    #[inline]
    pub fn iter(&self) -> VarVecIter<'_, T> {
        VarVecIter::new(self.view(), 0, self.len)
    }

    /// Returns a lazy iterator yielding one [`Option<T>`] for each input index, in input order.
    ///
    /// Each valid index yields `Some(element)` and each index at or beyond [`len`](VarVec::len)
    /// yields `None`. Advancing the iterator decodes the requested element. Consecutive reads
    /// within a sample block reuse the decoding position. Backward reads and reads in a later
    /// block seek to that block's sample. Strictly increasing indices decode each element at
    /// most once, and repeated indices remain in the output.
    ///
    /// The decoding cursor uses constant auxiliary space. Extend an existing buffer from the
    /// iterator to reuse that buffer across calls.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::VarVec;
    ///
    /// let vec: VarVec<u32> = (0..100).collect();
    /// let mut out = Vec::new();
    /// out.extend(vec.get_many([3, 4, 50, 7]).flatten());
    /// assert_eq!(out, [3, 4, 50, 7]);
    /// assert_eq!(vec.get_many([1, 100]).collect::<Vec<_>>(), [Some(1), None]);
    /// ```
    pub fn get_many<I: IntoIterator<Item = usize>>(
        &self,
        indices: I,
    ) -> VarVecGetMany<'_, T, I::IntoIter> {
        VarVecGetMany::new(self.view(), indices.into_iter())
    }

    /// Searches a sorted vector for `value`.
    ///
    /// Returns `Ok(index)` for the first matching element, or `Err(index)` for the insertion
    /// position if no element matches. The vector must be sorted in ascending order.
    /// See [`binary_search_by`](VarVec::binary_search_by) for the decoding cost.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::VarVec;
    ///
    /// let vec = VarVec::from_slice(&[1u32, 3, 3, 7]);
    /// assert_eq!(vec.binary_search(&3), Ok(1));
    /// assert_eq!(vec.binary_search(&5), Err(3));
    /// ```
    pub fn binary_search(&self, value: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|element| element.cmp(value))
    }

    /// Searches a sorted vector with a comparator.
    ///
    /// `compare` must return [`Ordering::Less`] for elements before the target,
    /// [`Ordering::Equal`] for matches, and [`Ordering::Greater`] for elements after it, in that
    /// order. Returns the first matching index, or the insertion position as `Err(index)`.
    ///
    /// Searches the elements at the samples, then scans one sample block. This requires
    /// `O(log(sample_count + 1) + sampling_rate)` codeword decodes.
    pub fn binary_search_by(
        &self,
        mut compare: impl FnMut(&T) -> Ordering,
    ) -> Result<usize, usize> {
        let view = self.view();
        let k = view.sampling_rate;
        let mut cursor = Cursor::new(view);
        // The first block whose first element is not less than the target; the target is in the
        // block before it, or it is the first element of this block.
        let mut low = 0;
        let mut high = view.samples.len();
        while low < high {
            let mid = low + (high - low) / 2;
            if compare(&T::from_word(cursor.read_at(mid * k))) == Ordering::Less {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        let start = low.saturating_sub(1) * k;
        let end = sample_block_end(low, k, view.len);
        for index in start..end {
            match compare(&T::from_word(cursor.read_at(index))) {
                Ordering::Less => {}
                Ordering::Equal => return Ok(index),
                Ordering::Greater => return Err(index),
            }
        }
        if end < view.len && compare(&T::from_word(cursor.read_at(end))) == Ordering::Equal {
            return Ok(end);
        }
        Err(end)
    }

    /// Searches a vector sorted by the key returned by `extract`.
    ///
    /// Returns the first matching index, or the insertion position as `Err(index)`. The
    /// extracted keys must be in ascending order. See
    /// [`binary_search_by`](VarVec::binary_search_by) for the decoding cost.
    pub fn binary_search_by_key<K: Ord>(
        &self,
        key: &K,
        mut extract: impl FnMut(T) -> K,
    ) -> Result<usize, usize> {
        self.binary_search_by(|&element| extract(element).cmp(key))
    }
}

/// Returns the element boundary of `sample`, clamped to the vector length.
fn sample_block_end(sample: usize, sampling_rate: usize, len: usize) -> usize {
    // The rounded boundary after the last sample can exceed usize even when len fits.
    sample.saturating_mul(sampling_rate).min(len)
}

/// Decodes the first `len` codewords of `data` with the checked decoder, and checks that their
/// words fit `element_bits` bits. Passes the index and the bit position of every
/// `sampling_rate`-th element to `sample`, and returns the bit position after the last codeword.
fn check_stream(
    data: &[u64],
    len: usize,
    sampling_rate: NonZeroUsize,
    codec: ValidCodec,
    element_bits: u32,
    mut sample: impl FnMut(usize, u64) -> Result<(), VarVecPartsError>,
) -> Result<u64, VarVecPartsError> {
    let mut decoder = CheckedDecoder::new(data, codec);
    let mut until_sample = 0;
    for index in 0..len {
        if until_sample == 0 {
            sample(index / sampling_rate.get(), decoder.bit_pos())?;
            until_sample = sampling_rate.get();
        }
        until_sample -= 1;
        let word = decoder.try_read().map_err(|error| match error {
            CodewordError::Truncated => VarVecPartsError::Truncated { index },
            CodewordError::Invalid => VarVecPartsError::InvalidCodeword { index },
        })?;
        if element_bits < u64::BITS && word >> element_bits != 0 {
            return Err(VarVecPartsError::WordOutOfRange { index, word });
        }
    }
    Ok(decoder.bit_pos())
}

impl<T: Storable + fmt::Debug, B: Backend> fmt::Debug for VarVec<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Compares decoded elements across storage backends.
impl<T: Storable + PartialEq, B: Backend, B2: Backend> PartialEq<VarVec<T, B2>> for VarVec<T, B> {
    fn eq(&self, other: &VarVec<T, B2>) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<T: Storable + Eq, B: Backend> Eq for VarVec<T, B> {}

impl<T: Storable + PartialEq, B: Backend> PartialEq<[T]> for VarVec<T, B> {
    fn eq(&self, other: &[T]) -> bool {
        self.len == other.len() && self.iter().eq(other.iter().copied())
    }
}

/// Hashes the elements, consistently with [`PartialEq`].
impl<T: Storable + Hash, B: Backend> Hash for VarVec<T, B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len);
        for element in self.iter() {
            element.hash(state);
        }
    }
}

/// An empty vector.
impl<T: Storable> Default for VarVec<T> {
    fn default() -> Self {
        Self::from_slice(&[])
    }
}

impl<T: Storable> From<&[T]> for VarVec<T> {
    /// Equivalent to [`VarVec::from_slice`].
    fn from(values: &[T]) -> Self {
        Self::from_slice(values)
    }
}

impl<T: Storable> FromIterator<T> for VarVec<T> {
    /// Buffers the values, then encodes them with an automatically selected codec and a
    /// sampling rate of 32.
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        Self::builder().build_from_iter(values)
    }
}

impl<'a, T: Storable, B: Backend> IntoIterator for &'a VarVec<T, B> {
    type Item = T;
    type IntoIter = VarVecIter<'a, T>;

    fn into_iter(self) -> VarVecIter<'a, T> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::sample_block_end;

    #[test]
    fn sample_block_end_matches_wide_arithmetic() {
        for len in 0usize..=64 {
            for rate in 1usize..=64 {
                for sample in 0..=len.div_ceil(rate) {
                    let expected = ((sample as u128) * (rate as u128)).min(len as u128);
                    assert_eq!(sample_block_end(sample, rate, len) as u128, expected);
                }
            }
        }
    }

    #[test]
    fn sample_block_end_clamps_overflowing_boundary() {
        for len in [usize::MAX / 2 + 2, usize::MAX - 1, usize::MAX] {
            for rate in [1, 2, 3, usize::MAX / 2 + 1, usize::MAX] {
                let count = len.div_ceil(rate);
                for sample in [0, count - 1, count] {
                    let expected = ((sample as u128) * (rate as u128)).min(len as u128);
                    assert_eq!(
                        sample_block_end(sample, rate, len) as u128,
                        expected,
                        "sample {sample}, rate {rate}, len {len}"
                    );
                }
            }
        }
    }
}
