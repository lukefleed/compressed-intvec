//! An immutable, zero-copy slice of an [`VarVec`].
//!
//! This module provides [`VarVecSlice`], a view into a contiguous portion of an
//! [`VarVec`]. Slices are immutable and do not own their data; they borrow it
//! from the parent `VarVec`.
//!
//! [`VarVec`]: crate::variable::VarVec

use super::{traits::Storable, VarVec, VarVecBitReader};
use crate::common::codec_reader::CodecReader;
use dsi_bitstream::{
    dispatch::{CodesRead, StaticCodeRead},
    prelude::{BitRead, BitSeek, Endianness},
};
use std::cmp::Ordering;
use std::fmt;
use std::iter::FusedIterator;
use std::ops::Range;

/// An immutable, zero-copy slice of an [`VarVec`].
///
/// This struct provides a view into a contiguous portion of an [`VarVec`]
/// without copying the underlying compressed data. It is created by the
/// [`slice`](super::VarVec::slice) or [`split_at`](super::VarVec::split_at)
/// methods on an [`VarVec`].
///
/// All operations on an [`VarVecSlice`] are relative to the start of the slice,
/// not the parent vector.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use compressed_intvec::variable::{VarVec, UVarVec};
///
/// let data: Vec<u32> = (0..100).collect();
/// let vec: UVarVec<u32> = VarVec::from_slice(&data)?;
///
/// // Create a slice of the elements from index 20 to 49
/// let slice = vec.slice(20, 30).expect("valid slice range");
///
/// assert_eq!(slice.len(), 30);
///
/// // Accessing an element of the slice
/// // Index 5 of the slice corresponds to index 25 of the original vector
/// assert_eq!(slice.get(5), Some(25));
///
/// // Iterating over the slice
/// let mut slice_sum = 0;
/// for value in slice.iter() {
///     slice_sum += value;
/// }
/// assert_eq!(slice_sum, (20..50).sum());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct VarVecSlice<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> {
    /// A reference to the parent vector.
    vec: &'a VarVec<T, E, B>,
    /// The starting index of the slice within the parent vector.
    start: usize,
    /// The number of elements in the slice.
    len: usize,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> VarVecSlice<'a, T, E, B> {
    /// Creates a new `VarVecSlice`.
    pub(super) fn new(vec: &'a VarVec<T, E, B>, range: Range<usize>) -> Self {
        Self {
            vec,
            start: range.start,
            len: range.len(),
        }
    }

    /// Returns the number of elements in the slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slice contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the element at the specified index within the slice, or `None` if
    /// the index is out of bounds.
    ///
    /// The index is relative to the start of the slice.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T>
    where
        for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        if index >= self.len {
            return None;
        }
        // The actual index into the parent vector is `self.start + index`.
        self.vec.get(self.start + index)
    }

    /// Returns the element at `index` within the slice without bounds checking.
    ///
    /// The index is relative to the start of the slice.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    /// The caller must ensure that `index < self.len()`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> T
    where
        for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        debug_assert!(index < self.len, "Index out of bounds");
        unsafe { self.vec.get_unchecked(self.start + index) }
    }

    /// Returns an iterator over the values in the slice.
    pub fn iter(&self) -> VarVecSliceIter<'_, T, E, B>
    where
        for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        VarVecSliceIter::new(self)
    }
}

impl<T, E, B> VarVecSlice<'_, T, E, B>
where
    T: Storable + Ord,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Binary searches this slice for a given element.
    ///
    /// If the value is found, returns `Ok(usize)` with the index of the
    /// matching element within the slice. If not found, returns `Err(usize)`
    /// with the insertion point to maintain order.
    pub fn binary_search(&self, value: &T) -> Result<usize, usize> {
        self.binary_search_by(|probe| probe.cmp(value))
    }

    /// Binary searches this slice with a custom comparison function.
    ///
    /// Uses a sequential reader (`VarVecSeqReader`) internally, which exploits
    /// locality when binary search probes converge to nearby positions within
    /// the same sample block.
    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> Ordering,
    {
        let mut low = 0;
        let mut high = self.len();
        let mut seq_reader = self.vec.seq_reader();

        while low < high {
            let mid = low + (high - low) / 2;
            // SAFETY: Bounds are checked by the loop invariants and the slice's
            // construction, so the index into the parent vector is always valid.
            let cmp = f(unsafe { seq_reader.get_unchecked(self.start + mid) });
            match cmp {
                Ordering::Less => low = mid + 1,
                Ordering::Equal => return Ok(mid),
                Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }

    /// Binary searches this slice with a key extraction function.
    #[inline]
    pub fn binary_search_by_key<K, F>(&self, b: &K, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(T) -> K,
        K: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }
}

/// An iterator over the decompressed values of an [`VarVecSlice`].
///
/// This struct is created by the [`iter`](VarVecSlice::iter) method.
///
/// This iterator maintains its own bitstream reader and codec dispatcher. It
/// seeks once to the sample point for the slice's start, skips forward to the
/// first element, and then decodes sequentially. This gives O(1) amortized cost
/// per `next()` call, making a full scan O(n) instead of O(n*k).
pub struct VarVecSliceIter<'a, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// The bitstream reader, positioned at the next element to decode.
    reader: VarVecBitReader<'a, E>,
    /// The codec dispatcher, resolved once at construction.
    code_reader: CodecReader<'a, E>,
    /// Total number of elements in the slice.
    len: usize,
    /// Number of elements already yielded.
    current_index: usize,
    /// Phantom marker for unused generic parameters.
    _markers: std::marker::PhantomData<(&'a B, T)>,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> VarVecSliceIter<'a, T, E, B>
where
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Creates a new sequential iterator for a given `VarVecSlice`.
    ///
    /// At construction, the iterator:
    /// 1. Seeks to the sample point for `slice.start`
    /// 2. Decodes forward to skip elements before the slice start
    /// 3. Is then positioned to decode elements sequentially via `next()`
    fn new(slice: &'a VarVecSlice<'a, T, E, B>) -> Self {
        let vec = slice.vec;
        let mut reader = VarVecBitReader::<E>::new(
            dsi_bitstream::impls::MemWordReader::new_inf(vec.data.as_ref()),
        );
        let code_reader = CodecReader::new(vec.encoding);

        if slice.len > 0 {
            // Compute the sample block for the slice's start index.
            let k = vec.k;
            let (sample_index, block_start) = if k.is_power_of_two() {
                let k_exp = k.trailing_zeros();
                let si = slice.start >> k_exp;
                (si, si << k_exp)
            } else {
                let si = slice.start / k;
                (si, si * k)
            };

            // SAFETY: `slice.start < vec.len` is guaranteed by VarVecSlice
            // construction, which implies `sample_index` is valid.
            let start_bit = unsafe { vec.samples.get_unchecked(sample_index) };
            reader.set_bit_pos(start_bit).unwrap();

            // Skip elements from the block start to the slice start.
            for _ in block_start..slice.start {
                code_reader.read(&mut reader).unwrap();
            }
        }

        Self {
            reader,
            code_reader,
            len: slice.len,
            current_index: 0,
            _markers: std::marker::PhantomData,
        }
    }
}

impl<T, E, B> Iterator for VarVecSliceIter<'_, T, E, B>
where
    T: Storable,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.len {
            return None;
        }
        let word = self.code_reader.read(&mut self.reader).unwrap();
        self.current_index += 1;
        Some(Storable::from_word(word))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len.saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }

    fn fold<Acc, F>(mut self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut acc = init;
        let remaining = self.len - self.current_index;
        for _ in 0..remaining {
            let word = self.code_reader.read(&mut self.reader).unwrap();
            acc = f(acc, T::from_word(word));
        }
        self.current_index = self.len;
        acc
    }
}

impl<T, E, B> ExactSizeIterator for VarVecSliceIter<'_, T, E, B>
where
    T: Storable,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn len(&self) -> usize {
        self.len.saturating_sub(self.current_index)
    }
}

impl<T, E, B> FusedIterator for VarVecSliceIter<'_, T, E, B>
where
    T: Storable,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
}

impl<T, E, B> fmt::Debug for VarVecSliceIter<'_, T, E, B>
where
    T: Storable,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> VarVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VarVecSliceIter")
            .field("remaining", &self.len.saturating_sub(self.current_index))
            .finish()
    }
}
