//! Zero-copy views over a subset of sequences in a [`SeqVec`].
//!
//! This module provides [`SeqVecSlice`], an immutable view that represents a
//! contiguous range of sequences within a [`SeqVec`] without copying any data.
//!
//! [`SeqVec`]: crate::seq::SeqVec

use super::iter::{SeqIter, SeqVecBitReader};
use super::SeqVec;
use crate::variable::traits::Storable;
use dsi_bitstream::{
    dispatch::CodesRead,
    prelude::{BitRead, BitSeek, Endianness},
};
use std::marker::PhantomData;

/// An immutable, zero-copy view over a contiguous range of sequences in a [`SeqVec`].
///
/// A slice does not own any data; it simply maintains a reference to the
/// parent [`SeqVec`] along with an offset and length. All operations on the
/// slice are translated to operations on the underlying data.
///
/// # Index Translation
///
/// When accessing a sequence via a slice, the index is translated:
/// - `slice.get(i)` accesses the sequence at index `start + i` in the parent.
///
/// # Examples
///
/// ```ignore
/// use compressed_intvec::seq::{SeqVec, LESeqVec};
///
/// let sequences: &[&[u32]] = &[&[0], &[1], &[2], &[3], &[4]];
/// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
///
/// // Create a slice of sequences 1, 2, 3
/// let slice = vec.slice(1, 3).unwrap();
///
/// assert_eq!(slice.num_sequences(), 3);
/// assert_eq!(slice.get_vec(0), Some(vec![1])); // Original index 1
/// assert_eq!(slice.get_vec(1), Some(vec![2])); // Original index 2
/// assert_eq!(slice.get_vec(2), Some(vec![3])); // Original index 3
/// ```
///
/// [`SeqVec`]: crate::seq::SeqVec
pub struct SeqVecSlice<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> {
    /// Reference to the parent SeqVec.
    parent: &'a SeqVec<T, E, B>,
    /// The starting sequence index in the parent.
    start: usize,
    /// The number of sequences in this slice.
    len: usize,
    /// Marker for the element type.
    _marker: PhantomData<(T, E)>,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVecSlice<'a, T, E, B> {
    /// Creates a new slice over the specified range of sequences.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `start + len > parent.num_sequences()`.
    #[inline]
    pub(crate) fn new(parent: &'a SeqVec<T, E, B>, start: usize, len: usize) -> Self {
        debug_assert!(
            start + len <= parent.num_sequences(),
            "slice range {}..{} out of bounds for {} sequences",
            start,
            start + len,
            parent.num_sequences()
        );

        Self {
            parent,
            start,
            len,
            _marker: PhantomData,
        }
    }

    /// Returns the number of sequences in this slice.
    #[inline]
    pub fn num_sequences(&self) -> usize {
        self.len
    }

    /// Returns `true` if this slice contains no sequences.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the starting index of this slice in the parent [`SeqVec`].
    #[inline]
    pub fn start_index(&self) -> usize {
        self.start
    }

    /// Returns a reference to the parent [`SeqVec`].
    #[inline]
    pub fn parent(&self) -> &'a SeqVec<T, E, B> {
        self.parent
    }

    /// Translates a local index to the parent's index space.
    ///
    /// Returns `None` if the local index is out of bounds.
    #[inline]
    fn translate_index(&self, local_index: usize) -> Option<usize> {
        if local_index >= self.len {
            None
        } else {
            Some(self.start + local_index)
        }
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVecSlice<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Returns an iterator over the elements of the sequence at local `index`.
    ///
    /// Returns `None` if `index >= num_sequences()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[10], &[20], &[30]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    /// let slice = vec.slice(1, 2).unwrap(); // Sequences [20], [30]
    ///
    /// let first: Vec<u32> = slice.get(0).unwrap().collect();
    /// assert_eq!(first, vec![20]);
    /// ```
    #[inline]
    pub fn get(&self, index: usize) -> Option<SeqIter<'a, T, E, B>> {
        let parent_index = self.translate_index(index)?;
        self.parent.get(parent_index)
    }

    /// Returns an iterator without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < num_sequences()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> SeqIter<'a, T, E, B> {
        debug_assert!(
            index < self.len,
            "index {} out of bounds for slice with {} sequences",
            index,
            self.len
        );
        let parent_index = self.start + index;
        self.parent.get_unchecked(parent_index)
    }

    /// Returns the elements of sequence `index` as a newly allocated `Vec`.
    ///
    /// Returns `None` if `index >= num_sequences()`.
    #[inline]
    pub fn get_vec(&self, index: usize) -> Option<Vec<T>> {
        self.get(index).map(|iter| iter.collect())
    }

    /// Decodes sequence `index` into the provided buffer.
    ///
    /// The buffer is cleared before use. Returns the number of elements
    /// decoded, or `None` if `index >= num_sequences()`.
    #[inline]
    pub fn get_into(&self, index: usize, buf: &mut Vec<T>) -> Option<usize> {
        let iter = self.get(index)?;
        buf.clear();
        buf.extend(iter);
        Some(buf.len())
    }

    /// Returns an iterator over all sequences in this slice.
    ///
    /// Each element of the returned iterator is a [`SeqIter`] for the
    /// corresponding sequence.
    #[inline]
    pub fn iter(&self) -> SliceIter<'a, '_, T, E, B> {
        SliceIter {
            slice: self,
            current: 0,
        }
    }

    /// Creates a sub-slice of this slice.
    ///
    /// Returns `None` if `start + len > num_sequences()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[0], &[1], &[2], &[3], &[4]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let slice = vec.slice(1, 4).unwrap(); // [1], [2], [3], [4]
    /// let subslice = slice.slice(1, 2).unwrap(); // [2], [3]
    ///
    /// assert_eq!(subslice.get_vec(0), Some(vec![2]));
    /// ```
    #[inline]
    pub fn slice(&self, start: usize, len: usize) -> Option<SeqVecSlice<'a, T, E, B>> {
        if start.saturating_add(len) > self.len {
            return None;
        }
        Some(SeqVecSlice::new(self.parent, self.start + start, len))
    }

    /// Splits this slice into two non-overlapping sub-slices at the given index.
    ///
    /// Returns `None` if `mid > num_sequences()`.
    #[inline]
    pub fn split_at(
        &self,
        mid: usize,
    ) -> Option<(SeqVecSlice<'a, T, E, B>, SeqVecSlice<'a, T, E, B>)> {
        if mid > self.len {
            return None;
        }
        let left = SeqVecSlice::new(self.parent, self.start, mid);
        let right = SeqVecSlice::new(self.parent, self.start + mid, self.len - mid);
        Some((left, right))
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> Clone for SeqVecSlice<'a, T, E, B> {
    fn clone(&self) -> Self {
        Self {
            parent: self.parent,
            start: self.start,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> Copy for SeqVecSlice<'a, T, E, B> {}

/// An iterator over all sequences in a [`SeqVecSlice`].
pub struct SliceIter<'a, 's, T: Storable, E: Endianness, B: AsRef<[u64]>>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Reference to the slice.
    slice: &'s SeqVecSlice<'a, T, E, B>,
    /// Current position in the iteration.
    current: usize,
}

impl<'a, 's, T: Storable, E: Endianness, B: AsRef<[u64]>> Iterator for SliceIter<'a, 's, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = SeqIter<'a, T, E, B>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.slice.len {
            return None;
        }

        let iter = unsafe { self.slice.get_unchecked(self.current) };
        self.current += 1;
        Some(iter)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, 's, T: Storable, E: Endianness, B: AsRef<[u64]>> ExactSizeIterator
    for SliceIter<'a, 's, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
}

impl<'a, 's, T: Storable, E: Endianness, B: AsRef<[u64]>> std::iter::FusedIterator
    for SliceIter<'a, 's, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
}

impl<'a, 's, T: Storable, E: Endianness, B: AsRef<[u64]>> IntoIterator
    for &'s SeqVecSlice<'a, T, E, B>
where
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = SeqIter<'a, T, E, B>;
    type IntoIter = SliceIter<'a, 's, T, E, B>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
