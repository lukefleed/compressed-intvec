//! Zero-copy slices of a [`SeqVec`].
//!
//! This module provides [`SeqVecSlice`], a view into a contiguous portion of a
//! [`SeqVec`]. Slices are immutable and do not own their data; they borrow it
//! from the parent `SeqVec`.
//!
//! [`SeqVec`]: crate::seq::SeqVec

use super::{iter::SeqVecBitReader, SeqIter, SeqVec};
use crate::variable::traits::Storable;
use dsi_bitstream::prelude::{BitRead, BitSeek, CodesRead, Endianness};
use std::cmp::Ordering;
use std::ops::Range;

/// An immutable, zero-copy slice of a [`SeqVec`].
///
/// This struct provides a view into a contiguous portion of a [`SeqVec`]
/// without copying the underlying compressed data. It is created by the
/// [`slice`](super::SeqVec::slice) or [`split_at`](super::SeqVec::split_at)
/// methods on a [`SeqVec`].
///
/// All operations on a [`SeqVecSlice`] are relative to the start of the slice,
/// not the parent vector. The slice contains a contiguous range of sequences.
///
/// # Examples
///
/// ```
/// use compressed_intvec::seq::{SeqVec, LESeqVec};
///
/// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5], &[6], &[7, 8, 9, 10]];
/// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
///
/// // Create a slice of sequences 1 through 2 (indices 1 and 2)
/// let slice = vec.slice(1, 2).unwrap();
///
/// assert_eq!(slice.len(), 2);
///
/// // Accessing a sequence within the slice
/// // Index 0 of the slice corresponds to sequence 1 of the original vector
/// let seq0: Vec<u32> = slice.get(0).unwrap().collect();
/// assert_eq!(seq0, vec![3, 4, 5]);
///
/// // Iterating over the slice
/// let all_values: Vec<Vec<u32>> = slice.iter()
///     .map(|seq| seq.collect())
///     .collect();
/// assert_eq!(all_values, vec![vec![3, 4, 5], vec![6]]);
/// ```
#[derive(Debug, Clone)]
pub struct SeqVecSlice<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> {
    /// A reference to the parent vector.
    vec: &'a SeqVec<T, E, B>,
    /// The starting sequence index within the parent vector.
    start: usize,
    /// The number of sequences in the slice.
    len: usize,
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVecSlice<'a, T, E, B> {
    /// Creates a new `SeqVecSlice`.
    pub(super) fn new(vec: &'a SeqVec<T, E, B>, range: Range<usize>) -> Self {
        debug_assert!(
            range.end <= vec.num_sequences(),
            "Slice range exceeds vector bounds"
        );
        Self {
            vec,
            start: range.start,
            len: range.len(),
        }
    }

    /// Returns the number of sequences in the slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slice contains no sequences.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an iterator over the elements of the sequence at `index` within
    /// the slice, or `None` if the index is out of bounds.
    ///
    /// The index is relative to the start of the slice.
    #[inline]
    pub fn get(&self, index: usize) -> Option<SeqIter<'_, T, E>>
    where
        for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        if index >= self.len {
            return None;
        }
        // SAFETY: The bounds check above ensures `index` is valid.
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Returns an iterator over the elements of the sequence at `index` within
    /// the slice without bounds checking.
    ///
    /// The index is relative to the start of the slice.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    /// The caller must ensure that `index < self.len()`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> SeqIter<'_, T, E>
    where
        for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        debug_assert!(index < self.len, "Index out of bounds");
        // Translate slice-relative index to parent vector index.
        let parent_index = self.start + index;
        self.vec.get_unchecked(parent_index)
    }

    /// Returns the sequence at `index` as a materialized `Vec<T>`, or `None`
    /// if the index is out of bounds.
    ///
    /// This method fully decodes the sequence and allocates a new vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5], &[6]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let slice = vec.slice(1, 2).unwrap();
    /// assert_eq!(slice.get_vec(0), Some(vec![3, 4, 5]));
    /// assert_eq!(slice.get_vec(1), Some(vec![6]));
    /// assert_eq!(slice.get_vec(2), None); // Out of bounds
    /// ```
    #[inline]
    pub fn get_vec(&self, index: usize) -> Option<Vec<T>>
    where
        for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        Some(self.get(index)?.collect())
    }

    /// Decodes the sequence at `index` into a reusable buffer.
    ///
    /// The buffer is cleared and then filled with the decoded sequence elements.
    /// Returns the number of elements decoded, or `None` if the index is out
    /// of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5], &[6]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let slice = vec.slice(1, 2).unwrap();
    ///
    /// let mut buf = Vec::new();
    /// assert_eq!(slice.get_into(0, &mut buf), Some(3));
    /// assert_eq!(buf, vec![3, 4, 5]);
    ///
    /// // Buffer is reused (cleared internally).
    /// assert_eq!(slice.get_into(1, &mut buf), Some(1));
    /// assert_eq!(buf, vec![6]);
    /// ```
    #[inline]
    pub fn get_into(&self, index: usize, buf: &mut Vec<T>) -> Option<usize>
    where
        for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        let iter = self.get(index)?;
        buf.clear();
        buf.extend(iter);
        Some(buf.len())
    }

    /// Returns an iterator over the sequences in the slice.
    pub fn iter(&self) -> SeqVecSliceIter<'_, T, E, B>
    where
        for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
            + CodesRead<E>
            + BitSeek<Error = core::convert::Infallible>,
    {
        SeqVecSliceIter::new(self)
    }
}

impl<T, E, B> SeqVecSlice<'_, T, E, B>
where
    T: Storable + PartialEq,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    /// Binary searches this slice for a sequence matching the given elements.
    ///
    /// If the sequence is found, returns `Ok(usize)` with the index of the
    /// matching sequence within the slice. If not found, returns `Err(usize)`
    /// with the insertion point to maintain order.
    ///
    /// The comparison is performed element-by-element with early termination:
    /// sequences with different lengths or any differing element are ordered
    /// according to the first difference encountered.
    ///
    /// # Performance Note
    ///
    /// Each binary search probe **fully decodes a compressed sequence** until a
    /// difference is found or the sequence ends. For `log(n)` probes, each
    /// decoding is O(m) where `m` is the sequence length. This is O(m * log n)
    /// overall.
    ///
    /// - If sequences are short or differ early, early termination makes this
    ///   very efficient.
    /// - If sequences are very long and similar, decoding cost dominates.
    /// - To extract a key from each sequence efficiently, use
    ///   [`binary_search_by_key`](Self::binary_search_by_key) to extract a
    ///   sortable key from the compressed data (e.g., first element) rather
    ///   than the full sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2], &[3, 4, 5], &[6, 7], &[8]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let slice = vec.slice(0, 4).unwrap();
    ///
    /// assert_eq!(slice.binary_search(&[3, 4, 5]), Ok(1));
    /// assert_eq!(slice.binary_search(&[6, 7]), Ok(2));
    /// assert_eq!(slice.binary_search(&[5]), Err(2)); // Would be inserted at index 2
    /// ```
    pub fn binary_search(&self, sequence: &[T]) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|probe_iter| {
            // Compare element-by-element with early exit on first difference.
            let mut probe_iter = probe_iter.peekable();
            let mut seq_iter = sequence.iter().peekable();

            loop {
                match (probe_iter.next(), seq_iter.next()) {
                    (Some(a), Some(b)) => {
                        let cmp = a.cmp(b);
                        if cmp != Ordering::Equal {
                            return cmp;
                        }
                    }
                    // Probe sequence ended first → Less
                    (None, Some(_)) => return Ordering::Less,
                    // Target sequence ended first → Greater
                    (Some(_), None) => return Ordering::Greater,
                    // Both ended → Equal
                    (None, None) => return Ordering::Equal,
                }
            }
        })
    }

    /// Binary searches this slice with a custom comparison function.
    ///
    /// The comparison function receives a [`SeqIter`] for the probe sequence
    /// and should return the ordering relative to the target.
    ///
    /// # Performance Note
    ///
    /// The closure receives a [`SeqIter`] for the probe sequence, which may
    /// require decoding the compressed data. Each probe can involve up to O(m)
    /// decoding operations where `m` is the sequence length.
    ///
    /// - For fast key extraction (first element, hash, etc.), decode only what
    ///   is needed within the closure and return early.
    /// - To search by a pre-computed key, use [`binary_search_by_key`](Self::binary_search_by_key)
    ///   instead, which is optimized for key-based comparisons.
    /// - For comparisons requiring the full sequence, full decoding is
    ///   unavoidable: O(m * log n).
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    /// use std::cmp::Ordering;
    ///
    /// let sequences: &[&[u32]] = &[&[1], &[1, 2], &[1, 2, 3], &[1, 2, 3, 4]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let slice = vec.slice(0, 4).unwrap();
    ///
    /// // Search by sequence length (decodes all elements)
    /// let result = slice.binary_search_by(|probe| {
    ///     let probe_len = probe.count();
    ///     probe_len.cmp(&3)
    /// });
    ///
    /// assert_eq!(result, Ok(2));
    ///
    /// // Search by first element only (early termination)
    /// let result = slice.binary_search_by(|mut probe| {
    ///     match probe.next() {
    ///         Some(first) => first.cmp(&1),
    ///         None => Ordering::Less,
    ///     }
    /// });
    /// ```
    #[inline]
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(SeqIter<'_, T, E>) -> Ordering,
    {
        let mut low = 0;
        let mut high = self.len();

        while low < high {
            let mid = low + (high - low) / 2;
            // SAFETY: Bounds are checked by the loop invariants and the slice's
            // construction, so the index is always valid.
            let cmp = f(unsafe { self.get_unchecked(mid) });
            match cmp {
                Ordering::Less => low = mid + 1,
                Ordering::Equal => return Ok(mid),
                Ordering::Greater => high = mid,
            }
        }
        Err(low)
    }

    /// Binary searches this slice with a key extraction function.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// let sequences: &[&[u32]] = &[&[10], &[20], &[30, 40], &[50]];
    /// let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();
    ///
    /// let slice = vec.slice(0, 4).unwrap();
    ///
    /// // Search by first element
    /// let result = slice.binary_search_by_key(&30, |mut probe| probe.next().unwrap());
    /// assert_eq!(result, Ok(2)); // Found sequence [30, 40]
    /// ```
    #[inline]
    pub fn binary_search_by_key<K, F>(&self, b: &K, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(SeqIter<'_, T, E>) -> K,
        K: Ord,
    {
        self.binary_search_by(|probe| f(probe).cmp(b))
    }
}

/// An iterator over the sequences of a [`SeqVecSlice`].
///
/// This struct is created by the [`iter`](SeqVecSlice::iter) method. Each
/// element of this iterator is a [`SeqIter`] for a sequence within the slice.
pub struct SeqVecSliceIter<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> {
    slice: &'a SeqVecSlice<'a, T, E, B>,
    /// Current front index for forward iteration.
    front: usize,
    /// Current back index (exclusive) for backward iteration.
    back: usize,
}

// --- IntoIterator Implementation ---

impl<'a, T, E, B> IntoIterator for &'a SeqVecSlice<'a, T, E, B>
where
    T: Storable,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = SeqIter<'a, T, E>;
    type IntoIter = SeqVecSliceIter<'a, T, E, B>;

    /// Returns an iterator over the sequences in the slice.
    ///
    /// This implementation allows slices to be used in `for` loops:
    ///
    /// ```no_run
    /// # use compressed_intvec::seq::{SeqVec, LESeqVec};
    /// # let vec: LESeqVec<u32> = SeqVec::from_slices(&[&[1, 2][..], &[3, 4, 5][..]]).unwrap();
    /// # let slice = vec.slice(0, 2).unwrap();
    /// for seq in &slice {
    ///     // Process each sequence
    /// }
    /// ```
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Storable, E: Endianness, B: AsRef<[u64]>> SeqVecSliceIter<'a, T, E, B> {
    /// Creates a new iterator for a given `SeqVecSlice`.
    fn new(slice: &'a SeqVecSlice<'a, T, E, B>) -> Self {
        Self {
            slice,
            front: 0,
            back: slice.len(),
        }
    }
}

impl<'a, T, E, B> Iterator for SeqVecSliceIter<'a, T, E, B>
where
    T: Storable,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    type Item = SeqIter<'a, T, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let index = self.front;
        self.front += 1;
        // SAFETY: The iterator's invariant ensures `index` is in bounds.
        Some(unsafe { self.slice.get_unchecked(index) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let remaining = self.back.saturating_sub(self.front);
        if n >= remaining {
            self.front = self.back;
            return None;
        }
        self.front += n;
        self.next()
    }

    #[inline]
    fn count(self) -> usize {
        self.back.saturating_sub(self.front)
    }
}

impl<'a, T, E, B> ExactSizeIterator for SeqVecSliceIter<'a, T, E, B>
where
    T: Storable,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn len(&self) -> usize {
        self.back.saturating_sub(self.front)
    }
}

impl<'a, T, E, B> std::iter::FusedIterator for SeqVecSliceIter<'a, T, E, B>
where
    T: Storable,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
}

impl<'a, T, E, B> DoubleEndedIterator for SeqVecSliceIter<'a, T, E, B>
where
    T: Storable,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        // SAFETY: The iterator's invariant ensures `back` is in bounds.
        Some(unsafe { self.slice.get_unchecked(self.back) })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let remaining = self.back.saturating_sub(self.front);
        if n >= remaining {
            self.back = self.front;
            return None;
        }
        self.back -= n;
        self.next_back()
    }
}

// --- PartialEq Implementations ---

impl<'a, T, E, B> PartialEq<SeqVecSlice<'a, T, E, B>> for SeqVecSlice<'a, T, E, B>
where
    T: Storable + PartialEq,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn eq(&self, other: &SeqVecSlice<'a, T, E, B>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter()
            .zip(other.iter())
            .all(|(seq_a, seq_b)| seq_a.eq(seq_b))
    }
}

impl<'a, T, E, B> Eq for SeqVecSlice<'a, T, E, B>
where
    T: Storable + Eq,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
}

impl<'a, T, E, B> PartialEq<SeqVec<T, E, B>> for SeqVecSlice<'a, T, E, B>
where
    T: Storable + PartialEq,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn eq(&self, other: &SeqVec<T, E, B>) -> bool {
        if self.len() != other.num_sequences() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a.eq(b))
    }
}

impl<'a, T, E, B> PartialEq<SeqVecSlice<'a, T, E, B>> for SeqVec<T, E, B>
where
    T: Storable + PartialEq,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn eq(&self, other: &SeqVecSlice<'a, T, E, B>) -> bool {
        other.eq(self)
    }
}

impl<'a, T, E, B> PartialEq<&SeqVec<T, E, B>> for SeqVecSlice<'a, T, E, B>
where
    T: Storable + PartialEq,
    E: Endianness,
    B: AsRef<[u64]>,
    for<'b> SeqVecBitReader<'b, E>: BitRead<E, Error = core::convert::Infallible>
        + CodesRead<E>
        + BitSeek<Error = core::convert::Infallible>,
{
    fn eq(&self, other: &&SeqVec<T, E, B>) -> bool {
        self.eq(*other)
    }
}
