//! Vectors whose elements all take the same number of bits.

mod atomic;
mod iter;
mod layout;
#[cfg(feature = "parallel")]
mod parallel;
#[cfg(feature = "serde")]
mod serde;

pub use atomic::AtomicFixedVec;
pub use iter::{FixedVecIntoIter, FixedVecIter};

use crate::Storable;
use std::{
    error::Error,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Bound, RangeBounds},
    sync::Arc,
};

/// The packed word storage of a [`FixedVec`].
///
/// - `Vec<u64>` owns the words and supports mutation and resizing.
/// - `Box<[u64]>` owns the words and supports mutation at a fixed length.
/// - `Arc<[u64]>` shares ownership of the words and provides read-only access.
/// - `&[u64]` provides a read-only view through a shared slice of the words.
///
/// This trait is sealed and cannot be implemented outside this crate. Its implementations
/// keep the word slice consistent across accesses, which the unchecked bit operations require.
pub trait Backend: AsRef<[u64]> + sealed::Sealed {}

impl Backend for Vec<u64> {}
impl Backend for Box<[u64]> {}
impl Backend for Arc<[u64]> {}
impl Backend for &[u64] {}

mod sealed {
    use std::sync::Arc;

    pub trait Sealed {}
    impl Sealed for Vec<u64> {}
    impl Sealed for Box<[u64]> {}
    impl Sealed for Arc<[u64]> {}
    impl Sealed for &[u64] {}
}

/// A requested bit width exceeds the element type's maximum width.
///
/// Returned by [`FixedVec::new`] and [`FixedVec::with_capacity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitWidthTooLarge {
    /// The requested bit width.
    pub bit_width: u32,
    /// The largest bit width of the element type: its [`Storable::BITS`], at most 64.
    pub max: u32,
}

impl fmt::Display for BitWidthTooLarge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bit width {} exceeds the maximum {} of the element type",
            self.bit_width, self.max
        )
    }
}

impl Error for BitWidthTooLarge {}

/// Invalid bit width, length or storage for a fixed-width vector.
///
/// Returned by [`FixedVec::from_parts`] and [`AtomicFixedVec::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FixedVecError {
    /// The bit width exceeds the largest that the element type allows.
    BitWidthTooLarge(BitWidthTooLarge),
    /// The total number of element bits overflows `usize`.
    LengthOverflow {
        /// The number of elements.
        len: usize,
        /// The bit width.
        bit_width: u32,
    },
    /// The buffer holds fewer words than the elements and the padding word need.
    BufferTooSmall {
        /// The number of words in the buffer.
        words: usize,
        /// The number of words required.
        required: usize,
    },
}

impl fmt::Display for FixedVecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            // Transparent: the inner error carries the whole message.
            Self::BitWidthTooLarge(error) => error.fmt(f),
            Self::LengthOverflow { len, bit_width } => {
                write!(f, "{len} elements of {bit_width} bits overflow usize")
            }
            Self::BufferTooSmall { words, required } => {
                write!(
                    f,
                    "buffer of {words} words is smaller than the {required} words required"
                )
            }
        }
    }
}

impl Error for FixedVecError {}

impl From<BitWidthTooLarge> for FixedVecError {
    fn from(error: BitWidthTooLarge) -> Self {
        Self::BitWidthTooLarge(error)
    }
}

/// A value's encoded word does not fit a vector's bit width.
///
/// Returned by [`FixedVec::try_push`] and [`FixedVec::try_set`], with the rejected value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueTooLarge<T> {
    /// The rejected value.
    pub value: T,
    /// The number of bits that [`Storable::to_word`] requires for the value.
    pub required: u32,
    /// The bit width of the vector.
    pub bit_width: u32,
}

impl<T: fmt::Debug> fmt::Display for ValueTooLarge<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "value {:?} needs {} bits, but the vector stores {} bits per element",
            self.value, self.required, self.bit_width
        )
    }
}

impl<T: fmt::Debug> Error for ValueTooLarge<T> {}

/// Returns the number of bits that `word` needs.
#[inline]
fn required_bits(word: u64) -> u32 {
    u64::BITS - word.leading_zeros()
}

/// Panics when a value needs `required` bits and the vector stores `bit_width`.
///
/// The arguments describe bit counts rather than the element value.
#[cold]
#[inline(never)]
#[track_caller]
fn value_too_large(required: u32, bit_width: usize) -> ! {
    panic!("value needs {required} bits, but the vector stores {bit_width} bits per element")
}

/// Panics because `index` is out of bounds for a vector of `len` elements.
#[cold]
#[inline(never)]
#[track_caller]
fn index_out_of_bounds(index: usize, len: usize) -> ! {
    panic!("index out of bounds: the len is {len} but the index is {index}")
}

/// Returns the largest bit width of `T`.
fn max_bit_width<T: Storable>() -> usize {
    (T::BITS as usize).min(layout::WORD_BITS)
}

/// Checks a requested bit width against `T`.
fn checked_bit_width<T: Storable>(bit_width: u32) -> Result<usize, BitWidthTooLarge> {
    let max = max_bit_width::<T>();
    // Lossless: `usize` has at least 32 bits (the assertion in `lib.rs`).
    if bit_width as usize <= max {
        Ok(bit_width as usize)
    } else {
        Err(BitWidthTooLarge {
            bit_width,
            max: max as u32,
        })
    }
}

/// Returns the word of `value`, or an error with the value if it does not fit `mask`.
#[inline]
fn checked_word<T: Storable>(
    value: T,
    bit_width: usize,
    mask: u64,
) -> Result<u64, ValueTooLarge<T>> {
    let word = value.to_word();
    if word & !mask == 0 {
        Ok(word)
    } else {
        Err(ValueTooLarge {
            value,
            required: required_bits(word),
            bit_width: bit_width as u32,
        })
    }
}

/// Returns the word of `value`.
///
/// # Panics
///
/// If the word does not fit `mask`.
#[inline]
#[track_caller]
fn fitting_word<T: Storable>(value: T, bit_width: usize, mask: u64) -> u64 {
    let word = value.to_word();
    if word & !mask != 0 {
        value_too_large(required_bits(word), bit_width);
    }
    word
}

/// Returns the `start..end` that `range` selects within `0..len`, or `None` if it is reversed or
/// exceeds `len`.
fn bounds(range: impl RangeBounds<usize>, len: usize) -> Option<(usize, usize)> {
    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.checked_add(1)?,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&end) => end.checked_add(1)?,
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };
    (start <= end && end <= len).then_some((start, end))
}

/// A vector where every integer occupies the same predetermined number of bits.
///
/// Use `FixedVec` when values fit a known bit width and you need constant-time indexed reads
/// and writes. The width is fixed at construction, and every value must fit it. `T` defines
/// the element encoding through [`Storable`]. Elements use `len * bit_width` bits, rounded up
/// to whole `u64` words, plus one padding word when the vector is non-empty.
///
/// The storage type `B` implements [`Backend`]. The default `Vec<u64>` supports resizing.
/// Converting to `Box<[u64]>` retains mutation at a fixed length, while `Arc<[u64]>` and
/// `&[u64]` provide read-only access.
///
/// [`new`](FixedVec::new) chooses a width explicitly, while
/// [`from_slice`](FixedVec::from_slice) and [`collect`](Iterator::collect) choose the smallest
/// width that holds the initial values.
/// [`as_borrowed`](FixedVec::as_borrowed) borrows an existing vector's packed storage, and
/// [`from_parts`](FixedVec::from_parts) creates a vector over external packed words.
///
/// Equality between two `FixedVec` values compares both their bit widths and their decoded
/// elements. Compare [`as_slice`](FixedVec::as_slice) views to compare only the elements.
///
/// # Serialization
///
/// With the `serde` feature, every backend supports serialization, and deserialization creates
/// a `FixedVec<T>` with owned `Vec<u64>` storage. Loading requires a supported format version,
/// matching signedness and a stored bit width that fits `T`. Every 1.x release loads the
/// vectors that it and earlier 1.x releases serialized.
///
/// Serialization omits padding words and clears unused trailing bits, while preserving encoded
/// element representations. Equal vectors with custom [`Storable`] aliases or equality may
/// therefore serialize differently.
///
/// # Examples
///
/// ```
/// use compressed_intvec::FixedVec;
///
/// let mut vec = FixedVec::from_slice(&[10u32, 20, 30]);
/// assert_eq!(vec.bit_width(), 5);
/// assert_eq!(vec.get(1), Some(20));
///
/// vec.set(1, 31);
/// vec.push(7);
/// assert_eq!(vec, [10, 31, 30, 7][..]);
/// ```
///
/// Built-in signed integer types use zig-zag encoding, so small magnitudes take few bits:
///
/// ```
/// use compressed_intvec::FixedVec;
///
/// let vec: FixedVec<i16> = [-2, -1, 0, 1].into_iter().collect();
/// assert_eq!(vec.bit_width(), 2);
/// assert_eq!(vec.iter().collect::<Vec<_>>(), [-2, -1, 0, 1]);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize, mem_dbg::MemDbg))]
pub struct FixedVec<T, B = Vec<u64>> {
    // Invariants, established by every constructor and kept by every method:
    // - `bit_width <= max_bit_width::<T>()` and `mask == layout::mask(bit_width)`;
    // - `layout::required_words(len, bit_width)` does not overflow, and
    //   `words.as_ref().len()` is at least that many words.
    // The unsafe reads and writes rely on the second one. Bits past `len * bit_width` are
    // unspecified: nothing reads them as elements.
    words: B,
    len: usize,
    bit_width: usize,
    mask: u64,
    element: PhantomData<fn() -> T>,
}

impl<T, B> FixedVec<T, B> {
    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of bits used to store each encoded element.
    #[inline]
    pub fn bit_width(&self) -> u32 {
        self.bit_width as u32
    }

    /// Consumes the vector and returns its storage, length and bit width, which
    /// [`from_parts`](FixedVec::from_parts) accepts back.
    ///
    /// The storage holds the elements packed as described in [`from_parts`](FixedVec::from_parts),
    /// followed by at least one word when the vector is non-empty. The bits past the last element
    /// are unspecified.
    pub fn into_parts(self) -> (B, usize, u32) {
        (self.words, self.len, self.bit_width as u32)
    }
}

impl<T, B: Backend> FixedVec<T, B> {
    /// Returns a read-only vector borrowing the packed storage.
    ///
    /// The result contains a shared slice of `self`'s words and a copy of its metadata, retaining
    /// the length and bit width. Creation takes constant time with zero allocations, and the
    /// returned vector borrows `self` for its lifetime.
    ///
    /// ```
    /// use compressed_intvec::FixedVec;
    ///
    /// let owned = FixedVec::from_slice(&[1u64, 2, 3]);
    /// let view = owned.as_borrowed();
    /// assert_eq!(view.get(1), Some(2));
    /// let boxed: FixedVec<u64, Box<[u64]>> = owned.into();
    /// assert_eq!(boxed.get(2), Some(3));
    /// ```
    #[inline]
    pub fn as_borrowed(&self) -> FixedVec<T, &[u64]> {
        FixedVec {
            words: self.words.as_ref(),
            len: self.len,
            bit_width: self.bit_width,
            mask: self.mask,
            element: PhantomData,
        }
    }
}

/// Converts the owned word buffer to boxed storage, preserving every stored word and the
/// vector's metadata. Boxed vectors support element mutation at a fixed length.
impl<T> From<FixedVec<T>> for FixedVec<T, Box<[u64]>> {
    fn from(vec: FixedVec<T>) -> Self {
        let FixedVec {
            words,
            len,
            bit_width,
            mask,
            element,
        } = vec;
        Self {
            words: words.into_boxed_slice(),
            len,
            bit_width,
            mask,
            element,
        }
    }
}

impl<T: Storable> FixedVec<T> {
    /// Creates an empty vector whose elements take `bit_width` bits.
    ///
    /// # Errors
    ///
    /// [`BitWidthTooLarge`] if `bit_width` exceeds [`Storable::BITS`] of `T`, or 64.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::FixedVec;
    ///
    /// let mut vec = FixedVec::<u32>::new(4)?;
    /// vec.push(15);
    /// let error = vec.try_push(16).unwrap_err();
    /// assert_eq!((error.value, error.required), (16, 5));
    /// # Ok::<(), compressed_intvec::BitWidthTooLarge>(())
    /// ```
    pub fn new(bit_width: u32) -> Result<Self, BitWidthTooLarge> {
        Self::with_capacity(bit_width, 0)
    }

    /// Creates an empty vector whose elements take `bit_width` bits, with room for `capacity`
    /// elements.
    ///
    /// # Errors
    ///
    /// [`BitWidthTooLarge`] if `bit_width` exceeds [`Storable::BITS`] of `T`, or 64.
    ///
    /// # Panics
    ///
    /// If `capacity` elements of `bit_width` bits need more than `usize::MAX` bits.
    pub fn with_capacity(bit_width: u32, capacity: usize) -> Result<Self, BitWidthTooLarge> {
        let bit_width = checked_bit_width::<T>(bit_width)?;
        let words = layout::required_words(capacity, bit_width).expect("capacity overflow");
        Ok(Self {
            words: Vec::with_capacity(words),
            len: 0,
            bit_width,
            mask: layout::mask(bit_width),
            element: PhantomData,
        })
    }

    /// Creates a vector holding `values`, with the smallest bit width that holds all of them.
    ///
    /// The bit width is 0 when every value is stored as the word 0, which includes an empty
    /// slice.
    ///
    /// # Panics
    ///
    /// If [`Storable::to_word`] returns a word wider than [`Storable::BITS`].
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::FixedVec;
    ///
    /// let vec = FixedVec::from_slice(&[1u64, 1000, 7]);
    /// assert_eq!(vec.bit_width(), 10);
    /// assert_eq!(vec.get(1), Some(1000));
    /// ```
    pub fn from_slice(values: &[T]) -> Self {
        let all_bits = values.iter().fold(0, |bits, value| bits | value.to_word());
        let bit_width = (u64::BITS - all_bits.leading_zeros()) as usize;
        assert!(
            bit_width <= max_bit_width::<T>(),
            "Storable::to_word returned a word wider than Storable::BITS"
        );
        let required = layout::required_words(values.len(), bit_width).expect("capacity overflow");
        let mut words = vec![0; required];
        layout::pack(
            &mut words,
            bit_width,
            values.iter().map(|value| value.to_word()),
        );
        Self {
            words,
            len: values.len(),
            bit_width,
            mask: layout::mask(bit_width),
            element: PhantomData,
        }
    }

    /// Returns the number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        if self.bit_width == 0 {
            // A width of 0 needs one word, the padding word, whatever the length.
            return if self.words.capacity() > 0 {
                usize::MAX
            } else {
                0
            };
        }
        let data_words = self.words.capacity().saturating_sub(1);
        data_words.saturating_mul(layout::WORD_BITS) / self.bit_width
    }

    /// Reserves room for at least `additional` more elements.
    ///
    /// # Panics
    ///
    /// If the new length, or the number of bits that it needs, overflows `usize`.
    pub fn reserve(&mut self, additional: usize) {
        let len = self.len.checked_add(additional).expect("capacity overflow");
        let required = layout::required_words(len, self.bit_width).expect("capacity overflow");
        self.words
            .reserve(required.saturating_sub(self.words.len()));
    }

    /// Shrinks the storage to the words needed for the elements and their padding.
    ///
    /// As with [`Vec::shrink_to_fit`], the allocator may retain additional capacity.
    pub fn shrink_to_fit(&mut self) {
        let required = layout::required_words(self.len, self.bit_width)
            .expect("the length of a vector does not overflow");
        self.words.truncate(required);
        self.words.shrink_to_fit();
    }

    /// Makes the storage hold `len` elements.
    #[inline]
    #[track_caller]
    fn grow_to(&mut self, len: usize) {
        let required = layout::required_words(len, self.bit_width).expect("capacity overflow");
        if self.words.len() < required {
            self.words.resize(required, 0);
        }
    }

    /// Appends `value`.
    ///
    /// # Panics
    ///
    /// If `value` does not fit the bit width; [`try_push`](FixedVec::try_push) returns an error
    /// instead. Also if the new length, or the number of bits that it needs, overflows `usize`.
    #[inline]
    #[track_caller]
    pub fn push(&mut self, value: T) {
        let word = fitting_word(value, self.bit_width, self.mask);
        self.push_word(word);
    }

    /// Appends `value`, or returns an error if it does not fit the bit width.
    ///
    /// # Errors
    ///
    /// [`ValueTooLarge`], with `value`, if `value` does not fit the bit width.
    ///
    /// # Panics
    ///
    /// If the new length, or the number of bits that it needs, overflows `usize`.
    #[inline]
    pub fn try_push(&mut self, value: T) -> Result<(), ValueTooLarge<T>> {
        let word = checked_word(value, self.bit_width, self.mask)?;
        self.push_word(word);
        Ok(())
    }

    #[inline]
    #[track_caller]
    fn push_word(&mut self, word: u64) {
        let new_len = self.len.checked_add(1).expect("capacity overflow");
        self.grow_to(new_len);
        // SAFETY: `grow_to` made the words hold `new_len` elements.
        unsafe {
            layout::write_in_order(&mut self.words, self.bit_width, self.mask, self.len, word)
        };
        self.len = new_len;
    }

    /// Removes the last element and returns it, or `None` if the vector is empty.
    pub fn pop(&mut self) -> Option<T> {
        let last = self.len.checked_sub(1)?;
        // SAFETY: `last < len`.
        let value = unsafe { self.get_unchecked(last) };
        self.len = last;
        Some(value)
    }

    /// Removes every element, keeping the capacity.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Changes the length to `new_len`, appending copies of `value` or dropping elements from
    /// the end.
    ///
    /// # Panics
    ///
    /// If the vector grows and `value` does not fit the bit width, or if the new length needs
    /// more than `usize::MAX` bits.
    #[track_caller]
    pub fn resize(&mut self, new_len: usize, value: T) {
        if new_len > self.len {
            let word = fitting_word(value, self.bit_width, self.mask);
            self.grow_to(new_len);
            for index in self.len..new_len {
                // SAFETY: `grow_to` made the words hold `new_len` elements.
                unsafe {
                    layout::write_in_order(&mut self.words, self.bit_width, self.mask, index, word)
                };
            }
        }
        self.len = new_len;
    }

    /// Appends `values`.
    ///
    /// # Panics
    ///
    /// If one of the values does not fit the bit width (the vector then retains its original
    /// contents), or if the new length, or the number of bits that it needs, overflows `usize`.
    #[track_caller]
    pub fn extend_from_slice(&mut self, values: &[T]) {
        let new_len = self
            .len
            .checked_add(values.len())
            .expect("capacity overflow");
        self.grow_to(new_len);
        // Each value's word is checked and written in one pass, so the word written is the word
        // checked. On a panic, `len` is unchanged, and the words written so far lie past it,
        // where the bits are unspecified, so the vector retains its contents.
        for (index, &value) in (self.len..).zip(values) {
            let word = fitting_word(value, self.bit_width, self.mask);
            // SAFETY: `grow_to` made the words hold `new_len` elements, and `index < new_len`.
            unsafe {
                layout::write_in_order(&mut self.words, self.bit_width, self.mask, index, word)
            };
        }
        self.len = new_len;
    }

    /// Inserts `value` at `index`, moving the elements after it one place up.
    ///
    /// This takes time linear in the number of elements after `index`.
    ///
    /// # Panics
    ///
    /// If `index > len`, if `value` does not fit the bit width, or if the new length, or the
    /// number of bits that it needs, overflows `usize`.
    #[track_caller]
    pub fn insert(&mut self, index: usize, value: T) {
        let len = self.len;
        assert!(
            index <= len,
            "insertion index (is {index}) should be <= len (is {len})"
        );
        let word = fitting_word(value, self.bit_width, self.mask);
        let new_len = len.checked_add(1).expect("capacity overflow");
        self.grow_to(new_len);
        let width = self.bit_width;
        layout::shift_right(&mut self.words, index * width, width, (len - index) * width);
        // SAFETY: `grow_to` made the words hold `new_len` elements, and `index <= len`.
        unsafe { layout::write(&mut self.words, width, self.mask, index, word) };
        self.len = new_len;
    }

    /// Removes the element at `index` and returns it, moving the elements after it one place
    /// down.
    ///
    /// This takes time linear in the number of elements after `index`.
    ///
    /// # Panics
    ///
    /// If `index >= len`.
    #[track_caller]
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len;
        assert!(
            index < len,
            "removal index (is {index}) should be < len (is {len})"
        );
        // SAFETY: `index < len`.
        let value = unsafe { self.get_unchecked(index) };
        let width = self.bit_width;
        layout::shift_left(
            &mut self.words,
            index * width,
            width,
            (len - index - 1) * width,
        );
        self.len = len - 1;
        value
    }

    /// Removes the element at `index` and returns it, moving the last element into its place.
    ///
    /// This takes constant time.
    ///
    /// # Panics
    ///
    /// If `index >= len`.
    #[track_caller]
    pub fn swap_remove(&mut self, index: usize) -> T {
        let len = self.len;
        assert!(
            index < len,
            "swap_remove index (is {index}) should be < len (is {len})"
        );
        // SAFETY: `index < len` and `len - 1 < len`.
        unsafe {
            let value = self.get_unchecked(index);
            let last = layout::read(&self.words, self.bit_width, self.mask, len - 1);
            layout::write(&mut self.words, self.bit_width, self.mask, index, last);
            self.len = len - 1;
            value
        }
    }
}

impl<T: Storable, B: Backend> FixedVec<T, B> {
    /// Creates a vector over `words`, which hold `len` elements of `bit_width` bits.
    ///
    /// Element `i` occupies the bits `[i * bit_width, (i + 1) * bit_width)` of the words, where
    /// bit `k` is bit `k % 64` of word `k / 64`. A non-empty vector needs
    /// `ceil(len * bit_width / 64) + 1` words, including one padding word. At width zero this
    /// is a single padding word. An empty vector accepts an empty buffer.
    ///
    /// [`into_parts`](FixedVec::into_parts) returns words in this form. Extra words, the padding
    /// word and the bits past the last element can hold anything. This constructor validates
    /// the bit width, length and buffer size, then retains the supplied packed words.
    ///
    /// # Errors
    ///
    /// - [`FixedVecError::BitWidthTooLarge`] if `bit_width` exceeds [`Storable::BITS`] of `T`, or
    ///   64;
    /// - [`FixedVecError::LengthOverflow`] if `len * bit_width` overflows `usize`;
    /// - [`FixedVecError::BufferTooSmall`] if `words` is too short.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::FixedVec;
    ///
    /// // Three elements of five bits, and the padding word.
    /// let words = [0b11110_10100_01010, 0];
    /// let view = FixedVec::<u32, &[u64]>::from_parts(&words[..], 3, 5)?;
    /// assert_eq!(view.get(2), Some(30));
    /// # Ok::<(), compressed_intvec::FixedVecError>(())
    /// ```
    pub fn from_parts(words: B, len: usize, bit_width: u32) -> Result<Self, FixedVecError> {
        let width = checked_bit_width::<T>(bit_width)?;
        let required = layout::required_words(len, width)
            .ok_or(FixedVecError::LengthOverflow { len, bit_width })?;
        let available = words.as_ref().len();
        if available < required {
            return Err(FixedVecError::BufferTooSmall {
                words: available,
                required,
            });
        }
        Ok(Self {
            words,
            len,
            bit_width: width,
            mask: layout::mask(width),
            element: PhantomData,
        })
    }

    /// Returns a view of all the elements.
    #[inline]
    pub fn as_slice(&self) -> FixedVecSlice<'_, T> {
        FixedVecSlice {
            words: self.words.as_ref(),
            start: 0,
            len: self.len,
            bit_width: self.bit_width,
            mask: self.mask,
            element: PhantomData,
        }
    }

    /// Returns the element at `index`, or `None` if `index >= len`.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        if index < self.len {
            // SAFETY: `index < len`.
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    /// Returns the element at `index`, requiring the caller to ensure that the index is in bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index < self.len()`. Calling this method with an out-of-bounds
    /// index is undefined behavior, even if the returned value is unused.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        debug_assert!(index < self.len);
        // SAFETY: `index < len` (caller), and the words hold `len` elements (invariant; `B` is
        // sealed, so `as_ref` returns the same words on every call).
        T::from_word(unsafe { layout::read(self.words.as_ref(), self.bit_width, self.mask, index) })
    }

    /// Returns the first element, or `None` if the vector is empty.
    pub fn first(&self) -> Option<T> {
        self.get(0)
    }

    /// Returns the last element, or `None` if the vector is empty.
    pub fn last(&self) -> Option<T> {
        self.get(self.len.checked_sub(1)?)
    }

    /// Returns an iterator yielding decoded elements by value.
    #[inline]
    pub fn iter(&self) -> FixedVecIter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns a borrowed view of `range`, or `None` if the range is reversed or out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::FixedVec;
    ///
    /// let vec: FixedVec<u32> = (0..10).collect();
    /// let middle = vec.slice(3..6).unwrap();
    /// assert_eq!(middle.iter().collect::<Vec<_>>(), [3, 4, 5]);
    /// assert!(vec.slice(8..11).is_none());
    /// ```
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Option<FixedVecSlice<'_, T>> {
        self.as_slice().slice(range)
    }

    /// Searches a sorted vector for `value` in logarithmic time.
    ///
    /// Returns `Ok(index)` for a match or `Err(index)` for its insertion position. If multiple
    /// elements match, any matching index may be returned. The vector must be sorted in
    /// ascending order, as for [`slice::binary_search`].
    pub fn binary_search(&self, value: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.as_slice().binary_search(value)
    }

    /// Searches a sorted vector with `compare`, as [`slice::binary_search_by`].
    ///
    /// `compare` must return the ordering of each element relative to the target, consistently
    /// with the vector's sort order. Returns a matching index or the target's insertion position.
    pub fn binary_search_by(
        &self,
        compare: impl FnMut(&T) -> std::cmp::Ordering,
    ) -> Result<usize, usize> {
        self.as_slice().binary_search_by(compare)
    }

    /// Searches a vector sorted by the keys returned by `extract`, as
    /// [`slice::binary_search_by_key`].
    ///
    /// Returns a matching index or the insertion position of `key`.
    pub fn binary_search_by_key<K: Ord>(
        &self,
        key: &K,
        extract: impl FnMut(T) -> K,
    ) -> Result<usize, usize> {
        self.as_slice().binary_search_by_key(key, extract)
    }

    /// Returns the index of the first element for which `predicate` is false.
    ///
    /// All elements for which `predicate` is true must precede those for which it is false,
    /// as for [`slice::partition_point`]. Returns `len` if every element satisfies the predicate.
    pub fn partition_point(&self, predicate: impl FnMut(&T) -> bool) -> usize {
        self.as_slice().partition_point(predicate)
    }
}

impl<T: Storable, B: Backend + AsMut<[u64]>> FixedVec<T, B> {
    #[inline]
    #[track_caller]
    fn check_index(&self, index: usize) {
        if index >= self.len {
            index_out_of_bounds(index, self.len);
        }
    }

    /// Replaces the element at `index` with `value`.
    ///
    /// # Panics
    ///
    /// If `index >= len`, or if `value` does not fit the bit width; [`try_set`](FixedVec::try_set)
    /// returns an error for the latter.
    #[inline]
    #[track_caller]
    pub fn set(&mut self, index: usize, value: T) {
        self.check_index(index);
        // SAFETY: `index < len`.
        unsafe { self.set_unchecked(index, value) };
    }

    /// Replaces the element at `index` with `value`, or returns an error if it does not fit the
    /// bit width.
    ///
    /// # Errors
    ///
    /// [`ValueTooLarge`], with `value`, if `value` does not fit the bit width.
    ///
    /// # Panics
    ///
    /// If `index >= len`.
    #[track_caller]
    pub fn try_set(&mut self, index: usize, value: T) -> Result<(), ValueTooLarge<T>> {
        self.check_index(index);
        let word = checked_word(value, self.bit_width, self.mask)?;
        // SAFETY: `index < len`, and the words hold `len` elements (invariant).
        unsafe { layout::write(self.words.as_mut(), self.bit_width, self.mask, index, word) };
        Ok(())
    }

    /// Replaces the element at `index` with `value`, requiring the caller to ensure that the
    /// index is in bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index < self.len()`. An out-of-bounds index is undefined behavior.
    ///
    /// # Panics
    ///
    /// If `value` does not fit the bit width.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: T) {
        debug_assert!(index < self.len);
        let word = fitting_word(value, self.bit_width, self.mask);
        // SAFETY: `index < len` (caller), and the words hold `len` elements (invariant; `B` is
        // sealed, so `as_mut` returns the same words on every call).
        unsafe { layout::write(self.words.as_mut(), self.bit_width, self.mask, index, word) };
    }

    /// Replaces the element at `index` with `value` and returns the old element.
    ///
    /// # Panics
    ///
    /// If `index >= len`, or if `value` does not fit the bit width.
    #[track_caller]
    pub fn replace(&mut self, index: usize, value: T) -> T {
        self.check_index(index);
        let word = fitting_word(value, self.bit_width, self.mask);
        let (width, mask) = (self.bit_width, self.mask);
        let words = self.words.as_mut();
        // The read and the write follow each other with no call between them, so the compiler
        // can hoist their two width tests out of a caller's loop together.
        // SAFETY: `index < len`, and the words hold `len` elements (invariant).
        let old = unsafe {
            let old = layout::read(words, width, mask, index);
            layout::write(words, width, mask, index, word);
            old
        };
        T::from_word(old)
    }

    /// Swaps the elements at `a` and `b`.
    ///
    /// # Panics
    ///
    /// If `a >= len` or `b >= len`.
    #[track_caller]
    pub fn swap(&mut self, a: usize, b: usize) {
        self.check_index(a);
        self.check_index(b);
        let (width, mask) = (self.bit_width, self.mask);
        let words = self.words.as_mut();
        // SAFETY: `a < len` and `b < len`, and the words hold `len` elements (invariant).
        unsafe {
            let value_a = layout::read(words, width, mask, a);
            let value_b = layout::read(words, width, mask, b);
            layout::write(words, width, mask, a, value_b);
            layout::write(words, width, mask, b, value_a);
        }
    }

    /// Replaces every element with `value`.
    ///
    /// # Panics
    ///
    /// If `value` does not fit the bit width.
    #[track_caller]
    pub fn fill(&mut self, value: T) {
        let word = fitting_word(value, self.bit_width, self.mask);
        let (width, mask, len) = (self.bit_width, self.mask, self.len);
        let words = self.words.as_mut();
        for index in 0..len {
            // SAFETY: `index < len`, and the words hold `len` elements (invariant).
            unsafe { layout::write_in_order(words, width, mask, index, word) };
        }
    }

    /// Replaces every element, in order, with a value that `f` returns.
    ///
    /// # Panics
    ///
    /// If a value does not fit the bit width. The elements before it are already replaced.
    #[track_caller]
    pub fn fill_with(&mut self, mut f: impl FnMut() -> T) {
        let (width, mask, len) = (self.bit_width, self.mask, self.len);
        let words = self.words.as_mut();
        for index in 0..len {
            let word = fitting_word(f(), width, mask);
            // SAFETY: `index < len`, and the words hold `len` elements (invariant).
            unsafe { layout::write_in_order(words, width, mask, index, word) };
        }
    }

    /// Replaces every element `x` with `f(x)`.
    ///
    /// # Panics
    ///
    /// If `f` returns a value that does not fit the bit width. The vector stays valid, and the
    /// updates completed before the panic remain stored. An element that straddles two words
    /// can hold bits of both its old and its new value. The same holds if `f` panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::FixedVec;
    ///
    /// let mut vec = FixedVec::<u32>::new(5)?;
    /// vec.extend_from_slice(&[1, 2, 3]);
    /// vec.map_in_place(|x| x * 10);
    /// assert_eq!(vec, [10, 20, 30][..]);
    /// # Ok::<(), compressed_intvec::BitWidthTooLarge>(())
    /// ```
    pub fn map_in_place(&mut self, mut f: impl FnMut(T) -> T) {
        let (width, mask, len) = (self.bit_width, self.mask, self.len);
        layout::map(self.words.as_mut(), len, width, mask, |word| {
            fitting_word(f(T::from_word(word)), width, mask)
        });
    }

    /// Rotates the elements so that the element at `mid` becomes the first.
    ///
    /// # Panics
    ///
    /// If `mid > len`.
    #[track_caller]
    pub fn rotate_left(&mut self, mid: usize) {
        let len = self.len;
        assert!(mid <= len, "mid (is {mid}) should be <= len (is {len})");
        // Reversing both parts and then the whole rotates in place.
        self.reverse_range(0, mid);
        self.reverse_range(mid, len);
        self.reverse_range(0, len);
    }

    /// Reverses the elements `[start, end)`, where `end <= len`.
    fn reverse_range(&mut self, start: usize, end: usize) {
        let (width, mask) = (self.bit_width, self.mask);
        let words = self.words.as_mut();
        let (mut low, mut high) = (start, end);
        while low + 1 < high {
            high -= 1;
            // SAFETY: `low < high < end <= len`, and the words hold `len` elements (invariant).
            unsafe {
                let a = layout::read(words, width, mask, low);
                let b = layout::read(words, width, mask, high);
                layout::write_in_order(words, width, mask, low, b);
                layout::write_in_order(words, width, mask, high, a);
            }
            low += 1;
        }
    }

    /// Rotates the elements so that the element at `len - k` becomes the first.
    ///
    /// # Panics
    ///
    /// If `k > len`.
    #[track_caller]
    pub fn rotate_right(&mut self, k: usize) {
        let len = self.len;
        assert!(k <= len, "k (is {k}) should be <= len (is {len})");
        self.rotate_left(len - k);
    }
}

impl<T: Storable + fmt::Debug, B: Backend> fmt::Debug for FixedVec<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl<T: Storable + PartialEq, B: Backend, B2: Backend> PartialEq<FixedVec<T, B2>>
    for FixedVec<T, B>
{
    /// Compares the bit widths and the elements. Two vectors with the same elements and
    /// different bit widths differ, because they accept different values. To compare only the
    /// elements, compare [`as_slice`](FixedVec::as_slice) views.
    fn eq(&self, other: &FixedVec<T, B2>) -> bool {
        self.bit_width == other.bit_width && self.as_slice() == other.as_slice()
    }
}

impl<T: Storable + Eq, B: Backend> Eq for FixedVec<T, B> {}

impl<T: Storable + PartialEq, B: Backend> PartialEq<[T]> for FixedVec<T, B> {
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice() == *other
    }
}

impl<T: Storable + Hash, B: Backend> Hash for FixedVec<T, B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bit_width.hash(state);
        self.as_slice().hash(state);
    }
}

impl<T: Storable> Default for FixedVec<T> {
    /// Creates an empty vector with the largest bit width of `T`, which holds every value.
    fn default() -> Self {
        let bit_width = max_bit_width::<T>();
        Self {
            words: Vec::new(),
            len: 0,
            bit_width,
            mask: layout::mask(bit_width),
            element: PhantomData,
        }
    }
}

impl<T: Storable> From<&[T]> for FixedVec<T> {
    /// Creates a vector with the smallest bit width, as [`FixedVec::from_slice`].
    fn from(values: &[T]) -> Self {
        Self::from_slice(values)
    }
}

impl<T: Storable> FromIterator<T> for FixedVec<T> {
    /// Collects the values with the smallest bit width, as [`FixedVec::from_slice`]. The values
    /// are buffered in a `Vec<T>` first.
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        let values: Vec<T> = values.into_iter().collect();
        Self::from_slice(&values)
    }
}

impl<T: Storable> Extend<T> for FixedVec<T> {
    /// Appends the values.
    ///
    /// # Panics
    ///
    /// If a value does not fit the bit width. The values before it are already appended.
    fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        let values = values.into_iter();
        self.reserve(values.size_hint().0);
        for value in values {
            self.push(value);
        }
    }
}

impl<'a, T: Storable, B: Backend> IntoIterator for &'a FixedVec<T, B> {
    type Item = T;
    type IntoIter = FixedVecIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Storable> IntoIterator for FixedVec<T> {
    type Item = T;
    type IntoIter = FixedVecIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        // SAFETY: the words hold `len` elements (invariant).
        unsafe { FixedVecIntoIter::new(self.words, self.len, self.bit_width, self.mask) }
    }
}

/// A borrowed, read-only view of a range in a [`FixedVec`].
///
/// [`FixedVec::slice`] creates a view sharing the vector's packed words. Indices are relative
/// to the start of the view, and access takes constant time.
///
/// A view is `Copy`. Iterators and subviews borrow the underlying words for `'a`, so they may
/// outlive the view that created them.
/// Equality between views compares decoded elements, regardless of bit width.
///
/// # Examples
///
/// ```
/// use compressed_intvec::FixedVec;
///
/// let vec: FixedVec<u32> = (0..100).collect();
/// let tail = vec.slice(90..).unwrap();
/// assert_eq!(tail.len(), 10);
/// assert_eq!(tail.get(0), Some(90));
/// assert_eq!(tail.binary_search(&95), Ok(5));
/// ```
#[derive(Clone, Copy)]
pub struct FixedVecSlice<'a, T> {
    // Invariant: the elements `[start, start + len)` are elements of a `FixedVec` whose words
    // are `words`, so `words` holds `start + len` elements.
    words: &'a [u64],
    start: usize,
    len: usize,
    bit_width: usize,
    mask: u64,
    element: PhantomData<fn() -> T>,
}

impl<'a, T: Storable> FixedVecSlice<'a, T> {
    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns element `index` of the view.
    ///
    /// # Safety
    ///
    /// `index < self.len`.
    #[inline(always)]
    unsafe fn element(&self, index: usize) -> T {
        debug_assert!(index < self.len);
        // SAFETY: `start + index < start + len`, and `words` holds `start + len` elements
        // (invariant).
        T::from_word(unsafe {
            layout::read(self.words, self.bit_width, self.mask, self.start + index)
        })
    }

    /// Returns the element at `index` of the view, or `None` if `index >= len`.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        // SAFETY: `index < len`.
        (index < self.len).then(|| unsafe { self.element(index) })
    }

    /// Returns the first element, or `None` if the view is empty.
    pub fn first(&self) -> Option<T> {
        self.get(0)
    }

    /// Returns the last element, or `None` if the view is empty.
    pub fn last(&self) -> Option<T> {
        self.get(self.len.checked_sub(1)?)
    }

    /// Returns an iterator yielding decoded elements by value.
    #[inline]
    pub fn iter(&self) -> FixedVecIter<'a, T> {
        // SAFETY: `words` holds `start + len` elements (invariant).
        unsafe {
            FixedVecIter::new(
                self.words,
                self.start,
                self.start + self.len,
                self.bit_width,
                self.mask,
            )
        }
    }

    /// Returns a subview of `range`, or `None` if the range is reversed or out of bounds.
    ///
    /// The range is relative to this view. The subview borrows the original vector's words.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Option<FixedVecSlice<'a, T>> {
        let (start, end) = bounds(range, self.len)?;
        Some(FixedVecSlice {
            start: self.start + start,
            len: end - start,
            ..*self
        })
    }

    /// Searches a sorted view for `value` in logarithmic time.
    ///
    /// Returns `Ok(index)` for a match or `Err(index)` for its insertion position, relative to
    /// the view. If multiple elements match, any matching index may be returned. The view
    /// must be sorted in ascending order, as for [`slice::binary_search`].
    pub fn binary_search(&self, value: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|element| element.cmp(value))
    }

    /// Searches a sorted view with `compare`, as [`slice::binary_search_by`].
    ///
    /// `compare` must return the ordering of each element relative to the target, consistently
    /// with the view's sort order. Returns a matching index or insertion position within the view.
    pub fn binary_search_by(
        &self,
        mut compare: impl FnMut(&T) -> std::cmp::Ordering,
    ) -> Result<usize, usize> {
        use std::cmp::Ordering::{Equal, Greater, Less};
        // The algorithm of `slice::binary_search_by`: the loop narrows `[base, base + size)` to
        // one element with a select instead of a branch on each comparison, and never exits
        // early, so the number of steps depends only on the length.
        let mut size = self.len;
        if size == 0 {
            return Err(0);
        }
        let mut base = 0;
        while size > 1 {
            let half = size / 2;
            let mid = base + half;
            // SAFETY: `mid < base + size <= len`.
            let order = compare(&unsafe { self.element(mid) });
            base = std::hint::select_unpredictable(order == Greater, base, mid);
            size -= half;
        }
        // SAFETY: `base < base + size <= len`, with `size == 1`.
        match compare(&unsafe { self.element(base) }) {
            Equal => Ok(base),
            Less => Err(base + 1),
            Greater => Err(base),
        }
    }

    /// Searches a view sorted by the keys returned by `extract`, as
    /// [`slice::binary_search_by_key`].
    ///
    /// Returns a matching index or the insertion position of `key`, relative to the view.
    pub fn binary_search_by_key<K: Ord>(
        &self,
        key: &K,
        mut extract: impl FnMut(T) -> K,
    ) -> Result<usize, usize> {
        self.binary_search_by(|&element| extract(element).cmp(key))
    }

    /// Returns the index of the first element for which `predicate` is false.
    ///
    /// All elements for which `predicate` is true must precede those for which it is false,
    /// as for [`slice::partition_point`]. Returns the view's length if every element satisfies
    /// the predicate. The result is relative to the view.
    pub fn partition_point(&self, mut predicate: impl FnMut(&T) -> bool) -> usize {
        self.binary_search_by(|element| {
            if predicate(element) {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        })
        .unwrap_or_else(|index| index)
    }
}

impl<T: Storable + fmt::Debug> fmt::Debug for FixedVecSlice<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Storable + PartialEq> PartialEq<FixedVecSlice<'_, T>> for FixedVecSlice<'_, T> {
    /// Compares the elements; the bit widths may differ.
    fn eq(&self, other: &FixedVecSlice<'_, T>) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<T: Storable + Eq> Eq for FixedVecSlice<'_, T> {}

impl<T: Storable + PartialEq> PartialEq<[T]> for FixedVecSlice<'_, T> {
    fn eq(&self, other: &[T]) -> bool {
        self.len == other.len() && self.iter().eq(other.iter().copied())
    }
}

impl<T: Storable + Hash> Hash for FixedVecSlice<'_, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len);
        for element in self.iter() {
            element.hash(state);
        }
    }
}

impl<'a, T: Storable> IntoIterator for FixedVecSlice<'a, T> {
    type Item = T;
    type IntoIter = FixedVecIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
