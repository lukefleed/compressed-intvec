//! The element types that every vector of this crate stores.

/// Converts copyable elements to and from the integer representation used by every vector.
///
/// Implement this trait to store an integer newtype in [`FixedVec`](crate::FixedVec),
/// [`AtomicFixedVec`](crate::AtomicFixedVec), [`VarVec`](crate::VarVec) or
/// [`SeqVec`](crate::SeqVec). Primitive integer types up to 64 bits, including `usize` and
/// `isize`, already implement it.
///
/// Vectors compress the `u64` word returned by [`to_word`](Storable::to_word) and reconstruct
/// an element with [`from_word`](Storable::from_word). Unsigned integers use their value as
/// the word. Signed integers use zig-zag encoding, which maps
/// `0, -1, 1, -2, 2, ...` to `0, 1, 2, 3, 4, ...`, so that values close to zero need few bits.
///
/// # Contract
///
/// An implementation must satisfy, for every value `x`:
///
/// - `x.to_word() < 2^BITS`;
/// - `Self::from_word(x.to_word()) == x`;
/// - `to_word` is deterministic: every call on `x` returns the same word.
///
/// The vectors of this crate pass `from_word` only words below `2^BITS`. Given any other word it
/// may return any value, and it should not panic. The implementations in this crate truncate.
///
/// The contract does not require `Self::from_word(word).to_word() == word` for every admitted
/// word. Distinct stored words may decode to the same element. Vector equality and hashing use
/// the element's equality and hash traits, while serialization preserves the stored element
/// words. [`AtomicFixedVec::compare_exchange`](crate::AtomicFixedVec::compare_exchange) compares
/// encoded words, so passing a loaded element back as `current` may fail when its stored word
/// differs from `current.to_word()`, including when the stored word is unchanged.
///
/// The unsafe code of this crate does not rely on this trait: an implementation that breaks the
/// contract makes the vectors return wrong values or panic, but never causes undefined behavior.
/// For example, some constructors call `to_word` twice on each value, once to choose a bit width
/// or a codec and once to store the value, so a `to_word` that is not deterministic can make them
/// store words that do not fit their choice: they then panic, or return a vector whose elements
/// differ from the input.
///
/// # Portability
///
/// `usize` and `isize` have `BITS` equal to the pointer width of the target. A serialized vector
/// loads as any element type of the same signedness whose `BITS` can hold its contents: for a
/// [`FixedVec`](crate::FixedVec) or an [`AtomicFixedVec`](crate::AtomicFixedVec), a bit width of
/// at most `BITS`; for a [`VarVec`](crate::VarVec) or a [`SeqVec`](crate::SeqVec), every word
/// below `2^BITS`. Loading fails if these bounds are exceeded. A vector of
/// `usize` written on a 64-bit target loads on a 32-bit target when its elements fit in 32 bits.
///
/// # Examples
///
/// ```
/// use compressed_intvec::{FixedVec, Storable};
///
/// #[derive(Clone, Copy, Debug, PartialEq)]
/// struct NodeId(u32);
///
/// impl Storable for NodeId {
///     const BITS: u32 = 32;
///     fn to_word(self) -> u64 {
///         self.0.into()
///     }
///     fn from_word(word: u64) -> Self {
///         NodeId(word as u32)
///     }
/// }
///
/// let ids = FixedVec::from_slice(&[NodeId(3), NodeId(7)]);
/// assert_eq!(ids.get(1), Some(NodeId(7)));
/// assert_eq!(ids.bit_width(), 3);
/// ```
pub trait Storable: Copy {
    /// The number of bits available for an element's integer representation.
    ///
    /// Every word returned by [`to_word`](Storable::to_word) must be below `2^BITS`.
    /// Fixed-width vectors use a bit width of at most this value or 64, whichever is smaller.
    const BITS: u32;

    /// Whether the stored word is the zig-zag encoding of a two's complement integer. Defaults to
    /// `false`.
    ///
    /// Determines the wrapping semantics of
    /// [`AtomicFixedVec::fetch_add`](crate::AtomicFixedVec::fetch_add) and
    /// [`fetch_sub`](crate::AtomicFixedVec::fetch_sub). Serialized vectors record this flag and
    /// require the same value when loaded, preventing signed and unsigned representations from
    /// being mixed.
    const SIGNED: bool = false;

    /// Converts `self` to its integer representation, which must be below `2^BITS`.
    fn to_word(self) -> u64;

    /// Reconstructs an element from its integer representation.
    ///
    /// Must recover every value passed through [`to_word`](Storable::to_word). Other words
    /// below `2^BITS` may also decode to that value, as described in the trait's contract.
    fn from_word(word: u64) -> Self;
}

/// Maps a two's complement integer to its zig-zag encoding.
#[inline]
pub(crate) fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

/// Inverts [`zigzag`].
#[inline]
pub(crate) fn unzigzag(word: u64) -> i64 {
    ((word >> 1) as i64) ^ -((word & 1) as i64)
}

impl Storable for u8 {
    const BITS: u32 = u8::BITS;
    const SIGNED: bool = false;
    #[inline]
    fn to_word(self) -> u64 {
        self.into()
    }
    #[inline]
    fn from_word(word: u64) -> Self {
        word as u8
    }
}

impl Storable for u16 {
    const BITS: u32 = u16::BITS;
    const SIGNED: bool = false;
    #[inline]
    fn to_word(self) -> u64 {
        self.into()
    }
    #[inline]
    fn from_word(word: u64) -> Self {
        word as u16
    }
}

impl Storable for u32 {
    const BITS: u32 = u32::BITS;
    const SIGNED: bool = false;
    #[inline]
    fn to_word(self) -> u64 {
        self.into()
    }
    #[inline]
    fn from_word(word: u64) -> Self {
        word as u32
    }
}

impl Storable for u64 {
    const BITS: u32 = u64::BITS;
    const SIGNED: bool = false;
    #[inline]
    fn to_word(self) -> u64 {
        self
    }
    #[inline]
    fn from_word(word: u64) -> Self {
        word
    }
}

impl Storable for usize {
    const BITS: u32 = usize::BITS;
    const SIGNED: bool = false;
    #[inline]
    fn to_word(self) -> u64 {
        // Lossless: `usize` has at most 64 bits (the assertion in `lib.rs`).
        self as u64
    }
    #[inline]
    fn from_word(word: u64) -> Self {
        word as usize
    }
}

impl Storable for i8 {
    const BITS: u32 = i8::BITS;
    const SIGNED: bool = true;
    #[inline]
    fn to_word(self) -> u64 {
        zigzag(self.into())
    }
    #[inline]
    fn from_word(word: u64) -> Self {
        unzigzag(word) as i8
    }
}

impl Storable for i16 {
    const BITS: u32 = i16::BITS;
    const SIGNED: bool = true;
    #[inline]
    fn to_word(self) -> u64 {
        zigzag(self.into())
    }
    #[inline]
    fn from_word(word: u64) -> Self {
        unzigzag(word) as i16
    }
}

impl Storable for i32 {
    const BITS: u32 = i32::BITS;
    const SIGNED: bool = true;
    #[inline]
    fn to_word(self) -> u64 {
        zigzag(self.into())
    }
    #[inline]
    fn from_word(word: u64) -> Self {
        unzigzag(word) as i32
    }
}

impl Storable for i64 {
    const BITS: u32 = i64::BITS;
    const SIGNED: bool = true;
    #[inline]
    fn to_word(self) -> u64 {
        zigzag(self)
    }
    #[inline]
    fn from_word(word: u64) -> Self {
        unzigzag(word)
    }
}

impl Storable for isize {
    const BITS: u32 = isize::BITS;
    const SIGNED: bool = true;
    #[inline]
    fn to_word(self) -> u64 {
        // Lossless: `isize` has at most 64 bits (the assertion in `lib.rs`).
        zigzag(self as i64)
    }
    #[inline]
    fn from_word(word: u64) -> Self {
        unzigzag(word) as isize
    }
}
