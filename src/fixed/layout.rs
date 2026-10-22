//! The bit layout shared by [`FixedVec`](super::FixedVec) and
//! [`AtomicFixedVec`](super::AtomicFixedVec).
//!
//! Element `i` of a vector with bit width `w` occupies the bits `[i * w, (i + 1) * w)` of the
//! word stream, and stream bit `k` is bit `k % 64` of word `k / 64` (little-endian bit order).
//! Both vector types follow this layout and reuse the words during conversion.

/// The number of bits in a storage word.
pub(crate) const WORD_BITS: usize = 64;

/// Returns the mask of the low `bit_width` bits. `bit_width` is at most 64.
#[inline]
pub(crate) fn mask(bit_width: usize) -> u64 {
    if bit_width == WORD_BITS {
        u64::MAX
    } else {
        (1 << bit_width) - 1
    }
}

/// Returns the number of words that `len` elements of `bit_width` bits need: the data words,
/// plus one padding word when `len > 0`, or `None` if the count overflows `usize`.
///
/// The padding word lets [`read`] load eight bytes starting at the byte of any element.
#[inline]
pub(crate) fn required_words(len: usize, bit_width: usize) -> Option<usize> {
    if len == 0 {
        return Some(0);
    }
    len.checked_mul(bit_width)?
        .div_ceil(WORD_BITS)
        .checked_add(1)
}

/// Returns the word that holds the first bit of element `index`, and the offset of that bit.
///
/// `index * bit_width` must fit `usize`, which holds for every `index < len` of a vector: its
/// invariant states that `required_words(len, bit_width)`, and so `len * bit_width`, does not
/// overflow.
#[inline]
pub(crate) fn locate(index: usize, bit_width: usize) -> (usize, usize) {
    let bit = index * bit_width;
    (bit / WORD_BITS, bit % WORD_BITS)
}

/// Returns the position of element `index` of 8, 16 or 32 bits among the `bit_width`-bit units
/// of the words, where unit `i` occupies the bytes `[i * bit_width / 8, (i + 1) * bit_width / 8)`
/// of memory.
///
/// Such an element is an aligned unit of its own size, in the word `index * bit_width / 64`. The
/// words are native-endian, and so is a load or store of one unit, so the unit holds the
/// element's bits in their order on every target. On little-endian targets, unit `i` holds the
/// stream bits `[i * bit_width, (i + 1) * bit_width)`, which are element `i`. On big-endian
/// targets, a word stores its most significant unit at its lowest address, so the units within
/// each word come in reverse order: element `index` is unit `index ^ (64 / bit_width - 1)`,
/// which flips only the position within the word.
#[inline(always)]
fn unit(index: usize, bit_width: usize) -> usize {
    if cfg!(target_endian = "little") {
        index
    } else {
        index ^ (WORD_BITS / bit_width - 1)
    }
}

/// Returns the `bit_width` bits of element `index`, as a word.
///
/// # Safety
///
/// `bit_width <= 64`, `mask == mask(bit_width)`, and
/// `required_words(index + 1, bit_width) <= words.len()`. Every index below the length of a
/// vector whose words satisfy its invariant meets the last condition.
#[inline(always)]
pub(crate) unsafe fn read(words: &[u64], bit_width: usize, mask: u64, index: usize) -> u64 {
    debug_assert!(
        required_words(index + 1, bit_width).is_some_and(|required| required <= words.len())
    );
    if bit_width == WORD_BITS {
        // SAFETY: with 64-bit elements, element `index` is word `index`, and
        // `index < required_words(index + 1, 64) <= words.len()`.
        return unsafe { *words.get_unchecked(index) };
    }
    if bit_width.is_power_of_two() && bit_width >= 8 {
        // Elements of 8, 16 and 32 bits are aligned units of their own size: one load of
        // exactly their bytes, which touches one cache line and needs no shift or mask.
        let at = words.as_ptr();
        // SAFETY: the unit of the element is in the element's word, which is below
        // `required_words(index + 1, bit_width) <= words.len()`, and the words are aligned to
        // 8 bytes, so the unit is aligned to its size.
        return unsafe {
            if bit_width > 16 {
                u64::from(*at.cast::<u32>().add(unit(index, 32)))
            } else if bit_width > 8 {
                u64::from(*at.cast::<u16>().add(unit(index, 16)))
            } else {
                u64::from(*at.cast::<u8>().add(unit(index, 8)))
            }
        };
    }
    let bit = index * bit_width;
    if bit_width.is_power_of_two() || bit_width == 0 {
        // Widths that divide 64 never straddle words: one aligned load, which never touches
        // two cache lines as an unaligned load can. Width 0 reads word 0 and masks it to 0.
        // SAFETY: the element's word is below `((index + 1) * bit_width).div_ceil(64) <
        // words.len()`.
        let word = unsafe { *words.get_unchecked(bit / WORD_BITS) };
        return (word >> (bit % WORD_BITS)) & mask;
    }
    // On little-endian targets, bytes follow the stream order. One unaligned load reads the
    // eight bytes from the byte of the element's first bit. It holds the whole element when
    // `bit % 8 + bit_width <= 64` for every element, which is the case for every width up to 58
    // and for 60 (where `bit % 8` is 0 or 4).
    if cfg!(target_endian = "little") {
        // Every load below stays within bytes `[byte, byte + 9)`, which are within `words`: the
        // element's bits end at `(index + 1) * bit_width > bit` (`bit_width >= 3` here), so the
        // data words hold at least `byte + 1` bytes, and the padding word eight more.
        let (bytes, byte, shift) = (words.as_ptr().cast::<u8>(), bit / 8, bit % 8);
        if bit_width <= 58 || bit_width == 60 {
            // SAFETY: bytes `[byte, byte + 8)`, see above.
            let word = unsafe { bytes.add(byte).cast::<u64>().read_unaligned() };
            return (word >> shift) & mask;
        }
        // Widths 59 and 61 to 63: the element's first byte, and the eight bytes after it, whose
        // shift is `8 - shift`, between 1 and 8, so no shift reaches 64. The first load differs
        // from the one above, so that the compiler does not merge the two paths into one loop
        // that tests the width on every element.
        // SAFETY: bytes `[byte, byte + 9)`, see above.
        let (first, rest) = unsafe {
            (
                *bytes.add(byte),
                bytes.add(byte + 1).cast::<u64>().read_unaligned(),
            )
        };
        return ((u64::from(first) >> shift) | (rest << (8 - shift))) & mask;
    }
    // Big-endian targets read the element's word and the next one, whether or not the element
    // straddles them: with random indices, a branch on the straddle mispredicts, and the word
    // after the element's first word always exists.
    let (word, offset) = (bit / WORD_BITS, bit % WORD_BITS);
    // SAFETY: `bit_width >= 1` here, so the element's bits end at `(index + 1) * bit_width >
    // 64 * word`, and `word + 1 <= ((index + 1) * bit_width).div_ceil(64) <
    // required_words(index + 1, bit_width) <= words.len()`.
    let (low, high) = unsafe { (*words.get_unchecked(word), *words.get_unchecked(word + 1)) };
    // `high << 1 << (63 - offset)` is `high << (64 - offset)`, and 0 when `offset` is 0.
    ((low >> offset) | (high << 1 << (63 - offset))) & mask
}

/// Stores `value` as element `index` through a read-modify-write of the surrounding storage.
///
/// Each call accesses the bytes or the two words around the element. With consecutive indices,
/// each call loads bytes that the previous call stored. [`write_in_order`] writes sequential
/// elements through their containing words.
///
/// A `value` that does not fit `mask` overwrites bits of the neighboring elements.
///
/// # Safety
///
/// As for [`read`].
#[inline(always)]
pub(crate) unsafe fn write(
    words: &mut [u64],
    bit_width: usize,
    mask: u64,
    index: usize,
    value: u64,
) {
    debug_assert!(
        required_words(index + 1, bit_width).is_some_and(|required| required <= words.len())
    );
    debug_assert_eq!(value & !mask, 0);
    if bit_width == WORD_BITS {
        // SAFETY: with 64-bit elements, element `index` is word `index`, below `words.len()` by
        // the precondition.
        unsafe { *words.get_unchecked_mut(index) = value };
        return;
    }
    if bit_width.is_power_of_two() && bit_width >= 8 {
        // Elements of 8, 16 and 32 bits: one store of exactly their unit, with no load.
        let at = words.as_mut_ptr();
        // SAFETY: the unit is within `words` and aligned, as in `read`.
        unsafe {
            if bit_width > 16 {
                *at.cast::<u32>().add(unit(index, 32)) = value as u32;
            } else if bit_width > 8 {
                *at.cast::<u16>().add(unit(index, 16)) = value as u16;
            } else {
                *at.cast::<u8>().add(unit(index, 8)) = value as u8;
            }
        }
        return;
    }
    let bit = index * bit_width;
    if bit_width.is_power_of_two() || bit_width == 0 {
        // SAFETY: the element's word is below `words.len()`, as in `read`.
        let word = unsafe { words.get_unchecked_mut(bit / WORD_BITS) };
        let offset = bit % WORD_BITS;
        *word = (*word & !(mask << offset)) | (value << offset);
        return;
    }
    if cfg!(target_endian = "little") {
        // Every access below stays within bytes `[byte, byte + 9)`, which are within `words`, as
        // in `read`.
        let (byte, shift) = (bit / 8, bit % 8);
        // SAFETY: byte `byte` is within `words`, see above.
        let at = unsafe { words.as_mut_ptr().cast::<u8>().add(byte) };
        // With random indices beyond the caches, a four-byte access is faster than an
        // eight-byte one, so every element that four bytes hold is written through four bytes.
        if bit_width <= 25 {
            // `shift + bit_width <= 32`: four bytes hold the element.
            // SAFETY: bytes `[byte, byte + 4)`, see above.
            unsafe {
                let at = at.cast::<u32>();
                let (mask, value) = ((mask as u32) << shift, (value as u32) << shift);
                at.write_unaligned((at.read_unaligned() & !mask) | value);
            }
            return;
        }
        if bit_width <= 58 || bit_width == 60 {
            // SAFETY: bytes `[byte, byte + 8)`, see above.
            unsafe {
                let at = at.cast::<u64>();
                at.write_unaligned((at.read_unaligned() & !(mask << shift)) | (value << shift));
            }
            return;
        }
        // Widths 59 and 61 to 63, as `read` loads them: the element's first byte, and the eight
        // bytes after it, whose shift `8 - shift` is between 1 and 8.
        // SAFETY: bytes `[byte, byte + 9)`, see above.
        unsafe {
            *at = (*at & ((1 << shift) - 1)) | (value << shift) as u8;
            let rest = at.add(1).cast::<u64>();
            let (mask, value) = (mask >> (8 - shift), value >> (8 - shift));
            rest.write_unaligned((rest.read_unaligned() & !mask) | value);
        }
        return;
    }
    // Big-endian targets write both words, whether or not the element straddles them: when it
    // does not, the bits for the second word are 0 and the word keeps its value.
    let (word, offset) = (bit / WORD_BITS, bit % WORD_BITS);
    // SAFETY: `word + 1 < words.len()`, as in `read`.
    let (low, high) = unsafe { (*words.get_unchecked(word), *words.get_unchecked(word + 1)) };
    // `x >> 1 >> (63 - offset)` is `x >> (64 - offset)`, and 0 when `offset` is 0.
    let (high_mask, high_value) = (mask >> 1 >> (63 - offset), value >> 1 >> (63 - offset));
    // SAFETY: as above.
    unsafe {
        *words.get_unchecked_mut(word) = (low & !(mask << offset)) | (value << offset);
        *words.get_unchecked_mut(word + 1) = (high & !high_mask) | high_value;
    }
}

/// Stores `value` as element `index`, for callers that write consecutive indices.
///
/// Elements of 8, 16 and 32 bits are stored as in [`write`]. For the other widths, it reads,
/// modifies and stores the element's word, and the next word only when the element straddles
/// them. The branch on the straddle follows a periodic pattern over consecutive
/// indices, which the processor predicts.
///
/// A `value` that does not fit `mask` overwrites bits of the neighboring elements.
///
/// # Safety
///
/// As for [`read`].
#[inline(always)]
pub(crate) unsafe fn write_in_order(
    words: &mut [u64],
    bit_width: usize,
    mask: u64,
    index: usize,
    value: u64,
) {
    debug_assert!(
        required_words(index + 1, bit_width).is_some_and(|required| required <= words.len())
    );
    debug_assert_eq!(value & !mask, 0);
    if bit_width == WORD_BITS {
        // SAFETY: with 64-bit elements, element `index` is word `index`, below `words.len()` by
        // the precondition.
        unsafe { *words.get_unchecked_mut(index) = value };
        return;
    }
    if bit_width.is_power_of_two() && bit_width >= 8 {
        // SAFETY: the precondition of `write` is the same.
        unsafe { write(words, bit_width, mask, index, value) };
        return;
    }
    let (word, offset) = locate(index, bit_width);
    // SAFETY: `word` holds the first bit of the element, so it is below `words.len()` as in
    // `read`.
    let low = unsafe { words.get_unchecked_mut(word) };
    *low = (*low & !(mask << offset)) | (value << offset);
    if offset + bit_width > WORD_BITS {
        let shift = WORD_BITS - offset;
        // SAFETY: the element continues into word `word + 1`, which is below `words.len()` as
        // in `read`.
        let high = unsafe { words.get_unchecked_mut(word + 1) };
        *high = (*high & !(mask >> shift)) | (value >> shift);
    }
}

/// Stores `values` as elements `0, 1, ...`, overwriting each data word that holds element bits.
///
/// Every value must fit `bit_width` bits, and `words` must hold the data words of all the values,
/// or the function panics. The padding word retains its contents.
pub(crate) fn pack(words: &mut [u64], bit_width: usize, values: impl IntoIterator<Item = u64>) {
    // `pending` holds the low `filled` bits of the next word.
    let mut pending = 0u64;
    let mut filled = 0;
    let mut next = 0;
    for value in values {
        pending |= value << filled;
        filled += bit_width;
        if filled >= WORD_BITS {
            words[next] = pending;
            next += 1;
            filled -= WORD_BITS;
            // The bits of `value` that did not fit start the next word.
            pending = if filled == 0 {
                0
            } else {
                value >> (bit_width - filled)
            };
        }
    }
    if filled > 0 {
        words[next] = pending;
    }
}

/// Moves the bits `[start + shift, start + shift + count)` to `[start, start + count)`.
///
/// `shift` is at most 64, and `count` is 0 when `shift` is 0. The source and destination bits
/// are inside `words`. When `shift` is 64, `start` and `count` are multiples of 64.
pub(crate) fn shift_left(words: &mut [u64], start: usize, shift: usize, count: usize) {
    if count == 0 {
        return;
    }
    if shift == WORD_BITS {
        // `start` is a multiple of 64, because it is an element boundary at width 64.
        let first = start / WORD_BITS;
        words.copy_within(first + 1..first + 1 + count / WORD_BITS, first);
        return;
    }
    let end = start + count;
    let (first, last) = (start / WORD_BITS, (end - 1) / WORD_BITS);
    // Word `i` takes its bits from words `i` and `i + 1`, which are still unchanged when word `i`
    // is written, because the loop runs upward.
    for i in first..=last {
        let low = words[i] >> shift;
        let high = words
            .get(i + 1)
            .map_or(0, |&next| next << (WORD_BITS - shift));
        let mut keep = 0;
        if i == first {
            keep |= mask(start % WORD_BITS);
        }
        if i == last && !end.is_multiple_of(WORD_BITS) {
            keep |= !mask(end % WORD_BITS);
        }
        words[i] = (words[i] & keep) | ((low | high) & !keep);
    }
}

/// Moves the bits `[start, start + count)` to `[start + shift, start + shift + count)`, and leaves
/// the bits `[start, start + shift)` unspecified.
///
/// `shift` is at most 64, and `count` is 0 when `shift` is 0. The source and destination bits
/// are inside `words`. When `shift` is 64, `start` and `count` are multiples of 64.
pub(crate) fn shift_right(words: &mut [u64], start: usize, shift: usize, count: usize) {
    if count == 0 {
        return;
    }
    if shift == WORD_BITS {
        let first = start / WORD_BITS;
        words.copy_within(first..first + count / WORD_BITS, first + 1);
        return;
    }
    let (dest_start, dest_end) = (start + shift, start + shift + count);
    let (first, last) = (dest_start / WORD_BITS, (dest_end - 1) / WORD_BITS);
    // Word `i` takes its bits from words `i - 1` and `i`, which are still unchanged when word `i`
    // is written, because the loop runs downward.
    for i in (first..=last).rev() {
        let high = words[i] << shift;
        let low = if i > 0 {
            words[i - 1] >> (WORD_BITS - shift)
        } else {
            0
        };
        let mut keep = 0;
        if i == first {
            keep |= mask(dest_start % WORD_BITS);
        }
        if i == last && !dest_end.is_multiple_of(WORD_BITS) {
            keep |= !mask(dest_end % WORD_BITS);
        }
        words[i] = (words[i] & keep) | ((low | high) & !keep);
    }
}

/// Replaces each of the first `len` elements with `f` of it, word by word.
///
/// `f` must enforce that each returned value fits `mask`: a wider value overwrites bits of
/// neighboring elements. If `f` panics, completed writes remain stored, and an element that
/// straddles two words can hold bits of both its old and its new value.
pub(crate) fn map(
    words: &mut [u64],
    len: usize,
    bit_width: usize,
    mask: u64,
    mut f: impl FnMut(u64) -> u64,
) {
    if len == 0 {
        return;
    }
    if bit_width == 0 {
        // Every element is 0; `f` still sees each one, so that it can check its result.
        for _ in 0..len {
            f(0);
        }
        return;
    }
    if WORD_BITS.is_multiple_of(bit_width) {
        // Elements never straddle words: rebuild each word from its elements.
        let per_word = WORD_BITS / bit_width;
        for (i, word) in words[..len.div_ceil(per_word)].iter_mut().enumerate() {
            let count = per_word.min(len - i * per_word);
            let old = *word;
            let mut new = 0;
            for j in 0..count {
                let shift = j * bit_width;
                new |= f((old >> shift) & mask) << shift;
            }
            *word = new;
        }
        return;
    }
    // Elements may straddle two words. `read` holds the unchanged value of word `i`, and
    // `written` the new bits of word `i` computed so far.
    let total_bits = len * bit_width;
    let last = total_bits.div_ceil(WORD_BITS) - 1;
    let mut read = words[0];
    let mut written = 0u64;
    let mut bit = 0;
    for i in 0..last {
        let word_end = (i + 1) * WORD_BITS;
        while bit + bit_width <= word_end {
            let offset = bit % WORD_BITS;
            written |= f((read >> offset) & mask) << offset;
            bit += bit_width;
        }
        let next = words[i + 1];
        let mut carry = 0;
        if bit < word_end {
            // The element starting at `bit` straddles words `i` and `i + 1`.
            let offset = bit % WORD_BITS;
            let in_low = word_end - bit;
            let new = f(((read >> offset) | (next << in_low)) & mask);
            written |= new << offset;
            carry = new >> in_low;
            bit += bit_width;
        }
        words[i] = written;
        read = next;
        written = carry;
    }
    while bit < total_bits {
        let offset = bit % WORD_BITS;
        written |= f((read >> offset) & mask) << offset;
        bit += bit_width;
    }
    words[last] = written;
}
