//! The mapping of data words to stripes and parking slots, at and around the sizes where
//! `Stripes::new` changes the number of stripes, blocks or slots.

use super::{MAX_SLOTS, MAX_STRIPES, STRIPES_PER_BLOCK, Stripes, WORDS_PER_STRIPE};

/// Checks that every data word of a vector of `data_words` words maps to one of its stripes,
/// that the parking slots divide the stripes, and that the words reach the stripes: all of them
/// within one block, and at least half of them across blocks, whose words fill at least half of
/// the blocks.
fn check(data_words: usize) {
    let stripes = Stripes::new(data_words, true);
    let n = stripes.versions.len();
    let slots = stripes.slots.len();
    assert!(
        n.is_power_of_two() && n <= MAX_STRIPES,
        "{data_words} words: {n} stripes"
    );
    assert!(
        slots.is_power_of_two() && slots <= MAX_SLOTS && n.is_multiple_of(slots),
        "{data_words} words: {slots} slots for {n} stripes"
    );
    let mut used = vec![false; n];
    for word in 0..data_words {
        let stripe = stripes.index(word);
        assert!(
            stripe < n,
            "{data_words} words: word {word} maps to stripe {stripe} of {n}"
        );
        used[stripe] = true;
    }
    let used = used.iter().filter(|&&used| used).count();
    if n <= STRIPES_PER_BLOCK {
        assert_eq!(used, n, "{data_words} words");
    } else {
        assert!(
            2 * used >= n,
            "{data_words} words: {used} of {n} stripes used"
        );
    }
}

#[test]
fn small_vectors_map_every_word_to_a_stripe() {
    // Up to 64 stripes, where the slots stop growing, and on to 64 stripes of 64 words each.
    let max = if cfg!(miri) {
        2 * WORDS_PER_STRIPE + 2
    } else {
        4096
    };
    (1..=max).for_each(check);
}

#[test]
#[cfg_attr(miri, ignore = "millions of words")]
fn large_vectors_map_every_word_to_a_stripe() {
    let one_block = WORDS_PER_STRIPE * STRIPES_PER_BLOCK;
    let all_stripes = WORDS_PER_STRIPE * MAX_STRIPES;
    // One block of stripes, two and four blocks, every stripe, and past the largest number.
    for size in [
        one_block,
        one_block + WORDS_PER_STRIPE,
        2 * one_block,
        2 * one_block + WORDS_PER_STRIPE,
        all_stripes,
        all_stripes + WORDS_PER_STRIPE,
        2 * all_stripes,
    ] {
        (size - 1..=size + 1).for_each(check);
    }
}
