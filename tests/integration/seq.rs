//! `SeqVec` against a `Vec<Vec<T>>`, for every code, with and without stored lengths, and its
//! parts.

use crate::codec::{assert_canonical, codecs, words};
use crate::{Elem, Rng, Trace, hash};
use compressed_intvec::{
    Backend, Codec, FixedVec, SeqIter, SeqVec, SeqVecParts, SeqVecPartsError, Storable,
};
use std::any::type_name;
use std::cell::Cell;
use std::sync::Arc;

/// Checks one sequence; its `size_hint` is exact when the lengths are stored, and a bound
/// otherwise.
#[track_caller]
fn check_sequence<T: Elem>(mut iter: SeqIter<'_, T>, model: &[T], stored: bool) {
    let mut rest = model;
    loop {
        let (low, high) = iter.size_hint();
        if stored {
            assert_eq!((low, high), (rest.len(), Some(rest.len())));
        } else {
            assert!(low <= rest.len() && high.is_some_and(|high| rest.len() <= high));
        }
        let Some((&x, tail)) = rest.split_first() else {
            break;
        };
        assert_eq!(iter.next(), Some(x));
        rest = tail;
    }
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

/// Checks every read of `vec` against `model`, with random calls to the iterator of sequences.
#[track_caller]
fn check<T: Elem, B: Backend>(vec: &SeqVec<T, B>, model: &[Vec<T>], stored: bool, rng: &mut Rng) {
    assert_eq!(vec.len(), model.len());
    assert_eq!(vec.is_empty(), model.is_empty());
    for (i, sequence) in model.iter().enumerate() {
        check_sequence(vec.get(i).unwrap(), sequence, stored);
        assert_eq!(vec.get(i).unwrap().count(), sequence.len());
        let mut partial = vec.get(i).unwrap();
        partial.next();
        assert_eq!(partial.count(), sequence.len().saturating_sub(1));
    }
    assert!(vec.get(model.len()).is_none());
    assert_eq!(vec.iter().count(), model.len());
    assert_eq!(
        vec.iter().last().map(Iterator::collect::<Vec<T>>),
        model.last().cloned()
    );
    assert_eq!(*vec, *model);
    assert_eq!(format!("{vec:?}"), format!("{model:?}"));

    let mut iter = vec.iter();
    let mut expected = model.iter();
    loop {
        assert_eq!(iter.len(), expected.len());
        let n = rng.below(4);
        let (got, want) = match rng.below(4) {
            0 => (iter.next_back(), expected.next_back()),
            1 => (iter.nth(n), expected.nth(n)),
            2 => (iter.nth_back(n), expected.nth_back(n)),
            _ => (iter.next(), expected.next()),
        };
        assert_eq!(got.map(Iterator::collect::<Vec<T>>).as_ref(), want);
        if want.is_none() {
            break;
        }
    }
}

#[test]
fn borrowing_and_boxing_preserve_storage() {
    #[track_caller]
    fn borrow<B: Backend>(
        data: B,
        offset_words: B,
        length_words: Option<B>,
        original: &SeqVecParts<Vec<u64>>,
        model: &[Vec<u64>],
    ) -> SeqVec<u64, B> {
        let (original_offsets, offset_count, offset_width) = original.offsets.clone().into_parts();
        let original_lengths = original.lengths.clone().map(FixedVec::into_parts);
        let data_pointer = data.as_ref().as_ptr();
        let offset_pointer = offset_words.as_ref().as_ptr();
        let length_pointer = length_words.as_ref().map(|words| words.as_ref().as_ptr());
        let offsets = FixedVec::from_parts(offset_words, offset_count, offset_width).unwrap();
        let lengths = length_words.map(|words| {
            let (_, count, width) = original_lengths.as_ref().unwrap();
            FixedVec::from_parts(words, *count, *width).unwrap()
        });
        let parent = SeqVec::from_parts(SeqVecParts {
            data,
            offsets,
            lengths,
            codec: original.codec,
        })
        .unwrap();
        let stored = original.lengths.is_some();
        let view = parent.as_borrowed();
        check(&view, model, stored, &mut Rng::new(1));
        let parts = view.into_parts();
        let (offsets, count, width) = parts.offsets.into_parts();
        let lengths = parts.lengths.map(FixedVec::into_parts);
        assert_eq!(parts.data.as_ptr(), data_pointer);
        assert_eq!(parts.data, original.data);
        assert_eq!(offsets.as_ptr(), offset_pointer);
        assert_eq!(offsets, original_offsets);
        assert_eq!((count, width), (offset_count, offset_width));
        assert_eq!(
            lengths.as_ref().map(|(words, _, _)| words.as_ptr()),
            length_pointer
        );
        assert_eq!(
            lengths,
            original_lengths
                .as_ref()
                .map(|(words, count, width)| (&words[..], *count, *width))
        );
        assert_eq!(parts.codec, original.codec);
        check(&parent, model, stored, &mut Rng::new(1));
        parent
    }

    let sequences: [&[u64]; 3] = [&[1, 2], &[], &[3]];
    for input in [&sequences[..], &[][..]] {
        let model: Vec<Vec<u64>> = input.iter().map(|sequence| sequence.to_vec()).collect();
        for stored in [false, true] {
            let mut builder = SeqVec::builder().codec(Codec::Delta);
            if stored {
                builder = builder.store_lengths();
            }
            let original = builder.build(input).unwrap().into_parts();
            let (offset_words, offset_count, offset_width) = original.offsets.clone().into_parts();
            let length_words = original.lengths.clone().map(FixedVec::into_parts);
            let owned = borrow(
                original.data.clone(),
                offset_words.clone(),
                length_words.as_ref().map(|(words, _, _)| words.clone()),
                &original,
                &model,
            );
            borrow(
                original.data.clone().into_boxed_slice(),
                offset_words.clone().into_boxed_slice(),
                length_words
                    .as_ref()
                    .map(|(words, _, _)| words.clone().into_boxed_slice()),
                &original,
                &model,
            );
            borrow(
                Arc::<[u64]>::from(&original.data[..]),
                Arc::<[u64]>::from(&offset_words[..]),
                length_words
                    .as_ref()
                    .map(|(words, _, _)| Arc::<[u64]>::from(&words[..])),
                &original,
                &model,
            );
            borrow(
                &original.data[..],
                &offset_words[..],
                length_words.as_ref().map(|(words, _, _)| &words[..]),
                &original,
                &model,
            );

            let boxed: SeqVec<u64, Box<[u64]>> = owned.into();
            check(&boxed, &model, stored, &mut Rng::new(1));
            check(&boxed.as_borrowed(), &model, stored, &mut Rng::new(1));
            let borrowed = boxed.as_borrowed().into_parts();
            let (borrowed_offsets, count, width) = borrowed.offsets.into_parts();
            let borrowed_lengths = borrowed.lengths.map(FixedVec::into_parts);
            assert_eq!(borrowed.data, original.data);
            assert_eq!(borrowed_offsets, offset_words);
            assert_eq!((count, width), (offset_count, offset_width));
            assert_eq!(
                borrowed_lengths,
                length_words
                    .as_ref()
                    .map(|(words, count, width)| (&words[..], *count, *width))
            );
            assert_eq!(borrowed.codec, original.codec);
            let data_pointer = borrowed.data.as_ptr();
            let offset_pointer = borrowed_offsets.as_ptr();
            let length_pointer = borrowed_lengths
                .as_ref()
                .map(|(words, _, _)| words.as_ptr());
            let parts = boxed.into_parts();
            let (offsets, count, width) = parts.offsets.into_parts();
            let lengths = parts.lengths.map(FixedVec::into_parts);
            assert_eq!(parts.data.as_ptr(), data_pointer);
            assert_eq!(offsets.as_ptr(), offset_pointer);
            assert_eq!(
                lengths.as_ref().map(|(words, _, _)| words.as_ptr()),
                length_pointer
            );
            assert_eq!(&*parts.data, original.data);
            assert_eq!(&*offsets, offset_words);
            assert_eq!((count, width), (offset_count, offset_width));
            assert_eq!(
                lengths
                    .as_ref()
                    .map(|(words, count, width)| (&words[..], *count, *width)),
                length_words
                    .as_ref()
                    .map(|(words, count, width)| (&words[..], *count, *width))
            );
            assert_eq!(parts.codec, original.codec);
        }
    }
}

/// Returns sequences of the words that `codec` encodes cheaply, stored as `T`: many short or
/// empty ones, and a few long ones.
fn sequences<T: Elem>(rng: &mut Rng, codec: Codec) -> Vec<Vec<T>> {
    let mask = u64::MAX >> (64 - T::BITS);
    (0..rng.below(60))
        .map(|_| {
            let len = match rng.below(8) {
                0 => 0,
                1 => rng.below(100),
                _ => rng.below(5),
            };
            words(rng, codec, len)
                .into_iter()
                .map(|w| T::from_word(w & mask))
                .collect()
        })
        .collect()
}

fn every_code<T: Elem>() {
    let mut rng = Rng::new(1);
    for codec in codecs() {
        for _ in 0..3 {
            let model = sequences::<T>(&mut rng, codec);
            let _trace = Trace(vec![format!(
                "{} with {codec:?}: {model:?}",
                type_name::<T>()
            )]);
            for stored in [false, true] {
                let mut builder = SeqVec::builder();
                if stored {
                    builder = builder.store_lengths();
                }
                let vec = builder.codec(codec).build(&model).unwrap();
                assert_eq!(vec.codec(), codec);
                check(&vec, &model, stored, &mut rng);
                let from_iter = builder.codec(codec).build_from_iter(&model).unwrap();
                assert_eq!(from_iter.into_parts().data, vec.clone().into_parts().data);
                check(&builder.build(&model), &model, stored, &mut rng);
                with_borrowed(vec.into_parts(), |view| {
                    check(&view, &model, stored, &mut rng)
                });
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn every_code_with_and_without_lengths() {
    every_code::<u64>();
    every_code::<i64>();
    every_code::<u8>();
    every_code::<i8>();
}

/// Rebuilds a vector over the borrowed words of `parts`, which `from_parts` checks again, and
/// passes it to `f`.
fn with_borrowed<T: Storable>(parts: SeqVecParts<Vec<u64>>, f: impl FnOnce(SeqVec<T, &[u64]>)) {
    let (offset_words, n, offset_width) = parts.offsets.into_parts();
    let lengths = parts.lengths.map(FixedVec::into_parts);
    f(SeqVec::from_parts(SeqVecParts {
        data: &parts.data[..],
        offsets: FixedVec::from_parts(&offset_words[..], n, offset_width).unwrap(),
        lengths: lengths
            .as_ref()
            .map(|(words, n, width)| FixedVec::from_parts(&words[..], *n, *width).unwrap()),
        codec: parts.codec,
    })
    .unwrap());
}

/// `from_slices`, `collect`, `From` and the builder without a code agree; equality and hashing
/// compare the sequences, not the code or the stored lengths.
#[test]
fn defaults_equality_and_hashing() {
    let model = vec![vec![1u32, 2], vec![], vec![3]];
    let vec = SeqVec::from_slices(&model);
    let collected: SeqVec<u32> = model.iter().collect();
    for other in [
        collected,
        SeqVec::from(&model[..]),
        SeqVec::builder().build(&model),
        SeqVec::builder().build_from_iter(&model),
    ] {
        assert_eq!(other.codec(), vec.codec());
        assert_eq!(other, vec);
    }
    assert!(SeqVec::<u32>::default().is_empty());

    let delta = SeqVec::builder()
        .store_lengths()
        .codec(Codec::Delta)
        .build(&model)
        .unwrap();
    assert_eq!(delta.codec(), Codec::Delta);
    assert_eq!(delta, vec);
    assert_eq!(hash(&delta), hash(&vec));
    with_borrowed::<u32>(delta.into_parts(), |view| assert_eq!(view, vec));
    assert_eq!(vec, model[..]);
    assert_ne!(vec, model[..2]);
    // The same elements, split differently.
    assert_ne!(vec, SeqVec::from_slices(&[vec![1u32], vec![2], vec![3]]));
    assert_ne!(vec, SeqVec::from_slices(&[vec![1u32, 2], vec![3], vec![]]));
}

const FIRST_SEQUENCE: [u64; 17] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1024];

struct ChangingSequence<'a>(&'a Cell<usize>);

impl AsRef<[u64]> for ChangingSequence<'_> {
    fn as_ref(&self) -> &[u64] {
        let calls = self.0.get();
        self.0.set(calls + 1);
        if calls == 0 {
            &FIRST_SEQUENCE
        } else {
            &[u64::MAX]
        }
    }
}

fn check_snapshot(vec: SeqVec<u64>, calls: &Cell<usize>, stored: bool) {
    assert_eq!(calls.get(), 1);
    assert_eq!(vec.len(), 1);
    assert_eq!(vec.get(0).unwrap().collect::<Vec<_>>(), FIRST_SEQUENCE);
    let parts = vec.into_parts();
    assert_eq!(parts.lengths.is_some(), stored);
    assert_eq!(
        SeqVec::<u64>::from_parts(parts).unwrap(),
        [FIRST_SEQUENCE][..]
    );
}

#[test]
fn from_slices_snapshots_changing_sequence() {
    let calls = Cell::new(0);
    let input = [ChangingSequence(&calls)];
    check_snapshot(SeqVec::from_slices(&input), &calls, false);
}

#[test]
fn automatic_build_snapshots_changing_sequence() {
    for stored in [false, true] {
        let calls = Cell::new(0);
        let input = [ChangingSequence(&calls)];
        let mut builder = SeqVec::builder();
        if stored {
            builder = builder.store_lengths();
        }
        check_snapshot(builder.build(&input), &calls, stored);
    }
}

#[test]
fn automatic_iterator_snapshots_changing_sequence() {
    for stored in [false, true] {
        let calls = Cell::new(0);
        let input = [ChangingSequence(&calls)];
        let mut builder = SeqVec::builder();
        if stored {
            builder = builder.store_lengths();
        }
        check_snapshot(builder.build_from_iter(input), &calls, stored);
    }
}

#[test]
fn from_parts_rejects_inconsistent_parts() {
    // Gamma takes 3 bits for 1 and 2, and 5 for 3: the sequences end at bits 6, 6 and 11.
    let model = [vec![1u64, 2], vec![], vec![3]];
    let parts = SeqVec::builder()
        .codec(Codec::Gamma)
        .build(&model)
        .unwrap()
        .into_parts();
    assert_eq!(parts.offsets, FixedVec::from_slice(&[0, 6, 6, 11]));
    let changed = |change: &dyn Fn(&mut SeqVecParts<Vec<u64>>)| {
        let mut parts = parts.clone();
        change(&mut parts);
        SeqVec::<u64>::from_parts(parts).unwrap_err()
    };
    let offsets = |offsets: &[u64]| FixedVec::from_slice(offsets);

    for codec in [
        Codec::Golomb { b: 0 },
        Codec::Zeta { k: 0 },
        Codec::Zeta { k: 64 },
        Codec::Rice { log2_b: 64 },
        Codec::ExpGolomb { k: 64 },
        Codec::Pi { k: 64 },
    ] {
        assert_eq!(
            changed(&|p| p.codec = codec),
            SeqVecPartsError::InvalidParameter(codec)
        );
    }
    assert_eq!(
        changed(&|p| p.offsets = offsets(&[])),
        SeqVecPartsError::NoOffsets
    );
    assert_eq!(
        changed(&|p| p.lengths = Some(offsets(&[2, 0]))),
        SeqVecPartsError::LengthCount {
            expected: 3,
            found: 2
        }
    );
    // The bits after the last codeword are zero, so gamma finds no end of its unary part.
    assert_eq!(
        changed(&|p| p.offsets = offsets(&[0, 6, 6, 200])),
        SeqVecPartsError::Truncated { sequence: 2 }
    );
    assert_eq!(
        changed(&|p| p.data.clear()),
        SeqVecPartsError::Truncated { sequence: 0 }
    );
    // 64 zeros and a one: the length of a gamma codeword whose value does not fit 64 bits.
    assert_eq!(
        changed(&|p| {
            p.data = vec![0, 1];
            p.offsets = offsets(&[0, 129]);
        }),
        SeqVecPartsError::InvalidCodeword { sequence: 0 }
    );
    // The second codeword of sequence 0 crosses its end.
    assert_eq!(
        changed(&|p| p.offsets = offsets(&[0, 5, 6, 11])),
        SeqVecPartsError::Misaligned {
            sequence: 0,
            expected: 5,
            found: 6
        }
    );
    // A stored length that stops before the end.
    assert_eq!(
        changed(&|p| p.lengths = Some(offsets(&[1, 0, 1]))),
        SeqVecPartsError::Misaligned {
            sequence: 0,
            expected: 6,
            found: 3
        }
    );

    let wide = SeqVec::from_slices(&[[1u64, 300]]).into_parts();
    assert_eq!(
        SeqVec::<u8>::from_parts(wide).unwrap_err(),
        SeqVecPartsError::WordOutOfRange {
            sequence: 0,
            word: 300
        }
    );
}

/// Offsets and lengths of bit width 0 make every sequence empty, so `from_parts` accepts any
/// number of them without decoding: a few words cannot make it run for long.
#[test]
fn empty_sequences_are_accepted_without_decoding() {
    let len = usize::MAX / 2;
    let offsets = FixedVec::<u64, &[u64]>::from_parts(&[0], len + 1, 0).unwrap();
    let lengths = FixedVec::<u64, &[u64]>::from_parts(&[0], len, 0).unwrap();
    for lengths in [None, Some(lengths)] {
        let vec = SeqVec::<u32, &[u64]>::from_parts(SeqVecParts {
            data: &[],
            offsets: offsets.clone(),
            lengths,
            codec: Codec::Gamma,
        })
        .unwrap();
        assert_eq!(vec.len(), len);
        assert_eq!(vec.get(len - 1).unwrap().count(), 0);
        assert_eq!(vec.iter().next_back().unwrap().next(), None);
    }
}

/// Corrupts one part at random.
fn corrupt(rng: &mut Rng, parts: &mut SeqVecParts<Vec<u64>>) {
    let bits = 64 * parts.data.len() as u64 + 1;
    let mut offsets: Vec<u64> = parts.offsets.iter().collect();
    match rng.below(6) {
        0 => {
            for _ in 0..1 + rng.below(4) {
                let i = rng.below(parts.data.len().max(1));
                if let Some(word) = parts.data.get_mut(i) {
                    *word ^= 1 << rng.below(64);
                }
            }
        }
        1 => parts.data.truncate(rng.below(parts.data.len() + 1)),
        2 => parts.data = (0..rng.below(8)).map(|_| rng.word()).collect(),
        3 => {
            let i = rng.below(offsets.len());
            if let Some(offset) = offsets.get_mut(i) {
                *offset = rng.next_u64() % bits;
            }
            parts.offsets = FixedVec::from_slice(&offsets);
        }
        4 => {
            // A sorted run of offsets, possibly at the boundaries of other codewords.
            let mut random: Vec<u64> = (0..offsets.len()).map(|_| rng.next_u64() % bits).collect();
            random.sort_unstable();
            parts.offsets = FixedVec::from_slice(&random);
        }
        _ => {
            parts.lengths = match parts.lengths.take() {
                Some(_) => None,
                None => {
                    let n = offsets.len() - 1;
                    let lengths: Vec<u64> = (0..n).map(|_| rng.next_u64() % 4).collect();
                    Some(FixedVec::from_slice(&lengths))
                }
            }
        }
    }
}

/// A corrupt vector is rejected, or reads the same through every access path, and each of its
/// sequences holds the codewords of what it reads; both outcomes occur.
#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn corrupt_parts_are_rejected_or_read_consistently() {
    let mut rng = Rng::new(2);
    let (mut accepted, mut rejected) = (0, 0);
    for codec in codecs() {
        for _ in 0..30 {
            let model = sequences::<u64>(&mut rng, codec);
            let mut builder = SeqVec::builder();
            if rng.coin() {
                builder = builder.store_lengths();
            }
            let mut parts = builder.codec(codec).build(&model).unwrap().into_parts();
            corrupt(&mut rng, &mut parts);
            let (data, offsets) = (parts.data.clone(), parts.offsets.clone());
            let Ok(vec) = SeqVec::<u64>::from_parts(parts) else {
                rejected += 1;
                continue;
            };
            accepted += 1;
            let read: Vec<Vec<u64>> = vec.iter().map(Iterator::collect).collect();
            assert_eq!(read.len(), vec.len());
            for (i, sequence) in read.iter().enumerate() {
                assert!(vec.get(i).unwrap().eq(sequence.iter().copied()));
                assert_eq!(vec.get(i).unwrap().count(), sequence.len());
                assert_canonical(codec, sequence, &data, offsets.get(i).unwrap());
            }
            let backwards: Vec<Vec<u64>> = vec.iter().rev().map(Iterator::collect).collect();
            assert!(backwards.iter().eq(read.iter().rev()));
        }
    }
    assert!(accepted > 0 && rejected > 0, "{accepted} {rejected}");
}
