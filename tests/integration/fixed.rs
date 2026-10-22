//! `FixedVec`, `FixedVecSlice` and their iterators, against a `Vec<T>` that receives the same
//! operations.

use crate::{
    Case, Elem, MIRI_DIVISOR, Rng, Trace, for_each_type, hash, max_width, panics, range, too_large,
    value, width_of,
};
use compressed_intvec::{
    Backend, BitWidthTooLarge, FixedVec, FixedVecError, FixedVecSlice, ValueTooLarge,
};
use std::any::type_name;
use std::fmt::Debug;
use std::ops::Bound;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

/// Checks every read of `vec` against `model`.
#[track_caller]
pub fn check<T: Elem, B: Backend>(vec: &FixedVec<T, B>, model: &[T]) {
    assert_eq!(vec.len(), model.len());
    assert_eq!(vec.is_empty(), model.is_empty());
    for (i, &x) in model.iter().enumerate() {
        assert_eq!(vec.get(i), Some(x), "index {i}");
        // SAFETY: `i < len`.
        assert_eq!(unsafe { vec.get_unchecked(i) }, x, "index {i}");
    }
    assert_eq!(vec.get(model.len()), None);
    assert_eq!(vec.first(), model.first().copied());
    assert_eq!(vec.last(), model.last().copied());
    assert!(vec.iter().eq(model.iter().copied()));
    assert_eq!(*vec, *model);
    assert_eq!(format!("{vec:?}"), format!("{model:?}"));
    check_slice(vec.as_slice(), model);
}

#[track_caller]
fn check_slice<T: Elem>(slice: FixedVecSlice<'_, T>, model: &[T]) {
    assert_eq!(slice.len(), model.len());
    assert_eq!(slice.is_empty(), model.is_empty());
    for (i, &x) in model.iter().enumerate() {
        assert_eq!(slice.get(i), Some(x), "index {i}");
    }
    assert_eq!(slice.get(model.len()), None);
    assert_eq!(slice.first(), model.first().copied());
    assert_eq!(slice.last(), model.last().copied());
    assert!(slice.iter().eq(model.iter().copied()));
    assert!(slice.iter().rev().eq(model.iter().rev().copied()));
    assert_eq!(slice, *model);
    assert_eq!(format!("{slice:?}"), format!("{model:?}"));
}

/// The operations of the model test, from the simplest to the most complex.
#[derive(Debug, Clone)]
enum Op<T> {
    Clear,
    ShrinkToFit,
    Pop,
    Reserve(usize),
    Push(T),
    /// A value that may not fit.
    TryPush(T),
    Set(usize, T),
    SetUnchecked(usize, T),
    /// A value that may not fit.
    TrySet(usize, T),
    Replace(usize, T),
    Swap(usize, usize),
    Insert(usize, T),
    Remove(usize),
    SwapRemove(usize),
    Resize(usize, T),
    ExtendFromSlice(Vec<T>),
    Extend(Vec<T>),
    Fill(T),
    FillWith(Vec<T>),
    /// Replaces each element with the next value of the range of the bit width, wrapping.
    MapInPlace,
    RotateLeft(usize),
    RotateRight(usize),
}

const KINDS: usize = 22;

/// Returns an operation of kind `kind` that is valid on a vector of `len` elements of
/// `bit_width` bits, or `None` if the kind needs an element and the vector is empty.
fn generate<T: Elem>(rng: &mut Rng, kind: usize, len: usize, bit_width: u32) -> Option<Op<T>> {
    let x = value(rng, bit_width);
    let maybe_too_large = match too_large(rng, bit_width) {
        Some(large) if rng.coin() => large,
        _ => x,
    };
    let index = (len > 0).then(|| rng.below(len));
    let extension = rng.below(20);
    Some(match kind {
        0 => Op::Clear,
        1 => Op::ShrinkToFit,
        2 => Op::Pop,
        3 => Op::Reserve(rng.below(100)),
        4 => Op::Push(x),
        5 => Op::TryPush(maybe_too_large),
        6 => Op::Set(index?, x),
        7 => Op::SetUnchecked(index?, x),
        8 => Op::TrySet(index?, maybe_too_large),
        9 => Op::Replace(index?, x),
        10 => Op::Swap(index?, rng.below(len)),
        11 => Op::Insert(rng.below(len + 1), x),
        12 => Op::Remove(index?),
        13 => Op::SwapRemove(index?),
        14 => Op::Resize(rng.below(len + 20), x),
        15 => Op::ExtendFromSlice(values(rng, extension, bit_width)),
        16 => Op::Extend(values(rng, extension, bit_width)),
        17 => Op::Fill(x),
        18 => Op::FillWith(values(rng, len, bit_width)),
        19 => Op::MapInPlace,
        20 => Op::RotateLeft(rng.below(len + 1)),
        21 => Op::RotateRight(rng.below(len + 1)),
        _ => unreachable!("{KINDS} kinds"),
    })
}

fn values<T: Elem>(rng: &mut Rng, n: usize, bit_width: u32) -> Vec<T> {
    (0..n).map(|_| value(rng, bit_width)).collect()
}

/// The error of storing `x` in `bit_width` bits, if it does not fit.
fn fits<T: Elem>(x: T, bit_width: u32) -> Result<(), ValueTooLarge<T>> {
    let required = width_of(x);
    if required <= bit_width {
        Ok(())
    } else {
        Err(ValueTooLarge {
            value: x,
            required,
            bit_width,
        })
    }
}

/// Applies `op` to `vec` and `model` if it changes the elements in place, and returns whether
/// it did. These are the operations of every backend that implements `AsMut<[u64]>`.
fn apply_in_place<T: Elem, B: Backend + AsMut<[u64]>>(
    op: &Op<T>,
    vec: &mut FixedVec<T, B>,
    model: &mut [T],
) -> bool {
    let bit_width = vec.bit_width();
    let (lo, hi) = range::<T>(bit_width);
    match *op {
        Op::Set(i, x) => {
            vec.set(i, x);
            model[i] = x;
        }
        Op::SetUnchecked(i, x) => {
            // SAFETY: the generator draws `i < len`.
            unsafe { vec.set_unchecked(i, x) };
            model[i] = x;
        }
        Op::TrySet(i, x) => {
            let expected = fits(x, bit_width);
            assert_eq!(vec.try_set(i, x), expected);
            if expected.is_ok() {
                model[i] = x;
            }
        }
        Op::Replace(i, x) => assert_eq!(vec.replace(i, x), std::mem::replace(&mut model[i], x)),
        Op::Swap(a, b) => {
            vec.swap(a, b);
            model.swap(a, b);
        }
        Op::Fill(x) => {
            vec.fill(x);
            model.fill(x);
        }
        Op::FillWith(ref values) => {
            let mut next = values.iter().copied();
            vec.fill_with(|| next.next().expect("one value per element"));
            model.copy_from_slice(values);
        }
        Op::MapInPlace => {
            let step = |x: T| T::from_int(lo + (x.to_int() - lo + 1) % (hi - lo + 1)).unwrap();
            vec.map_in_place(step);
            model.iter_mut().for_each(|x| *x = step(*x));
        }
        Op::RotateLeft(mid) => {
            vec.rotate_left(mid);
            model.rotate_left(mid);
        }
        Op::RotateRight(k) => {
            vec.rotate_right(k);
            model.rotate_right(k);
        }
        _ => return false,
    }
    true
}

fn apply<T: Elem>(op: &Op<T>, vec: &mut FixedVec<T>, model: &mut Vec<T>) {
    if apply_in_place(op, vec, model) {
        return;
    }
    match *op {
        Op::Clear => {
            vec.clear();
            model.clear();
        }
        Op::ShrinkToFit => {
            vec.shrink_to_fit();
            assert!(vec.capacity() >= model.len());
        }
        Op::Pop => assert_eq!(vec.pop(), model.pop()),
        Op::Reserve(n) => {
            vec.reserve(n);
            assert!(vec.capacity() >= model.len() + n);
        }
        Op::Push(x) => {
            vec.push(x);
            model.push(x);
        }
        Op::TryPush(x) => {
            let expected = fits(x, vec.bit_width());
            assert_eq!(vec.try_push(x), expected);
            if expected.is_ok() {
                model.push(x);
            }
        }
        Op::Insert(i, x) => {
            vec.insert(i, x);
            model.insert(i, x);
        }
        Op::Remove(i) => assert_eq!(vec.remove(i), model.remove(i)),
        Op::SwapRemove(i) => assert_eq!(vec.swap_remove(i), model.swap_remove(i)),
        Op::Resize(n, x) => {
            vec.resize(n, x);
            model.resize(n, x);
        }
        Op::ExtendFromSlice(ref values) => {
            vec.extend_from_slice(values);
            model.extend_from_slice(values);
        }
        Op::Extend(ref values) => {
            vec.extend(values.iter().copied());
            model.extend(values);
        }
        _ => unreachable!("an operation in place"),
    }
}

/// Runs random operations on a vector of `bit_width` bits, with the kinds of a random subset.
fn run<T: Elem>(seed: u64, bit_width: u32) {
    let mut rng = Rng::new(seed);
    let enabled = loop {
        let mask = rng.next_u64() & ((1 << KINDS) - 1);
        if mask != 0 {
            break mask;
        }
    };
    let mut trace = Trace(vec![format!(
        "{} with bit width {bit_width}, seed {seed}, kinds {enabled:#b}",
        type_name::<T>()
    )]);
    let mut vec = FixedVec::<T>::new(bit_width).unwrap();
    let mut model = Vec::new();
    for _ in 0..100 {
        let kind = rng.below(KINDS);
        if (enabled >> kind) & 1 == 0 {
            continue;
        }
        let Some(op) = generate(&mut rng, kind, model.len(), bit_width) else {
            continue;
        };
        trace.0.push(format!("{op:?}"));
        apply(&op, &mut vec, &mut model);
        assert_eq!(vec.bit_width(), bit_width);
        check(&vec, &model);
    }
}

fn random_operations<T: Elem>() {
    for bit_width in 0..=max_width::<T>() {
        for seed in 0..8 {
            run::<T>(seed << 8 | u64::from(bit_width), bit_width);
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn random_operations_on_every_type_and_bit_width() {
    for_each_type!(random_operations);
}

/// The operations in place of a vector over `Box<[u64]>`, the other backend that can change its
/// words.
#[test]
fn random_operations_in_place_on_a_boxed_vector() {
    let mut rng = Rng::new(1);
    for bit_width in 0..=64 {
        let mut model: Vec<u64> = (0..rng.below(200))
            .map(|_| value(&mut rng, bit_width))
            .collect();
        let mut owned = FixedVec::new(bit_width).unwrap();
        owned.extend_from_slice(&model);
        let len = owned.len();
        let mut vec: FixedVec<u64, Box<[u64]>> = owned.into();
        let mut trace = Trace(vec![format!("bit width {bit_width}")]);
        for _ in 0..100 / MIRI_DIVISOR {
            let kind = rng.below(KINDS);
            let Some(op) = generate(&mut rng, kind, len, bit_width) else {
                continue;
            };
            trace.0.push(format!("{op:?}"));
            if apply_in_place(&op, &mut vec, &mut model) {
                check(&vec, &model);
            }
        }
    }
}

#[test]
fn borrowing_and_boxing_preserve_storage() {
    #[track_caller]
    fn borrow<B: Backend>(
        words: B,
        original: &[u64],
        len: usize,
        width: u32,
        model: &[u64],
    ) -> FixedVec<u64, B> {
        let pointer = words.as_ref().as_ptr();
        let parent = FixedVec::from_parts(words, len, width).unwrap();
        let view = parent.as_borrowed();
        check(&view, model);
        let (borrowed, borrowed_len, borrowed_width) = view.into_parts();
        assert_eq!(borrowed.as_ptr(), pointer);
        assert_eq!(borrowed, original);
        assert_eq!((borrowed_len, borrowed_width), (len, width));
        check(&parent, model);
        parent
    }

    for (words, len, width, model) in [
        (&[57, 0][..], 3, 2, &[1, 2, 3][..]),
        (&[3, u64::MAX, 0x1234][..], 1, 2, &[3][..]),
        (&[][..], 0, 0, &[][..]),
        (&[u64::MAX, 0x1234][..], 3, 0, &[0, 0, 0][..]),
    ] {
        let owned = borrow(words.to_vec(), words, len, width, model);
        borrow(words.to_vec().into_boxed_slice(), words, len, width, model);
        borrow(Arc::<[u64]>::from(words), words, len, width, model);
        borrow(words, words, len, width, model);

        let boxed: FixedVec<u64, Box<[u64]>> = owned.into();
        check(&boxed, model);
        let view = boxed.as_borrowed();
        check(&view, model);
        let (borrowed, borrowed_len, borrowed_width) = view.into_parts();
        assert_eq!(borrowed, words);
        assert_eq!((borrowed_len, borrowed_width), (len, width));
        let pointer = borrowed.as_ptr();
        let (boxed_words, boxed_len, boxed_width) = boxed.into_parts();
        assert_eq!(pointer, boxed_words.as_ptr());
        assert_eq!(&*boxed_words, words);
        assert_eq!((boxed_len, boxed_width), (len, width));
    }
}

fn constructors<T: Elem>() {
    let max = max_width::<T>();
    for bit_width in 0..=max {
        let vec = FixedVec::<T>::new(bit_width).unwrap();
        assert_eq!(vec.bit_width(), bit_width);
        check(&vec, &[]);
        assert!(
            FixedVec::<T>::with_capacity(bit_width, 100)
                .unwrap()
                .capacity()
                >= 100
        );
    }
    for bit_width in [max + 1, u32::MAX] {
        let error = BitWidthTooLarge { bit_width, max };
        assert_eq!(FixedVec::<T>::new(bit_width).unwrap_err(), error);
        assert_eq!(
            FixedVec::<T>::with_capacity(bit_width, 0).unwrap_err(),
            error
        );
    }
    let default = FixedVec::<T>::default();
    assert_eq!(default.bit_width(), max);
    check(&default, &[]);

    // `from_slice`, `collect` and `From` choose the smallest bit width that holds every value.
    let mut rng = Rng::new(2);
    for _ in 0..100 / MIRI_DIVISOR {
        let bit_width = rng.below(max as usize + 1) as u32;
        let model: Vec<T> = (0..rng.below(50))
            .map(|_| value(&mut rng, bit_width))
            .collect();
        let smallest = model.iter().map(|&x| width_of(x)).max().unwrap_or(0);
        for vec in [
            FixedVec::from_slice(&model),
            model.iter().copied().collect(),
            FixedVec::from(&model[..]),
        ] {
            assert_eq!(vec.bit_width(), smallest);
            check(&vec, &model);
        }
    }
}

#[test]
fn constructors_of_every_type() {
    for_each_type!(constructors);
}

/// Drives `iter` and `model` with the same random calls at both ends, including skips.
#[track_caller]
fn same_iter<I>(mut iter: I, model: &[u64], rng: &mut Rng)
where
    I: DoubleEndedIterator<Item = u64> + ExactSizeIterator + Clone + Debug,
{
    let mut rest = model;
    while !rest.is_empty() {
        assert_eq!(iter.len(), rest.len());
        assert_eq!(iter.size_hint(), (rest.len(), Some(rest.len())));
        assert_eq!(format!("{iter:?}"), format!("{rest:?}"));
        assert!(iter.clone().eq(rest.iter().copied()));
        // Short skips stay within the current word; long ones reach past the end.
        let n = if rng.coin() {
            rng.below(4)
        } else {
            rng.below(rest.len() + 2)
        };
        let (got, expected) = match rng.below(4) {
            0 => {
                let expected = rest.first().copied();
                rest = &rest[1..];
                (iter.next(), expected)
            }
            1 => {
                let expected = rest.last().copied();
                rest = &rest[..rest.len() - 1];
                (iter.next_back(), expected)
            }
            2 => {
                let expected = rest.get(n).copied();
                rest = rest.get(n + 1..).unwrap_or_default();
                (iter.nth(n), expected)
            }
            _ => {
                let end = rest.len().checked_sub(n + 1);
                let expected = end.map(|i| rest[i]);
                rest = &rest[..end.unwrap_or(0)];
                (iter.nth_back(n), expected)
            }
        };
        assert_eq!(got, expected);
    }
    assert_eq!(iter.len(), 0);
    for _ in 0..2 {
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }
}

#[test]
fn iterators_from_both_ends() {
    let mut rng = Rng::new(3);
    for bit_width in 0..=64 {
        for _ in 0..if cfg!(miri) { 1 } else { 10 } {
            let model: Vec<u64> = (0..rng.below(200))
                .map(|_| value(&mut rng, bit_width))
                .collect();
            let mut vec = FixedVec::new(bit_width).unwrap();
            vec.extend_from_slice(&model);
            same_iter(vec.iter(), &model, &mut rng);
            let start = rng.below(model.len() + 1);
            let end = start + rng.below(model.len() - start + 1);
            same_iter(
                vec.slice(start..end).unwrap().iter(),
                &model[start..end],
                &mut rng,
            );
            same_iter(vec.into_iter(), &model, &mut rng);
        }
    }
}

/// Returns a bound that is often at or past the end of `0..len`, or `usize::MAX`.
fn bound(rng: &mut Rng, len: usize) -> Bound<usize> {
    let at = match rng.below(4) {
        0 => usize::MAX,
        1 => len + 1,
        _ => rng.below(len + 1),
    };
    match rng.below(3) {
        0 => Bound::Unbounded,
        1 => Bound::Included(at),
        _ => Bound::Excluded(at),
    }
}

/// `slice` accepts the ranges that `<[T]>::get` accepts, and selects the same elements.
#[test]
fn slices_select_the_ranges_of_std_slices() {
    let mut rng = Rng::new(4);
    for _ in 0..2000 / MIRI_DIVISOR {
        let bit_width = rng.below(65) as u32;
        let model: Vec<u64> = (0..rng.below(100))
            .map(|_| value(&mut rng, bit_width))
            .collect();
        let mut vec = FixedVec::new(bit_width).unwrap();
        vec.extend_from_slice(&model);
        let outer = (bound(&mut rng, model.len()), bound(&mut rng, model.len()));
        let (Some(slice), Some(expected)) = (vec.slice(outer), model.get(outer)) else {
            assert_eq!(
                vec.slice(outer).is_some(),
                model.get(outer).is_some(),
                "{outer:?}"
            );
            continue;
        };
        check_slice(slice, expected);
        let inner = (
            bound(&mut rng, expected.len()),
            bound(&mut rng, expected.len()),
        );
        match (slice.slice(inner), expected.get(inner)) {
            (Some(slice), Some(expected)) => check_slice(slice, expected),
            (got, expected) => assert_eq!(got.is_some(), expected.is_some(), "{inner:?}"),
        }
    }
}

/// `binary_search` finds the value if it is present, and its insertion point otherwise.
#[track_caller]
fn check_search(result: Result<usize, usize>, sorted: &[i32], x: i32) {
    match result {
        Ok(i) => assert_eq!(sorted[i], x),
        Err(i) => {
            assert!(!sorted.contains(&x), "{x} is present");
            assert_eq!(i, sorted.partition_point(|&y| y < x));
        }
    }
}

/// Searches with signed elements, whose order is not the order of their zig-zag words.
#[test]
fn searches_of_sorted_vectors() {
    let mut rng = Rng::new(5);
    for _ in 0..300 / MIRI_DIVISOR {
        let mut model: Vec<i32> = (0..rng.below(100))
            .map(|_| rng.below(21) as i32 - 10)
            .collect();
        model.sort_unstable();
        let vec = FixedVec::from_slice(&model);
        let start = rng.below(model.len() + 1);
        let slice = vec.slice(start..).unwrap();
        let halves: Vec<i32> = model.iter().map(|y| y.div_euclid(2)).collect();
        for x in -12..12 {
            check_search(vec.binary_search(&x), &model, x);
            check_search(slice.binary_search(&x), &model[start..], x);
            check_search(vec.binary_search_by(|y| y.cmp(&x)), &model, x);
            check_search(
                vec.binary_search_by_key(&x.div_euclid(2), |y| y.div_euclid(2)),
                &halves,
                x.div_euclid(2),
            );
            assert_eq!(
                vec.partition_point(|&y| y < x),
                model.partition_point(|&y| y < x)
            );
            assert_eq!(
                slice.partition_point(|&y| y < x),
                model[start..].partition_point(|&y| y < x)
            );
        }
    }
}

/// `binary_search_by` returns a matching index where `slice::binary_search_by` finds a match, and
/// the same insertion point where it does not, on views with runs of equal elements, the empty
/// view among them.
#[test]
fn binary_search_by_agrees_with_std() {
    let mut rng = Rng::new(7);
    for _ in 0..300 / MIRI_DIVISOR {
        let mut model: Vec<u64> = (0..rng.below(70)).map(|_| rng.below(12) as u64).collect();
        model.sort_unstable();
        let vec = FixedVec::from_slice(&model);
        let start = rng.below(model.len() + 1);
        let end = start + rng.below(model.len() - start + 1);
        let (slice, expected) = (vec.slice(start..end).unwrap(), &model[start..end]);
        for x in 0..13 {
            match (
                slice.binary_search_by(|y| y.cmp(&x)),
                expected.binary_search_by(|y| y.cmp(&x)),
            ) {
                (Ok(i), Ok(_)) => assert_eq!(expected[i], x),
                (Err(i), Err(j)) => assert_eq!(i, j),
                (ours, std) => panic!("{ours:?} and {std:?} for {x} in {expected:?}"),
            }
        }
    }
}

/// A vector compares its bit width and its elements, a view only its elements, and each hash
/// follows its equality.
#[test]
fn equality_and_hashing() {
    let values = [1u32, 2, 3];
    let narrow = FixedVec::from_slice(&values);
    let mut same = FixedVec::new(2).unwrap();
    same.extend_from_slice(&values);
    let mut wide = FixedVec::new(10).unwrap();
    wide.extend_from_slice(&values);
    let (words, len, bit_width) = narrow.clone().into_parts();
    let view = FixedVec::<u32, &[u64]>::from_parts(&words, len, bit_width).unwrap();

    assert_eq!(narrow, same);
    assert_eq!(hash(&narrow), hash(&same));
    assert_eq!(narrow, view);
    assert_eq!(hash(&narrow), hash(&view));
    assert_ne!(narrow, wide);
    assert_eq!(wide, values[..]);
    assert_eq!(narrow.as_slice(), wide.as_slice());
    assert_eq!(hash(&narrow.as_slice()), hash(&wide.as_slice()));

    let shifted = FixedVec::from_slice(&[5u32, 1, 2, 3]);
    assert_eq!(shifted.slice(1..).unwrap(), narrow.as_slice());
    assert_eq!(hash(&shifted.slice(1..).unwrap()), hash(&narrow.as_slice()));
    assert_ne!(shifted.slice(..3).unwrap(), narrow.as_slice());
    assert_ne!(narrow, FixedVec::from_slice(&[1u32, 2]));
    assert_ne!(
        narrow.as_slice(),
        FixedVec::from_slice(&[1u32, 2]).as_slice()
    );
}

fn bit(words: &[u64], k: usize) -> bool {
    (words[k / 64] >> (k % 64)) & 1 == 1
}

/// `from_parts` reads the layout that it documents (element `i` takes the bits
/// `[i * w, (i + 1) * w)`, bit `k` is bit `k % 64` of word `k / 64`, and a non-empty vector has
/// one more word), with any bits past the last element, over every backend; `into_parts`
/// returns words in that layout.
#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn parts_follow_the_documented_layout() {
    let mut rng = Rng::new(6);
    for bit_width in 0..=64 {
        let w = bit_width as usize;
        for len in [0, 1, 2, 63, 64, 65, rng.below(300)] {
            let model: Vec<u64> = (0..len).map(|_| value(&mut rng, bit_width)).collect();
            let required = if len == 0 {
                0
            } else {
                (len * w).div_ceil(64) + 1
            };
            let mut words: Vec<u64> = (0..required).map(|_| rng.next_u64()).collect();
            for k in 0..len * w {
                let (i, j) = (k / w, k % w);
                words[k / 64] &= !(1 << (k % 64));
                words[k / 64] |= ((model[i] >> j) & 1) << (k % 64);
            }

            check(
                &FixedVec::<u64, &[u64]>::from_parts(&words, len, bit_width).unwrap(),
                &model,
            );
            check(
                &FixedVec::<u64>::from_parts(words.clone(), len, bit_width).unwrap(),
                &model,
            );
            check(
                &FixedVec::<u64, Box<[u64]>>::from_parts(words.clone().into(), len, bit_width)
                    .unwrap(),
                &model,
            );
            check(
                &FixedVec::<u64, Arc<[u64]>>::from_parts(words.clone().into(), len, bit_width)
                    .unwrap(),
                &model,
            );
            if required > 0 {
                assert_eq!(
                    FixedVec::<u64, &[u64]>::from_parts(&words[..required - 1], len, bit_width)
                        .unwrap_err(),
                    FixedVecError::BufferTooSmall {
                        words: required - 1,
                        required
                    }
                );
            }

            let mut vec = FixedVec::<u64>::new(bit_width).unwrap();
            vec.extend_from_slice(&model);
            let (parts, parts_len, parts_width) = vec.into_parts();
            assert_eq!((parts_len, parts_width), (len, bit_width));
            assert!(parts.len() >= required);
            assert!((0..len * w).all(|k| bit(&parts, k) == bit(&words, k)));

            // `set` writes the same layout.
            let mut vec = FixedVec::<u64>::from_parts(vec![0; required], len, bit_width).unwrap();
            for (index, &x) in model.iter().enumerate() {
                vec.set(index, x);
            }
            let parts = vec.into_parts().0;
            assert!((0..len * w).all(|k| bit(&parts, k) == bit(&words, k)));
        }
    }
}

#[test]
fn from_parts_rejects_inconsistent_parts() {
    let too_wide =
        |bit_width, max| FixedVecError::BitWidthTooLarge(BitWidthTooLarge { bit_width, max });
    assert_eq!(
        FixedVec::<u8, &[u64]>::from_parts(&[0, 0], 1, 9).unwrap_err(),
        too_wide(9, 8)
    );
    assert_eq!(
        FixedVec::<u64, &[u64]>::from_parts(&[0, 0], 1, 65).unwrap_err(),
        too_wide(65, 64)
    );
    assert_eq!(
        FixedVec::<u64, &[u64]>::from_parts(&[], usize::MAX, 2).unwrap_err(),
        FixedVecError::LengthOverflow {
            len: usize::MAX,
            bit_width: 2
        }
    );
    // A non-empty vector of bit width 0 still needs the padding word, and nothing more.
    assert_eq!(
        FixedVec::<u64, &[u64]>::from_parts(&[], 5, 0).unwrap_err(),
        FixedVecError::BufferTooSmall {
            words: 0,
            required: 1
        }
    );
    let zeros = FixedVec::<u64, &[u64]>::from_parts(&[u64::MAX], usize::MAX, 0).unwrap();
    assert_eq!(zeros.len(), usize::MAX);
    assert_eq!(zeros.get(usize::MAX - 1), Some(0));
    assert_eq!(zeros.iter().next_back(), Some(0));
}

fn check_length_overflow(grow: impl Fn(&mut FixedVec<u64>)) {
    // Zero-width elements need only one padding word, even at the largest logical length.
    let mut vec = FixedVec::from_parts(vec![u64::MAX], usize::MAX - 1, 0).unwrap();
    grow(&mut vec);
    assert_eq!(vec.len(), usize::MAX);
    assert_eq!(vec.last(), Some(0));

    let mut vec = FixedVec::from_parts(vec![u64::MAX], usize::MAX, 0).unwrap();
    let capacity = vec.capacity();
    let result = catch_unwind(AssertUnwindSafe(|| grow(&mut vec)));
    assert_eq!(vec.capacity(), capacity);
    assert_eq!(vec.into_parts(), (vec![u64::MAX], usize::MAX, 0));
    let panic = result.expect_err("growing past usize::MAX must panic");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied());
    assert_eq!(message, Some("capacity overflow"));
}

#[test]
fn push_at_max_length_preserves_state() {
    check_length_overflow(|vec| vec.push(0));
}

#[test]
fn try_push_at_max_length_preserves_state() {
    check_length_overflow(|vec| vec.try_push(0).unwrap());
}

#[test]
fn insert_at_max_length_preserves_state() {
    check_length_overflow(|vec| vec.insert(0, 0));
}

#[test]
fn insert_at_end_at_max_length_preserves_state() {
    check_length_overflow(|vec| vec.insert(vec.len(), 0));
}

/// Values that do not fit return an error from the `try_` methods and panic elsewhere, and the
/// methods state what a panic leaves behind.
#[test]
fn values_that_do_not_fit_and_indices_out_of_bounds() {
    let mut vec = FixedVec::<u32>::new(4).unwrap();
    vec.extend_from_slice(&[1, 2, 3]);
    let error = ValueTooLarge {
        value: 16,
        required: 5,
        bit_width: 4,
    };
    assert_eq!(vec.try_push(16), Err(error));
    assert_eq!(vec.try_set(0, 16), Err(error));
    assert_eq!(vec, [1, 2, 3][..]);

    // Signed elements of 3 bits hold -4..=3.
    let mut signed = FixedVec::<i8>::new(3).unwrap();
    for x in [4, -5] {
        assert_eq!(
            signed.try_push(x),
            Err(ValueTooLarge {
                value: x,
                required: 4,
                bit_width: 3
            })
        );
    }
    signed.extend_from_slice(&[-4, 3]);
    assert_eq!(signed, [-4, 3][..]);

    let cases: [Case<FixedVec<u32>>; 18] = [
        ("push", |v| v.push(16)),
        ("set out of bounds", |v| v.set(3, 0)),
        ("set", |v| v.set(0, 16)),
        // SAFETY: `0 < len`.
        ("set_unchecked", |v| unsafe { v.set_unchecked(0, 16) }),
        ("try_set out of bounds", |v| {
            let _ = v.try_set(3, 0);
        }),
        ("replace out of bounds", |v| {
            let _ = v.replace(3, 0);
        }),
        ("replace", |v| {
            let _ = v.replace(0, 16);
        }),
        ("swap", |v| v.swap(0, 3)),
        ("insert past the end", |v| v.insert(4, 0)),
        ("insert", |v| v.insert(0, 16)),
        ("remove", |v| {
            let _ = v.remove(3);
        }),
        ("swap_remove", |v| {
            let _ = v.swap_remove(3);
        }),
        ("resize", |v| v.resize(10, 16)),
        ("fill", |v| v.fill(16)),
        ("map_in_place", |v| v.map_in_place(|x| x + 14)),
        ("rotate_left", |v| v.rotate_left(4)),
        ("rotate_right", |v| v.rotate_right(4)),
        ("reserve", |v| v.reserve(usize::MAX)),
    ];
    for (name, case) in cases {
        let mut v = vec.clone();
        panics(name, || case(&mut v));
    }
    panics("with_capacity", || {
        let _ = FixedVec::<u64>::with_capacity(64, usize::MAX);
    });

    // `extend_from_slice` appends nothing, `extend` the values before the one that does not
    // fit, and `fill_with` replaces the elements before it.
    let mut v = vec.clone();
    panics("extend_from_slice", || v.extend_from_slice(&[5, 16]));
    assert_eq!(v, [1, 2, 3][..]);
    let mut v = vec.clone();
    panics("extend", || v.extend([5, 16, 7]));
    assert_eq!(v, [1, 2, 3, 5][..]);
    let mut v = vec.clone();
    let mut fills = [9, 16, 9].into_iter();
    panics("fill_with", || v.fill_with(|| fills.next().unwrap()));
    assert_eq!(v, [9, 2, 3][..]);
}

/// `Backend` is sealed: the unchecked reads trust the slice that `as_ref` returns, and a type of
/// another crate could return a shorter one. trybuild compares the whole output of the
/// compiler with `tests/compile_fail/*.stderr`; after a compiler release changes the message,
/// review the new one and regenerate it with `TRYBUILD=overwrite`.
#[test]
#[cfg_attr(miri, ignore = "trybuild runs the compiler")]
fn foreign_types_cannot_be_backends() {
    trybuild::TestCases::new().compile_fail("tests/compile_fail/*.rs");
}
