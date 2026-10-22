//! `VarVec` against a `Vec<T>`, for every code and several sampling rates, and its parts.

use crate::codec::{assert_canonical, codecs, words};
use crate::{Elem, MIRI_DIVISOR, Rng, Trace, hash, nz};
use compressed_intvec::{
    Backend, Codec, FixedVec, Storable, VarVec, VarVecParts, VarVecPartsError,
};
use std::any::type_name;
use std::sync::Arc;

/// Checks every read of `vec` against `model`, with random calls to the iterators.
#[track_caller]
fn check<T: Elem, B: Backend>(vec: &VarVec<T, B>, model: &[T], rng: &mut Rng) {
    assert_eq!(vec.len(), model.len());
    assert_eq!(vec.is_empty(), model.is_empty());
    for (i, &x) in model.iter().enumerate() {
        assert_eq!(vec.get(i), Some(x), "index {i}");
    }
    assert_eq!(vec.get(model.len()), None);
    assert!(vec.iter().eq(model.iter().copied()));
    assert!(vec.iter().rev().eq(model.iter().rev().copied()));
    assert_eq!(vec.iter().count(), model.len());
    assert_eq!(vec.iter().last(), model.last().copied());
    assert_eq!(*vec, *model);
    assert_eq!(format!("{vec:?}"), format!("{model:?}"));

    let reach = 2 * vec.sampling_rate().get().min(50);
    let mut iter = vec.iter();
    let mut expected = model.iter().copied();
    loop {
        assert_eq!(iter.len(), expected.len());
        let (got, want) = match rng.below(5) {
            0 => (iter.next_back(), expected.next_back()),
            1 => {
                let n = rng.below(reach);
                (iter.nth(n), expected.nth(n))
            }
            2 => {
                let n = rng.below(reach);
                (iter.nth_back(n), expected.nth_back(n))
            }
            _ => (iter.next(), expected.next()),
        };
        assert_eq!(got, want);
        if want.is_none() {
            break;
        }
    }
    let skip = rng.below(model.len() + 1);
    let take = rng.below(model.len() + 1);
    assert!(
        vec.iter()
            .skip(skip)
            .take(take)
            .eq(model.iter().skip(skip).take(take).copied())
    );

    // In any order, with repeats and indices out of bounds, and in increasing order.
    let mut indices: Vec<usize> = (0..60)
        .map(|_| match rng.below(8) {
            0 => usize::MAX,
            1 => model.len(),
            _ => rng.below(model.len() + 1),
        })
        .collect();
    for _ in 0..2 {
        let expected: Vec<Option<T>> = indices.iter().map(|&i| model.get(i).copied()).collect();
        let many = vec.get_many(indices.iter().copied());
        assert_eq!(many.len(), indices.len());
        assert_eq!(many.collect::<Vec<_>>(), expected);
        indices.sort_unstable();
    }
}

#[test]
fn borrowing_and_boxing_preserve_storage() {
    #[track_caller]
    fn borrow<B: Backend>(
        data: B,
        sample_words: B,
        original: &VarVecParts<Vec<u64>>,
        model: &[u64],
    ) -> VarVec<u64, B> {
        let (original_samples, sample_count, sample_width) = original.samples.clone().into_parts();
        let data_pointer = data.as_ref().as_ptr();
        let sample_pointer = sample_words.as_ref().as_ptr();
        let samples = FixedVec::from_parts(sample_words, sample_count, sample_width).unwrap();
        let parent = VarVec::from_parts(VarVecParts {
            data,
            samples,
            sampling_rate: original.sampling_rate,
            len: original.len,
            codec: original.codec,
        })
        .unwrap();
        let view = parent.as_borrowed();
        check(&view, model, &mut Rng::new(1));
        let parts = view.into_parts();
        let (samples, count, width) = parts.samples.into_parts();
        assert_eq!(parts.data.as_ptr(), data_pointer);
        assert_eq!(parts.data, original.data);
        assert_eq!(samples.as_ptr(), sample_pointer);
        assert_eq!(samples, original_samples);
        assert_eq!((count, width), (sample_count, sample_width));
        assert_eq!(
            (parts.len, parts.codec, parts.sampling_rate),
            (original.len, original.codec, original.sampling_rate)
        );
        check(&parent, model, &mut Rng::new(1));
        parent
    }

    for model in [&[0u64, 1, 7, 1024][..], &[][..]] {
        let original = VarVec::builder()
            .codec(Codec::Delta)
            .sampling_rate(nz(3))
            .build(model)
            .unwrap()
            .into_parts();
        let (sample_words, sample_count, sample_width) = original.samples.clone().into_parts();
        let owned = borrow(
            original.data.clone(),
            sample_words.clone(),
            &original,
            model,
        );
        borrow(
            original.data.clone().into_boxed_slice(),
            sample_words.clone().into_boxed_slice(),
            &original,
            model,
        );
        borrow(
            Arc::<[u64]>::from(&original.data[..]),
            Arc::<[u64]>::from(&sample_words[..]),
            &original,
            model,
        );
        borrow(&original.data[..], &sample_words[..], &original, model);

        let boxed: VarVec<u64, Box<[u64]>> = owned.into();
        check(&boxed, model, &mut Rng::new(1));
        check(&boxed.as_borrowed(), model, &mut Rng::new(1));
        let borrowed = boxed.as_borrowed().into_parts();
        let (borrowed_samples, count, width) = borrowed.samples.into_parts();
        assert_eq!(borrowed.data, original.data);
        assert_eq!(borrowed_samples, sample_words);
        assert_eq!((count, width), (sample_count, sample_width));
        assert_eq!(
            (borrowed.len, borrowed.codec, borrowed.sampling_rate),
            (original.len, original.codec, original.sampling_rate)
        );
        let data_pointer = borrowed.data.as_ptr();
        let sample_pointer = borrowed_samples.as_ptr();
        let parts = boxed.into_parts();
        let (samples, count, width) = parts.samples.into_parts();
        assert_eq!(parts.data.as_ptr(), data_pointer);
        assert_eq!(samples.as_ptr(), sample_pointer);
        assert_eq!(&*parts.data, original.data);
        assert_eq!(&*samples, sample_words);
        assert_eq!((count, width), (sample_count, sample_width));
        assert_eq!(
            (parts.len, parts.codec, parts.sampling_rate),
            (original.len, original.codec, original.sampling_rate)
        );
    }
}

/// Lengths at the samples around `rate`, and one at random.
fn lengths(rng: &mut Rng, rate: usize) -> [usize; 5] {
    let near = rate.min(100);
    [0, 1, near, near + 1, rng.below(150)]
}

const RATES: [usize; 6] = [1, 2, 3, 7, 32, usize::MAX];

/// Every code, sampling rate and length, with the words that each code encodes cheaply, stored
/// as `T`.
fn every_code<T: Elem>() {
    let mut rng = Rng::new(1);
    let mask = u64::MAX >> (64 - T::BITS);
    for codec in codecs() {
        for rate in RATES {
            for len in lengths(&mut rng, rate) {
                let model: Vec<T> = words(&mut rng, codec, len)
                    .into_iter()
                    .map(|w| T::from_word(w & mask))
                    .collect();
                let _trace = Trace(vec![format!(
                    "{} with {codec:?}, rate {rate}: {model:?}",
                    type_name::<T>()
                )]);
                let builder = VarVec::builder().sampling_rate(nz(rate));
                let vec = builder.codec(codec).build(&model).unwrap();
                assert_eq!((vec.codec(), vec.sampling_rate()), (codec, nz(rate)));
                check(&vec, &model, &mut rng);
                let from_iter = builder.codec(codec).build_from_iter(model.iter().copied());
                assert_eq!(
                    from_iter.unwrap().into_parts().data,
                    vec.clone().into_parts().data
                );
                check(&builder.build(&model), &model, &mut rng);
                with_borrowed(vec.into_parts(), |view| check(&view, &model, &mut rng));
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn every_code_sampling_rate_and_length() {
    every_code::<u64>();
    every_code::<i64>();
    every_code::<u8>();
    every_code::<i8>();
}

/// Rebuilds a vector over the borrowed words of `parts`, which `from_parts` checks again, and
/// passes it to `f`.
fn with_borrowed<T: Storable>(parts: VarVecParts<Vec<u64>>, f: impl FnOnce(VarVec<T, &[u64]>)) {
    let (words, len, bit_width) = parts.samples.into_parts();
    let samples = FixedVec::from_parts(&words[..], len, bit_width).unwrap();
    f(VarVec::from_parts(VarVecParts {
        data: &parts.data[..],
        samples,
        sampling_rate: parts.sampling_rate,
        len: parts.len,
        codec: parts.codec,
    })
    .unwrap());
}

/// `binary_search` returns the first match, or the insertion point; signed elements, whose
/// order is not the order of their zig-zag words.
#[test]
fn binary_search_finds_the_first_match() {
    let mut rng = Rng::new(2);
    for rate in [1, 2, 3, 7, 32] {
        for _ in 0..40 / MIRI_DIVISOR {
            let mut model: Vec<i32> = (0..rng.below(200))
                .map(|_| rng.below(31) as i32 - 15)
                .collect();
            model.sort_unstable();
            let vec = VarVec::builder().sampling_rate(nz(rate)).build(&model);
            for x in -17..17 {
                let first = model.partition_point(|&y| y < x);
                let expected = if model.get(first) == Some(&x) {
                    Ok(first)
                } else {
                    Err(first)
                };
                assert_eq!(vec.binary_search(&x), expected, "{x} in {model:?}");
                assert_eq!(vec.binary_search_by(|y| y.cmp(&x)), expected);
                let half = x.div_euclid(2);
                let first = model.partition_point(|&y| y.div_euclid(2) < half);
                let expected = match model.get(first) {
                    Some(&y) if y.div_euclid(2) == half => Ok(first),
                    _ => Err(first),
                };
                assert_eq!(
                    vec.binary_search_by_key(&half, |y| y.div_euclid(2)),
                    expected
                );
            }
        }
    }
}

#[cfg(target_pointer_width = "32")]
#[test]
#[ignore = "allocates 256 MiB and validates over two billion codewords"]
fn binary_search_last_sample_boundary_on_32_bit() {
    let rate = 1usize << 31;
    let len = rate + 1;
    // Gamma encodes zero as one bit set to 1. Each sample starts at its element's index.
    let vec = VarVec::<u64>::from_parts(VarVecParts {
        data: vec![u64::MAX; len.div_ceil(64)],
        samples: FixedVec::from_slice(&[0, rate as u64]),
        sampling_rate: nz(rate),
        len,
        codec: Codec::Gamma,
    })
    .unwrap();
    assert_eq!(vec.get(0), Some(0));
    assert_eq!(vec.get(rate), Some(0));
    assert_eq!(vec.binary_search(&1), Err(len));
}

/// `from_slice`, `collect`, `From` and the builder without a code agree, with the sampling rate
/// 32; equality and hashing ignore the code and the sampling rate.
#[test]
fn defaults_equality_and_hashing() {
    let values: Vec<u32> = (0..100).map(|x| x * x % 37).collect();
    let vec = VarVec::from_slice(&values);
    assert_eq!(vec.sampling_rate(), nz(32));
    let collected: VarVec<u32> = values.iter().copied().collect();
    for other in [
        collected,
        VarVec::from(&values[..]),
        VarVec::builder().build(&values),
        VarVec::builder().build_from_iter(values.iter().copied()),
    ] {
        assert_eq!(
            (other.codec(), other.sampling_rate()),
            (vec.codec(), nz(32))
        );
        assert_eq!(other, vec);
    }
    assert!(VarVec::<u32>::default().is_empty());

    let delta = VarVec::builder()
        .sampling_rate(nz(5))
        .codec(Codec::Delta)
        .build(&values)
        .unwrap();
    let gamma = VarVec::builder()
        .codec(Codec::Gamma)
        .sampling_rate(nz(1))
        .build(&values)
        .unwrap();
    assert_eq!(
        (delta.codec(), delta.sampling_rate()),
        (Codec::Delta, nz(5))
    );
    assert_eq!(
        (gamma.codec(), gamma.sampling_rate()),
        (Codec::Gamma, nz(1))
    );
    assert_eq!(delta, gamma);
    assert_eq!(hash(&delta), hash(&gamma));
    with_borrowed::<u32>(gamma.into_parts(), |view| assert_eq!(view, delta));
    assert_ne!(vec, VarVec::from_slice(&values[1..]));
    assert_ne!(vec, VarVec::from_slice(&[0; 100]));
}

#[test]
fn from_parts_rejects_inconsistent_parts() {
    let values = [3u64, 1, 4, 1, 5, 9, 2, 6, 5, 3];
    let parts = VarVec::builder()
        .codec(Codec::Gamma)
        .sampling_rate(nz(4))
        .build(&values)
        .unwrap()
        .into_parts();
    let samples: Vec<u64> = parts.samples.iter().collect();
    let changed = |change: &dyn Fn(&mut VarVecParts<Vec<u64>>)| {
        let mut parts = parts.clone();
        change(&mut parts);
        VarVec::<u64>::from_parts(parts).unwrap_err()
    };

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
            VarVecPartsError::InvalidParameter(codec)
        );
    }
    assert_eq!(
        changed(&|p| p.samples = FixedVec::from_slice(&samples[..2])),
        VarVecPartsError::SampleCount {
            expected: 3,
            found: 2
        }
    );
    assert_eq!(
        changed(&|p| p.sampling_rate = nz(5)),
        VarVecPartsError::SampleCount {
            expected: 2,
            found: 3
        }
    );
    assert_eq!(
        changed(&|p| {
            let mut shifted = samples.clone();
            shifted[1] += 1;
            p.samples = FixedVec::from_slice(&shifted);
        }),
        VarVecPartsError::SampleMismatch {
            sample: 1,
            expected: samples[1],
            found: samples[1] + 1
        }
    );
    // The bits after the last codeword are zero, so gamma finds no end of its unary part.
    assert_eq!(
        changed(&|p| p.len = 11),
        VarVecPartsError::Truncated { index: 10 }
    );
    assert_eq!(
        changed(&|p| p.data.clear()),
        VarVecPartsError::Truncated { index: 0 }
    );
    // 64 zeros and a one: the length of a gamma codeword whose value does not fit 64 bits.
    assert_eq!(
        changed(&|p| {
            p.data = vec![0, 1];
            p.len = 1;
            p.samples = FixedVec::from_slice(&[0]);
        }),
        VarVecPartsError::InvalidCodeword { index: 0 }
    );

    let wide = VarVec::from_slice(&[1u64, 300]).into_parts();
    assert_eq!(
        VarVec::<u8>::from_parts(wide).unwrap_err(),
        VarVecPartsError::WordOutOfRange {
            index: 1,
            word: 300
        }
    );
    // The zig-zag word of -200.
    let negative = VarVec::from_slice(&[-200i64]).into_parts();
    assert_eq!(
        VarVec::<i8>::from_parts(negative).unwrap_err(),
        VarVecPartsError::WordOutOfRange {
            index: 0,
            word: 399
        }
    );
}

/// Corrupts one part at random.
fn corrupt(rng: &mut Rng, parts: &mut VarVecParts<Vec<u64>>) {
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
        3 => parts.len = rng.below(2 * parts.len + 5),
        4 => {
            let mut samples: Vec<u64> = parts.samples.iter().collect();
            let i = rng.below(samples.len().max(1));
            if let Some(sample) = samples.get_mut(i) {
                *sample = rng.next_u64() % (64 * parts.data.len() as u64 + 1);
            }
            parts.samples = FixedVec::from_slice(&samples);
        }
        _ => parts.sampling_rate = nz(1 + rng.below(8)),
    }
}

/// A corrupt vector is rejected, or reads the same through every access path, and its stream
/// holds the codewords of what it reads; both outcomes occur.
#[test]
fn corrupt_parts_are_rejected_or_read_consistently() {
    let mut rng = Rng::new(3);
    let (mut accepted, mut rejected) = (0, 0);
    for codec in codecs() {
        for _ in 0..30 {
            let n = rng.below(100);
            let values = words(&mut rng, codec, n);
            let mut parts = VarVec::builder()
                .codec(codec)
                .sampling_rate(nz(1 + rng.below(8)))
                .build(&values)
                .unwrap()
                .into_parts();
            corrupt(&mut rng, &mut parts);
            let data = parts.data.clone();
            let Ok(vec) = VarVec::<u64>::from_parts(parts) else {
                rejected += 1;
                continue;
            };
            accepted += 1;
            let read: Vec<u64> = vec.iter().collect();
            assert_eq!(read.len(), vec.len());
            assert!(vec.iter().rev().eq(read.iter().rev().copied()));
            assert!(read.iter().enumerate().all(|(i, &x)| vec.get(i) == Some(x)));
            let backwards = vec.get_many((0..read.len()).rev());
            assert!(backwards.eq(read.iter().rev().map(|&x| Some(x))));
            assert_canonical(codec, &read, &data, 0);
        }
    }
    assert!(accepted > 0 && rejected > 0, "{accepted} {rejected}");
}
