//! The parallel iterators, with the `parallel` feature, against the sequential ones: rayon
//! splits the elements at every index when the pieces may hold one element.

use crate::{Rng, nz, value};
use compressed_intvec::{Codec, FixedVec, SeqVec, VarVec};
use rayon::prelude::*;

#[track_caller]
fn same<I: IndexedParallelIterator<Item = u64>>(make: impl Fn() -> I, model: &[u64]) {
    let reversed: Vec<u64> = model.iter().rev().copied().collect();
    let (skip, take) = (model.len() / 3, model.len() / 2);
    assert_eq!(make().len(), model.len());
    assert_eq!(make().collect::<Vec<_>>(), model);
    assert_eq!(make().with_max_len(1).collect::<Vec<_>>(), model);
    assert_eq!(make().rev().with_max_len(1).collect::<Vec<_>>(), reversed);
    assert_eq!(
        make().skip(skip).take(take).collect::<Vec<_>>(),
        model[skip..][..take]
    );
    assert!(make().enumerate().all(|(i, x)| model[i] == x));
}

#[test]
fn fixed_vectors() {
    let mut rng = Rng::new(1);
    for bit_width in 0..=64 {
        for len in [0, 1, 1000] {
            let model: Vec<u64> = (0..len).map(|_| value(&mut rng, bit_width)).collect();
            let mut vec = FixedVec::new(bit_width).unwrap();
            vec.extend_from_slice(&model);
            same(|| vec.par_iter(), &model);
        }
    }
}

#[test]
fn variable_vectors() {
    let mut rng = Rng::new(2);
    for codec in [Codec::Gamma, Codec::VByteLe, Codec::Zeta { k: 3 }] {
        for rate in [1, 2, 7, 32, usize::MAX] {
            for len in [0, 1, 31, 32, 33, 1000] {
                let model: Vec<u64> = (0..len).map(|_| rng.word() >> 1).collect();
                let vec = VarVec::builder()
                    .codec(codec)
                    .sampling_rate(nz(rate))
                    .build(&model)
                    .unwrap();
                same(|| vec.par_iter(), &model);
            }
        }
    }
}

#[test]
fn sequence_vectors() {
    let mut rng = Rng::new(3);
    for stored in [false, true] {
        let model: Vec<Vec<u64>> = (0..500)
            .map(|_| (0..rng.below(6)).map(|_| rng.word() >> 1).collect())
            .collect();
        let mut builder = SeqVec::builder();
        if stored {
            builder = builder.store_lengths();
        }
        let vec = builder.build(&model);
        let collect = |max_len| {
            vec.par_iter()
                .with_max_len(max_len)
                .map(Iterator::collect::<Vec<u64>>)
                .collect::<Vec<_>>()
        };
        assert_eq!(collect(1), model);
        assert_eq!(collect(usize::MAX), model);
        let reversed: Vec<Vec<u64>> = vec.par_iter().rev().map(Iterator::collect).collect();
        assert!(reversed.iter().eq(model.iter().rev()));
    }
}
