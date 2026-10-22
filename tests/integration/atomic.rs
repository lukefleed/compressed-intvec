//! `AtomicFixedVec`: every operation against a `Vec<T>` on one thread, then the guarantees that
//! hold between threads.

use crate::{
    AliasedFlag, Case, Elem, MIRI_DIVISOR, Rng, Trace, fixed::check, for_each_type, max_width,
    panics, range, value,
};
use compressed_intvec::{AtomicFixedVec, BitWidthTooLarge, FixedVec, FixedVecError};
use std::any::type_name;
use std::sync::atomic::Ordering::{self, AcqRel, Acquire, Relaxed, Release, SeqCst};
use std::thread;

const LOADS: [Ordering; 3] = [Relaxed, Acquire, SeqCst];
const STORES: [Ordering; 3] = [Relaxed, Release, SeqCst];
const ALL: [Ordering; 5] = [Relaxed, Release, Acquire, AcqRel, SeqCst];

#[test]
fn compare_exchange_compares_encoded_words() {
    use AliasedFlag::{Off, On};

    for (width, len, index, words) in [(2, 1, 0, vec![3, 0]), (33, 2, 1, vec![3u64 << 33, 0, 0])] {
        let fixed = FixedVec::<AliasedFlag>::from_parts(words, len, width).unwrap();
        let atomic = AtomicFixedVec::from(fixed);
        let loaded = atomic.load(index, Relaxed);
        assert_eq!(loaded, On);
        assert_eq!(
            atomic.compare_exchange(index, loaded, Off, Relaxed, Relaxed),
            Err(On)
        );
        assert_eq!(atomic.load(index, Relaxed), On);
        assert_eq!(
            atomic.fetch_update(index, Relaxed, Relaxed, |_| Some(Off)),
            Ok(On)
        );
        assert_eq!(atomic.load(index, Relaxed), Off);
        if index == 1 {
            assert_eq!(atomic.load(0, Relaxed), Off);
        }
    }
}

fn pick(rng: &mut Rng, orders: &[Ordering]) -> Ordering {
    orders[rng.below(orders.len())]
}

/// Wraps `x` into the range of `bit_width` bits: modulo `2^w` for unsigned elements, and into
/// the two's complement range of `w` bits for signed ones.
fn wrap<T: Elem>(x: i128, bit_width: u32) -> T {
    let (lo, hi) = range::<T>(bit_width);
    T::from_int(lo + (x - lo).rem_euclid(hi - lo + 1)).unwrap()
}

/// Applies one random operation to `vec` and `model`, compares what it returns, and returns its
/// description.
fn step<T: Elem>(rng: &mut Rng, vec: &AtomicFixedVec<T>, model: &mut [T]) -> String {
    let w = vec.bit_width();
    let i = rng.below(model.len());
    let old = model[i];
    let x: T = value(rng, w);
    let op = rng.below(13);
    let description = format!("op {op} at {i} with {x:?}");
    let new = match op {
        0 => {
            assert_eq!(vec.load(i, pick(rng, &LOADS)), old);
            old
        }
        1 => {
            vec.store(i, x, pick(rng, &STORES));
            x
        }
        2 => {
            assert_eq!(vec.swap(i, x, pick(rng, &ALL)), old);
            x
        }
        3 => {
            let current = if rng.coin() { old } else { value(rng, w) };
            let result = vec.compare_exchange(i, current, x, pick(rng, &ALL), pick(rng, &LOADS));
            if current == old {
                assert_eq!(result, Ok(old));
                x
            } else {
                assert_eq!(result, Err(old));
                old
            }
        }
        4 => {
            let replace = rng.coin();
            let mut seen = Vec::new();
            let result = vec.fetch_update(i, pick(rng, &ALL), pick(rng, &LOADS), |y| {
                seen.push(y);
                replace.then_some(x)
            });
            // A weak exchange may fail spuriously, and the closure then runs again.
            assert!(
                !seen.is_empty() && seen.iter().all(|&y| y == old),
                "{seen:?}"
            );
            assert_eq!(result, if replace { Ok(old) } else { Err(old) });
            if replace { x } else { old }
        }
        5 => {
            assert_eq!(vec.fetch_add(i, x, pick(rng, &ALL)), old);
            wrap(old.to_int() + x.to_int(), w)
        }
        6 => {
            assert_eq!(vec.fetch_sub(i, x, pick(rng, &ALL)), old);
            wrap(old.to_int() - x.to_int(), w)
        }
        7 => {
            assert_eq!(vec.fetch_and(i, x, pick(rng, &ALL)), old);
            old & x
        }
        8 => {
            assert_eq!(vec.fetch_or(i, x, pick(rng, &ALL)), old);
            old | x
        }
        9 => {
            assert_eq!(vec.fetch_xor(i, x, pick(rng, &ALL)), old);
            old ^ x
        }
        10 => {
            assert_eq!(vec.fetch_max(i, x, pick(rng, &ALL)), old);
            old.max(x)
        }
        11 => {
            assert_eq!(vec.fetch_min(i, x, pick(rng, &ALL)), old);
            old.min(x)
        }
        _ => {
            // Several operations on the same element, which then holds the last result.
            vec.store(i, x, Relaxed);
            assert_eq!(vec.fetch_add(i, x, Relaxed), x);
            wrap(2 * x.to_int(), w)
        }
    };
    model[i] = new;
    description
}

fn random_operations<T: Elem>() {
    let mut rng = Rng::new(1);
    for bit_width in 0..=max_width::<T>() {
        for _ in 0..if cfg!(miri) { 1 } else { 4 } {
            let len = 1 + rng.below(100);
            let mut model: Vec<T> = (0..len).map(|_| value(&mut rng, bit_width)).collect();
            let mut start = FixedVec::new(bit_width).unwrap();
            start.extend_from_slice(&model);
            let vec = AtomicFixedVec::from(start);
            let mut trace = Trace(vec![format!(
                "{} with bit width {bit_width}, {model:?}",
                type_name::<T>()
            )]);
            for _ in 0..200 / MIRI_DIVISOR {
                trace.0.push(step(&mut rng, &vec, &mut model));
                let i = rng.below(len);
                assert_eq!(vec.load(i, Relaxed), model[i]);
            }
            assert_eq!(vec.len(), len);
            assert_eq!(format!("{vec:?}"), format!("{model:?}"));
            check(&FixedVec::from(vec), &model);
        }
    }
}

#[test]
fn random_operations_on_every_type_and_bit_width() {
    for_each_type!(random_operations);
}

fn constructors<T: Elem>() {
    let max = max_width::<T>();
    for bit_width in 0..=max {
        let vec = AtomicFixedVec::<T>::new(bit_width, 70).unwrap();
        assert_eq!(
            (vec.len(), vec.bit_width(), vec.is_empty()),
            (70, bit_width, false)
        );
        let zero = T::from_int(0).unwrap();
        assert!((0..70).all(|i| vec.load(i, Relaxed) == zero));
    }
    assert_eq!(
        AtomicFixedVec::<T>::new(max + 1, 1).unwrap_err(),
        FixedVecError::BitWidthTooLarge(BitWidthTooLarge {
            bit_width: max + 1,
            max
        })
    );
    assert_eq!(
        AtomicFixedVec::<T>::new(2, usize::MAX).unwrap_err(),
        FixedVecError::LengthOverflow {
            len: usize::MAX,
            bit_width: 2
        }
    );
    let mut rng = Rng::new(2);
    let model: Vec<T> = (0..50).map(|_| value(&mut rng, max)).collect();
    for vec in [
        AtomicFixedVec::from_slice(&model),
        AtomicFixedVec::from(&model[..]),
    ] {
        assert_eq!(vec.bit_width(), FixedVec::from_slice(&model).bit_width());
        check(&FixedVec::from(vec), &model);
    }
    assert!(AtomicFixedVec::<T>::from_slice(&[]).is_empty());
}

#[test]
fn constructors_of_every_type() {
    for_each_type!(constructors);
}

/// The panics that the methods document; `fetch_update` leaves the element unchanged when its
/// closure returns a value that does not fit.
#[test]
fn panics_on_values_indices_and_orderings() {
    let mut vec = AtomicFixedVec::<u32>::new(5, 3).unwrap();
    vec.store(1, 7, Relaxed);
    let cases: [Case<AtomicFixedVec<u32>>; 22] = [
        ("load out of bounds", |v| {
            let _ = v.load(3, Relaxed);
        }),
        ("store out of bounds", |v| v.store(3, 0, Relaxed)),
        ("swap out of bounds", |v| {
            let _ = v.swap(3, 0, Relaxed);
        }),
        ("fetch_add out of bounds", |v| {
            let _ = v.fetch_add(3, 0, Relaxed);
        }),
        ("fetch_update out of bounds", |v| {
            let _ = v.fetch_update(3, Relaxed, Relaxed, Some);
        }),
        ("store", |v| v.store(0, 32, Relaxed)),
        ("swap", |v| {
            let _ = v.swap(0, 32, Relaxed);
        }),
        ("compare_exchange", |v| {
            let _ = v.compare_exchange(0, 0, 32, Relaxed, Relaxed);
        }),
        ("fetch_add", |v| {
            let _ = v.fetch_add(0, 32, Relaxed);
        }),
        ("fetch_sub", |v| {
            let _ = v.fetch_sub(0, 32, Relaxed);
        }),
        ("fetch_and", |v| {
            let _ = v.fetch_and(0, 32, Relaxed);
        }),
        ("fetch_or", |v| {
            let _ = v.fetch_or(0, 32, Relaxed);
        }),
        ("fetch_xor", |v| {
            let _ = v.fetch_xor(0, 32, Relaxed);
        }),
        ("fetch_max", |v| {
            let _ = v.fetch_max(0, 32, Relaxed);
        }),
        ("fetch_min", |v| {
            let _ = v.fetch_min(0, 32, Relaxed);
        }),
        ("load with Release", |v| {
            let _ = v.load(0, Release);
        }),
        ("load with AcqRel", |v| {
            let _ = v.load(0, AcqRel);
        }),
        ("store with Acquire", |v| v.store(0, 0, Acquire)),
        ("store with AcqRel", |v| v.store(0, 0, AcqRel)),
        ("compare_exchange failing with Release", |v| {
            let _ = v.compare_exchange(0, 0, 1, SeqCst, Release);
        }),
        ("compare_exchange failing with AcqRel", |v| {
            let _ = v.compare_exchange(0, 0, 1, SeqCst, AcqRel);
        }),
        ("fetch_update fetching with Release", |v| {
            let _ = v.fetch_update(0, Relaxed, Release, Some);
        }),
    ];
    for (name, case) in cases {
        panics(name, || case(&mut vec));
    }
    panics("fetch_update", || {
        let _ = vec.fetch_update(1, Relaxed, Relaxed, |_| Some(32));
    });
    assert_eq!(format!("{vec:?}"), "[0, 7, 0]");
}

/// The closure of `fetch_update` runs without a lock of the vector held, so it may access the
/// vector, including the straddling elements of the same lock stripe. Elements of 33 bits: 1
/// and 3 straddle two words.
#[test]
fn fetch_update_closure_may_access_the_vector() {
    let vec = AtomicFixedVec::<u64>::new(33, 4).unwrap();
    let result = vec.fetch_update(1, SeqCst, SeqCst, |x| {
        vec.store(3, vec.load(1, SeqCst) + 5, SeqCst);
        vec.fetch_add(1, 0, SeqCst);
        Some(x + 1)
    });
    assert_eq!(result, Ok(0));
    assert_eq!(format!("{vec:?}"), "[0, 1, 0, 5]");
}

/// Bit widths that make elements straddle two words, and others that do not.
const WIDTHS: [u32; 7] = [1, 7, 16, 33, 61, 63, 64];

/// Readers see one of the values that writers store, never a mix of the bits of two, while the
/// neighbors in the same words change too.
#[test]
fn reads_are_never_torn() {
    for bit_width in WIDTHS {
        let mask = u64::MAX >> (64 - bit_width);
        let (a, b) = (0x5555_5555_5555_5555 & mask, 0xAAAA_AAAA_AAAA_AAAA & mask);
        let vec = AtomicFixedVec::<u64>::new(bit_width, 64).unwrap();
        (0..64).for_each(|i| vec.store(i, a, Relaxed));
        thread::scope(|scope| {
            for parity in 0..2 {
                let vec = &vec;
                scope.spawn(move || {
                    for round in 0..2000 / MIRI_DIVISOR {
                        for i in (parity..64).step_by(2) {
                            vec.store(i, if round % 2 == 0 { b } else { a }, Release);
                        }
                    }
                });
                scope.spawn(move || {
                    for _ in 0..2000 / MIRI_DIVISOR {
                        for i in 0..64 {
                            let x = vec.load(i, Acquire);
                            assert!(x == a || x == b, "width {bit_width}: torn {x:#x}");
                        }
                    }
                });
            }
        });
    }
}

/// Increments from several threads, through every kind of read-modify-write, are never lost:
/// they commute, so the final value is the number of increments, wrapped at the bit width.
#[test]
fn concurrent_increments_are_never_lost() {
    const ROUNDS: u64 = 500 / MIRI_DIVISOR as u64;
    for bit_width in WIDTHS {
        let vec = AtomicFixedVec::<u64>::new(bit_width, 64).unwrap();
        let mask = u64::MAX >> (64 - bit_width);
        let increments: [fn(&AtomicFixedVec<u64>, usize, u64); 4] = [
            |v, i, _| {
                let _ = v.fetch_add(i, 1, AcqRel);
            },
            // Subtracting `2^w - 1` adds 1 modulo `2^w`.
            |v, i, mask| {
                let _ = v.fetch_sub(i, mask, AcqRel);
            },
            |v, i, mask| {
                v.fetch_update(i, AcqRel, Acquire, |x| Some(x.wrapping_add(1) & mask))
                    .unwrap();
            },
            |v, i, mask| {
                let mut current = v.load(i, Relaxed);
                while let Err(actual) =
                    v.compare_exchange(i, current, current.wrapping_add(1) & mask, AcqRel, Relaxed)
                {
                    current = actual;
                }
            },
        ];
        thread::scope(|scope| {
            for increment in increments {
                let vec = &vec;
                scope.spawn(move || {
                    for _ in 0..ROUNDS {
                        (0..64).for_each(|i| increment(vec, i, mask));
                    }
                });
            }
        });
        let expected = (4 * ROUNDS) & mask;
        assert!(
            (0..64).all(|i| vec.load(i, Relaxed) == expected),
            "width {bit_width}: {vec:?}"
        );
    }
}

/// Each thread owns the elements `i` with `i % 4 == thread`, which share their words with the
/// elements of the other threads: every read sees the owner's last write.
#[test]
fn neighbors_never_disturb_each_other() {
    for bit_width in WIDTHS {
        let vec = AtomicFixedVec::<u64>::new(bit_width, 128).unwrap();
        thread::scope(|scope| {
            for owner in 0..4 {
                let vec = &vec;
                scope.spawn(move || {
                    let mut rng = Rng::new(owner as u64);
                    let mut mine = vec![0; 128];
                    for _ in 0..20_000 / MIRI_DIVISOR {
                        let i = 4 * rng.below(32) + owner;
                        let x = value::<u64>(&mut rng, bit_width);
                        match rng.below(3) {
                            0 => vec.store(i, x, Release),
                            1 => assert_eq!(vec.swap(i, x, AcqRel), mine[i]),
                            _ => assert_eq!(vec.fetch_xor(i, x ^ mine[i], AcqRel), mine[i]),
                        }
                        mine[i] = x;
                        assert_eq!(vec.load(i, Acquire), x, "width {bit_width}, element {i}");
                    }
                });
            }
        });
    }
}

/// `SeqCst` stores and loads of two straddling elements that share a word are sequentially
/// consistent: in the store-buffering pattern, at least one thread sees the other's store.
#[test]
fn seqcst_straddling_accesses_are_sequentially_consistent() {
    for _ in 0..1000 / MIRI_DIVISOR {
        // Elements 1 and 3 of 33 bits straddle words 0 and 1, and words 1 and 2.
        let vec = AtomicFixedVec::<u64>::new(33, 4).unwrap();
        let (seen_3, seen_1) = thread::scope(|scope| {
            let first = scope.spawn(|| {
                vec.store(1, 1, SeqCst);
                vec.load(3, SeqCst)
            });
            let second = scope.spawn(|| {
                vec.store(3, 1, SeqCst);
                vec.load(1, SeqCst)
            });
            (first.join().unwrap(), second.join().unwrap())
        });
        assert!(
            seen_3 == 1 || seen_1 == 1,
            "both threads missed the other's store"
        );
    }
}

/// The values that the threads return from successful exchanges of one straddling element, with
/// its final value, are its initial value and the values written, each exactly once: no
/// exchange is lost or applied twice. `exchange(vec, value)` writes `value` and returns the
/// element it replaced.
fn check_history(exchange: fn(&AtomicFixedVec<u64>, u64) -> u64) {
    const THREADS: u64 = 3;
    const ROUNDS: u64 = if cfg!(miri) { 500 } else { 100_000 };
    // Distinct values for distinct tags, other than the initial value, most of them with bits
    // in both words of element 1 at widths 33 and 61.
    let tagged = |tag: u64| tag | (tag & 3) << 31;
    for bit_width in [33, 61] {
        let vec = AtomicFixedVec::<u64>::new(bit_width, 4).unwrap();
        // Element 1 straddles words 0 and 1 at both widths; its initial value sets both ends.
        let initial = (1 << (bit_width - 1)) | 1;
        vec.store(1, initial, Relaxed);
        let mut seen: Vec<u64> = thread::scope(|scope| {
            let threads: Vec<_> = (0..THREADS)
                .map(|thread| {
                    let vec = &vec;
                    scope.spawn(move || {
                        (0..ROUNDS)
                            .map(|round| exchange(vec, tagged(thread * ROUNDS + round + 1)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            threads
                .into_iter()
                .flat_map(|thread| thread.join().unwrap())
                .collect()
        });
        seen.push(vec.load(1, Relaxed));
        seen.sort_unstable();
        let mut expected: Vec<u64> = (1..=THREADS * ROUNDS).map(tagged).collect();
        expected.push(initial);
        expected.sort_unstable();
        assert!(
            seen == expected,
            "width {bit_width}: the exchanges do not form one history"
        );
    }
}

#[test]
fn concurrent_swaps_form_one_history() {
    check_history(|vec, value| vec.swap(1, value, AcqRel));
}

#[test]
fn concurrent_compare_exchanges_form_one_history() {
    check_history(|vec, value| {
        let mut current = vec.load(1, Relaxed);
        loop {
            match vec.compare_exchange(1, current, value, AcqRel, Relaxed) {
                Ok(old) => break old,
                Err(actual) => current = actual,
            }
        }
    });
}

/// Increments on a vector large enough for the production stripe constants to give many stripes
/// in many blocks, with every parking slot shared by many stripes. Each thread adds 1 to every
/// element and then hammers a few straddling elements whose first words are multiples of 64,
/// whose stripes share one slot, while a reader checks that their values never decrease.
#[test]
#[cfg_attr(miri, ignore = "millions of operations")]
fn increments_on_many_stripes_are_never_lost() {
    const LEN: usize = 1 << 20;
    const THREADS: usize = 4;
    const HOT_ROUNDS: usize = 1 << 20;
    let hot: Vec<usize> = (0..LEN)
        .filter(|&i| {
            let bit = 33 * i;
            (bit / 64).is_multiple_of(64) && bit % 64 + 33 > 64
        })
        .take(8)
        .collect();
    let vec = AtomicFixedVec::<u64>::new(33, LEN).unwrap();
    let done = std::sync::atomic::AtomicBool::new(false);
    let added: Vec<Vec<u64>> = thread::scope(|scope| {
        scope.spawn(|| {
            let mut last = vec![0; hot.len()];
            while !done.load(Relaxed) {
                for (last, &i) in last.iter_mut().zip(&hot) {
                    let x = vec.load(i, Acquire);
                    assert!(x >= *last, "element {i} went from {last} to {x}");
                    *last = x;
                }
            }
        });
        let adders: Vec<_> = (0..THREADS)
            .map(|thread| {
                let (vec, hot) = (&vec, &hot);
                scope.spawn(move || {
                    // Each thread walks the vector from its own quarter.
                    (0..LEN).for_each(|i| {
                        vec.fetch_add((i + thread * LEN / THREADS) % LEN, 1, AcqRel);
                    });
                    let mut added = vec![0; hot.len()];
                    let mut rng = Rng::new(thread as u64);
                    for _ in 0..HOT_ROUNDS {
                        let k = rng.below(hot.len());
                        vec.fetch_add(hot[k], 1, AcqRel);
                        added[k] += 1;
                    }
                    added
                })
            })
            .collect();
        let added = adders.into_iter().map(|a| a.join().unwrap()).collect();
        done.store(true, Relaxed);
        added
    });
    let mut expected = vec![THREADS as u64; LEN];
    for added in &added {
        for (&i, &n) in hot.iter().zip(added) {
            expected[i] += n;
        }
    }
    let wrong = (0..LEN).find(|&i| vec.load(i, Relaxed) != expected[i]);
    assert_eq!(wrong, None, "an increment was lost");
}
