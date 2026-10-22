//! The `Storable` implementations of the primitive integers: unsigned values are stored as they
//! are, signed ones with the zig-zag encoding `0, -1, 1, -2, ... -> 0, 1, 2, 3, ...`.

use crate::{Case, Elem, Rng, for_each_type, max_width, signed};
use compressed_intvec::{AtomicFixedVec, FixedVec, SeqVec, Storable, ValueTooLarge, VarVec};
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::Ordering::Relaxed;

#[test]
fn storage_conversions_do_not_convert_elements() {
    thread_local! {
        static CONVERSIONS: Cell<usize> = const { Cell::new(0) };
    }

    #[derive(Clone, Copy)]
    struct Counted(u64);

    impl Storable for Counted {
        const BITS: u32 = 64;

        fn to_word(self) -> u64 {
            CONVERSIONS.with(|count| count.set(count.get() + 1));
            self.0
        }

        fn from_word(word: u64) -> Self {
            CONVERSIONS.with(|count| count.set(count.get() + 1));
            Self(word)
        }
    }

    let values = [Counted(1), Counted(2)];
    let fixed = FixedVec::from_slice(&values);
    let variable = VarVec::from_slice(&values);
    let sequences = SeqVec::builder().store_lengths().build(&[&values[..]]);
    let variable_codec = variable.codec();
    let sequence_codec = sequences.codec();
    CONVERSIONS.with(|count| count.set(0));

    let borrowed_fixed = fixed.as_borrowed();
    let borrowed_variable = variable.as_borrowed();
    let borrowed_sequences = sequences.as_borrowed();
    assert_eq!((borrowed_fixed.len(), borrowed_fixed.bit_width()), (2, 2));
    assert_eq!(
        (borrowed_variable.len(), borrowed_variable.codec()),
        (2, variable_codec)
    );
    assert_eq!(
        (borrowed_sequences.len(), borrowed_sequences.codec()),
        (1, sequence_codec)
    );

    let boxed_fixed: FixedVec<Counted, Box<[u64]>> = fixed.into();
    let boxed_variable: VarVec<Counted, Box<[u64]>> = variable.into();
    let boxed_sequences: SeqVec<Counted, Box<[u64]>> = sequences.into();
    assert_eq!((boxed_fixed.len(), boxed_fixed.bit_width()), (2, 2));
    assert_eq!(
        (boxed_variable.len(), boxed_variable.codec()),
        (2, variable_codec)
    );
    assert_eq!(
        (boxed_sequences.len(), boxed_sequences.codec()),
        (1, sequence_codec)
    );
    CONVERSIONS.with(|count| assert_eq!(count.get(), 0));
}

#[track_caller]
fn check<T: Elem>(x: T) {
    let n = x.to_int();
    let expected = if n >= 0 && signed::<T>() {
        2 * n
    } else if n >= 0 {
        n
    } else {
        -2 * n - 1
    };
    let word = x.to_word();
    assert_eq!(i128::from(word), expected, "{x:?}");
    assert_eq!(T::from_word(word), x, "{x:?}");
}

fn constants<T: Elem>() {
    assert_eq!(T::BITS, max_width::<T>());
    assert_eq!(T::SIGNED, signed::<T>());
}

#[test]
fn constants_describe_the_type() {
    for_each_type!(constants);
}

#[test]
#[cfg_attr(miri, ignore = "safe code only")]
fn every_value_of_the_small_types() {
    (0..=u8::MAX).for_each(check);
    (i8::MIN..=i8::MAX).for_each(check);
    (0..=u16::MAX).for_each(check);
    (i16::MIN..=i16::MAX).for_each(check);
}

fn ends_and_random<T: Elem>() {
    let (min, max) = if signed::<T>() {
        (
            -(1 << (max_width::<T>() - 1)),
            (1 << (max_width::<T>() - 1)) - 1,
        )
    } else {
        (0, (1 << max_width::<T>()) - 1)
    };
    let mut rng = Rng::new(1);
    let random = (0..10_000).map(|_| min + i128::from(rng.next_u64()) % (max - min + 1));
    for n in [min, min + 1, -1, 0, 1, max - 1, max]
        .into_iter()
        .chain(random)
    {
        if let Some(x) = T::from_int(n) {
            check(x);
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "safe code only")]
fn ends_and_random_values_of_every_type() {
    for_each_type!(ends_and_random);
}

/// Runs `f` and returns the message of its panic, or `None` if it returns.
fn panic_message(f: impl FnOnce()) -> Option<String> {
    let payload = catch_unwind(AssertUnwindSafe(f)).err()?;
    let text = payload
        .downcast_ref::<&str>()
        .map(|message| message.to_string());
    Some(
        payload
            .downcast_ref::<String>()
            .cloned()
            .or(text)
            .unwrap_or_default(),
    )
}

/// Checks that `f` returns, or panics with the message that the documentation gives for a value
/// that does not fit a bit width.
#[track_caller]
fn returns_or_does_not_fit(label: &str, f: impl FnOnce()) {
    if let Some(message) = panic_message(f) {
        assert!(
            message.starts_with("value needs ") && message.ends_with(" bits per element"),
            "{label}: {message}"
        );
    }
}

/// A `Storable` whose `to_word` returns a 41-bit word, although `BITS` is 4.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Wide;

impl Storable for Wide {
    const BITS: u32 = 4;

    fn to_word(self) -> u64 {
        1 << 40
    }

    fn from_word(_: u64) -> Self {
        Self
    }
}

const WIDER_THAN_BITS: &str = "Storable::to_word returned a word wider than Storable::BITS";
const WIDE_DOES_NOT_FIT: &str = "value needs 41 bits, but the vector stores 4 bits per element";

/// Every check of a fixed-width vector rejects the word that `to_word` returns, as documented;
/// the vectors of codes store it, and nothing reads past their streams.
#[test]
fn a_to_word_wider_than_bits_is_rejected_or_stored() {
    assert_eq!(
        panic_message(|| {
            let _ = FixedVec::from_slice(&[Wide]);
        })
        .as_deref(),
        Some(WIDER_THAN_BITS)
    );
    assert_eq!(
        panic_message(|| {
            let _ = AtomicFixedVec::from_slice(&[Wide]);
        })
        .as_deref(),
        Some(WIDER_THAN_BITS)
    );

    let mut vec = FixedVec::<Wide>::from_parts(vec![0; 2], 2, 4).unwrap();
    let error = ValueTooLarge {
        value: Wide,
        required: 41,
        bit_width: 4,
    };
    assert_eq!(vec.try_push(Wide), Err(error));
    assert_eq!(vec.try_set(1, Wide), Err(error));
    let cases: [Case<FixedVec<Wide>>; 7] = [
        ("push", |v| v.push(Wide)),
        ("extend_from_slice", |v| v.extend_from_slice(&[Wide])),
        ("extend", |v| v.extend([Wide])),
        ("insert", |v| v.insert(0, Wide)),
        ("resize", |v| v.resize(3, Wide)),
        ("set", |v| v.set(0, Wide)),
        ("fill", |v| v.fill(Wide)),
    ];
    for (name, case) in cases {
        let message = panic_message(|| case(&mut vec));
        assert_eq!(message.as_deref(), Some(WIDE_DOES_NOT_FIT), "{name}");
        assert_eq!(vec.len(), 2, "{name}");
    }

    let atomic = AtomicFixedVec::<Wide>::new(4, 2).unwrap();
    assert_eq!(
        panic_message(|| atomic.store(0, Wide, Relaxed)).as_deref(),
        Some(WIDE_DOES_NOT_FIT)
    );
    assert_eq!(
        panic_message(|| {
            let _ = atomic.swap(1, Wide, Relaxed);
        })
        .as_deref(),
        Some(WIDE_DOES_NOT_FIT)
    );
    assert_eq!(atomic.len(), 2);

    let variable = VarVec::from_slice(&[Wide, Wide]);
    assert_eq!(variable.iter().count(), 2);
    assert!(variable.get(1).is_some());
    let sequences = SeqVec::from_slices(&[[Wide, Wide]]);
    assert_eq!(sequences.get(0).map(Iterator::count), Some(2));
}

thread_local! {
    static FLAKY_CALLS: Cell<u64> = const { Cell::new(0) };
}

/// A `Storable` whose `to_word` returns 0 and `u64::MAX` on alternate calls, whatever the value.
#[derive(Clone, Copy, Debug)]
struct Flaky(u64);

impl Storable for Flaky {
    const BITS: u32 = 64;

    fn to_word(self) -> u64 {
        let calls = FLAKY_CALLS.with(|calls| calls.replace(calls.get() + 1));
        if calls.is_multiple_of(2) { 0 } else { u64::MAX }
    }

    fn from_word(word: u64) -> Self {
        Self(word)
    }
}

/// A `to_word` that is not deterministic breaks the contract, so the vectors may return wrong
/// values or panic. Each mutator of `FixedVec` checks the word that it stores, so it either
/// stores it or panics as documented, and every vector stays usable.
#[test]
fn a_to_word_that_is_not_deterministic_leaves_the_vectors_usable() {
    // `extend_from_slice` retains the original elements when it panics; with a single value,
    // the word that it checks is the word that it stores.
    for start in 0..2 {
        let mut vec = FixedVec::<Flaky>::from_parts(vec![0; 2], 3, 1).unwrap();
        FLAKY_CALLS.with(|calls| calls.set(start));
        returns_or_does_not_fit("extend_from_slice", || vec.extend_from_slice(&[Flaky(0)]));
        assert!(vec.iter().take(3).all(|element| element.0 == 0));
        returns_or_does_not_fit("extend_from_slice", || {
            vec.extend_from_slice(&[Flaky(0); 3]);
        });
        assert!(vec.iter().take(3).all(|element| element.0 == 0));
    }

    let cases: [Case<FixedVec<Flaky>>; 11] = [
        ("push", |v| v.push(Flaky(0))),
        ("try_push", |v| {
            let _ = v.try_push(Flaky(0));
        }),
        ("extend", |v| v.extend([Flaky(0); 3])),
        ("insert", |v| v.insert(1, Flaky(0))),
        ("resize", |v| v.resize(5, Flaky(0))),
        ("set", |v| v.set(0, Flaky(0))),
        ("try_set", |v| {
            let _ = v.try_set(0, Flaky(0));
        }),
        ("replace", |v| {
            let _ = v.replace(0, Flaky(0));
        }),
        ("fill", |v| v.fill(Flaky(0))),
        ("fill_with", |v| v.fill_with(|| Flaky(0))),
        ("map_in_place", |v| v.map_in_place(|x| x)),
    ];
    for start in 0..2 {
        let mut vec = FixedVec::<Flaky>::from_parts(vec![0; 2], 3, 1).unwrap();
        FLAKY_CALLS.with(|calls| calls.set(start));
        for (name, case) in cases {
            returns_or_does_not_fit(name, || case(&mut vec));
            assert_eq!(vec.iter().count(), vec.len(), "{name}");
        }
    }

    let values = [Flaky(0); 3];
    for start in 0..2 {
        FLAKY_CALLS.with(|calls| calls.set(start));
        let fixed = FixedVec::from_slice(&values);
        assert_eq!(fixed.iter().count(), 3);
        let atomic = AtomicFixedVec::from_slice(&values);
        returns_or_does_not_fit("store", || atomic.store(0, Flaky(0), Relaxed));
        assert_eq!(atomic.len(), 3);
        // The vectors of codes choose a codec in one pass and encode in another, so they may
        // panic; any vector that they return decodes.
        if let Ok(variable) = catch_unwind(|| VarVec::from_slice(&values)) {
            assert_eq!(variable.iter().count(), 3);
        }
        if let Ok(sequences) = catch_unwind(|| SeqVec::from_slices(&[values])) {
            assert_eq!(sequences.get(0).map(Iterator::count), Some(3));
        }
    }
}

/// A `Storable` whose `from_word` panics.
#[derive(Clone, Copy, Debug)]
struct Unreadable(u8);

impl Storable for Unreadable {
    const BITS: u32 = 8;

    fn to_word(self) -> u64 {
        self.0.into()
    }

    fn from_word(_: u64) -> Self {
        panic!("from_word panics")
    }
}

/// A panic in `from_word` reaches the caller, and leaves the vector usable.
#[test]
fn a_panicking_from_word_reaches_the_caller() {
    #[track_caller]
    fn from_word_panics(label: &str, f: impl FnOnce()) {
        assert_eq!(
            panic_message(f).as_deref(),
            Some("from_word panics"),
            "{label}"
        );
    }

    let values = [Unreadable(1), Unreadable(200), Unreadable(3)];
    let mut fixed = FixedVec::from_slice(&values);
    let cases: [Case<FixedVec<Unreadable>>; 8] = [
        ("get", |v| {
            let _ = v.get(0);
        }),
        ("iter", |v| {
            let _ = v.iter().next();
        }),
        ("into_iter", |v| {
            let _ = v.clone().into_iter().next_back();
        }),
        ("replace", |v| {
            let _ = v.replace(1, Unreadable(4));
        }),
        ("map_in_place", |v| v.map_in_place(|x| x)),
        ("pop", |v| {
            let _ = v.pop();
        }),
        ("remove", |v| {
            let _ = v.remove(0);
        }),
        ("swap_remove", |v| {
            let _ = v.swap_remove(0);
        }),
    ];
    for (name, case) in cases {
        from_word_panics(name, || case(&mut fixed));
        assert_eq!(fixed.len(), 3, "{name}");
        fixed.set(2, Unreadable(5));
    }

    let atomic = AtomicFixedVec::from_slice(&values);
    from_word_panics("load", || {
        let _ = atomic.load(0, Relaxed);
    });
    from_word_panics("swap", || {
        let _ = atomic.swap(0, Unreadable(6), Relaxed);
    });
    atomic.store(1, Unreadable(7), Relaxed);
    assert_eq!(atomic.len(), 3);

    let variable = VarVec::from_slice(&values);
    from_word_panics("VarVec::get", || {
        let _ = variable.get(2);
    });
    from_word_panics("VarVec::iter", || {
        let _ = variable.iter().next();
    });
    assert_eq!(variable.len(), 3);

    let sequences = SeqVec::from_slices(&[values]);
    from_word_panics("SeqVec::get", || {
        {
            let _ = sequences.get(0).and_then(|mut sequence| sequence.next());
        };
    });
    assert_eq!(sequences.len(), 1);
}
