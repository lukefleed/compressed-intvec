//! Model checks of the synchronization of `AtomicFixedVec` with loom, which runs every
//! interleaving of a few threads under the memory model of Rust. Under `cfg(loom)` the crate
//! takes its atomics and locks from loom (`src/sync.rs`):
//! `RUSTFLAGS="--cfg loom" cargo test --release --test loom`.
//!
//! The vectors hold 4 elements of 33 bits: elements 0 and 2 lie inside words 0 and 1, and
//! elements 1 and 3 straddle two words. Under loom each data word has its own stripe and one
//! parking slot serves every stripe, so element 1 lies on stripe 0, element 3 on stripe 1, and
//! the two stripes share the slot.
//!
//! Each model yields in the main thread before its own operations: without the yield, loom 0.7.2
//! did not reach the schedules in which the spawned threads run first, and the models passed
//! with the validation of the sequence lock removed. The models fail when the reader skips that
//! validation or the writer ends with a `Relaxed` store; loom does not reach the reorderings that
//! removing either fence of the sequence lock allows.

#![cfg(loom)]

use compressed_intvec::AtomicFixedVec;
use loom::cell::UnsafeCell;
use loom::sync::Arc;
use loom::thread;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};

/// Runs `f` under every interleaving with at most three preemptions, which bounds the time.
fn model(f: impl Fn() + Sync + Send + 'static) {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.check(f);
}

fn vector() -> Arc<AtomicFixedVec<u64>> {
    Arc::new(AtomicFixedVec::new(33, 4).unwrap())
}

/// A reader sees the old or the new value of a straddling element, never a mix of their bits,
/// while the elements next to it, in the same words, change too. `Relaxed`, so that only the
/// sequence lock orders the accesses.
#[test]
fn straddling_reads_are_never_torn() {
    model(|| {
        let v = vector();
        let (old, new) = (0x1_5555_5555, 0x0_AAAA_AAAA);
        v.store(1, old, Relaxed);
        let writer = {
            let v = v.clone();
            thread::spawn(move || v.store(1, new, Relaxed))
        };
        let neighbors = {
            let v = v.clone();
            thread::spawn(move || {
                v.fetch_add(0, 1, Relaxed);
                v.fetch_add(2, 1, Relaxed);
            })
        };
        thread::yield_now();
        let x = v.load(1, Relaxed);
        assert!(x == old || x == new, "torn read {x:#x}");
        writer.join().unwrap();
        neighbors.join().unwrap();
        let after = (v.load(0, Relaxed), v.load(1, Relaxed), v.load(2, Relaxed));
        assert_eq!(after, (1, new, 1));
    });
}

/// Two read-modify-writes of a straddling element both take effect.
#[test]
fn straddling_updates_are_never_lost() {
    model(|| {
        let v = vector();
        let adder = {
            let v = v.clone();
            thread::spawn(move || {
                let _ = v.fetch_add(1, 1, AcqRel);
            })
        };
        thread::yield_now();
        v.fetch_update(1, AcqRel, Acquire, |x| Some(x + 3)).unwrap();
        adder.join().unwrap();
        assert_eq!(v.load(1, Relaxed), 4);
    });
}

/// A value that one thread writes and another reads; loom reports every access to it that the
/// synchronization under test leaves unordered.
struct Message(UnsafeCell<u64>);

// SAFETY: loom checks every access to the cell, and fails the model on an unordered one.
unsafe impl Sync for Message {}

/// A release store publishes the writes before it to an acquire load that sees it, for an
/// element inside a word and for a straddling one.
#[test]
fn release_stores_publish_to_acquire_loads() {
    for index in [0, 1] {
        model(move || {
            let v = vector();
            let message = Arc::new(Message(UnsafeCell::new(0)));
            let writer = {
                let (v, message) = (v.clone(), message.clone());
                thread::spawn(move || {
                    // SAFETY: the only write, before the release store.
                    message.0.with_mut(|m| unsafe { *m = 42 });
                    v.store(index, 1, Release);
                })
            };
            thread::yield_now();
            if v.load(index, Acquire) == 1 {
                // SAFETY: the acquire load saw the release store, which follows the write.
                assert_eq!(message.0.with(|m| unsafe { *m }), 42);
            }
            writer.join().unwrap();
        });
    }
}

/// Two swaps of a straddling element form one history: the values that they return, with the
/// final value, are the initial value and the two written ones.
#[test]
fn straddling_swaps_form_one_history() {
    model(|| {
        let v = vector();
        let (initial, a, b) = (0x1_8000_0001, 0x1_5555_5555, 0x0_AAAA_AAAA);
        v.store(1, initial, Relaxed);
        let swapper = {
            let v = v.clone();
            thread::spawn(move || v.swap(1, a, AcqRel))
        };
        thread::yield_now();
        let mut seen = [
            v.swap(1, b, AcqRel),
            swapper.join().unwrap(),
            v.load(1, Relaxed),
        ];
        seen.sort_unstable();
        let mut expected = [initial, a, b];
        expected.sort_unstable();
        assert_eq!(seen, expected);
    });
}

/// Updates on two stripes that share a parking slot: a release of one stripe wakes the waiters
/// of both, and a waiter whose stripe is still held sleeps again. Each thread adds 1 to a
/// straddling element, two on each stripe, and no update is lost.
#[test]
fn stripes_that_share_a_parking_slot() {
    model(|| {
        let v = vector();
        let adders: Vec<_> = [1, 3, 3]
            .into_iter()
            .map(|index| {
                let v = v.clone();
                thread::spawn(move || {
                    v.fetch_add(index, 1, Relaxed);
                })
            })
            .collect();
        thread::yield_now();
        v.fetch_add(1, 1, Relaxed);
        adders.into_iter().for_each(|adder| adder.join().unwrap());
        let after: Vec<u64> = (0..4).map(|i| v.load(i, Relaxed)).collect();
        assert_eq!(after, [0, 2, 0, 2]);
    });
}
