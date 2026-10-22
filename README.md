# compressed-intvec

[![crates.io](https://img.shields.io/crates/v/compressed-intvec.svg)](https://crates.io/crates/compressed-intvec)
[![docs](https://docs.rs/compressed-intvec/badge.svg)](https://docs.rs/compressed-intvec/latest/compressed_intvec/)
[![rustc](https://img.shields.io/badge/rustc-1.88+-red.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![CI](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml/badge.svg)](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml)
![license](https://img.shields.io/crates/l/compressed-intvec)

A Rust library that provides space-efficient, in-memory representations for
integer vectors. The library is designed to reduce the memory footprint of standard `Vec`
collections of integers while retaining indexed access and iteration. It offers three complementary data structures:

* [`FixedVec`] for fixed-width encoding with constant-time reads and updates by
  index. [`AtomicFixedVec`] provides the atomic counterpart for concurrent access.
* [`VarVec`] for variable-length encoding with automatic code selection and
  sampled indexed access.
* [`SeqVec`] for storing sequences of integers, such as adjacency lists, with
  indexed access and lazy decoding.

The elements can be any primitive integer up to 64 bits, or a type of your own
that implements [`Storable`] by converting its values to and from `u64` words,
so a `FixedVec<NodeId>` of an identifier newtype returns `NodeId` values.

`FixedVec`, `VarVec` and `SeqVec` also take a storage type, `Vec<u64>` by
default, which decides who owns the packed words:

* `Vec<u64>` owns them, and lets a `FixedVec` grow.
* `Box<[u64]>` owns them without spare capacity, and a `FixedVec` can still
  change its elements.
* `Arc<[u64]>` shares them, so cloning the vector copies no data.
* `&[u64]` borrows them from another owner.

## FixedVec

`FixedVec` stores every integer in the same number of bits, so it suits data
with a bounded range, such as node identifiers, compact labels and counters.
Reads and updates by index take constant time, through an API modelled on
`Vec`.

The bit width is fixed when the vector is built. `from_slice` and `collect`
choose the smallest width that holds the input, five bits for values from 0 to
31, and `FixedVec::new` takes the width explicitly. Every later value must fit
that width, so `set` and `push` panic on a value that does not fit, while
`try_set` and `try_push` return a `ValueTooLarge` error that holds the rejected
value.

`get_unchecked` and `set_unchecked` skip the index check, and they are the
crate's only `unsafe` functions.

```rust
use compressed_intvec::FixedVec;

let mut values = FixedVec::from_slice(&[3u64, 7, 31, 0]);
assert_eq!(values.bit_width(), 5);
assert_eq!(values.get(2), Some(31));

values.set(1, 12);
values.push(9);

let view = values.as_borrowed(); // The same vector over borrowed words.
assert_eq!(view.iter().collect::<Vec<_>>(), [3, 12, 31, 0, 9]);
```

## AtomicFixedVec

`AtomicFixedVec` is a `FixedVec` whose words are atomic, so threads can read
and update its elements through a shared reference. As in `FixedVec`, the bit
width is fixed at construction and writers panic on a value that does not fit
it, while `fetch_add` and `fetch_sub` wrap at it.

Each operation is atomic for the element it indexes, at every bit width, so a
reader observes either the old value or the new one. The vector needs two
mechanisms for that, because an element can lie inside one word or straddle
two:

* An element inside one word changes through a compare-and-swap on that word.
* An element that straddles two words is guarded by a lock shared by a group
  of words. Writers take the lock, and readers retry when a writer changed the
  element while they read it.

The operations are those of `AtomicU64`, with the same orderings: `load`,
`store`, `swap`, `compare_exchange`, `fetch_update` and the arithmetic and
bitwise `fetch_*` operations. `From` converts between `FixedVec` and
`AtomicFixedVec` by value, so the words are never visible to atomic and plain
accesses at the same time.

```rust
use compressed_intvec::AtomicFixedVec;
use std::{sync::atomic::Ordering, thread};

// 1000 counters of 21 bits, about a third of which straddle two words.
let counters = AtomicFixedVec::<u32>::new(21, 1000).expect("21 bits fit in a u32");
thread::scope(|scope| {
    for _ in 0..4 {
        scope.spawn(|| {
            for i in 0..1000 {
                counters.fetch_add(i, 1, Ordering::Relaxed);
            }
        });
    }
});
assert!((0..1000).all(|i| counters.load(i, Ordering::Relaxed) == 4));
```

## VarVec

`VarVec` stores each integer as a codeword whose length grows with its value,
so it suits data where small values are common and a few large ones would make
every element of a `FixedVec` wide. The codes come from [`dsi-bitstream`], and
the vector is immutable once built.

The codewords form one bitstream, and an index records the bit position of
every `sampling_rate`-th element, 32 by default. `get(i)` seeks to the sample
before `i` and decodes forward, so it decodes at most `sampling_rate`
codewords, and a larger rate shrinks the index at the cost of a slower `get`.
Iteration decodes each element once.

The available codes are unary, Elias gamma, delta and omega, variable-length
bytes in either byte order, Rice, Golomb, exponential Golomb, zeta and pi, and
the [`Codec`] documentation describes each of them. By default the builder
picks the code with the smallest estimated size. An explicit code fits data
whose distribution is known. Rice and Golomb codes suit geometrically
distributed values, such as the gaps between sorted random positions, and zeta
codes suit power-law distributions, such as the degrees of a web graph.

The builder records the choice in its type. With the automatic choice, `build`
returns the vector, because only codes that encode every value are candidates.
After `.codec(c)`, `build` returns a `Result`, whose `CodecError` reports a
parameter out of range or the first value that the code cannot encode.

Full-width values need no special handling, because the automatic choice skips
the codes that cannot encode them:

```rust
use compressed_intvec::VarVec;

let mut input = vec![0u64; 1024];
input[513] = u64::MAX;

let values = VarVec::from_slice(&input);
assert_eq!(values.get(512), Some(0));
assert_eq!(values.get(513), Some(u64::MAX));
assert_eq!(values.iter().sum::<u64>(), u64::MAX);
```

## SeqVec

`SeqVec` applies the codes of `VarVec` to a vector of integer sequences, such
as graph adjacency lists or posting lists, which would otherwise be a
`Vec<Vec<T>>`. The sequences keep their order and can have any length,
including zero, and the vector is immutable once built.

The values of all sequences share one bitstream, and an index stores `n + 1`
bit offsets for `n` sequences, the start of each sequence and the end of the
last one. `get(i)` reads two offsets, so it returns in constant time a
`SeqIter` that decodes the values of sequence `i` as it advances. The builder
chooses the code as it does for `VarVec`.

The builder can also store the length of every sequence, so `count` on a
`SeqIter` takes constant time and its size hint is exact. Without stored
lengths, `count` decodes the rest of the sequence.

For a graph, each sequence contains one node's neighbors:

```rust
use compressed_intvec::SeqVec;

let lists: [&[u32]; 3] = [&[1, 2], &[], &[0, 1]];
let graph = SeqVec::builder().store_lengths().build(&lists);

assert_eq!(graph.len(), 3);
assert_eq!(graph.get(0).unwrap().count(), 2);
assert_eq!(graph.get(1).unwrap().count(), 0);
assert_eq!(graph.get(2).unwrap().sum::<u32>(), 1);
```

## Building vectors from external data

A vector built from words outside the crate, such as words read from a file,
goes through `from_parts`, which checks the words before the vector exists.
Corrupt input therefore returns an error in every build profile, without a
panic or an endless loop, and `get` and the iterators decode without checking
again. The fields of every vector are private, so the builders and
`from_parts` are the only ways to obtain one.

The checks depend on the vector:

* `FixedVec::from_parts` checks that the bit width fits the element type and
  that the buffer holds enough words for the length, in constant time.
* `VarVec::from_parts` and `SeqVec::from_parts` decode the whole stream,
  rejecting a truncated stream or a codeword whose value does not fit in 64
  bits, and compare every stored sample or offset with the position the
  decoder reached. They also check that every value fits the element type, so
  they take time linear in the data.

With the `serde` feature, deserialization reads the format version before the
rest of the data, requires the same element signedness, and rejects unknown
fields and any set bit outside the elements and codewords. It then builds the
vector through the same checks, and `VarVec` rebuilds its samples while it
decodes the stream, so the serialized form does not carry them.

## Extra features

Enable Cargo features for the integrations you need:

| Feature | Provides |
| --- | --- |
| `parallel` | Indexed parallel iterators through [`rayon`]. |
| `serde` | Serialization of packed vectors and validated deserialization through [`serde`]. |
| `mem_dbg` | Memory size measurements and a tree of the vector's fields and buffers through [`mem_dbg`], outside the semver guarantee. |

### mem_dbg

With the `mem_dbg` feature, the vectors implement `MemSize` and `MemDbg` from
[`mem_dbg`]. `MemSize` measures the vector and its owned heap buffers. `MemDbg`
expands that footprint into a tree, showing each field's size, Rust type and
share of the total.

This example stores the same million integers in a `Vec<u64>` and a
`VarVec<u64>`, with one large value every 1,000 elements, and prints both
reports with allocated capacity:

```rust,ignore
use compressed_intvec::VarVec;
use mem_dbg::{DbgFlags, MemDbg};

let input: Vec<u64> = (0..1_000_000)
    .map(|i| if i % 1000 == 0 { 1 << 40 } else { 0 })
    .collect();
let values = VarVec::from_slice(&input);

let flags = DbgFlags::default() | DbgFlags::CAPACITY | DbgFlags::HUMANIZE;
input.mem_dbg(flags).unwrap();
values.mem_dbg(flags).unwrap();
```

On a 64-bit target, the two vectors report:

```text
8.000 MB 100.00% ⏺: alloc::vec::Vec<u64>
213.4 kB 100.00% ⏺: compressed_intvec::variable::VarVec<u64>
131.3 kB  61.52% ├╴data: alloc::vec::Vec<u64>
82.09 kB  38.47% ├╴samples: compressed_intvec::fixed::FixedVec<u64>
82.06 kB  38.46% │ ├╴words: alloc::vec::Vec<u64>
    8  B   0.00% │ ├╴len: usize
    8  B   0.00% │ ├╴bit_width: usize
    8  B   0.00% │ ├╴mask: u64
    0  B   0.00% │ ╰╴element: core::marker::PhantomData<fn() -> u64>
    8  B   0.00% ├╴sampling_rate: core::num::nonzero::NonZero<usize>
    8  B   0.00% ├╴len: usize
   16  B   0.01% ├╴codec: compressed_intvec::common::codec::ValidCodec
    0  B   0.00% ╰╴element: core::marker::PhantomData<fn() -> u64>
```

For this data, `VarVec<u64>` takes 97.3% less memory than `Vec<u64>`,
including its sampled index.

[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/struct.FixedVec.html
[`AtomicFixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/struct.AtomicFixedVec.html
[`VarVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/struct.VarVec.html
[`SeqVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/struct.SeqVec.html
[`Codec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/enum.Codec.html
[`Storable`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/trait.Storable.html
[`dsi-bitstream`]: https://docs.rs/dsi-bitstream/
[`rayon`]: https://docs.rs/rayon/
[`serde`]: https://docs.rs/serde/
[`mem_dbg`]: https://docs.rs/mem_dbg/
