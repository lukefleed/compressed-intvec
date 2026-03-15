# Compressed Integer Vectors

[![crates.io](https://img.shields.io/crates/v/compressed-intvec.svg)](https://crates.io/crates/compressed-intvec)
[![rust](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml/badge.svg)](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml)
[![docs](https://docs.rs/compressed-intvec/badge.svg)](https://docs.rs/compressed-intvec)
[![downloads](https://img.shields.io/crates/d/compressed-intvec)](https://crates.io/crates/compressed-intvec)
![license](https://img.shields.io/crates/l/compressed-intvec)
[![Line count](https://tokei.rs/b1/github/lukefleed/compressed-intvec?type=Rust)](https://github.com/lukefleed/compressed-intvec)

A Rust library providing space-efficient, in-memory integer vectors.

Standard `Vec<u64>` uses 64 bits per element regardless of the actual
values stored. When most values are small or follow a skewed distribution,
a large fraction of those bits is wasted. This crate exploits that
redundancy with two complementary strategies: *fixed-width encoding*
packs each element into the minimum number of bits needed to represent
the largest value (a vector of values in \[0, 1000) needs only 10 bits
per element instead of 64). *Variable-length encoding* goes further by
assigning shorter codes to more frequent values using instantaneous codes
from [`dsi-bitstream`], adapting to the actual data distribution.

- [`FixedVec`]: fixed-width bit-packed elements with O(1) mutable
  access. With small bit widths it can outperform `Vec<u64>` thanks to
  better cache utilization. Provides [`AtomicFixedVec`] for thread-safe
  concurrent access with atomic read-modify-write operations.

- [`VarVec`]: variable-length instantaneous codes (Gamma, Delta, Rice,
  Zeta, ...) with automatic codec selection. Maintains a sampling table
  of bit positions every *k*-th element. Random access jumps to the
  nearest sample and decodes forward, giving amortized O(1) access.

- [`SeqVec`]: compressed sequences of integers indexed by rank, decoded
  lazily through zero-allocation iterators. Designed for adjacency lists,
  document-term vectors, and similar collections of variable-length
  integer lists.

### When to Use Which

| | [`FixedVec`] | [`VarVec`] | [`SeqVec`] |
|---|---|---|---|
| Encoding | Fixed-width bits | Variable-length codes | Variable-length codes |
| Access | O(1) read/write | O(k) random read | O(m) per sequence |
| Mutation | Yes | No | No |
| Atomics | Yes ([`AtomicFixedVec`]) | No | No |
| Best for | Uniform-range data | Skewed distributions | Adjacency lists, sequences |

All structures support [Rayon] parallel iterators, [`mem_dbg`] memory
inspection, and optional [Serde] serialization.

## Quick Start

All common types and traits are available through the [`prelude`].

### [`FixedVec`]

```rust
use compressed_intvec::prelude::*;

let data: &[u32] = &[10, 20, 30, 40, 50];
let mut vec: UFixedVec<u32> = FixedVec::builder()
    .build(data)
    .unwrap();

assert_eq!(vec.get(2), Some(30));

// FixedVec supports in-place mutation.
vec.set(2, 35);
assert_eq!(vec.get(2), Some(35));
```

### [`VarVec`]

```rust
use compressed_intvec::prelude::*;

let skewed_data: &[u64] = &[5, 8, 13, 1000, 7, 6, 10_000, 10, 2, 3];

// Auto selects the best compression codec for the data.
let varvec: LEVarVec = VarVec::builder()
    .codec(Codec::Auto)
    .build(skewed_data)
    .unwrap();

assert_eq!(varvec.len(), skewed_data.len());
assert_eq!(varvec.get(3), Some(1000));
```

### [`SeqVec`]

```rust
use compressed_intvec::prelude::*;

let sequences: &[&[u32]] = &[
    &[1, 2, 3],
    &[10, 20],
    &[100, 200, 300, 400],
    &[],
];

let vec: LESeqVec<u32> = SeqVec::builder()
    .codec(Codec::Auto)
    .build(sequences)
    .unwrap();

assert_eq!(vec.num_sequences(), 4);

let seq1: Vec<u32> = vec.get(1).unwrap().collect();
assert_eq!(seq1, vec![10, 20]);

for (idx, seq_iter) in vec.iter().enumerate() {
    let seq: Vec<u32> = seq_iter.collect();
    println!("Sequence {}: {:?}", idx, seq);
}
```

## [`AtomicFixedVec`]

Thread-safe variant of [`FixedVec`] with atomic read-modify-write
operations (`load`, `store`, `fetch_add`, `compare_exchange`, ...).

*   **Lock-free path**: when an element fits in a single `u64` word
    (guaranteed for power-of-two bit widths), operations use lock-free
    atomic instructions.
*   **Locked path**: when an element spans two words, a striped lock
    pool ensures atomicity without a global lock.

For write-heavy workloads, prefer power-of-two bit widths to stay on the
lock-free path. See the [module documentation][atomic-docs] for examples.

[atomic-docs]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/atomic/index.html

## Compression Codecs

[`VarVec`] and [`SeqVec`] use instantaneous codes from [`dsi-bitstream`].
The [`Codec`] enum controls the strategy:

| Codec | Best for | Implied distribution |
| :--- | :--- | :--- |
| **`Auto`** | Unknown distribution | Analyzes data at build time |
| `Gamma` | Small non-negative integers | ≈ 1/2x² |
| `Delta` | Moderate integers | ≈ 1/2x(log x)² |
| `Zeta(k)` | Power-law (web graphs, etc.) | ≈ 1/x^(1+1/k) |
| `Rice` / `Golomb` | Geometric distributions | ≈ 1/rˣ |
| `VByteLe` / `VByteBe` | General-purpose, fast decode | ≈ 1/x^(8/7) |
| `Omega` | Very large numbers | ≈ 1/x |
| `Unary` | Near-zero values | Geometric with high P(0) |
| `Explicit(Codes)` | Direct [`dsi-bitstream`] codes | — |

`Codec::Auto` analyzes the full input and selects the best codec. This
introduces a one-time cost at construction, but the amortized cost is
negligible for read-heavy workloads. See the [`Codec`] documentation for
encoding details and parameter selection.

## [`VarVec`] Access Patterns

The access strategy has significant performance implications:

| Method | Pattern | Mechanism |
| :--- | :--- | :--- |
| [`get`] | Single lookup | Seek to nearest sample, decode forward |
| [`get_many`] | Batch from slice | Sorts indices for monotonic scan |
| [`get_many_from_iter`] | Streaming indices | Stateful sequential reader |
| [`reader()`][`VarVecReader`] | Repeated random | Amortizes reader setup cost |
| [`seq_reader()`][`VarVecSeqReader`] | Sequential / clustered | Cursor-based, skips seeks in same block |

Prefer [`get_many`] for batch lookups and [`seq_reader()`][`VarVecSeqReader`]
for sorted or clustered access patterns. See the method documentation for
examples.

## Memory Analysis

All structures implement [`MemDbg`] and [`MemSize`] for memory inspection:

```rust
use compressed_intvec::prelude::*;
use mem_dbg::{DbgFlags, MemDbg};
use rand::{rngs::SmallRng, RngExt, SeedableRng};

fn generate_random_vec(size: usize, max: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size).map(|_| rng.random_range(0..max)).collect()
}

fn main() {
    let data = generate_random_vec(1_000_000, 1 << 20);

    let auto_varvec = LEVarVec::builder()
        .codec(Codec::Auto)
        .build(&data)
        .unwrap();

    println!("VarVec (Auto):");
    auto_varvec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE | DbgFlags::RUST_LAYOUT);

    let fixed_vec = LEFixedVec::builder()
        .bit_width(BitWidth::Minimal)
        .build(&data)
        .unwrap();

    println!("\nFixedVec (minimal bit width):");
    fixed_vec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE | DbgFlags::RUST_LAYOUT);
}
```

For 1M elements uniformly distributed in \[0, 2²⁰):

| Representation | Size | Reduction |
| :--- | ---: | ---: |
| `Vec<u64>` | 8.00 MB | — |
| `VarVec` (Gamma) | 4.73 MB | 41% |
| `VarVec` (Auto → Zeta(10)) | 2.85 MB | 64% |
| `FixedVec` (20 bits) | 2.50 MB | 69% |

The `mem_dbg` output shows the internal layout of each structure,
including overhead for metadata and sampling:

```text
Size of the VarVec with Auto encoding:
2.846 MB 100.00% ⏺
2.749 MB  96.57% ├╴data
    8  B   0.00% ├╴k
    8  B   0.00% ├╴len
   16  B   0.00% ├╴encoding
97.72 kB   3.43% ├╴samples
97.70 kB   3.43% │ ├╴bits
    8  B   0.00% │ ├╴bit_width
    8  B   0.00% │ ├╴mask
    8  B   0.00% │ ├╴len
    0  B   0.00% │ ╰╴_phantom
    0  B   0.00% ╰╴_markers

Size of the FixedVec with minimal bit width:
2.500 MB 100.00% ⏺
2.500 MB 100.00% ├╴bits
    8  B   0.00% ├╴bit_width
    8  B   0.00% ├╴mask
    8  B   0.00% ├╴len
    0  B   0.00% ╰╴_phantom
```

## Features

- `parallel` (default): parallel iterators and batch access via [Rayon].
- `serde`: [Serde] support for all vector types.
- `arch-dependent-storable`: `Storable` implementations for `usize` and
  `isize`. **Warning**: breaks cross-architecture data portability; a
  `VarVec<usize>` serialized on a 64-bit system may panic on a 32-bit
  system.

## Benchmarks

[Criterion] and [iai-callgrind] benchmarks compare against
[`sux::BitFieldVec`] and [`succinct::IntVector`]:

```bash
cargo bench --bench bench_random_access       # fixed: random access
cargo bench --bench bench_var_random_access   # variable: random access
cargo bench --bench bench_seq_access_patterns # seq: access patterns
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE)

[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html
[`VarVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html
[`SeqVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/seq/struct.SeqVec.html
[`AtomicFixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/atomic/struct.AtomicFixedVec.html
[`prelude`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/prelude/index.html
[`Codec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/codec/enum.Codec.html
[`get`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html#method.get
[`get_many`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html#method.get_many
[`get_many_from_iter`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html#method.get_many_from_iter
[`VarVecReader`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/reader/struct.VarVecReader.html
[`VarVecSeqReader`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/seq_reader/struct.VarVecSeqReader.html
[`dsi-bitstream`]: https://crates.io/crates/dsi-bitstream
[`MemDbg`]: https://docs.rs/mem_dbg/latest/mem_dbg/trait.MemDbg.html
[`MemSize`]: https://docs.rs/mem_dbg/latest/mem_dbg/trait.MemSize.html
[`mem_dbg`]: https://crates.io/crates/mem_dbg
[Rayon]: https://docs.rs/rayon/latest/rayon/
[Serde]: https://serde.rs/
[Criterion]: https://docs.rs/criterion/latest/criterion/
[iai-callgrind]: https://docs.rs/iai-callgrind/latest/iai_callgrind/
[`sux::BitFieldVec`]: https://docs.rs/sux/latest/sux/bits/bit_field_vec/index.html
[`succinct::IntVector`]: https://docs.rs/succinct/latest/succinct/int_vec/trait.IntVec.html
