# Compressed Integer Vectors

[![crates.io](https://img.shields.io/crates/v/compressed-intvec.svg)](https://crates.io/crates/compressed-intvec)
[![rust](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml/badge.svg)](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml)
[![docs](https://docs.rs/compressed-intvec/badge.svg)](https://docs.rs/compressed-intvec)
[![downloads](https://img.shields.io/crates/d/compressed-intvec)](https://crates.io/crates/compressed-intvec)
![license](https://img.shields.io/crates/l/compressed-intvec)
[![Line count](https://tokei.rs/b1/github/lukefleed/compressed-intvec?type=Rust,Python)](https://github.com/lukefleed/compressed-intvec)

A Rust library that provides space-efficient, in-memory representations for integer vectors. It offers three complementary data structures: [`FixedVec`] for fixed-width encoding with _blazing fast_ mutable and atomic access, [`VarVec`] for variable-length encoding with high compression and amortized O(1) random access, and [`SeqVec`] for storing sequences of integers with indexed access.

The library is designed to reduce the memory footprint of standard [`std::vec::Vec`] collections of integers while retaining performant access patterns.

## Core Structures: [`FixedVec`], [`VarVec`], and [`SeqVec`]

The library provides three distinct vector types, each based on a different encoding principle and access pattern. Choosing the right one depends on the specific use case, performance requirements, and data characteristics.

### [`FixedVec`]: Fixed-Width Encoding>

Implements a vector where every integer occupies the same, predetermined number of bits.

*   **O(1) Random Access**: The memory location of any element is determined by a direct bit-offset calculation, resulting in minimal-overhead access. With low bit widths (e.g., 8, 16, 32), [`FixedVec`] can be faster than [`std::vec::Vec`] for random access due to better cache utilization.
*   **Mutability**: Supports in-place modifications after creation through an API similar to [`std::vec::Vec`] (e.g., `push`, `set`, `pop`).
*   **Atomic Operations**: Provides [`AtomicFixedVec`], a thread-safe variant that supports atomic read-modify-write operations for concurrent environments.

### [`VarVec`]: Variable-Width Element Encoding

Implements a vector using variable-length instantaneous codes (e.g., Gamma, Delta, Rice, Zeta) to represent each integer with amortized O(1) random access.

*   **High Compression Ratios**: Achieves significant space savings for data with non-uniform distributions.
*   **Automatic Codec Selection**: Can analyze the data to select the most effective compression codec automatically.
*   **Amortized O(1) Random Access**: Enables fast random access by sampling the bit positions of elements at a configurable interval (`k`).

### [`SeqVec`]: Variable-Length Sequence Encoding

Implements a vector of sequences where each sequence is stored in compressed form and accessed as a whole, with efficient indexed access to sequences.

*   **Sequence-Oriented Access**: Optimized for workloads where entire sequences are retrieved together.
*   **Minimal Overhead**: Stores only the bit offset of sequence boundaries; sequence lengths are computed on-the-fly or stored optionally.
*   **Flexible Codec**: Supports the same compression codecs as [`VarVec`] for variable-length encoding of sequence elements.
*   **Zero-Copy Iteration**: Provides zero-allocation iterators over sequence elements.

[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html
[`VarVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html
[`SeqVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/seq/struct.SeqVec.html
[`AtomicFixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/atomic/struct.AtomicFixedVec.html
[`std::vec::Vec`]: https://doc.rust-lang.org/std/vec/struct.Vec.html

## Quick Start

The following examples show some use cases for [`FixedVec`], [`VarVec`], and [`SeqVec`]. All common types and traits are available through the [`prelude`].

### Example: [`FixedVec`] for Uniform Data and Mutable Access


```rust
use compressed_intvec::prelude::*;

// Data where values are within a relatively small, uniform range.
let data: &[u32] = &[10, 20, 30, 40, 50];

// The builder automatically infers the minimal bit width required.
let mut vec: UFixedVec<u32> = FixedVec::builder()
    .build(data)
    .unwrap();

assert_eq!(vec.get(2), Some(30));

// `FixedVec` supports in-place mutation.
vec.set(2, 35);
assert_eq!(vec.get(2), Some(35));
```

### Example: [`VarVec`] for High-Ratio Compression

```rust
use compressed_intvec::prelude::*;

// Skewed data with a mix of small and large values.
let skewed_data: &[u64] = &[5, 8, 13, 1000, 7, 6, 10_000, 10, 2, 3];

// The builder can automatically select the best compression codec.
let varvec: LEVarVec = VarVec::builder()
    .codec(Codec::Auto)
    .build(skewed_data)
    .unwrap();

assert_eq!(varvec.len(), skewed_data.len());
assert_eq!(varvec.get(3), Some(1000));
```

### Example: [`SeqVec`] for Sequence Collections

```rust
use compressed_intvec::prelude::*;

let sequences: &[&[u32]] = &[
    &[1, 2, 3],
    &[10, 20],
    &[100, 200, 300, 400],
    &[], // Empty sequences are supported
];

// Use the builder to choose a codec. Here we explicitly set
// `store_lengths(false)` to show the default behaviour where lengths are
// not stored, so length queries require decoding.
// Use `store_lengths(true)` when you frequently need O(1) length queries; it stores per-sequence lengths at a small additional memory cost.
let vec: LESeqVec<u32> = SeqVec::builder()
    .codec(Codec::Auto)
    .store_lengths(false)
    .build(sequences)
    .unwrap();

assert_eq!(vec.num_sequences(), 4);
assert!(!vec.has_stored_lengths());
assert_eq!(vec.sequence_len(0), None);

// Access a sequence by index
let seq1: Vec<u32> = vec.get(1).unwrap().collect();
assert_eq!(seq1, vec![10, 20]);

// Iterate over all sequences
for (idx, seq_iter) in vec.iter().enumerate() {
    let seq: Vec<u32> = seq_iter.collect();
    println!("Sequence {}: {:?}", idx, seq);
}
```

[`prelude`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/prelude/index.html
[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html
[`VarVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html
[`SeqVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/seq/struct.SeqVec.html
[`FixedVec::builder`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html#method.builder
[`VarVec::builder`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html#method.builder
[`UFixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/type.UFixedVec.html
[`LEVarVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/type.LEVarVec.html
[`Codec::Auto`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/codec/enum.Codec.html#variant.Auto

## Atomic Operations with [`AtomicFixedVec`]

For concurrent applications, the library provides [`AtomicFixedVec`], a thread-safe variant of [`FixedVec`]. It supports atomic read-modify-write (RMW) operations, enabling safe and efficient manipulation of shared integer data across multiple threads. The atomicity guarantees depend on the element's bit width and its alignment within the underlying `u64` storage words.

*   **Lock-Free Path**: When an element is fully contained within a single `u64` word (guaranteed for power-of-two bit widths), operations are performed using lock-free atomic instructions. Performance here is optimal as no locks are involved.
*   **Locked Path**: When an element's bits span across the boundary of two `u64` words (common for non-power-of-two bit widths), operations are protected by a fine-grained mutex from a striped lock pool. This ensures atomicity for the two-word update without resorting to a global lock.

Ideal for shared counters, parallel data processing, and any scenario requiring multiple threads to read from and write to the same integer vector concurrently. For write-heavy workloads, configuring the bit width to a power of two (e.g., 8, 16, 32) is recommended to ensure all operations remain on the lock-free path.

### Example: Concurrent Atomic Operations

```rust
use compressed_intvec::prelude::*;
use std::sync::Arc;
use std::thread;
use std::sync::atomic::Ordering;

// A vector with a single counter, initialized to 0.
// A bit width of 17 is sufficient to hold the final count (80000).
let vec = Arc::new(
    UAtomicFixedVec::<u32>::builder()
        .bit_width(BitWidth::Explicit(17))
        .build(&[0])
        .unwrap(),
);

const NUM_THREADS: u32 = 8;
const INCREMENTS_PER_THREAD: u32 = 10_000;

let mut handles = vec![];
for _ in 0..NUM_THREADS {
    let vec_clone = Arc::clone(&vec);
    handles.push(thread::spawn(move || {
        for _ in 0..INCREMENTS_PER_THREAD {
            // Atomically increment the counter.
            vec_clone.fetch_add(0, 1, Ordering::SeqCst);
        }
    }));
}

for handle in handles {
    handle.join().unwrap();
}

let final_value = vec.get(0).unwrap();
assert_eq!(final_value, NUM_THREADS * INCREMENTS_PER_THREAD);
```

## Compression with [`VarVec`]

[`VarVec`] is the optimal choice when data is not uniformly distributed and minimizing memory usage is a priority. It uses variable-length codes to represent integers.

The compression strategy is controlled by the [`Codec`] enum, passed to the builder.

### Choosing the Right Codec

For _not too large_ use cases, the recommended strategy is [`Codec::Auto`], which analyzes the data to select the most space-efficient codec. Note that this has a one-time cost during construction that is not negligible for very large datasets or frequent builds. You can also specify a codec explicitly based on your data characteristics.

| `Codec` Variant | Description & Encoding Strategy | Optimal Data Distribution |
| :--- | :--- | :--- |
| **`Auto`** | Analyzes the data to choose the best variable-length code, balancing build time and compression ratio. | Agnostic; adapts to the input data. |
| `Gamma` (γ) | A universal, parameter-free code. Encodes `n` using the unary code of log₂(*n*+1), followed by the remaining bits of `n`+1. | Implied distribution is ≈ 1/(2*x*²). Optimal for data skewed towards small non-negative integers. |
| `Delta` (δ) | A universal, parameter-free code. Encodes `n` using the γ code of  log₂(*n*+1) , making it more efficient than γ for larger values. | Implied distribution is ≈ 1/(2*x*(log *x*)²). |
| `Rice` | A fast, tunable version of Golomb codes where the parameter *b* must be a power of two. Encodes `n` by splitting it into a quotient (stored in unary) and a remainder (stored in binary). | Geometric distributions. |
| `Golomb` | A tunable code, more general than Rice. Encodes `n` by splitting it into a quotient (stored in unary) and a remainder (stored using a minimal binary code). | Geometric distributions. Implied distribution is ≈ 1/*r*ˣ. |
| `Zeta` (ζ) | A tunable code for power-law data. Encodes `n` based on log₂(*n*+1)/*k* in unary, followed by a minimal binary code for the remainder. | Power-law distributions (e.g., word frequencies, node degrees). Implied distribution is ≈ 1/*x*<sup>1+1/*k*</sup>. |
| `VByteLe`/`Be`| A byte-aligned code that uses a continuation bit to store integers in a variable number of bytes. Fast to decode. The big-endian variant is lexicographical. | Implied distribution is ≈ 1/*x*<sup>8/7</sup>. Good for general-purpose integer data. |
| `Omega` (ω) | A universal, parameter-free code that recursively encodes the length of the binary representation of `n`. | Implied distribution is approximately 1/*x*. Compact for very large numbers. |
| `Unary` | The simplest code. Encodes `n` as `n` zero-bits followed by a one-bit. | Geometric distributions with a very high probability of small values (e.g., boolean flags). |
| `Explicit` | An escape hatch to use any code from the [`dsi-bitstream::codes::Codes`][dsi-bitstream-codes] enum. | Advanced use cases requiring specific, unlisted codes. |

[dsi-bitstream-codes]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/enum.Codes.html

### Automatic Selection with [`Codec::Auto`]

The `Auto` strategy removes the guesswork from codec selection. During the build phase, it analyzes the input data and selects the codec that offers the best compression ratio. This introduces a one-time cost for the analysis at construction time. Use the `Auto` codec when you want to create a [`VarVec`] once and read it many times, as the amortized cost of the analysis is negligible compared to the space savings and performance of subsequent reads.

If you need to create multiple [`VarVec`] instances at run-time, consider using a specific codec that matches your data distribution to avoid the overhead of analysis.

[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html
[`IntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html
[`Codec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/codec/enum.Codec.html
[`Codec::Auto`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/codec/enum.Codec.html#variant.Auto
[dsi-bitstream-codes]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/enum.Codes.html
[`mem-dbg`]: https://docs.rs/mem-dbg/latest/mem_dbg/

## [`VarVec`] Access Patterns

The access strategy for a compressed [`VarVec`] has significant performance implications. The library provides several methods, each optimized for a specific access pattern. Using the appropriate method is key to achieving high throughput.

### [`get_many`]: Batch Access from a Slice

For retrieving a batch of elements from a slice of indices.

*   **Mechanism**: This method sorts the provided indices to transform a random access pattern into a single, monotonic scan over the compressed data. This approach minimizes expensive bitstream seek operations and leverages data locality.

This should be your preferred method for any batch lookup when all indices are known and can be stored in a slice.

```rust
use compressed_intvec::prelude::*;
use rand::RngExt;

let data: Vec<u64> = (0..10_000).collect();
let varvec: LEVarVec = VarVec::builder()
    .codec(Codec::Delta)
    .k(32)
    .build(&data)
    .unwrap();

// Indices can be in any order.
let indices_to_get: Vec<usize> = (0..100).map(|_| rand::rng().random_range(0..10_000)).collect();

// `get_many` retrieves all values in one optimized pass.
let values = varvec.get_many(&indices_to_get).unwrap();

assert_eq!(values, indices_to_get.iter().map(|&i| data[i]).collect::<Vec<_>>());
```

### [`get_many_from_iter`]: Access from an Iterator

For retrieving elements from a streaming iterator of indices.

*   **Mechanism**: Processes indices on-the-fly using a stateful [`variable::IntVecSeqReader`] internally, which is optimized for streams with sequential locality.

Use when indices cannot be collected into a slice, for example due to memory constraints.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let varvec: LEVarVec = VarVec::from_slice(&data).unwrap();

// Process indices from a streaming source, such as a range iterator.
let values: Vec<u64> = varvec.get_many_from_iter(500..505).unwrap();

assert_eq!(values, vec![500, 501, 502, 503, 504]);
```

### Dynamic Lookups: `VarVecReader` and `VarVecSeqReader`

There are interactive scenarios where lookup indices are not known in advance. The library provides two reader types to handle such cases:

#### [`VarVecReader`]: Stateless Random Access

A **stateless** reader for efficient, repeated random lookups.

*   **Mechanism**: Amortizes the setup cost of the bitstream reader across multiple calls. Each `get` operation performs an independent seek from the nearest sample point.

Optimal for sparse, unpredictable access patterns where there is no locality between consecutive lookups.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let varvec: LEVarVec = VarVec::from_slice(&data).unwrap();

// Create a stateless reader for random access.
let mut reader = varvec.reader();
assert_eq!(reader.get(500).unwrap(), Some(500));
assert_eq!(reader.get(10).unwrap(), Some(10));
```

#### [`VarVecSeqReader`]: Stateful Sequential Access

A **stateful** reader optimized for access patterns with sequential locality.

*   **Mechanism**: Maintains an internal cursor. If a requested index is forward and within the same sample block, it decodes from the last position, avoiding a full seek.

Optimal for iterating through sorted or clustered indices where consecutive lookups are near each other.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let varvec: LEVarVec = VarVec::from_slice(&data).unwrap();

// Create a stateful reader for sequential access.
let mut seq_reader = varvec.seq_reader();
assert_eq!(seq_reader.get(500).unwrap(), Some(500));
// This second call is faster as it decodes forward from index 500.
assert_eq!(seq_reader.get(505).unwrap(), Some(505));
```

[`get`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html#method.get
[`get_many`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html#method.get_many
[`get_many_from_iter`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html#method.get_many_from_iter
[`VarVecReader`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/reader/struct.VarVecReader.html
[`VarVecSeqReader`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/seq_reader/struct.VarVecSeqReader.html

## Storing Sequences with [`SeqVec`]

For workloads where data is naturally organized as multiple variable-length sequences and access patterns retrieve entire sequences, [`SeqVec`] provides an optimized solution. Each sequence is stored in compressed form using the same instantaneous codes as [`VarVec`], and all sequences are concatenated into a single compressed bitstream.

Common applications include compressed graph representations (adjacency lists), document-term associations, and any scenario with variable-length collections.

### Basic Usage

```rust
use compressed_intvec::seq::{SeqVec, LESeqVec};

let sequences: &[&[u32]] = &[
    &[1, 2, 3],
    &[10, 20],
    &[100, 200, 300, 400],
];

let vec: LESeqVec<u32> = SeqVec::from_slices(sequences).unwrap();

assert_eq!(vec.num_sequences(), 3);

// Retrieve an entire sequence by index.
let second_seq: Vec<u32> = vec.get(1).unwrap().collect();
assert_eq!(second_seq, vec![10, 20]);

// Iterate over all sequences.
for (idx, seq_iter) in vec.iter().enumerate() {
    let seq: Vec<u32> = seq_iter.collect();
    println!("Sequence {}: {:?}", idx, seq);
}
```

### Codec Customization

Like [`VarVec`], [`SeqVec`] supports the same compression codecs and can automatically select the best one:

```rust
use compressed_intvec::seq::{SeqVec, LESeqVec, Codec};

let sequences: Vec<Vec<u64>> = vec![
    vec![1, 1, 1, 2, 3],
    vec![100, 200, 300],
];

let vec: LESeqVec<u64> = SeqVec::builder()
    .codec(Codec::Zeta { k: Some(3) })
    .build(&sequences)
    .unwrap();

assert_eq!(vec.num_sequences(), 2);
```

### Storing Sequence Lengths

By default, sequence lengths are computed on-the-fly by decoding elements until the next sequence boundary is reached. For scenarios where O(1) length queries are beneficial, lengths can be stored explicitly:

```rust
use compressed_intvec::seq::{SeqVec, LESeqVec};

let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[]];

let vec: LESeqVec<u32> = SeqVec::builder()
    .store_lengths(true)
    .build(sequences)
    .unwrap();

// O(1) length query instead of O(length).
let len = vec.sequence_len(0).unwrap();
assert_eq!(len, 3);
```

## Memory Analysis

The library integrates [`mem-dbg`] to provide memory usage statistics, allowing you to compare the size of different encoding strategies easily. This is particularly useful for understanding the trade-offs between [`FixedVec`] and [`IntVec`] in terms of memory efficiency.

```rust
use compressed_intvec::prelude::*;
use mem_dbg::{DbgFlags, MemDbg};
use rand::{rngs::SmallRng, RngExt, SeedableRng};

// Generates a vector with uniformly random values.
fn generate_random_vec(size: usize, max: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size).map(|_| rng.random_range(0..max)).collect()
}

fn main() {
    let data = generate_random_vec(1_000_000, 1 << 20);

    println!("Size of the uncompressed Vec<u64>:");
    data.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE | DbgFlags::RUST_LAYOUT);

    // Create a VarVec with Gamma encoding.
    let gamma_varvec = LEVarVec::builder()
        .codec(Codec::Gamma)
        .build(&data)
        .unwrap();

    println!("\nSize of the VarVec with gamma encoding:");
    gamma_varvec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE | DbgFlags::RUST_LAYOUT);

    // Let the library analyze the data and choose the best codec.
    let auto_varvec = LEVarVec::builder()
        .codec(Codec::Auto)
        .build(&data)
        .unwrap();

    println!("\nSize of the VarVec with Auto encoding:");
    auto_varvec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE | DbgFlags::RUST_LAYOUT);
    println!("\nCodec selected by Auto: {:?}", auto_varvec.encoding());

    // Create a FixedVec with minimal bit width (20 bits)
    let fixed_vec = LEFixedVec::builder()
        .bit_width(BitWidth::Minimal)
        .build(&data)
        .unwrap();

    println!("\nSize of the FixedVec with minimal bit width:");
    fixed_vec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE | DbgFlags::RUST_LAYOUT);
}
```

The output displays memory breakdown for a 1,000,000-element vector of `u64` integers, uniformly distributed between 0 and 2<sup>20</sup>

- `Vec<u64>`: 8.00 MB (8 bytes per element)
- [`VarVec`] with Gamma encoding: 4.73 MB (41% reduction)
- [`VarVec`] with Auto codec selection: 2.85 MB (64% reduction, selected Zeta k=10)
- [`FixedVec`] with minimal bit width: 2.50 MB (69% reduction, 20 bits per element)

The memory analysis also shows the internal structure of each data type, including storage overhead for metadata, sampling structures, and encoding parameters.

```text
Size of the uncompressed Vec<u64>:
8.000 MB 100.00% ⏺

Size of the VarVec with gamma encoding:
4.727 MB 100.00% ⏺
4.625 MB  97.85% ├╴data
101.6 kB   2.15% ├╴samples
101.6 kB   2.15% │ ├╴bits
    8  B   0.00% │ ├╴bit_width
    8  B   0.00% │ ├╴mask
    8  B   0.00% │ ├╴len
    0  B   0.00% │ ╰╴_phantom
    8  B   0.00% ├╴k
    8  B   0.00% ├╴len
   16  B   0.00% ├╴encoding
    0  B   0.00% ╰╴_markers

Size of the VarVec with Auto encoding:
2.846 MB 100.00% ⏺
2.749 MB  96.57% ├╴data
97.72 kB   3.43% ├╴samples
97.70 kB   3.43% │ ├╴bits
    8  B   0.00% │ ├╴bit_width
    8  B   0.00% │ ├╴mask
    8  B   0.00% │ ├╴len
    0  B   0.00% │ ╰╴_phantom
    8  B   0.00% ├╴k
    8  B   0.00% ├╴len
   16  B   0.00% ├╴encoding
    0  B   0.00% ╰╴_markers

Codec selected by Auto: Zeta { k: 10 }

Size of the FixedVec with minimal bit width:
2.500 MB 100.00% ⏺
2.500 MB 100.00% ├╴bits
    8  B   0.00% ├╴bit_width
    8  B   0.00% ├╴mask
    8  B   0.00% ├╴len
    0  B   0.00% ╰╴_phantom
```

[Rayon]: https://docs.rs/rayon/latest/rayon/
[Serde]: https://serde.rs/
[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html
[`VarVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html
[`SeqVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/seq/struct.SeqVec.html
[`AtomicFixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/atomic/struct.AtomicFixedVec.html

## Benchmarks

The library includes benchmarks for [`FixedVec`], [`VarVec`], and [`SeqVec`]. It also tests performance against other library implementations of compressed integer storage: [`sux::BitFieldVec`] and [`succinct::IntVector`]. These benchmarks are in the `benches` directory and can be run via

```bash
cargo bench
```

The benchmarks measure the performance of random access, batch access, sequential access, and memory usage for various data distributions and vector sizes.

[`sux::BitFieldVec`]: https://docs.rs/sux/latest/sux/bits/bit_field_vec/index.htmll
[`succinct::IntVector`]: https://docs.rs/succinct/latest/succinct/int_vec/trait.IntVec.html

## Optional Features: Storing `usize` and `isize`

By default, [`variable::VarVec`] only supports integer types with a fixed size (e.g., `u32`, `i64`). This guarantees that compressed data is portable across different machine architectures (e.g., from a 64-bit server to a 32-bit embedded device).

The `arch-dependent-storable` feature flag enables [`Storable`] implementations for `usize` and `isize`. When activated, you can create a `VarVec<usize>` directly.

**Warning**: This feature breaks data portability. A `VarVec<usize>` created on a 64-bit system containing values larger than `u32::MAX` will cause a panic if deserialized or read on a 32-bit system. Only enable this feature if you can guarantee that your application and its data will only ever run on a single target architecture (e.g., `x86_64`).

Enable it in your `Cargo.toml`:

```toml
compressed-intvec = { version = "0.6.0", features = ["arch-dependent-storable"] }
```

[`Storable`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/traits/trait.Storable.html
[`variable::VarVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.VarVec.html

# TODO

* [ ] Add support for [`epsilon-serde`](https://crates.io/crates/epserde)
