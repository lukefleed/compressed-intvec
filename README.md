# Compressed-IntVec: High-Performance Compressed Integer Vectors

[![crates.io](https://img.shields.io/crates/v/compressed-intvec.svg)](https://crates.io/crates/compressed-intvec)
[![rust](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml/badge.svg)](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml)
[![docs](https://docs.rs/compressed-intvec/badge.svg)](https://docs.rs/compressed-intvec)
[![downloads](https://img.shields.io/crates/d/compressed-intvec)](https://crates.io/crates/compressed-intvec)
![license](https://img.shields.io/crates/l/compressed-intvec)

A high-performance Rust library for compressed integer vectors with fast random access.

`compressed-intvec` provides space-efficient, in-memory representations for vectors of `u64` and `i64` integers. It leverages a variety of instantaneous codes from the [dsi-bitstream] library, offering a flexible trade-off between compression ratio and random access speed.

## Features

-   **Efficient Compression**: Utilizes a range of sophisticated bit-level codes, including Gamma (γ), Delta (δ), Rice, and Zeta (ζ), alongside optimal fixed-width encoding.
-   **Fast Random Access**: Achieves O(1) access for fixed-width encoding and fast, tunable access for variable-length codes via a sampling mechanism.
-   **Smart Codec Selection**: Features an [`Auto`] mode that analyzes your data to pick the most space-efficient compression scheme automatically. No guesswork required.
-   **Signed Integer Support**: Provides [`SIntVec`], a specialized vector for `i64` data that uses ZigZag encoding to efficiently compress values centered around zero.
-   **Parallel Processing**: Offers parallel iterators and random access methods ([`par_iter`], [`par_get_many`]) under the `parallel` feature flag to leverage multi-core systems.
-   **Flexible Builders**: Construct vectors from slices with automatic parameter detection, or from iterators for streaming large datasets.
-   **Endianness Support**: Provides [`LEIntVec`] (Little-Endian) and [`BEIntVec`] (Big-Endian) type aliases for platform-specific optimizations.

## Quick Example

Getting started is simple with the builder pattern and the provided [`prelude`].

```rust
use compressed_intvec::prelude::*;

// 1. Compress unsigned integers with automatic codec selection.
let data: &[u64] = &[10, 20, 30, 40, 50, 60, 70, 80, 90];

// The builder automatically detects the best codec for this data.
let intvec = LEIntVec::builder(data)
    .codec(CodecSpec::Auto) // Let the library choose the best strategy.
    .k(4)
    .build()
    .unwrap();

assert_eq!(intvec.len(), data.len());
assert_eq!(intvec.get(2), Some(30));


// 2. Compress signed integers.
let signed_data: &[i64] = &[-10, -20, -30, 40, 50, 60, -70, 80, 90];

// For SIntVec, you must manually specify a codec. Delta is a safe default.
// The library transparently handles negative values using ZigZag encoding.
let sintvec = LESIntVec::builder(signed_data)
    .codec(CodecSpec::Delta)
    .build()
    .unwrap();

assert_eq!(sintvec.get(1), Some(-20));
```

## Available Codecs

Choosing the right codec is key to maximizing compression. The [`CodecSpec`] enum allows you to either select one manually or let the library decide with `CodecSpec::Auto`.


| Codec Variant | Description & Encoding Strategy                                                                                                                                              | Optimal Data Distribution                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **`Auto`**          | **Recommended default.** Analyzes the data to choose the best variable-length code, balancing build time and compression ratio.                                              | Agnostic; adapts to the input data.                      |
| `FixedLength`       | Encodes each integer using a fixed number of bits. Provides O(1) random access with minimal overhead. The fastest but can be memory-inefficient for highly skewed data.         | **Uniformly distributed data** within a known range.     |
| `Gamma` (γ)         | A universal, parameter-free code. Encodes `n` by representing its length in unary, followed by the value itself. Simple and effective for small numbers.                       | Data skewed towards small non-negative integers.         |
| `Delta` (δ)         | A universal, parameter-free code. Encodes `n` by representing its length in γ-code, making it more efficient than γ for larger values.                                         | Data skewed towards small non-negative integers.         |
| `Rice`              | A fast, tunable code. Encodes `n` by splitting it into a quotient (stored in unary) and a remainder (stored in binary).                                                        | Data with a geometric distribution (e.g., run-lengths).  |
| `Zeta` (ζ)          | A tunable code for power-law data. Encodes `n` by breaking it into blocks of `k` bits and storing the number of blocks in unary.                                               | Word frequencies, node degrees in scale-free networks.   |
| `Unary`             | The simplest possible code. Encodes `n` as `n` zero-bits followed by a one.                                                                                                  | Extremely skewed distributions (e.g., boolean flags).    |
| `Explicit`          | An escape hatch to use any code from the [`dsi-bitstream::codes::Codes`][dsi-bitstream-codes] enum directly (e.g., `Omega` for recursive length encoding, `VByte` for byte-aligned values). | Advanced use cases requiring specific, unlisted codes.   |


## Memory Analysis and Automated Codec Selection

The library integrates [`mem-dbg`] to provide detailed memory layout analysis, which is essential for verifying the effectiveness of different compression strategies. This allows for empirical validation of codec choices.

The following example demonstrates the utility of `CodecSpec::Auto` by comparing its result against a generic codec choice for a vector of 10,000 random integers.


```rust
use compressed_intvec::prelude::*;
use mem_dbg::{DbgFlags, MemDbg};
use rand::{rngs::SmallRng, Rng, SeedableRng};

/// Generates a vector with uniformly random values.
fn generate_random_vec(size: usize, max: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size).map(|_| rng.random_range(0..max)).collect()
}

fn main() {
    let data = generate_random_vec(10_000, 1 << 20);

    println!("Size of the uncompressed Vec<u64> of data:");
    data.mem_dbg(DbgFlags::empty() | DbgFlags::HUMANIZE);

    // Create a LEIntVec with a generic Gamma encoding.
    let gamma_intvec = LEIntVec::builder(&data)
        .codec(CodecSpec::Gamma)
        .build()
        .unwrap();

    println!("\nSize of the LEIntVec with Gamma encoding:");
    gamma_intvec.mem_dbg(DbgFlags::empty() | DbgFlags::HUMANIZE);

    // Let the library analyze the data and choose the best codec.
    let auto_intvec = LEIntVec::builder(&data)
        .codec(CodecSpec::Auto)
        .build()
        .unwrap();

    println!("\nSize of the LEIntVec with Auto encoding:");
    auto_intvec.mem_dbg(DbgFlags::empty() | DbgFlags::HUMANIZE);
    let codec_used = auto_intvec.encoding();
    println!("\nCodec used for Auto encoding: {:?}", codec_used);
}
```

This produces the following output:

```text
Size of the uncompressed Vec<u64> of data:
80.02 kB ⏺

Size of the LEIntVec with Gamma encoding:
47.59 kB ⏺
46.27 kB ├╴data
1.276 kB ├╴samples
   16  B ├╴k
    8  B ├╴len
   16  B ├╴encoding
    0  B ╰╴endian

Size of the LEIntVec with Auto encoding:
28.84 kB ⏺
27.53 kB ├╴data
1.276 kB ├╴samples
   16  B ├╴k
    8  B ├╴len
   16  B ├╴encoding
    0  B ╰╴endian

Codec used for Auto encoding: Dsi(Zeta { k: 10 })
```
The analysis of the output is as follows:
- The uncompressed `Vec<u64>` occupies **80.02 KB**.
- `CodecSpec::Gamma` reduces the size to **47.59 KB**, a 40.5% reduction.
- `CodecSpec::Auto` further reduces the size to **28.84 KB**, achieving a 64% total reduction.

The final line of the output indicates that the `Auto` spec analyzed the data's distribution and selected `Zeta` coding with `k=10` as the most space-efficient strategy for this dataset. This capability allows the library to adapt its compression strategy to the data, and the `mem_dbg` integration provides the means to directly inspect and validate the outcome.

## High-Performance Access Patterns

While [`get`] is convenient for single, isolated lookups, calling it repeatedly in a loop is inefficient. For performance-critical scenarios involving multiple lookups, the library provides optimized methods that avoid the primary bottleneck: the repeated creation of internal bitstream readers.

### Dynamic Lookups with a Reusable `IntVecReader`

Each call to `intvec.get()` creates and subsequently discards a new bitstream reader, incurring a small but significant setup overhead. When lookups are performed in a loop, this overhead accumulates and can degrade performance.

For scenarios involving multiple dynamic lookups, especially when the access sequence isn't known in advance, the solution is to use a stateful, reusable [`IntVecReader`]. You create it once with `intvec.reader()` and then call `get()` on it multiple times. This amortizes the reader's setup cost across all lookups.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec = LEIntVec::builder(&data).build().unwrap();

// Create a single, reusable reader for all lookups.
let mut reader = intvec.reader();

// Perform multiple, non-sequential lookups efficiently.
// This pattern is ideal for traversals where the next index depends on the
// current value.
assert_eq!(reader.get(500).unwrap(), Some(500));
assert_eq!(reader.get(10).unwrap(), Some(10));
assert_eq!(reader.get(9000).unwrap(), Some(9000));
```

### Efficient Batch Access with `get_many`

When you have a predefined set of indices to retrieve, the [`get_many`] method is the most efficient sequential approach.

For variable-length codes, it sorts the requested indices to perform a single, monotonic scan over the compressed data. This approach minimizes expensive seeking operations and avoids redundant decoding, making it substantially faster than other patterns for batch lookups.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec = LEIntVec::builder(&data)
    .codec(CodecSpec::Delta)
    .k(64)
    .build()
    .unwrap();

// Indices can be in any order.
let indices_to_get = &[500, 10, 9000, 1000, 2000];

// get_many efficiently retrieves all values in one pass.
let values = intvec.get_many(indices_to_get).unwrap();

assert_eq!(values, vec![500, 10, 9000, 1000, 2000]);
```

### Parallel Access with `par_get_many`

The `parallel` feature, which is **enabled by default**, provides access to [`par_get_many`]. This method uses the [Rayon] crate to parallelize lookups across multiple CPU cores.

It works by dividing the slice of indices among worker threads, with each thread performing lookups independently. While this can lead to some redundant decoding if multiple threads access data within the same sample block, the throughput gained from parallelism often results in significant speedups for large batches of randomly distributed indices.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec = LEIntVec::builder(&data)
    .codec(CodecSpec::Delta)
    .k(64)
    .build()
    .unwrap();

let indices_to_get = &[500, 10, 9000, 1000, 2000];

// par_get_many retrieves all values in parallel.
let parallel_values = intvec.par_get_many(indices_to_get).unwrap();

assert_eq!(parallel_values, vec![500, 10, 9000, 1000, 2000]);
```

If you need to disable this feature to reduce dependencies, you can do so in your `Cargo.toml`:

```toml
[dependencies]
compressed-intvec = { version = "0.4.0", default-features = false }
```

## Core Concepts

### The Trade-off: Instantaneous Codes and Random Access

The core challenge stems from how different encoding strategies store data.

#### Instantaneous Codes vs. Fixed-Length Encoding

The library offers two fundamental encoding families:

1.  **Fixed-Length Encoding:**
    This is the simplest and fastest strategy. Every integer is encoded using the *exact same number of bits*.
    -   **How it Works**: To access the element at `index`, the library performs a single calculation: `bit_offset = index * num_bits`. The bitstream reader can seek directly to this position and read the value.
    -   **Performance**: This makes random access a true **O(1)** operation with minimal overhead.
    -   **When to Use It**: It is the optimal choice when your data is **uniformly distributed** within a known range. For example, random numbers between 0 and 1000 can all be stored in 10 bits each. This leads to efficient storage and the fastest possible access.

2.  **Instantaneous Codes (Variable-Length):**
    Codes like `Gamma`, `Delta`, and `Zeta` are *variable-length*. They achieve high compression by using fewer bits for smaller or more frequent integers and more bits for larger, rarer ones.
    -   **The Challenge**: This efficiency comes at a cost. Because each element has a different bit-length, it is impossible to mathematically calculate the starting position of an arbitrary element.
    -   **When to Use It**: These codes excel when data follows a **specific, skewed distribution** (e.g., a power-law or geometric distribution). Here the compression ratio can be significantly better than fixed-length encoding, but random access becomes more complex.

### Why Sampling is Necessary for Variable-Length Codes

To solve the random access problem for instantaneous codes, `compressed-intvec` uses a **sampling mechanism**.

-   **How it Works**: During construction, the library stores the bit position of every `k`-th element in a separate lookup table (the `samples` array). To retrieve the value at a given `index`, the accessor performs the following steps:
    1.  It identifies the closest preceding sample point by calculating `sample_index = index / k`.
    2.  It looks up the bit offset of that sample point.
    3.  It seeks the internal bitstream reader to that offset.
    4.  It decodes elements sequentially from that point until it reaches the target `index`.

This is where the `k` parameter becomes critical:

-   **Small `k`** (e.g., 16): Creates more sample points. This increases memory overhead but makes random access ([`get`]) faster, as the sequential scan is always short (at most `k-1` elements).
-   **Large `k`** (e.g., 128): Creates fewer sample points, reducing memory overhead. However, random access becomes slower as the sequential scan can be much longer.

Choosing a `k` between 16 and 64 is often a good balance for large datasets.

### Endianness

The library provides [`LEIntVec`] (Little-Endian) and [`BEIntVec`] (Big-Endian) type aliases. While using the native endianness of your CPU is usually the fastest, the optimal choice can depend on the specific codec and hardware architecture. Benchmarking is the best way to determine the most performant option for your workload.

[dsi-bitstream]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/
[dsi-bitstream-codes]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/enum.Codes.html
[Rayon]: https://docs.rs/rayon/latest/rayon/
[`mem-dbg`]: https://docs.rs/mem-dbg/latest/mem_dbg/
[`prelude`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/prelude/index.html
[`SIntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/sintvec/struct.SIntVec.html
[`LEIntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/intvec/type.LEIntVec.html
[`BEIntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/intvec/type.BEIntVec.html
[`IntVecReader`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/intvec/struct.IntVecReader.html
[`CodecSpec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/codec_spec/enum.CodecSpec.html
[`Auto`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/codec_spec/enum.CodecSpec.html#variant.Auto
[`get`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/intvec/struct.IntVec.html#method.get
[`reader`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/intvec/struct.IntVec.html#method.reader
[`get_many`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/intvec/struct.IntVec.html#method.get_many
[`par_iter`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/intvec/struct.IntVec.html#method.par_iter
[`par_get_many`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/intvec/struct.IntVec.html#method.par_get_many
