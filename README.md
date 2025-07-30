# Compressed Integer Vectors

[![crates.io](https://img.shields.io/crates/v/compressed-intvec.svg)](https://crates.io/crates/compressed-intvec)
[![rust](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml/badge.svg)](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml)
[![docs](https://docs.rs/compressed-intvec/badge.svg)](https://docs.rs/compressed-intvec)
[![downloads](https://img.shields.io/crates/d/compressed-intvec)](https://crates.io/crates/compressed-intvec)
![license](https://img.shields.io/crates/l/compressed-intvec)

A Rust library that provides space-efficient, in-memory representations for vectors of `u64` and `i64` integers. It leverages a variety of instantaneous codes from the [dsi-bitstream] library, offering a flexible trade-off between compression ratio and random access speed.

## Features

-   **Efficient Compression**: Utilizes a range of sophisticated bit-level codes, including Gamma (γ), Delta (δ), Rice, and Zeta (ζ), alongside optimal fixed-width encoding.
-   **Fast Random Access**: Achieves O(1) access for fixed-width encoding and fast, tunable access for variable-length codes via a sampling mechanism.
-   **Smart Codec Selection**: Features an [`Auto`] mode that analyzes your data to pick the most space-efficient compression scheme automatically. No guesswork required. Note that this introduces a small overhead during the initial build phase.
-   **Signed Integer Support**: Provides [`SIntVec`], a specialized vector for `i64` data that uses ZigZag encoding to efficiently compress values centered around zero.
-   **Parallel Processing**: Offers parallel iterators and random access methods ([`par_iter`], [`par_get_many`]) under the `parallel` feature flag to leverage multi-core systems.
-   **Flexible Builders**: Construct vectors from slices with automatic parameter detection, or from iterators for streaming large datasets.
-   **Endianness Support**: Provides [`LEIntVec`] (Little-Endian) and [`BEIntVec`] (Big-Endian) type aliases for platform-specific optimizations.

## Quick Example

This example demonstrates how to use the library to compress and access unsigned integers and signed integers efficiently.

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

Choosing the right codec is key to maximizing compression. The [`CodecSpec`] enum allows you to either select one manually or let the library decide with [`CodecSpec::Auto`]. Note that the `Auto` codec introduces a small overhead during the initial build phase, as it analyzes a sample (up to 10k elements) of the data to determine the best compression strategy. This is a one-time cost that pays off in reduced memory usage and improved access speed.


| Codec Variant | Description & Encoding Strategy | Optimal Data Distribution |
| :--- | :--- | :--- |
| **`Auto`** | **Recommended default.** Analyzes the data to choose the best variable-length code, balancing build time and compression ratio. | Agnostic; adapts to the input data. |
| `FixedLength` | Encodes each integer using a fixed number of bits. Provides O(1) random access. The fastest but can be memory-inefficient for highly skewed data. | **Uniformly distributed data** within a known range. |
| `Gamma` (γ) | A universal, parameter-free code. Encodes `n` by representing its length in unary, followed by the value itself. Simple and effective for small numbers. | Data skewed towards small non-negative integers. |
| `Delta` (δ) | A universal, parameter-free code. Encodes `n` by representing its length in γ-code, making it more efficient than γ for larger values. | Data skewed towards small non-negative integers. |
| `Rice` | A fast, tunable code. Encodes `n` by splitting it into a quotient (stored in unary) and a remainder (stored in binary). | Data with a geometric distribution (e.g., run-lengths). |
| `Golomb` | A tunable code, more general than Rice, that splits `n` into a quotient and remainder. | Data with a geometric distribution. |
| `Zeta` (ζ) | A tunable code for power-law data. Encodes `n` by breaking it into blocks of `k` bits and storing the number of blocks in unary. | Word frequencies, node degrees in scale-free networks. |
| `VByteLe`/`Be`| A byte-aligned code that uses a continuation bit to store integers in a variable number of bytes. Very fast for values that fit in 1-4 bytes. | General-purpose integer data. |
| `Omega` (ω) | A universal, recursive code that encodes the length of the number's binary representation. Extremely compact for very large numbers. | The implied distribution is essentially it is as close as possible to ≈ 1/x (as there is no code for that distribution). |
| `Unary` | The simplest possible code. Encodes `n` as `n` zero-bits followed by a one. | Extremely skewed distributions (e.g., boolean flags). |
| `Explicit` | An escape hatch to use any code from the [`dsi-bitstream::codes::Codes`][dsi-bitstream-codes] enum not directly listed here. | Advanced use cases requiring specific, unlisted codes. |



## Memory Analysis and Automated Codec Selection

The library integrates [`mem-dbg`] to provide detailed memory layout analysis, which is essential for verifying the effectiveness of different compression strategies. This allows for empirical validation of codec choices.

The following example demonstrates the utility of [`CodecSpec::Auto`] by comparing its result against a generic codec choice for a vector of 10,000 random integers.


```rust
use compressed_intvec::prelude::*;
use mem_dbg::{DbgFlags, MemDbg};
use rand::{rngs::SmallRng, Rng, SeedableRng};

// Generates a vector with uniformly random values.
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
- [`CodecSpec::Gamma`] reduces the size to **47.59 KB**, a 40.5% reduction.
- [`CodecSpec::Auto`] further reduces the size to **28.84 KB**, achieving a 64% total reduction.

The final line of the output indicates that the `Auto` spec analyzed the data's distribution and selected `Zeta` coding with `k=10` as the most space-efficient strategy for this dataset. This capability allows the library to adapt its compression strategy to the data, and the [`mem_dbg`] integration provides the means to directly inspect and validate the outcome.

## Access Patterns

The standard method to access a single element in a compressed [`IntVec`] instance is through the [`get`] method.

```rust
use compressed_intvec::prelude::*;

let data = vec![10, 20, 30, 40, 50];
let intvec = LEIntVec::builder(&data)
    .codec(CodecSpec::Auto)
    .k(4)
    .build()
    .unwrap();

assert_eq!(intvec.get(2), Some(30));
```

Accessing multiple elements however, requires more attention to performance. A naive loop of `intvec.get()` calls is highly inefficient due to the repeated creation and destruction of internal readers. This library provides a suite of specialized methods tailored to different access scenarios.

### Batch Access from a Slice: `get_many`

When the complete set of indices to be retrieved is available upfront in a slice or `Vec`, [`get_many`] is the most robust and performant choice.

This method first sorts the indices, transforming a random access pattern into a single, monotonic scan over the compressed data. This approach minimizes expensive bitstream seeks and leverages data locality. Usually, [`get_many`] consistently outperforms loop-based alternatives across all access patterns, including sorted, clustered, and even fully random indices.

> **If you have a slice of indices, always prefer `get_many`.**

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec = LEIntVec::builder(&data)
    .codec(CodecSpec::Delta)
    .k(64)
    .build()
    .unwrap();

// Indices can be in any order; get_many will sort them internally.
let indices_to_get = &[500, 10, 9000, 1000, 2000];

// get_many efficiently retrieves all values in one optimized pass.
let values = intvec.get_many(indices_to_get).unwrap();

assert_eq!(values, vec![500, 10, 9000, 1000, 2000]);
```

### Access from a Stream: `get_many_from_iter`

In scenarios where indices are provided by a streaming source (an iterator) and cannot be collected into a vector due to memory constraints, we provide [`get_many_from_iter`]. This is the method to use when you need to process indices on-the-fly and you expect them to be somewhat ordered or clustered.

This method uses a stateful [`IntVecSeqReader`] internally, which is optimized for streams that have sequential locality.


```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec = LEIntVec::builder(&data).build().unwrap();

// Process indices from a streaming source, like a range iterator.
let values = intvec.get_many_from_iter(500..505).unwrap();

assert_eq!(values, vec![500, 501, 502, 503, 504]);
```

### Parallel Access: `par_get_many`

The `parallel` feature, enabled by default, provides [`par_get_many`]. This method uses the [Rayon] crate to distribute the lookup work across multiple CPU cores, offering significant throughput gains for large batches of indices on multi-core systems.

It parallelizes the lookups by giving each thread a reusable reader and a portion of the indices. This "embarrassingly parallel" approach is most effective for large, randomly distributed workloads where the benefits of parallelism outweigh the cost of some potentially redundant decoding.

> **Use `par_get_many` for large batches of indices. This is by far the fastest method**

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

### Dynamic Lookups: `IntVecReader` and `IntVecSeqReader`

For situations where we need to access elements dynamically, the library provides two distinct readers for this purpose:

-   [`IntVecReader`]: A **stateless** reader optimized for fully random, unpredictable access patterns.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec = LEIntVec::builder(&data)
    .codec(CodecSpec::Delta)
    .k(64)
    .build()
    .unwrap();

// Create a stateless reader for random access.
let mut reader = intvec.reader();
let indices_to_get = &[500, 10, 9000, 1000, 2000];
for &index in indices_to_get {
    // Use the stateless reader to get values.
    let value = reader.get(index).unwrap().unwrap();
    assert_eq!(value, index as u64);
}

// NOTE: in this case using `get_many` would be more efficient, but this demonstrates the reader's usage.
```

-   [`IntVecSeqReader`]: A **stateful** reader optimized for access patterns with sequential locality.

```rust
use compressed_intvec::prelude::*;

use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec = LEIntVec::builder(&data)
    .codec(CodecSpec::Delta)
    .k(64)
    .build()
    .unwrap();

let mut reader = intvec.seq_reader();
let indices_to_get = &[500, 10, 9000, 1000, 2000];
for &index in indices_to_get {
    let value = reader.get(index).unwrap().unwrap();
    assert_eq!(value, index as u64);
}

// NOTE: in this case using `get_many` would be more efficient, but this demonstrates the reader's usage.
```


## Instantaneous Codes vs. Fixed-Length Encoding

The library offers two fundamental encoding families:

1.  **Fixed-Length Encoding:**
    This is the simplest and fastest strategy. Every integer is encoded using the *exact same number of bits*.
    -   **How it Works**: To access the element at `index`, the library performs a single calculation: `bit_offset = index * num_bits`. The bitstream reader can seek directly to this position and read the value.
    -   **Performance**: This makes random access a true **O(1)** operation with minimal overhead.
    -   **When to Use It**: It is the optimal choice when your data is **uniformly distributed** within a known range. For example, random numbers between 0 and 1000 can all be stored in 10 bits each. This leads to efficient storage and the fastest possible access.

    ```rust
    use compressed_intvec::prelude::*;

    let data: Vec<u64> = (0..1000).collect();
    // Each number is < 1024, so fits in 10 bits
    let intvec = LEIntVec::builder(&data)
        .codec(CodecSpec::FixedLength {
            num_bits: Some(10),
        })
        .build()
        .unwrap();

    assert_eq!(intvec.into_vec(), data);

    // Otherwise, let the library choose the minimum number of bits required (with a small overhead at build time to find the maximum value in the data).
    let intvec = LEIntVec::builder(&data)
        .codec(CodecSpec::FixedLength { num_bits: None })
        .build()
        .unwrap();

    assert_eq!(intvec.into_vec(), data);
    ```

2.  **Instantaneous Codes (Variable-Length):**
    Codes like `Gamma`, `Delta`, and `Zeta` are *variable-length*. They achieve high compression by using fewer bits for smaller or more frequent integers and more bits for larger, rarer ones. This efficiency comes at a cost. Because each element has a different bit-length, it is impossible to mathematically calculate the starting position of an arbitrary element. That's why we say these codes are **instantaneous**: you cannot decode a value without reading the entire preceding bitstream.
    -   **When to Use It**: These codes excel when data follows a **specific, skewed distribution** (e.g., a power-law or geometric distribution). Here the compression ratio can be significantly better than fixed-length encoding, but random access becomes more complex.


    ```rust
    use compressed_intvec::prelude::*;

    let data: Vec<u64> = (0..1000).collect();

    // We can use codecs from the dsi-bitstream that are parameter-dependent like Rice encoding.
    let intvec = LEIntVec::builder(&data)
        .codec(CodecSpec::Rice { log2_b: Some(4) })
        .build()
        .unwrap();
    assert_eq!(intvec.into_vec(), data);

    // Or codecs that are byte-aligned like VByte, for faster access.
    let intvec = LEIntVec::builder(&data)
        .codec(CodecSpec::VByteLe)
        .build()
        .unwrap();
    assert_eq!(intvec.into_vec(), data);
    ```

### Why Sampling is Necessary for Variable-Length Codes

To solve the random access problem for instantaneous codes, `compressed-intvec` uses a **sampling mechanism**.

-   **How it Works**: During construction, the library stores the bit position of every `k`-th element in a separate lookup table (the `samples` array). To retrieve the value at a given `index`, the accessor performs the following steps:
    *  It identifies the closest preceding sample point by calculating `sample_index = index / k`.
    *  It looks up the bit offset of that sample point.
    *  It seeks the internal bitstream reader to that offset.
    *  It decodes elements sequentially from that point until it reaches the target `index`.

This is where the `k` parameter becomes critical:

-   **Small `k`** (e.g., 16): Creates more sample points. This increases memory overhead but makes random access ([`get`]) faster, as the sequential scan is always short (at most `k-1` elements).
-   **Large `k`** (e.g., 128): Creates fewer sample points, reducing memory overhead. However, random access becomes slower as the sequential scan can be much longer.

Choosing a `k` between 16 and 64 is often a good balance for large datasets.

### Endianness

The library provides [`LEIntVec`] (Little-Endian) and [`BEIntVec`] (Big-Endian) type aliases. While using the native endianness of your CPU is usually the fastest, the optimal choice can depend on the specific codec and hardware architecture. Benchmarking is the best way to determine the most performant option for your workload.

## Cargo Features

### `parallel` (Default)

Enables parallel operations for full-vector decompression and batch lookups using the [Rayon] crate. This provides `par_iter()` and `par_get_many()` methods on `IntVec` and `SIntVec`. This feature is **enabled by default**. If you do not require parallel processing or wish to minimize dependencies, you can disable the default features in your `Cargo.toml`:

```toml
[dependencies]
compressed-intvec = { version = "0.4.0", default-features = false }
```

### `serde`

Enables serialization and deserialization for `IntVec` and `SIntVec` by implementing the `Serialize` and `Deserialize` traits from the [Serde] framework. This allows compressed vectors to be efficiently stored or transmitted. This feature is **optional** and must be explicitly enabled:

```toml
[dependencies]
compressed-intvec = { version = "0.4.0", features = ["serde"] }
```

You can also enable multiple features. For example, to use both `serde` and `parallel` support (which is the default configuration plus `serde`):

```toml
[dependencies]
compressed-intvec = { version = "0.4.0", features = ["parallel", "serde"] }
```
## TODO For Future Releases

* [ ] Add [`ε-serde`](https://crates.io/crates/epserde) support.
* [ ] Add SIMD optimizations for faster decoding in [`CodecSpec::FixedLength`] and [`CodecSpec::VByte`].
* [ ] Consider adding optional Elias–Fano compression for the samples vector. This could further reduce memory usage, but would increase computational overhead for the [`get`] method. The trade-off may not justify the relatively small compression gain.


[dsi-bitstream]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/
[dsi-bitstream-codes]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/
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
[`CodecSpec::Auto`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/codec_spec/enum.CodecSpec.html#variant.Auto
[`CodecSpec::Gamma`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/codec_spec/enum.CodecSpec.html#variant.Gamma
[`CodecSpec::Delta`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/codec_spec/enum.CodecSpec.html#variant.Delta
[`CodecSpec::Zeta`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/codec_spec/enum.CodecSpec.html#variant.Zeta
[`CodecSpec::FixedLength`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/codec_spec/enum.CodecSpec.html#variant.FixedLength
[`CodecSpec::VByte`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/codec_spec/enum.CodecSpec.html#variant.VByteLe
[Serde]: https://serde.rs/
[`IntVecSeqReader`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/intvec/struct.IntVecSeqReader.html
[`get_many_from_iter`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/intvec/struct.IntVec.html#method.get_many_from_iter
