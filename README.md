# Compressed Integer Vectors

[![crates.io](https://img.shields.io/crates/v/compressed-intvec.svg)](https://crates.io/crates/compressed-intvec)
[![rust](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml/badge.svg)](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml)
[![docs](https://docs.rs/compressed-intvec/badge.svg)](https://docs.rs/compressed-intvec)
[![downloads](https://img.shields.io/crates/d/compressed-intvec)](https://crates.io/crates/compressed-intvec)
![license](https://img.shields.io/crates/l/compressed-intvec)
[![Line count](https://tokei.rs/b1/github/lukefleed/compressed-intvec?type=Rust,Python)](https://github.com/lukefleed/compressed-intvec)

A Rust library providing space-efficient, in-memory representations for integer vectors. It offers two complementary data structures: [`FixedVec`], which uses fixed-width encoding for O(1) mutable and atomic access, and [`IntVec`], which uses variable-length instantaneous codes for high-ratio compression.

The library is designed to reduce the memory footprint of standard [`std::vec::Vec`] collections of integers while retaining performant access patterns suitable for data-intensive applications.

## Core Structures: `FixedVec` vs. `IntVec`

The library provides two distinct vector types, each based on a different encoding principle. Choosing the right one depends on your specific use case, performance requirements, and data characteristics.

### `FixedVec`: Fixed-Width Encoding

Implements a vector where every integer occupies the same, predetermined number of bits.

*   **Key Features**:
    *   **O(1) Random Access**: The memory location of any element is determined by a direct bit-offset calculation (`index * bit_width`), resulting in minimal-overhead access.
    *   **Mutability**: Supports in-place modifications after creation through an API similar to [`std::vec::Vec`] (e.g., `push`, `set`, `pop`).
    *   **Atomic Operations**: Provides [`AtomicFixedVec`], a thread-safe variant that supports atomic read-modify-write operations for concurrent environments.

*   **Use Cases**:
    *   Data with a uniform or near-uniform distribution where all values fit within a known bit width.
    *   Applications where low-latency random access is the primary performance requirement.
    *   Scenarios requiring in-place vector modification or concurrent atomic updates.

With low bit widths (e.g., 8, 16, 32), `FixedVec` is 2-3x faster than `std::vec::Vec` for random access.

### `IntVec`: Variable-Width Encoding

Implements a vector using variable-length instantaneous codes (e.g., Gamma, Delta, Rice, Zeta) to represent each integer.

*   **Key Features**:
    *   **High Compression Ratios**: Achieves significant space savings for data with non-uniform distributions.
    *   **Automatic Codec Selection**: Can analyze the data to select the most effective compression codec automatically.
    *   **Amortized O(1) Random Access**: Enables fast random access by sampling the bit positions of elements at a configurable interval (`k`).

*   **Use Cases**:
    *   Applications where minimizing memory usage is the primary goal.
    *   Read-only datasets with skewed distributions.
    *   Large-scale data processing where the vector is built once and read many times.

### Summary

| Feature             | [`FixedVec`]                                     | [`IntVec`]                                     |
| :------------------ | :--------------------------------------------- | :------------------------------------------- |
| **Encoding**        | Fixed-Width                                    | Variable-Width (Instantaneous Codes)         |
| **Random Access**   | O(1) (direct computation)                      | O(k) (amortized via sampling)                |
| **Mutability**      | Yes (`push`, `set`, etc.)                      | No (immutable after creation)                |
| **Atomic Support**  | Yes ([`AtomicFixedVec`])                       | No (read-only concurrency)                   |
| **Ideal Data**      | Uniformly distributed                          | Non-uniformly (skewed) distributed           |
| **Primary Goal**    | Speed and flexibility                          | Compression ratio                            |

[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html
[`IntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html
[`AtomicFixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/atomic/struct.AtomicFixedVec.html
[`std::vec::Vec`]: https://doc.rust-lang.org/std/vec/struct.Vec.html

## Quick Start

The following examples demonstrate the primary use cases for [`FixedVec`] and [`IntVec`]. All common types and traits are available through the [`prelude`].

### Example: `FixedVec` for Uniform Data and Mutable Access

Use [`FixedVec`] when data is uniformly distributed or when you require in-place modification or atomic operations.

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

### Example: `IntVec` for High-Ratio Compression

Use [`IntVec`] when minimizing memory usage is the priority and the data is read-only. It is most effective on data with skewed distributions.

```rust
use compressed_intvec::prelude::*;

// Skewed data with a mix of small and large values.
let skewed_data: &[u64] = &[5, 8, 13, 1000, 7, 6, 10_000, 10, 2, 3];

// The builder can automatically select the best compression codec.
let intvec: LEIntVec = IntVec::builder(skewed_data)
    .codec(VariableCodecSpec::Auto)
    .build()
    .unwrap();

assert_eq!(intvec.len(), skewed_data.len());
assert_eq!(intvec.get(3), Some(1000));
```

[`prelude`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/prelude/index.html
[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html
[`IntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html
[`FixedVec::builder`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html#method.builder
[`IntVec::builder`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html#method.builder
[`UFixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/type.UFixedVec.html
[`LEIntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/type.LEIntVec.html
[`VariableCodecSpec::Auto`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/codec/enum.VariableCodecSpec.html#variant.Auto

## Deep Dive: Maximizing Compression with `IntVec`

[`IntVec`] is the optimal choice when data that is not uniformly distributed and we want to minimize memory usage. It uses variable-length codes to represent integers, allowing for significant space savings compared to fixed-width encodings.

The compression strategy is controlled by the [`VariableCodecSpec`] enum, passed to the builder.

### Choosing the Right Codec

For most use cases, the recommended strategy is [`VariableCodecSpec::Auto`], which analyzes the data to select the most space-efficient codec. However, you can also specify a codec explicitly based on your data characteristics.

| `VariableCodecSpec` Variant | Description & Encoding Strategy | Optimal Data Distribution |
| :--- | :--- | :--- |
| **`Auto`** | **Recommended default.** Analyzes the data to choose the best variable-length code, balancing build time and compression ratio. | Agnostic; adapts to the input data. |
| `Gamma` (γ) | A universal, parameter-free code. Encodes `n` by representing its length in unary, followed by the value itself. | Data skewed towards small non-negative integers. |
| `Delta` (δ) | A universal, parameter-free code. Encodes `n` by representing its length in γ-code, making it more efficient than γ for larger values. | Data skewed towards small non-negative integers. |
| `Rice` | A fast, tunable code. Encodes `n` by splitting it into a quotient (stored in unary) and a remainder (stored in binary). | Data with a geometric distribution. |
| `Golomb` | A tunable code, more general than Rice, that splits `n` into a quotient and remainder. | Data with a geometric distribution. |
| `Zeta` (ζ) | A tunable code for power-law data. Encodes `n` by breaking it into blocks and storing the number of blocks in unary. | Word frequencies, node degrees in scale-free networks. |
| `VByteLe`/`Be`| A byte-aligned code that uses a continuation bit to store integers in a variable number of bytes. Fast decoding. | General-purpose integer data. |
| `Omega` (ω) | A universal, recursive code that encodes the length of the number's binary representation. Compact for very large numbers. | Implied distribution is approximately 1/x. |
| `Unary` | The simplest code. Encodes `n` as `n` zero-bits followed by a one. | Extremely skewed distributions (e.g., boolean flags). |
| `Explicit` | An escape hatch to use any code from the [`dsi-bitstream::codes::Codes`][dsi-bitstream-codes] enum. | Advanced use cases requiring specific, unlisted codes. |

### Automatic Selection with `VariableCodecSpec::Auto`

The `Auto` strategy removes the guesswork from codec selection. During the build phase, it analyzes a sample of the input data and selects the codec that offers the best compression ratio. This one-time analysis cost often leads to significant memory savings.

The library integrates [`mem-dbg`] to provide memory usage statistics, allowing you to compare the size of different encoding strategies easily.

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

    println!("Size of the uncompressed Vec<u64>:");
    data.mem_dbg(DbgFlags::PERCENTAGE | DbgFlags::HUMANIZE);

    // Create an IntVec with a generic Gamma encoding.
    let gamma_intvec = LEIntVec::builder(&data)
        .codec(VariableCodecSpec::Gamma)
        .build()
        .unwrap();

    println!("\nSize of the IntVec with Gamma encoding:");
    gamma_intvec.mem_dbg(DbgFlags::PERCENTAGE | DbgFlags::HUMANIZE);

    // Let the library analyze the data and choose the best codec.
    let auto_intvec = LEIntVec::builder(&data)
        .codec(VariableCodecSpec::Auto)
        .build()
        .unwrap();

    println!("\nSize of the IntVec with Auto encoding:");
    auto_intvec.mem_dbg(DbgFlags::PERCENTAGE | DbgFlags::HUMANIZE);
    println!("\nCodec selected by Auto: {:?}", auto_intvec.encoding());
}
```

This produces the following output:

```text
Size of the uncompressed Vec<u64>:
80.02 kB 100.00% ⏺

Size of the IntVec with Gamma encoding:
47.10 kB 100.00% ⏺
46.27 kB  98.23% ├╴data
  800  B   1.70% ├╴samples
  776  B   1.65% │ ├╴bits
    8  B   0.02% │ ├╴bit_width
    8  B   0.02% │ ├╴mask
    8  B   0.02% │ ├╴len
    0  B   0.00% │ ╰╴_phantom
    8  B   0.02% ├╴k
    8  B   0.02% ├╴len
   16  B   0.03% ├╴encoding
                 │ ╰╴Variant: Gamma
    0  B   0.00% ╰╴_markers

Size of the IntVec with Auto encoding:
28.33 kB 100.00% ⏺
27.53 kB  97.18% ├╴data
  768  B   2.71% ├╴samples
  744  B   2.63% │ ├╴bits
    8  B   0.03% │ ├╴bit_width
    8  B   0.03% │ ├╴mask
    8  B   0.03% │ ├╴len
    0  B   0.00% │ ╰╴_phantom
    8  B   0.03% ├╴k
    8  B   0.03% ├╴len
   16  B   0.06% ├╴encoding
                 │ ╰╴Variant: Zeta { k: 10 }
    0  B   0.00% ╰╴_markers

Codec selected by Auto: Zeta { k: 10 }
```

[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html
[`IntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html
[`VariableCodecSpec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/codec/enum.VariableCodecSpec.html
[`VariableCodecSpec::Auto`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/codec/enum.VariableCodecSpec.html#variant.Auto
[dsi-bitstream-codes]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/enum.Codes.html
[`mem-dbg`]: https://docs.rs/mem-dbg/latest/mem_dbg/

## `IntVec` Access Patterns

The access strategy for a compressed [`IntVec`] has significant performance implications. The library provides several methods, each optimized for a specific access pattern. Using the appropriate method is key to achieving high throughput.

### [`get_many`]: Batch Access from a Slice

For retrieving a batch of elements from a slice of indices.

*   **Mechanism**: This method sorts the provided indices to transform a random access pattern into a single, monotonic scan over the compressed data. This approach minimizes expensive bitstream seek operations and leverages data locality.
*   **Use Case**: The preferred method for any batch lookup when all indices are known and can be stored in a slice.

```rust
use compressed_intvec::prelude::*;
use rand::Rng; // For random number generation

let data: Vec<u64> = (0..10_000).collect();
let intvec: LEIntVec = IntVec::builder(&data)
    .codec(VariableCodecSpec::Delta)
    .k(64)
    .build()
    .unwrap();

// Indices can be in any order.
let indices_to_get: Vec<usize> = (0..100).map(|_| rand::rng().gen_range(0..10_000)).collect();

// `get_many` retrieves all values in one optimized pass.
let values = intvec.get_many(&indices_to_get).unwrap();

assert_eq!(values, indices_to_get.iter().map(|&i| data[i]).collect::<Vec<_>>());
```

### [`get_many_from_iter`]: Access from an Iterator

For retrieving elements from a streaming iterator of indices.

*   **Mechanism**: Processes indices on-the-fly using a stateful [`IntVecSeqReader`] internally, which is optimized for streams with sequential locality.
*   **Use Case**: Use when indices cannot be collected into a slice, for example due to memory constraints.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec: LEIntVec = IntVec::builder(&data).build().unwrap();

// Process indices from a streaming source, such as a range iterator.
let values: Vec<u64> = intvec.get_many_from_iter(500..505).unwrap();

assert_eq!(values, vec![500, 501, 502, 503, 504]);
```

### Dynamic Lookups: `IntVecReader` and `IntVecSeqReader`

For dynamic or interactive scenarios where lookup indices are not known in advance.

#### [`IntVecReader`]: Stateless Random Access

A **stateless** reader for efficient, repeated random lookups.

*   **Mechanism**: Amortizes the setup cost of the bitstream reader across multiple calls. Each `get` operation performs an independent seek from the nearest sample point.
*   **Use Case**: Optimal for sparse, unpredictable access patterns where there is no locality between consecutive lookups.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec: LEIntVec = IntVec::builder(&data).build().unwrap();

// Create a stateless reader for random access.
let mut reader = intvec.reader();
assert_eq!(reader.get(500).unwrap(), Some(500));
assert_eq!(reader.get(10).unwrap(), Some(10));
```

#### [`IntVecSeqReader`]: Stateful Sequential Access

A **stateful** reader optimized for access patterns with sequential locality.

*   **Mechanism**: Maintains an internal cursor. If a requested index is forward and within the same sample block, it decodes from the last position, avoiding a full seek.
*   **Use Case**: Optimal for iterating through sorted or clustered indices where consecutive lookups are near each other.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec: LEIntVec = IntVec::builder(&data).build().unwrap();

// Create a stateful reader for sequential access.
let mut seq_reader = intvec.seq_reader();
assert_eq!(seq_reader.get(500).unwrap(), Some(500));
// This second call is faster as it decodes forward from index 500.
assert_eq!(seq_reader.get(505).unwrap(), Some(505));
```

[`get`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html#method.get
[`get_many`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html#method.get_many
[`get_many_from_iter`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html#method.get_many_from_iter
[`IntVecReader`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/reader/struct.IntVecReader.html
[`IntVecSeqReader`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/seq_reader/struct.IntVecSeqReader.html

## Concurrency with `AtomicFixedVec`

For concurrent applications, the library provides [`AtomicFixedVec`], a thread-safe variant of [`FixedVec`]. It supports atomic read-modify-write (RMW) operations, enabling safe and efficient manipulation of shared integer data across multiple threads.

*   **Mechanism**: The atomicity guarantees depend on the element's bit width and its alignment within the underlying `u64` storage words.
    *   **Lock-Free Path**: When an element is fully contained within a single `u64` word (common for power-of-two bit widths), operations are performed using lock-free atomic hardware instructions. Performance is optimal as no locks are involved.
    *   **Locked Path**: When an element's bits span across the boundary of two `u64` words, operations are protected by a fine-grained mutex from a striped lock pool. This ensures atomicity for the two-word update without resorting to a global lock.

*   **Use Case**: Ideal for shared counters, parallel data processing, and any scenario requiring multiple threads to read from and write to the same integer vector concurrently. For write-heavy workloads, configuring the bit width to a power of two (e.g., 8, 16, 32) is recommended to ensure all operations remain on the lock-free path.

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

## Cargo Features

The library's functionality can be customized through the following Cargo features.

### `parallel` (Default Feature)

Enables parallel operations using the [Rayon] crate.Provides `par_iter` and `par_get_many` methods on [`FixedVec`] and [`IntVec`], as well as `par_iter` and `par_iter_mut` on [`AtomicFixedVec`].

This feature is enabled by default, as it significantly improves performance for batch operations on large vectors.

### `serde`

Enables serialization and deserialization for [`FixedVec`], [`AtomicFixedVec`], and [`IntVec`] by implementing the `Serialize` and `Deserialize` traits from the [Serde] framework.

This feature is not enabled by default.


[Rayon]: https://docs.rs/rayon/latest/rayon/
[Serde]: https://serde.rs/
[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html
[`IntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html
[`AtomicFixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/atomic/struct.AtomicFixedVec.html
