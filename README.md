# Compressed Integer Vectors

[![crates.io](https://img.shields.io/crates/v/compressed-intvec.svg)](https://crates.io/crates/compressed-intvec)
[![rust](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml/badge.svg)](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml)
[![docs](https://docs.rs/compressed-intvec/badge.svg)](https://docs.rs/compressed-intvec)
[![downloads](https://img.shields.io/crates/d/compressed-intvec)](https://crates.io/crates/compressed-intvec)
![license](https://img.shields.io/crates/l/compressed-intvec)
[![Line count](https://tokei.rs/b1/github/lukefleed/compressed-intvec?type=Rust,Python)](https://github.com/lukefleed/compressed-intvec)

A Rust library that provides space-efficient, in-memory representations for integer vectors. It offers two complementary data structures: [`FixedVec`], which uses fixed-width encoding for O(1) mutable and atomic access, and [`IntVec`], which uses variable-length instantaneous codes for high-ratio compression and amortized O(1) random access.

The library is designed to reduce the memory footprint of standard [`std::vec::Vec`] collections of integers while retaining performant access patterns.

## Core Structures: [`FixedVec`] and [`IntVec`]

The library provides two distinct vector types, each based on a different encoding principle. Choosing the right one depends on your specific use case, performance requirements, and data characteristics.

### [`FixedVec`]: Fixed-Width Encoding

Implements a vector where every integer occupies the same, predetermined number of bits.

*   **Key Features**:
    *   **O(1) Random Access**: The memory location of any element is determined by a direct bit-offset calculation, resulting in minimal-overhead access. With low bit widths (e.g., 8, 16, 32), [`FixedVec`] can be 2-3x faster than [`std::vec::Vec`] for random access.
    *   **Mutability**: Supports in-place modifications after creation through an API similar to [`std::vec::Vec`] (e.g., `push`, `set`, `pop`).
    *   **Atomic Operations**: Provides [`AtomicFixedVec`], a thread-safe variant that supports atomic read-modify-write operations for concurrent environments.

### [`IntVec`]: Variable-Width Encoding

Implements a vector using variable-length instantaneous codes (e.g., Gamma, Delta, Rice, Zeta) to represent each integer.

*   **Key Features**:
    *   **High Compression Ratios**: Achieves significant space savings for data with non-uniform distributions.
    *   **Automatic Codec Selection**: Can analyze the data to select the most effective compression codec automatically.
    *   **Amortized O(1) Random Access**: Enables fast random access by sampling the bit positions of elements at a configurable interval (`k`).

[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html
[`IntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html
[`AtomicFixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/atomic/struct.AtomicFixedVec.html
[`std::vec::Vec`]: https://doc.rust-lang.org/std/vec/struct.Vec.html

## Quick Start

The following examples shows some use cases for [`FixedVec`] and [`IntVec`]. All common types and traits are available through the [`prelude`].

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

### Example: `IntVec` for High-Ratio Compression

```rust
use compressed_intvec::prelude::*;

// Skewed data with a mix of small and large values.
let skewed_data: &[u64] = &[5, 8, 13, 1000, 7, 6, 10_000, 10, 2, 3];

// The builder can automatically select the best compression codec.
let intvec: LEIntVec = IntVec::builder()
    .codec(VariableCodecSpec::Auto)
    .build(skewed_data)
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

## Compression with [`IntVec`]

[`IntVec`] is the optimal choice when data that is not uniformly distributed and we want to minimize memory usage. It uses variable-length codes to represent integers.

The compression strategy is controlled by the [`VariableCodecSpec`] enum, passed to the builder.

### Choosing the Right Codec

For most use cases, the recommended strategy is [`VariableCodecSpec::Auto`], which analyzes the data to select the most space-efficient codec. However, you can also specify a codec explicitly based on your data characteristics.

| `VariableCodecSpec` Variant | Description & Encoding Strategy | Optimal Data Distribution |
| :--- | :--- | :--- |
| **`Auto`** | **Recommended default.** Analyzes the data to choose the best variable-length code, balancing build time and compression ratio. | Agnostic; adapts to the input data. |
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

### Automatic Selection with [`VariableCodecSpec::Auto`]

The `Auto` strategy removes the guesswork from codec selection. During the build phase, it analyzes the input data and selects the codec that offers the best compression ratio. This introduces a (non negligible) one-time cost for the analysis at construction time. Use the `Auto` codec when you want to create an [`IntVec`] once and read it many times, as the amortized cost of the analysis is negligible compared to the space savings and the performance of subsequent reads.

If you need to create multiple [`IntVec`] instances at run-time, consider using a specific codec that matches your data distribution to avoid the overhead of analysis.



[`FixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/struct.FixedVec.html
[`IntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html
[`VariableCodecSpec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/codec/enum.VariableCodecSpec.html
[`VariableCodecSpec::Auto`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/codec/enum.VariableCodecSpec.html#variant.Auto
[dsi-bitstream-codes]: https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/enum.Codes.html
[`mem-dbg`]: https://docs.rs/mem-dbg/latest/mem_dbg/

## [`IntVec`] Access Patterns

The access strategy for a compressed [`IntVec`] has significant performance implications. The library provides several methods, each optimized for a specific access pattern. Using the appropriate method is key to achieving high throughput.

### [`get_many`]: Batch Access from a Slice

For retrieving a batch of elements from a slice of indices.

*   **Mechanism**: This method sorts the provided indices to transform a random access pattern into a single, monotonic scan over the compressed data. This approach minimizes expensive bitstream seek operations and leverages data locality.

This should be your preferred method for any batch lookup when all indices are known and can be stored in a slice.

```rust
use compressed_intvec::prelude::*;
use rand::Rng; // For random number generation

let data: Vec<u64> = (0..10_000).collect();
let intvec: LEIntVec = IntVec::builder()
    .codec(VariableCodecSpec::Delta)
    .k(32)
    .build(&data)
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

Use when indices cannot be collected into a slice, for example due to memory constraints.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec: LEIntVec = IntVec::from_slice(&data).unwrap();

// Process indices from a streaming source, such as a range iterator.
let values: Vec<u64> = intvec.get_many_from_iter(500..505).unwrap();

assert_eq!(values, vec![500, 501, 502, 503, 504]);
```

### Dynamic Lookups: `IntVecReader` and `IntVecSeqReader`

There are interactive scenarios where lookup indices are not known in advance. The library provides two reader types to handle such cases:

#### [`IntVecReader`]: Stateless Random Access

A **stateless** reader for efficient, repeated random lookups.

*   **Mechanism**: Amortizes the setup cost of the bitstream reader across multiple calls. Each `get` operation performs an independent seek from the nearest sample point.

Optimal for sparse, unpredictable access patterns where there is no locality between consecutive lookups.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec: LEIntVec = IntVec::from_slice(&data).unwrap();

// Create a stateless reader for random access.
let mut reader = intvec.reader();
assert_eq!(reader.get(500).unwrap(), Some(500));
assert_eq!(reader.get(10).unwrap(), Some(10));
```

#### [`IntVecSeqReader`]: Stateful Sequential Access

A **stateful** reader optimized for access patterns with sequential locality.

*   **Mechanism**: Maintains an internal cursor. If a requested index is forward and within the same sample block, it decodes from the last position, avoiding a full seek.

Optimal for iterating through sorted or clustered indices where consecutive lookups are near each other.

```rust
use compressed_intvec::prelude::*;

let data: Vec<u64> = (0..10_000).collect();
let intvec: LEIntVec = IntVec::from_slice(&data).unwrap();

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

## Concurrency with [`AtomicFixedVec`]

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

## Memory Analysis

The library integrates [`mem-dbg`] to provide memory usage statistics, allowing you to compare the size of different encoding strategies easily. This is particularly useful for understanding the trade-offs between [`FixedVec`] and [`IntVec`] in terms of memory efficiency.

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
    let data = generate_random_vec(1_000_000, 1 << 20);

    println!("Size of the uncompressed Vec<u64>:");
    data.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE | DbgFlags::RUST_LAYOUT);

    // Create an IntVec with Gamma encoding.
    let gamma_intvec = LEIntVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&data)
        .unwrap();

    println!("\nSize of the IntVec with gamma encoding:");
    gamma_intvec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE | DbgFlags::RUST_LAYOUT);

    // Let the library analyze the data and choose the best codec.
    let auto_intvec = LEIntVec::builder()
        .codec(VariableCodecSpec::Auto)
        .build(&data)
        .unwrap();

    println!("\nSize of the IntVec with Auto encoding:");
    auto_intvec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE | DbgFlags::RUST_LAYOUT);
    println!("\nCodec selected by Auto: {:?}", auto_intvec.encoding());

    // Create a FixedVec with minimal bit width (20 bits)
    let fixed_intvec = LEFixedVec::builder()
        .bit_width(BitWidth::Minimal)
        .build(&data)
        .unwrap();

    println!("\nSize of the FixedVec with minimal bit width:");
    fixed_intvec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE | DbgFlags::RUST_LAYOUT);
}
```

The output displays memory breakdown for a 1,000,000-element vector of `u64` integers, uniformly distributed between 0 and 2<sup>20</sup>

- `Vec<u64>`: 80.02 kB (8 bytes per element)
- [`IntVec`] with Gamma encoding: 47.10 kB (41% reduction)
- [`IntVec`] with Auto codec selection: 28.33 kB (65% reduction, selected Zeta k=10)
- [`FixedVec`] with minimal bit width: 25.06 kB (69% reduction, 20 bits per element)

The memory analysis also shows the internal structure of each data type, including storage overhead for metadata, sampling structures, and encoding parameters.

```text
Size of the uncompressed Vec<u64>:
8.000 MB 100.00% ⏺

Size of the IntVec with gamma encoding:
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

Size of the IntVec with Auto encoding:
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
[`IntVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/variable/struct.IntVec.html
[`AtomicFixedVec`]: https://docs.rs/compressed-intvec/latest/compressed_intvec/fixed/atomic/struct.AtomicFixedVec.html

## Benchmarks

The library includes benchmarks for both [`FixedVec`] and [`IntVec`]. It also tests the performance against two other libraries implementation of a fixed-width vector: [`sux::BitFieldVec`] and [`succinct::IntVector`]. These benchmarks are in the `benches` directory and can be run via

```bash
cargo bench
```

The benchmarks measure the performance of random access, batch access, iter_access, and memory usage for various data distributions and vector sizes. For a visual representation of the random access performance, run the following command:

```bash
pip3 install -r python/requirements.txt
python3 python/plot.py --random-access
```

The resulting SVGs file will be saved in the `images` directory.

[`sux::BitFieldVec`]: https://docs.rs/sux/latest/sux/bits/bit_field_vec/index.htmll
[`succinct::IntVector`]: https://docs.rs/succinct/latest/succinct/int_vec/trait.IntVec.html