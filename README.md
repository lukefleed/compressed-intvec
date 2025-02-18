
# Compressed Integer Vector Library

[![rust](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml/badge.svg)](https://github.com/lukefleed/compressed-intvec)

The library provides a compressed representation for vectors of unsigned 64-bit integers by utilizing several variable‑length coding methods. It is engineered to offer both efficient compression and fast random access to individual elements.

## Overview

This library leverages several encoding schemes including Gamma, Delta, Exp‑Golomb, Zeta, Rice, and their parameterized variants. These codecs are implemented in the library [dsi-bitstream](https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/index.html) and are used to compress and decompress integers in the vector.

The key features include:

- **Multiple Codecs:** Choose between codecs like `GammaCodec`, `DeltaCodec`, `ExpGolombCodec`, and more can be found in the ['codecs'](src/codecs.rs) module.
- **Endian Flexibility:** Offers both big-endian and little-endian representations using `BEIntVec` and `LEIntVec` respectively.
- **Sampling Support:** Users may provide a sampling period to balance decoding speed and memory footprint.
- **Benchmarks and Tests:** Integrated benchmarks in benches and comprehensive tests in tests ensure reliability and performance.


## Usage Examples

### Creating a Big-Endian Compressed Vector

Using ExpGolombCodec with a runtime parameter:

```rust
use compressed_intvec::BEIntVec;
use compressed_intvec::codecs::DeltaCodec;

let input = vec![1, 5, 3, 1991, 42];
let intvec = BEIntVec::<DeltaCodec>::from(&input, 2).unwrap();

let value = intvec.get(3);
assert_eq!(value, Some(1991));

let decoded = intvec.into_vec();
assert_eq!(decoded, input);
```

### Creating a Little-Endian Compressed Vector

Using GammaCodec without extra codec parameters:

```rust
use compressed_intvec::LEIntVec;
use compressed_intvec::codecs::GammaCodec;

let input = vec![10, 20, 30, 40, 50];
let intvec = LEIntVec::<GammaCodec>::from(input, 2);

assert_eq!(intvec.get(2), Some(30));
```

## Codecs and Customization

The library provides several codecs in `codecs.rs`:

- **GammaCodec:** No extra runtime parameters. Ideal for data with moderate spread.
- **DeltaCodec:** Often used when values are clustered.
- **ExpGolombCodec:** Requires a runtime parameter; ideal for values with a wide range.
- **Parameterized Codecs:** Such as `ParamDeltaCodec` and `ParamGammaCodec` allow customization via compile‑time flags.

Each codec implements the `Codec` trait and encodes/decodes values at the bit level. See the inline documentation in codecs.rs for additional details.

### Choosing the Right Codec

The efficiency of a codec is highly dependent on the underlying data distribution, so selecting the appropriate codec is crucial for achieving optimal compression. Here are general guidelines to help you choose:

- **Skewed Distributions:** If the data is skewed, Rice coding is usually effective. In this case, set the Rice parameter to the floor of the base-2 logarithm of the mean value.
- **Power Law Distributions:** For data following a power law (e.g., \(P(x) \propto x^{-2}\)), Gamma coding is typically the most efficient.
- **Uniform Distributions:** When the data is uniformly distributed over the range \([0, u64::MAX)\), minimal binary coding offers the best performance.

For further details, refer to the literature on entropy coding.

---

### Why Choosing the Right Codec Matters: An Example

Consider a vector of `u64` values uniformly distributed in the range \([0, u64::MAX)\). The following function generates this vector:

```rust
fn generate_uniform_vec(size: usize, max: u64) -> Vec<u64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let uniform = Uniform::new(0, max).unwrap();
    (0..size).map(|_| uniform.sample(&mut rng)).collect()
}
```

The size of the vector before compression is measured as follows:

```rust
let input_vec = generate_uniform_vec(1000, u64::MAX);

println!("Size of the standard Vec<u64>");
let _ = input_vec.mem_dbg(DbgFlags::empty());
```

This outputs:

```
Size of the standard Vec<u64>
8024 B ⏺
```

Next, we compress the vector using both `MinimalBinaryCodec` and `DeltaCodec`:

```rust
let minimal_intvec = BEIntVec::<MinimalBinaryCodec>::from_with_param(&input_vec, 32, 10);
let delta_intvec = BEIntVec::<DeltaCodec>::from(&input_vec, 32);

println!("Size of the compressed IntVec with MinimalBinaryCodec");
minimal_intvec.mem_dbg(DbgFlags::empty());

println!("\nSize of the compressed IntVec with DeltaCodec");
delta_intvec.mem_dbg(DbgFlags::empty());
```

The compression results are:

```
Size of the compressed IntVec with MinimalBinaryCodec
832 B ⏺
528 B ├╴data
280 B ├╴samples
  0 B ├╴codec
  8 B ├╴k
  8 B ├╴len
  8 B ├╴codec_param
  0 B ╰╴endian

Size of the compressed IntVec with DeltaCodec
9584 B ⏺
9288 B ├╴data
 280 B ├╴samples
   0 B ├╴codec
   8 B ├╴k
   8 B ├╴len
   0 B ├╴codec_param
   0 B ╰╴endian
```

As shown, `MinimalBinaryCodec` is the most efficient for uniformly distributed data, reducing the size to 832 bytes—an order of magnitude smaller than the original vector. In contrast, `DeltaCodec` actually increases the size to 9584 bytes, as it is poorly suited for uniform distributions.

## Benchmarks

Benchmarks are provided to evaluate both the random access speed and space occupancy of compressed vectors.

- **Random Access Benchmarks:** Located in random_access.rs, these benchmarks measure the time required to access individual elements.
- **Space Occupancy Benchmarks:** Located in bench_size.rs, these benchmarks report the memory footprint of various codec configurations and compressed representations.

To run the benchmarks, execute:

```bash
cargo bench
```



The results are output to the terminal and also written to CSV files (e.g. `benchmark_space.csv`).




## Additional Resources

- Codec Documentation in [dsi-bitstream](https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/index.html)
