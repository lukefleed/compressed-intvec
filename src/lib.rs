//!  # Compressed Integer Vector Library
//!
//! [![Crates.io](https://img.shields.io/crates/v/compressed-intvec.svg)](https://crates.io/crates/compressed-intvec)
//! [![CI](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml/badge.svg)](https://github.com/lukefleed/compressed-intvec/actions/workflows/rust.yml)
//! [![Docs](https://docs.rs/compressed-intvec/badge.svg)](https://docs.rs/compressed-intvec)
//!
//! This Rust library provides efficient compression for vectors of `u64` integers by leveraging
//! variable-length coding from the [dsi-bitstream](https://docs.rs/dsi-bitstream) crate. It offers
//! rapid random access via configurable sampling, thus striking a balance between speed and memory
//! usage. Users can select between big-endian (`BEIntVec`) and little-endian (`LEIntVec`) encoding
//! based on their interoperability requirements.
//!
//! ## Features
//!
//! - **Efficient Compression:** Utilizes codecs such as Gamma, Delta, and Rice to compress data effectively.
//! - **Fast Random Access:** Provides O(1) access by storing periodic full positions (sampling).
//! - **Memory Profiling:** Integrates with [`mem-dbg`](https://crates.io/crates/mem-dbg) for memory analysis.
//! - **Flexible Codecs:** Offers a variety of codecs to best suit the distribution of your data.
//!
//! The sampling parameter dictates how often a full positional index is stored to accelerate random access.
//! A larger sampling parameter reduces memory overhead at the cost of slightly increased access time.
//! For many datasets, a value such as `32` serves as an effective trade-off.
//!
//! ### Example: Gamma Coding
//!
//! Gamma coding, introduced by Elias in the 1960s, is a universal code that represents an integer
//! using a unary prefix followed by its binary representation. For any integer `x`, the code length
//! is $O(\log x)$, just marginally longer than its binary form. For instance, the integer `9` is encoded as `0001001`.
//!
//! ```rust
//! use compressed_intvec::intvec::BEIntVec;
//! use compressed_intvec::codecs::GammaCodec;
//!
//! let vec = vec![1, 3, 6, 8, 13, 3];
//! let sampling_param = 2; // A modest sampling parameter for a small vector
//! let compressed_be = BEIntVec::<GammaCodec>::from(&vec, sampling_param).unwrap();
//!
//! assert_eq!(compressed_be.get(3), 8);
//!
//! for (i, val) in compressed_be.iter().enumerate() {
//!     assert_eq!(val, vec[i]);
//! }
//! ```
//!
//! Alternatively, for little-endian encoding:
//!
//! ```rust
//! use compressed_intvec::intvec::LEIntVec;
//! use compressed_intvec::codecs::GammaCodec;
//!
//! let vec = vec![1, 3, 6, 8, 13, 3];
//! let compressed_le = LEIntVec::<GammaCodec>::from(&vec, 2).unwrap();
//!
//! for (i, val) in compressed_le.iter().enumerate() {
//!     assert_eq!(val, vec[i]);
//! }
//! ```
//!
//! ## Supported Codecs
//!
//! The library offers several codecs for compressing integer vectors:
//!
//! | Codec Name           | Description                                                                                                                                                                  |
//! | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
//! | `GammaCodec`         | Gamma coding without requiring extra runtime parameters.                                                                                                                   |
//! | `DeltaCodec`         | Delta coding without additional parameters.                                                                                                                                |
//! | `ExpGolombCodec`     | Exp-Golomb coding, which requires an extra parameter (e.g., a parameter `k`).                                                                                               |
//! | `ZetaCodec`          | Zeta coding with additional runtime parameters ζ.                                                                                                                          |
//! | `RiceCodec`          | Rice coding, ideal for skewed distributions. The Rice parameter is often chosen as the floor of $\log_2$ of the mean of the values.                                         |
//! | `MinimalBinaryCodec` | Minimal binary (truncated binary) coding for values in `[0, u)`. Optimal for uniformly distributed data. See [truncated binary encoding](https://en.wikipedia.org/wiki/Truncated_binary_encoding). |
//! | `ParamZetaCodec`     | A parameterized variant of Zeta coding using compile-time flags.                                                                                                             |
//! | `ParamDeltaCodec`    | A parameterized variant of Delta coding using compile-time flags.                                                                                                            |
//! | `ParamGammaCodec`    | A parameterized variant of Gamma coding using compile-time flags.                                                                                                            |
//!
//! For codecs requiring extra parameters, use the `from_with_params` method. For example, with Rice coding:
//!
//! ```rust
//! use compressed_intvec::intvec::BEIntVec;
//! use compressed_intvec::codecs::RiceCodec;
//!
//! let vec = vec![1, 3, 6, 8, 13, 3];
//! let rice_param = 3; // Example Rice parameter
//! let sampling_param = 2;
//! let compressed = BEIntVec::<RiceCodec>::from_with_param(&vec, sampling_param, rice_param).unwrap();
//! ```
//!
//! Choosing the correct codec is essential for optimal compression. Rice coding is effective for skewed data,
//! whereas Minimal Binary coding excels with uniformly distributed values.
//!
//! ## Endianness
//!
//! The library supports both big-endian (`BEIntVec`) and little-endian (`LEIntVec`) formats. Although the
//! choice of endianness affects the byte order of the compressed data, it does not impact performance.
//!
//! ## Memory Analysis
//!
//! Both the big-endian and little-endian implementations support the [MemDbg/MemSize](https://docs.rs/mem-dbg/) traits
//! from the [`mem-dbg`](https://crates.io/crates/mem-dbg) crate. For example, running
//! `mem_dbg(DbgFlags::empty())` on a large `BEIntVec` might produce:
//!
//! ```bash
//! 11536 B ⏺
//! 10864 B ├╴data
//!   656 B ├╴samples
//!     0 B ├╴codec
//!     8 B ├╴k
//!     8 B ├╴len
//!     0 B ├╴codec_param
//!     0 B ╰╴endian
//! ```
//!
//! Consider compressing a vector of uniformly distributed `u64` values over `[0, u64::MAX)` and measure the memory usage.
//!
//! ```no_run
//! use rand::distr::{Uniform, Distribution};
//! use rand::SeedableRng;
//! use mem_dbg::{MemDbg, DbgFlags};
//!
//! fn generate_uniform_vec(size: usize, max: u64) -> Vec<u64> {
//!     let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//!     let uniform = Uniform::new(0, max).unwrap();
//!     (0..size).map(|_| uniform.sample(&mut rng)).collect()
//! }
//! let input_vec = generate_uniform_vec(1000, u64::MAX);
//!
//! println!("Size of the standard Vec<u64>");
//! input_vec.mem_dbg(DbgFlags::empty());
//! ```
//!
//! This outputs:
//!
//! ```bash
//! Size of the standard Vec<u64>
//! 8024 B ⏺
//! ```
//!
//! Next, let's compress the vector using `MinimalBinaryCodec` and `DeltaCodec` with a sampling parameter of `32`:
//!
//! ```no_run
//! use compressed_intvec::intvec::BEIntVec;
//! use compressed_intvec::codecs::{MinimalBinaryCodec, DeltaCodec};
//! use mem_dbg::{MemDbg, DbgFlags};
//!
//! fn generate_uniform_vec(size: usize, max: u64) -> Vec<u64> {
//!     let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//!     let uniform = Uniform::new(0, max).unwrap();
//!     (0..size).map(|_| uniform.sample(&mut rng)).collect()
//! }
//! use rand::distr::{Uniform, Distribution};
//! use rand::SeedableRng;
//!
//! let input_vec = generate_uniform_vec(1000, u64::MAX);
//!
//! let minimal_intvec = BEIntVec::<MinimalBinaryCodec>::from_with_param(&input_vec, 32, 10).unwrap();
//! let delta_intvec = BEIntVec::<DeltaCodec>::from(&input_vec, 32).unwrap();
//!
//! println!("Size of the standard Vec<u64>");
//! input_vec.mem_dbg(DbgFlags::empty());
//!
//! println!("\nSize of the compressed IntVec with MinimalBinaryCodec");
//! minimal_intvec.mem_dbg(DbgFlags::empty());
//!
//! println!("\nSize of the compressed IntVec with DeltaCodec");
//! delta_intvec.mem_dbg(DbgFlags::empty());
//! ```
//!
//! This code snippet should output:
//!
//! ```bash
//! Size of the standard Vec<u64>
//! 8024 B ⏺
//!
//! Size of the compressed IntVec with MinimalBinaryCodec
//! 832 B ⏺
//! 528 B ├╴data
//! 280 B ├╴samples
//!   0 B ├╴codec
//!   8 B ├╴k
//!   8 B ├╴len
//!   8 B ├╴codec_param
//!   0 B ╰╴endian
//!
//! Size of the compressed IntVec with DeltaCodec
//! 9584 B ⏺
//! 9288 B ├╴data
//!  280 B ├╴samples
//!    0 B ├╴codec
//!    8 B ├╴k
//!    8 B ├╴len
//!    0 B ├╴codec_param
//!    0 B ╰╴endian
//! ```
//!
//! As demonstrated, `MinimalBinaryCodec` can reduce the memory footprint dramatically, whereas
//! `DeltaCodec` may lead to an increased size when applied to uniformly distributed data.
pub mod codecs;
pub mod intvec;
