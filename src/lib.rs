//! # Compressed IntVec Library
//!
//! This library provides a compressed representation for vectors of unsigned 64-bit
//! integers using variable‑length coding methods. The key structure, [`IntVec`](src/intvec.rs),
//! compresses the data into a bitstream while storing sample offsets for fast random access.
//!
//! ## Features
//!
//! - **Multiple Codecs:** Use various encoding schemes such as Gamma, Delta, Exp‑Golomb,
//!   Zeta, Rice, and their parameterized variants. All codecs implement the
//!   [`Codec`](src/codecs.rs) trait.
//! - **Endianness Support:** Dedicated types are available for both big‑endian and little‑endian
//!   representations. See [`BEIntVec`](src/intvec.rs) and [`LEIntVec`](src/intvec.rs).
//! - **Memory Debugging:** Integration with the [`mem_dbg`](https://crates.io/crates/mem_dbg)
//!   crate helps in estimating the memory footprint of compressed data structures.
//!
//! ## Modules
//!
//! - **`codecs`** ([src/codecs.rs](src/codecs.rs)): Contains the generic [`Codec`](src/codecs.rs)
//!   trait and its implementations for various coding schemes. Each codec defines both
//!   encoding and decoding functionalities.
//! - **`intvec`** ([src/intvec.rs](src/intvec.rs)): Implements the main [`IntVec`](src/intvec.rs)
//!   structure, along with big‑endian and little‑endian variants, constructors, element retrieval,
//!   full decoding, and iterator support.
//!
//! ## Usage Example
//!
//! To compress a vector of integers using gamma coding in big‑endian format:
//!
//! ```rust
//! use compressed_intvec::BEIntVec;
//! use compressed_intvec::codecs::GammaCodec;
//!
//! let input = vec![1, 2, 3, 4, 5];
//! // Create a compressed vector with a sampling period of 2 (every 2nd value is stored as a sample)
//! let intvec = BEIntVec::<GammaCodec>::from(input.clone(), 2).unwrap();
//!
//! // Random access: decode the 4th element
//! assert_eq!(intvec.get(3), Some(4));
//!
//! // Decode the full original vector (note: this may be expensive)
//! assert_eq!(intvec.into_vec(), input);
//! ```
//!
//! For little‑endian encoding, use [`LEIntVec`](src/intvec.rs) similarly:
//!
//! ```rust
//! use compressed_intvec::LEIntVec;
//! use compressed_intvec::codecs::GammaCodec;
//!
//! let input = vec![10, 20, 30, 40];
//! let intvec = LEIntVec::<GammaCodec>::from(input.clone(), 2).unwrap();
//! assert_eq!(intvec.get(2), Some(30));
//! ```
//!
//! ## Codecs Overview
//!
//! The [`codecs`](src/codecs.rs) module provides implementations of the [`Codec`](src/codecs.rs)
//! trait using different encoding algorithms:
//!
//! - **GammaCodec:** Uses gamma coding without requiring extra runtime parameters.
//! - **DeltaCodec:** Uses delta coding, likewise without extra parameters.
//! - **ExpGolombCodec:** Requires an extra parameter (e.g. a parameter `k`) for encoding/decoding.
//! - **ZetaCodec:** Uses additional runtime parameters ζ.
//! - **RiceCodec:** Uses a Rice parameter for encoding/decoding. Ideal for skewed distributions, you likely want to the set the rice prameter as the floor of the base‑2 logarithm of the mean value of the data.
//! - **MinimalBinaryCodec:** A minimal binary code with upper bound `u > 0` ([truncated binary encoding](https://en.wikipedia.org/wiki/Truncated_binary_encoding)). This is optimal for uniformly distributed data in the range [0, u).
//! - **Parameterized Codecs:** Variants like `ParamZetaCodec`, `ParamDeltaCodec`, and `ParamGammaCodec`
//!   use compile‑time flags (e.g. lookup table usage) to control internal behavior. If you choose to use this, consider that the tables are hardcoded in the library [dsi-dsi-bitstream](https://crates.io/crates/dsi-dsi-bitstream) (in the file `_tables.rs`). If you want to change the tables, you need to clone the repository and change the tables with the python script `gen_code_tables.py` and then compile the library. See
//!
//! **Theoretical Consideration:** The efficiency of each codec is closely tied to the data distribution.
//!
//! - **Skewed Distributions:** When the data is skewed, Rice coding generally outperforms Gamma coding.
//!   In these cases, choose the Rice parameter as the floor of the base‑2 logarithm of the mean value.
//! - **Power Law Distributions:** If the data roughly follows a power law (i.e. P(x) ∝ x⁻²),
//!   Gamma coding is often the optimal choice.
//! - **Uniform Distributions:** For data uniformly distributed across the range [0, u64::MAX),
//!  minimal binary coding is the most efficient.
//!
//! For more in‐depth information, please consult the literature on entropy coding.
//!
//! ## The `IntVec` Structure
//!
//! The [`IntVec`](src/intvec.rs) type encapsulates:
//!
//! - **Compressed Data:** A [`Vec<u64>`](std::vec::Vec) containing the compressed bitstream.
//! - **Sampling Offsets:** A [`Vec<usize>`] tracking bit offsets for every k‑th value to allow
//!   fast random access.
//! - **Codec Parameters:** Any additional parameters required by the selected codec.
//!
//! The main functionalities include:
//!
//! - **Construction:** Use `from_with_param` (or the convenience method `from` for codecs without
//!   runtime parameters) to build a compressed vector.
//! - **Random Access:** Retrieve a value using `get(index)` by decoding only the required part of the stream.
//! - **Full Decoding:** Convert back to a standard vector with `into_vec()`. This operation decodes every value.
//! - **Iteration:** The `iter()` method returns an iterator that decodes values on the fly.
//!
//! ## Crate Features
//!
//! - **`mem-debug` (default):** Enables memory debugging features using the [`mem_dbg`](https://crates.io/crates/mem_dbg) crate.
//! - **`serde` (disabled by default):** Enables serialization/deserialization support using the [`serde`](https://crates.io/crates/serde) crate.

pub mod codecs;
pub mod intvec;
