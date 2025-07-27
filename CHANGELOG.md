# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2025-07-27

This is a release representing a complete architectural overhaul of the
library. It introduces a more ergonomic and powerful builder-based API, adds
support for signed integers and parallel processing, and provides
intelligent, automatic codec selection. These changes result in significant
usability and performance gains but include foundational breaking changes.

### New

*   **Automatic Codec Selection (`CodecSpec::Auto`)**: The builder can now
    intelligently select the most space-efficient compression codec for a given
    dataset. When `CodecSpec::Auto` is used, the builder performs a statistical
    analysis on a sample of the input data to choose the optimal variable-length
    code (e.g., Gamma, Delta, Zeta). This eliminates the need for manual tuning
    and ensures excellent compression ratios out-of-the-box.

*   **Signed Integer Vector (`SIntVec`)**: Introduced `SIntVec`, a new data
    structure specifically designed for compressing vectors of `i64`. It
    transparently uses ZigZag encoding to map signed integers to unsigned
    integers, allowing standard compression codes to work efficiently on data
    distributions centered around zero.

*   **High-Performance Batch Access Methods**: Introduced two new methods for
    efficiently retrieving multiple elements at once:
    *   `get_many()`: An optimized sequential method for batch lookups.
      For variable-length codes, it sorts the requested indices to perform a
      single, monotonic forward scan over the data, which minimizes expensive
      seek operations and avoids redundant decoding.
    *   `par_get_many()`: A parallel version of `get_many` (enabled by the
      `parallel` feature) that distributes lookups across multiple CPU cores,
      offering significant speedups for large batches of indices.

*   **Parallel Processing (`parallel` feature)**: Added a `parallel` feature,
    enabled by default, which leverages the Rayon library to accelerate
    operations on multi-core systems. In addition to `par_get_many`, this
    feature also provides:
    *   `par_iter()`: A parallel iterator for high-throughput full-vector
      decompression, which is particularly effective with computationally
      intensive codecs.

*   **Iterator-Based Builder**: Introduced `IntVec::from_iter_builder` to
    construct an `IntVec` directly from a streaming iterator. This approach is
    highly memory-efficient, making it suitable for datasets that are too large
    to fit into memory. It requires manual codec parameter specification, as
    data cannot be pre-analyzed.

*   **Stateful Reader (`IntVecReader`)**: Added `IntVec::reader()`, which returns
    a reusable `IntVecReader`. This stateful reader is designed for efficient
    dynamic random access, as it amortizes the setup cost of the bitstream
    reader across multiple `get()` calls, making it ideal for access patterns
    where lookup indices are not known in advance.

*   **Prelude Module**: Added a new `prelude` module (`compressed_intvec::prelude::*`)
    to simplify imports of the most commonly used types and traits, such as
    `LEIntVec`, `SIntVec`, and `CodecSpec`.

### Changed

*   **BREAKING: Complete API Redesign with Builder Pattern**: The core API has
    been fundamentally redesigned around a builder pattern for ergonomics and flexibility.
    *   The previous `from()` and `from_with_param()` methods have been entirely
        replaced by `IntVec::builder()`.
    *   Codec selection is no longer performed with generic type parameters
        (e.g., `LEIntVec<GammaCodec>`). Instead, the compression strategy is now
        specified at runtime via the `builder.codec(CodecSpec)` method. This
        change is what enables dynamic and automatic codec selection.

*   **BREAKING: Project Structure Refactoring**: The `src` directory has been
    restructured.
    *   Core logic is now split into dedicated modules: `src/intvec/`,
        `src/sintvec/`, and `src/codec_spec.rs`.
    *   The internal implementation of `IntVec` is further organized into
        `builder.rs`, `iter.rs`, `reader.rs`, `parallel.rs`, and `serde.rs`.

*   **Dependencies**: Updated key dependencies, including `dsi-bitstream` to
    version `0.5.0` and `mem_dbg` to `0.3.0`. Added `rayon` as a core dependency
    for the new, default-enabled `parallel` feature.

### Improved

*   **`FixedLength` Encoding Strategy**: The new `CodecSpec::FixedLength` replaces
    the old `MinimalBinaryCodec`. This new implementation is more powerful and
    explicit:
    *   It can now automatically detect the minimum required bit width for a
      dataset when `num_bits: None` is specified.
    *   It is now a distinct encoding strategy, separate from the DSI-based
      codes, with a highly optimized O(1) access path that does not require a
      sampling table.

*   **Memory-Efficient Sample Storage**: For bit-level encodings, the `IntVec`
    now stores sample offsets in a `Vec<u32>` if the total bit-length of the
    data is less than `u32::MAX`, falling back to `Vec<u64>` only when strictly
    necessary. This reduces memory overhead for small to medium-sized vectors.

*   **Robust Serde Implementation**: The `serde` implementation is now handled
    manually using serializable proxy ("shadow") structs. This makes
    serialization more robust, removes previous limitations, and decouples the
    library from the `dsi-bitstream` dependency, which does not provide `serde`
    traits for its `Codes` enum.

*   **Comprehensive Benchmarking Suite**: The benchmarking infrastructure has been
    completely revamped to provide more accurate performance
    data.
    *   Benchmarks are now run against multiple data distributions (Uniform,
        Geometric, Power-Law) to provide a cleaner view of codec performance.
    *   A new benchmark, `bench_parallel`, specifically measures the performance
        and scalability of the new parallel access methods.
    *   Plotting scripts have been consolidated into a single `python/plot.py`
        that generates interactive and static plots directly from Criterion's
        JSON output, removing the need for intermediate CSV files for most
        benchmarks (except for `bench_size`)

### Removed

*   **BREAKING: Generic Codec API**: The old API, which relied on generic codec
    structs (e.g., `GammaCodec`, `DeltaCodec`, `RiceCodec`, `MinimalBinaryCodec`)
    and the associated `Codec` trait, has been completely removed in favor of the
    new builder pattern and the `CodecSpec` enum.

### Fixed

*   **Fixed `serde` Serialization**: The `serde` implementation now correctly
    serializes and deserializes the internal state of `IntVec` and `SIntVec`,
    ensuring that all codec parameters and data are preserved accurately.
