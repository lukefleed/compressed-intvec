# Change Log

All notable changes to this project will be documented in this file.

## [0.6.0] - Unreleased

### BREAKING

*   **`IntVec` renamed to `VarVec`**: The primary type in the `variable`
    module is now `VarVec`. All related type aliases have been updated
    (e.g., `LEIntVec` → `LEVarVec`, `SIntVec` → `SVarVec`). The old
    names are kept as deprecated aliases for one release cycle.

*   **`VariableCodecSpec` renamed to `Codec`**: The codec enum for
    `VarVec` and `SeqVec` is now `variable::Codec`. The old name
    `VariableCodecSpec` is kept as a deprecated alias.

*   **`dsi-bitstream` upgraded from 0.5 to 0.9**: Codec enum variants
    with parameters changed from struct syntax to tuple syntax
    (e.g., `Codes::Zeta { k: 3 }` → `Codes::Zeta(3)`). `MemWordReader`
    now requires an `INF` const generic for infallible reads.
    `anyhow::Error` in dispatch paths replaced by `DispatchError`.

*   **Serde format changed for `VarVec` and `SeqVec`**: The `encoding`
    field is now serialized via `Display`/`FromStr` (e.g., `"Zeta(3)"`)
    instead of the previous struct-based JSON format. Data serialized
    with 0.5.x cannot be deserialized with 0.6.0.

*   **`rand` upgraded from 0.9 to 0.10**: The `Rng` trait methods
    `random_range` and `random` moved to `RngExt`. Affects downstream
    code using `rand` types from the public API or benchmarks.
    `rand_distr` upgraded from 0.5 to 0.6 in lockstep.

*   **`UnsignedInt` removed from `Word` trait bounds**: Resolves method
    ambiguity with `num-primitive` (brought in by dsi-bitstream 0.9).
    Code relying on `UnsignedInt` methods through the `Word` bound must
    now bound on `UnsignedInt` explicitly.

### New

*   **`seq` module**: New `SeqVec` type for compressed sequences of
    variable-length integer vectors. Designed for adjacency lists,
    inverted indices, and document term lists. Bit offsets are stored at
    sequence boundaries, so `get(i)` returns a lazy `SeqIter` in O(1).
    Includes `SeqVecBuilder`, `SeqVecFromIterBuilder`, `SeqIter`,
    `SeqVecIter` (consuming), `SeqVecReader` (reusable random access),
    `SeqVecSlice` (zero-copy view), serde support, and full parallel
    iteration via rayon.

*   **`SeqVec` optional stored lengths**: `SeqVecBuilder` accepts a
    `lengths_capacity_hint` to pre-allocate a secondary `FixedVec`
    storing the length of each sequence. When present, `SeqIter`
    implements `ExactSizeIterator` and iteration is faster.

*   **`SeqVec` parallel methods**: `par_iter()`, `par_decode_many()`,
    `par_for_each()`, `par_for_each_many()`, and `par_for_each_reduce()`
    provide zero-allocation parallel access over sequences.
    `par_for_each_reduce` accumulates results across threads with a
    user-supplied fold/reduce pair.

*   **`SeqVecSlice` zero-copy view**: `SeqVec::slice(range)` and
    `SeqVec::split_at(mid)` return `SeqVecSlice` without copying data.
    `SeqVecSlice` supports iteration, random access, binary search, and
    `IntoIterator`.

*   **`sfixed_vec!` and `sseq_vec!` macros**: Convenience macros for
    constructing signed `FixedVec` and `SeqVec` instances. Complement
    the existing `fixed_vec!` and `seq_vec!` macros.

*   **`CodecWriter` internal writer** (`src/common/codec_writer.rs`):
    Shared dispatch writer for variable-length codes used by both
    `VarVec` and `SeqVec` builders. Mirrors the existing `CodecReader`
    pattern for symmetric hot-path dispatch.

*   **Shared serde utilities** (`src/common/serde.rs`): Common
    serialization helpers (bit-buffer serde, length list serde) shared
    between `VarVec` and `SeqVec`, replacing duplicated proxy code.

### Changed

*   Upgraded `dsi-bitstream` from 0.5.0 to 0.9.0. Key upstream changes:
    compile-time table validation, native `Serialize`/`Deserialize` for
    `Codes`, `Codes::canonicalize()`, `DispatchError` replacing
    `anyhow::Error`, `MemWordReader` with `INF` const generic.

*   Upgraded `rand` from 0.9 to 0.10 and `rand_distr` from 0.5 to 0.6.

*   Removed the `CodesSerde` proxy enum previously in
    `src/common/serde.rs`. `Codes` now implements serde natively in
    dsi-bitstream 0.9; the proxy is no longer needed.

*   Codec resolution in `VarVec` and `SeqVec` builders now analyzes
    iterators directly (single-pass) rather than collecting data first,
    reducing peak memory during construction with `Codec::Auto`.

*   `SeqVec` sequence access methods renamed from `get`/`get_vec` to
    `decode`/`decode_vec` for clarity. The old names were never stable.

### Improved

*   Codec auto-selection now calls `Codes::canonicalize()` to ensure
    canonical forms (e.g., `Golomb(2^n)` → `Rice(n)`,
    `Zeta(1)` → `Gamma`).

*   `SeqIter` fast-path when sequence lengths are stored: skips
    bit-counting and provides an exact `size_hint` with zero decoding
    overhead for `len()`.

*   `SeqVecReader` index calculations use bit-shift when the sampling
    rate is a power of two, reducing arithmetic overhead per access.

*   Parallel iteration in `VarVec` and `SeqVec` pre-allocates decode
    buffers based on known sequence lengths, reducing allocations.

*   All public iterator types now implement `FusedIterator`.

*   All public types now implement `Debug`.

### Fixed

*   **`FixedVecIntoIter` undefined behaviour**: The consuming iterator
    held a transmuted `'static` reference to a stack-local `FixedVec`
    that was subsequently partially moved, leaving a dangling pointer.
    The `FixedVec` is now heap-allocated via `Box` before transmuting,
    giving it a stable address for the lifetime of the iterator.

*   **`SeqVecBuilder` error handling**: `build()` now propagates errors
    from sequence-length validation correctly instead of silently
    discarding them.

*   **`seq_vec!()` empty macro**: The empty invocation `seq_vec!()`
    previously produced an invalid state; it now correctly produces a
    `SeqVec` with zero sequences.

## [0.5.1] - 2025-09-16

### New

*   Added a new `arch-dependent-storable` feature flag to enable `Storable` implementations for `usize` and `isize` in `variable::IntVec`. This allows for creating vectors of architecture-dependent types, but sacrifices data portability guarantees across platforms with different pointer widths.
*   Added a new benchmark suite (`bench_random_write`) and corresponding plotting scripts to measure and compare random write performance for `fixed::FixedVec` against other libraries.

### Improved

*   Optimized `FixedVec::set_unchecked` for significantly improved write performance. The implementation now includes a fast path for elements with a bit width equal to the storage word size and uses `split_at_mut_unchecked` to reduce bounds checking overhead for writes that span word boundaries.
*   Updated crate description and keywords in `Cargo.toml` for better discoverability and clarity on crates.io.
*   Replaced the `num_cpus` dependency with `std::thread::available_parallelism`.

### Fixed

*   Corrected minor type errors in the `variable` module test suite.

## [0.5.0] - 11-08-2025

This release introduces a fundamental architectural restructuring of the library, splitting the implementation into two distinct data structures: `FixedVec` for fixed-width encoding and `IntVec` for variable-width encoding. This separation enables significant new features, including full mutability, atomic operations, and improved performance, but constitutes a major breaking change to the public API.

### BREAKING

*   **Architectural Separation of Encodings**: The core `IntVec` structure has been split into two distinct implementations based on the encoding strategy.
    *   **`fixed::FixedVec`**: A new data structure exclusively for fixed-width integer encoding. It provides a mutable, `Vec`-like API, O(1) random access, and a thread-safe atomic variant (`AtomicFixedVec`). This component replaces the previous `CodecSpec::FixedLength` functionality.
    *   **`variable::IntVec`**: The refactored successor to the original `IntVec`, now located in the `variable` module. It is dedicated exclusively to variable-length instantaneous codes (e.g., Gamma, Delta, Zeta). It remains immutable after creation.

*   **Module and API Restructuring**: The project's module structure has been fundamentally changed.
    *   The top-level `intvec` and `sintvec` source modules have been removed. All functionality is now organized under the `fixed` and `variable` modules.
    *   The `CodecSpec` enum has been renamed to `VariableCodecSpec` and is now located in `src/variable/codec.rs`. It is used only for `variable::IntVec`.
    *   `fixed::FixedVec` is now configured using a new `BitWidth` enum, which is distinct from `VariableCodecSpec`.

*   **Removal of `SIntVec` and Introduction of Generic Types**: The standalone `SIntVec` struct has been removed.
    *   Both `FixedVec` and `IntVec` are now fully generic over all primitive integer types (e.g., `u8`, `i16`, `u32`, `i64`).
    *   Signed integer support is now provided directly through the `Storable` trait, which transparently handles ZigZag encoding.
    *   New type aliases such as `SFixedVec<T>` (for signed `FixedVec`) and `SIntVec<T>` (for signed `IntVec`) are provided.

*   **Builder API Modification**: The builder pattern has been changed for all vector types. The input data slice is now passed to the final `.build()` method instead of the constructor.
    *   **Old**: `LEIntVec::builder(&data).codec(...).build()`
    *   **New**: `LEIntVec::builder().codec(...).build(&data)`

*   **Codec Analysis Behavior Change**: The analysis strategy for `VariableCodecSpec::Auto` has been changed.
    *   It now analyzes the **entire** input dataset to determine the optimal codec, whereas the previous implementation used a 10,000-element sample for large datasets.
    *   This change improves compression ratio accuracy at the cost of increased construction time for large inputs.

### New

*   **`fixed::FixedVec` Data Structure**: Introduced a new vector implementation for fixed-width integer encoding, located in the `fixed` module.
    *   **Mutable API**: Provides a mutable interface including methods such as `push`, `pop`, `set`, `insert`, `remove`, `resize`, `map_in_place`, and `fill`.
    *   **Generic Implementation**: The structure is generic over the element type `T` (e.g., `u16`, `i32`), the storage word `W` (e.g., `u64`, `usize`), the endianness `E`, and the backing buffer `B` (`Vec<W>` or `&[W]`).
    *   **Zero-Copy Slicing**: Added an extensive API for creating immutable and mutable zero-copy views (`slice`, `split_at`, `chunks`, `windows`).
    *   **Unaligned Access**: A new method, `get_unaligned_unchecked`, was added for random access via unaligned memory reads.
    *   **Convenience Constructors**: Implements `FromIterator`, `TryFrom<&[T]>`, and is supported by a new `fixed_vec!` macro.

*   **`fixed::atomic::AtomicFixedVec` Data Structure**: Added a thread-safe variant of `FixedVec` for concurrent applications.
    *   **Atomic Operations**: Provides an API analogous to standard atomic types, including `load`, `store`, `swap`, `compare_exchange`.
    *   **Atomic RMW Operations**: Implements atomic Read-Modify-Write (RMW) methods such as `fetch_add`, `fetch_sub`, `fetch_and`, `fetch_or`, `fetch_xor`, `fetch_max`, and `fetch_min`.
    *   **Hybrid Atomicity Model**: Utilizes lock-free atomic instructions for elements fully contained within a single `u64` storage word. For elements that span two words, it uses a striped locking mechanism to ensure atomicity without a global lock.
    *   **Parallel Mutation**: Supports parallel in-place modification via a `par_iter_mut` method.
    *   **Construction**: Supported by its own builder and the `atomic_fixed_vec!` macro.

*   **Generic Integer Type Support**: Both `FixedVec` and `IntVec` now support all primitive integer types (e.g., `u8`, `i16`, `u32`, `i64`). This is managed by the `Storable` trait, which abstracts the conversion to and from the underlying storage words.

*   **`variable::IntVecSeqReader`**: Introduced a new stateful reader for `variable::IntVec`, optimized for access patterns with high locality.
    *   **Stateful Cursor**: The reader maintains an internal cursor of its current decoding position.
    *   **Optimized Forward Decoding**: If a requested index is at or after the cursor's position and within the same sample block, the reader decodes forward from its last position, avoiding a seek operation. For non-sequential access, it falls back to a seek-based approach.

*   **Convenience Macros**:
    *   **`fixed_vec!`**: A new macro for `vec!`-like initialization of `fixed::FixedVec`.
    *   **`atomic_fixed_vec!`**: A new macro for `vec!`-like initialization of `fixed::atomic::AtomicFixedVec`.
    *   The `int_vec!` and `sint_vec!` macros have been updated to construct `variable::IntVec` instances.

### Improved

*   **`variable::IntVec` Reader Robustness**: The internal codec dispatcher for `variable::IntVec` has been re-implemented with a hybrid strategy. It now uses a fast, function-pointer-based path for common codecs and parameters, but includes a `match`-based fallback path for less common parameterizations. This change guarantees that any validly constructed `IntVec` can be read without panicking due to an unsupported codec configuration.

*   **API Unification and Ergonomics**:
    *   The `Storable` trait now provides a unified mechanism for handling both signed and unsigned integer types across `FixedVec` and `IntVec`, removing the need for a separate `SIntVec` struct and resulting in a more consistent API.
    *   The `prelude` module has been expanded to export all common types, traits, and aliases from both the `fixed` and `variable` modules, simplifying imports.

*   **Benchmarking Suite**: The benchmarking infrastructure has been significantly expanded and restructured.
    *   Benchmarks are now organized into `fixed` and `variable` directories, with new suites covering atomic operations, mutable operations, and various access patterns (random, sorted, clustered).
    *   Performance is now compared against external libraries (`sux`, `succinct`) to provide a clearer context for the library's performance characteristics.

*   **`serde` Implementation**: The `serde` implementations have been updated to support the new `FixedVec`, `AtomicFixedVec`, and restructured `IntVec` types, ensuring correct serialization and deserialization for the new architecture.

## [0.4.0] - 2025-07-27

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
