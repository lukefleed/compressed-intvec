# **Technical Roadmap: `fixed` Module Evolution**


## **Core Structure Refactoring: Unified Genericity**

**Objective**: To establish a single, fully generic `FixedVec` data structure that serves as the foundation for all subsequent functionality. This phase focuses on abstracting the core components—element type, storage word, endianness, and backend—through a trait-based design.

#### **1.1. Abstraction Traits**

The core of the generic design will be two primary traits that decouple the user-facing data type from the internal storage representation. These will reside in a new `src/fixed/traits.rs` module.

*   **`trait Word`**:
    *   **Concept**: This trait abstracts over the primitive unsigned integer types used for the underlying bit buffer. It establishes a contract for what constitutes a "machine word" for storage purposes.
    *   **Technical Specification**:
        ```rust
        pub trait Word:
            common_traits::UnsignedInt
            + Copy
            + Send + Sync
            + std::fmt::Debug
            + 'static
        {
            const BITS: usize;
        }
        ```
    *   **Implementation**: Provide `impl Word` for `u8`, `u16`, `u32`, `u64`, `u128`, and `usize`. The `BITS` constant will be defined using `std::mem::size_of::<Self>() * 8`.

*   **`trait Storable<W: Word>`**:
    *   **Concept**: This trait defines the bidirectional conversion between the user-facing element type `T` and the storage `Word` type `W`. It is the mechanism that enables the `FixedVec` to transparently handle signed/unsigned data, different integer sizes, and ZigZag encoding.
    *   **Technical Specification**:
        ```rust
        pub trait Storable<W: Word>: Copy + Sized {
            fn into_word(self) -> W;
            fn from_word(word: W) -> Self;
        }
        ```
    *   **Implementation**:
        *   **For Unsigned Types (`uN`)**: The implementation will be generic. `into_word` will perform a cast (`self.into()`), and `from_word` will use `word.try_into()`. This requires bounds such as `W: From<T> + TryInto<T>`.
        *   **For Signed Types (`iN`)**: The implementation will encapsulate the ZigZag encoding logic. `into_word` will call a generic `to_nat()` function that converts the signed value to its unsigned ZigZag representation and then casts it to `W`. `from_word` will perform the reverse `to_int()` operation. This will require `T` to have an associated unsigned type (via `common_traits::SignedInt`) that can be converted to `W`.

#### **1.2. Unified `FixedVec` Data Structure**

The existing `FixedVec` and `SFixedVec` will be replaced by a single, more powerful struct.

*   **Concept**: A data structure generic over all four key architectural dimensions.
*   **Technical Specification**:
    *   The new definition will be:
        ```rust
        pub struct FixedVec<
            T: Storable<W>,
            W: Word,
            E: Endianness,
            B: AsRef<[W]> = Vec<W>,
        > {
            bits: B,
            bit_width: usize,
            mask: W,
            len: usize,
            _phantom: PhantomData<(T, W, E)>,
        }
        ```
    *   **Core Logic**: The internal methods for bit manipulation (`get_unchecked`, `set_unchecked`) will be implemented once. All arithmetic related to word boundaries will use `W::BITS`. All data conversion will be delegated to the `T: Storable<W>` trait methods. The `E: Endianness` parameter will be used to dispatch to the correct bit-reading logic, as is currently done.

#### **1.3. Generic `FixedVecBuilder`**

The builder must be updated to handle the new generic structure.

*   **Concept**: A builder capable of constructing an owned `FixedVec<T, W, E, Vec<W>>` from a slice of any `Storable` type.
*   **Technical Specification**:
    *   **Signature**: `pub struct FixedVecBuilder<'a, T, W, E>`.
    *   **Construction Logic**: The `build(input: &'a [T])` method will perform two conceptual passes:
        1.  **Analysis Pass**: It iterates through the `input` slice, calling `val.into_word()` on each element to convert it to its storage representation (`W`). It tracks the maximum value among these converted words to determine the minimal `bit_width` required. This pass is essential for the `BitWidth::Minimal` and `BitWidth::PowerOfTwo` strategies.
        2.  **Encoding Pass**: It allocates the `bits: Vec<W>` buffer. It then iterates through the `input` a second time, converting each `T` to `W` and writing the resulting bits into the buffer using the calculated `bit_width`.
    *   The builder will retain the fluent `.bit_width(BitWidth)` API.

#### **1.4. API Ergonomics through Type Aliases**

To hide the complexity of the fully generic `FixedVec<T, W, E, B>` signature from the end-user, a set of intuitive type aliases will be provided in the prelude.

*   **Concept**: Provide simple, intention-revealing names for the most common configurations.
*   **Technical Specification**:
    *   **Primary Aliases**: The default word size `W` will be set to `usize`, which adapts to the target machine's architecture for optimal performance.
        ```rust
        pub type UFixedVec<T, B = Vec<usize>> = FixedVec<T, usize, LittleEndian, B>;
        pub type SFixedVec<T, B = Vec<usize>> = FixedVec<T, usize, LittleEndian, B>;
        ```
    *   **Concrete Aliases (for convenience and backward compatibility)**:
        ```rust
        // Little Endian (recommended default)
        pub type LEFixedVec<B = Vec<u64>> = FixedVec<u64, u64, LittleEndian, B>;
        pub type LESFixedVec<B = Vec<i64>> = FixedVec<i64, u64, LittleEndian, B>;

        // Big Endian
        pub type BEFixedVec<B = Vec<u64>> = FixedVec<u64, u64, BigEndian, B>;
        pub type BESFixedVec<B = Vec<i64>> = FixedVec<i64, u64, BigEndian, B>;
        ```
    *   Users requiring non-standard configurations (e.g., `i32` elements in a `u32` buffer) can still construct the full `FixedVec<i32, u32, ...>` type directly.


### **Behavior Specialization via Backend Traits**

**Objective**: To layer mutable and resizable functionalities onto the core `FixedVec` structure in a type-safe manner. The behavior of a `FixedVec` instance will be determined at compile time by the traits implemented by its backend parameter `B`. This avoids runtime costs and creates a clear, predictable API contract.

#### **2.1. Immutable View (`B: AsRef<[W]>`)**

This is the baseline behavior, applicable to all `FixedVec` instances, including owned vectors and borrowed slices.

*   **Concept**: An immutable `FixedVec` provides read-only access to the underlying bit-packed data. This is the core functionality upon which all other behaviors are built.
*   **Technical Specification**:
    *   A single, primary `impl` block will contain all read-only methods:
        ```rust
        impl<T, W, E, B> FixedVec<T, W, E, B>
        where
            T: Storable<W>,
            W: Word,
            E: Endianness,
            B: AsRef<[W]>,
        {
            // Read-only methods reside here.
        }
        ```
    *   **API Methods**:
        *   `len() -> usize`
        *   `is_empty() -> bool`
        *   `bit_width() -> usize`
        *   `get(index: usize) -> T`
        *   `iter() -> FixedVecIter<...>`
        *   `as_limbs() -> &[W]`
    *   **Implementation**: These methods will use `self.bits.as_ref()` to get a read-only slice of the underlying buffer.

#### **2.2. Mutable View (`B: AsMut<[W]>`)**

This specialization enables in-place modification for backends that allow it, such as `&mut [W]` or `Vec<W>`.

*   **Concept**: This layer adds element-wise mutation capabilities. It allows for changing the values within the vector but does not permit changing its length.
*   **Technical Specification**:
    *   A separate `impl` block will be constrained by `AsMut<[W]>`:
        ```rust
        impl<T, W, E, B> FixedVec<T, W, E, B>
        where
            T: Storable<W>,
            W: Word,
            E: Endianness,
            B: AsRef<[W]> + AsMut<[W]>, // The key constraint
        {
            // In-place mutation methods reside here.
        }
        ```
    *   **API Methods**:
        *   `set(index: usize, value: T)`: The primary method for changing a value.
        *   `replace(index: usize, value: T) -> T`: Sets a new value and returns the old one.
        *   `as_mut_limbs() -> &mut [W]`: Provides mutable access to the raw underlying buffer.
    *   **Implementation**:
        *   The `set` method's logic is the most critical. It must perform a read-modify-write operation on the underlying word(s) in `self.bits.as_mut()`:
            1.  Calculate the bit position and identify the target word(s).
            2.  Read the current word(s).
            3.  Convert the input `value: T` to its storage representation `w: W` using `T::into_word()`.
            4.  Create a bitmask to clear the target bit range in the word(s).
            5.  Apply the mask using a bitwise `AND`.
            6.  Shift the new `w` into the correct position and apply it using a bitwise `OR`.
            7.  Write the modified word(s) back to the buffer.
        *   This operation must be carefully implemented to correctly handle values that span across word boundaries.

#### **2.3. Resizable Vector (`B = Vec<W>`)**

This is the most capable variant, providing the full dynamic array functionality familiar from `std::vec::Vec`.

*   **Concept**: This specialization is exclusive to the owned `FixedVec` where the backend is `Vec<W>`. It provides methods to grow and shrink the vector.
*   **Technical Specification**:
    *   A highly specific `impl` block will target `Vec<W>` directly:
        ```rust
        impl<T, W, E> FixedVec<T, W, E, Vec<W>>
        where
            T: Storable<W>,
            W: Word,
            E: Endianness,
        {
            // Resizing and capacity management methods reside here.
        }
        ```
    *   **API Methods**:
        *   `new(bit_width: usize)`: Creates an empty vector.
        *   `with_capacity(bit_width: usize, capacity: usize)`: Creates an empty vector with pre-allocated capacity.
        *   `capacity() -> usize`: Returns the total number of elements the vector can hold without reallocating.
        *   `reserve(additional: usize)`: Ensures capacity for at least `additional` more elements.
        *   `shrink_to_fit()`: Reduces capacity to match the current length.
        *   `push(value: T)`: Appends an element.
        *   `pop() -> Option<T>`: Removes and returns the last element.
        *   `resize(new_len: usize, value: T)`: Changes the vector's length, filling new slots with `value`.
        *   `clear()`: Removes all elements.
        *   `insert(index: usize, value: T)` and `remove(index: usize) -> T`.
    *   **Implementation**:
        *   **Capacity Management**: `capacity()` and `reserve()` will translate element counts into the required number of words (`W`) and delegate to the underlying `Vec<W>`'s capacity management. The formula `(elements * bit_width).div_ceil(W::BITS)` will be central.
        *   **`push`**: This method will first check if `self.len + 1` exceeds `self.capacity()`. If so, it will trigger a reallocation (e.g., doubling the capacity). Then, it will use a `BufBitWriter` positioned at the end of the current bitstream to efficiently write the new element.
        *   **`insert`/`remove`**: These are the most complex operations due to the need for bit-shifting. The implementation will perform the following conceptual steps:
            1.  Calculate the bit position of the `index`.
            2.  Shift all subsequent bits in the `bits` buffer to the right (for `insert`) or left (for `remove`) by `self.bit_width` positions. This is a non-trivial, low-level bit manipulation task that must be heavily tested.
            3.  For `insert`, write the new element into the created gap.
            4.  Update `self.len`.
        *   The documentation for `insert` and `remove` will explicitly state their O(n) complexity, where `n` is the number of elements after the index.

By layering functionality in this manner, we create a single, coherent `FixedVec` type whose capabilities are discovered and enforced by the compiler, resulting in a design that is both powerful and safe.

## **Concurrent and Atomic Access (Corrected Plan)**

**Objective**: To provide a thread-safe `AtomicFixedVec` that guarantees 100% atomic correctness for all operations. It will achieve this by transparently selecting the optimal synchronization strategy: a high-performance **lock-free CAS loop** for single-word operations, and a highly efficient, fine-grained **Seqlock mechanism** for multi-word operations.


#### **3.2. Hybrid Atomicity Strategy: The `AtomicAccess` Trait (Refined)**

The fundamental strategy remains a compile-time dispatch via a private trait. The refinement lies in the implementation of the multi-word fallback.

1.  **Implement `AtomicAccess` - Strategy 1: Lock-Free (Single-Word)**
    *   **Concept**: An implementation optimized for configurations where an element is guaranteed to be contained within a single `AtomicWord`.
    *   **Applicability**: Same as before: `bit_width` that ensures no boundary crossing (e.g., powers of two).
    *   **Technical Implementation**: Remains unchanged. It will use a `fetch_update` (CAS) loop on the single `AtomicW` in the buffer. This is the fastest path and is already fully correct.

2.  **Implement `AtomicAccess` - Strategy 2: Seqlock-Based (Multi-Word)**
    *   **Concept**: Provide 100% atomicity for elements that span two `AtomicWord` boundaries by using a Seqlock mechanism for each word. This avoids torn reads and provides atomicity without the heavy cost and potential for writer starvation of a traditional `RwLock`.
    *   **Applicability**: This will be the implementation for all configurations not covered by the lock-free strategy.
    *   **Data Structure for the Backend**:
        *   Instead of a simple `Vec<AtomicW>`, the backend will be a `Vec<SeqLockWord<W>>`.
        *   The `SeqLockWord<W>` struct will contain:
            ```rust
            struct SeqLockWord<W: Word> {
                version: AtomicUsize,
                data: UnsafeCell<W>, // For interior mutability without ARC/Mutex overhead
            }
            ```
    *   **Technical Implementation**:
        *   **`atomic_store(index, value)`**:
            1.  Calculate the two word indices (`idx1`, `idx2`) and bit offsets involved.
            2.  Acquire locks in a consistent order to prevent deadlock (e.g., always lock the smaller index first). This can be a simple spinlock or a lightweight mutex guarding the `version` counters if contention is expected to be extremely high, but typically the atomic operations on `version` are sufficient.
            3.  For each word `w` at `idx1` and `idx2`:
                a. Increment `w.version` to an odd number (signaling a write is in progress).
                b. Ensure memory ordering with a release fence.
            4.  Perform the read-modify-write sequence on the `data` fields of both words (via `UnsafeCell::get()`).
            5.  Ensure memory ordering with another release fence.
            6.  Increment both `version` counters again to an even number (signaling the write is complete).
        *   **`atomic_load(index)`**:
            1.  Calculate the two word indices (`idx1`, `idx2`).
            2.  Enter a read loop:
                a. Read `version1_start` from `seqlock1.version`.
                b. Read `version2_start` from `seqlock2.version`.
                c. Ensure both versions are even (not currently being written to). If not, spin/retry.
                d. Ensure memory ordering with an acquire fence.
                e. Read the data from both `seqlock1.data` and `seqlock2.data`.
                f. Ensure memory ordering with another acquire fence.
                g. Read `version1_end` from `seqlock1.version` and `version2_end` from `seqlock2.version`.
            3.  If `version1_start == version1_end` and `version2_start == version2_end`, the read was consistent and successful. Combine the data from the two words and return the result.
            4.  If the versions do not match, a write occurred during the read. Discard the data and repeat the loop.
        *   **`atomic_compare_exchange`**: This operation becomes more complex. It must acquire exclusive write access (e.g., via a spinlock on the version counters or by transitioning them to a special "locked" state), perform a `load`, compare the value, and if it matches, perform the `store` logic before releasing the locks.

#### **3.3. Public API on `AtomicFixedVec`**

The public API remains unchanged. The user calls `load`, `store`, etc., and the `where Self: AtomicAccess<...>` bound ensures the compiler transparently dispatches to the correct underlying implementation (lock-free CAS or Seqlock).

**Advantages of this Refined Plan**:

*   **Correctness**: The Seqlock mechanism guarantees 100% atomicity for multi-word operations, preventing the torn reads that are possible in the reference implementation. This is a significant improvement in safety.
*   **Performance**: For the multi-word case, Seqlocks are generally much faster than `RwLock` for read-heavy workloads, as readers do not block each other or the writer. They simply retry if a conflict occurs. This is a substantial performance gain over a coarse-grained lock.
*   **Superior Design**: This hybrid approach demonstrates a deep understanding of concurrent data structures, offering the absolute best-case performance (lock-free) where possible, and falling back to a highly efficient and correct mechanism (Seqlock) where necessary. This unequivocally surpasses the reference implementation in both safety and sophistication.

## **API Feature Parity and Enhancement**

**Objective**: To enrich the `FixedVec` ecosystem with a comprehensive set of ergonomic and high-performance APIs. This phase focuses on integrating seamlessly with Rust's standard traits, providing advanced tools for performance-critical applications, and ensuring smooth interoperability between different `FixedVec` variants.


#### **4.1. Low-Level Access API**

Provide advanced, performance-oriented tools for users who need fine-grained control over memory access patterns.

1.  **Unaligned Memory Access**:
    *   **Concept**: Offer a safe and an unsafe pathway to leverage faster, unaligned reads when the data layout permits.
    *   **Technical Specification**:
        *   **Builder Method**: Add `.with_unaligned_padding(true)` to `FixedVecBuilder`. When enabled, it allocates one extra `W` at the end of the `bits` buffer. A private flag `has_padding: bool` will be stored in the `FixedVec`.
        *   **Safe API**: `pub fn try_get_unaligned(&self, index: usize) -> Result<T, UnalignedReadError>`. This method will perform runtime checks:
            1.  Verify `self.has_padding` is true.
            2.  Verify `self.bit_width` is compatible with unaligned reads (e.g., `bit_width <= W::BITS - (size_of::<W>()-1)*8`).
            3.  If checks pass, it will internally call the `unsafe` version.
        *   **Unsafe API**: `pub unsafe fn get_unaligned_unchecked(&self, index: usize) -> T`. This method will directly use `core::ptr::read_unaligned` on a pointer derived from the `bits` buffer. Its documentation will clearly state the required invariants (padding and `bit_width`).

2.  **CPU Cache Prefetching**:
    *   **Concept**: Allow users to hint to the CPU that specific data will be needed soon, reducing memory latency.
    *   **Technical Specification**:
        *   `pub fn prefetch(&self, index: usize)`: A safe method that performs no action if the index is out of bounds.
        *   **Implementation**: It will calculate the byte offset corresponding to the element at `index`, get a pointer to the start of the `bits` buffer, and use `core::intrinsics::prefetch_read_data(ptr, locality_level)`.

3.  **Direct Memory Address Access**:
    *   **Concept**: Provide a mechanism to get a pointer to the storage word containing an element, enabling custom, user-defined memory operations.
    *   **Technical Specification**:
        *   `pub fn addr_of(&self, index: usize) -> Option<*const W>`: A safe method that returns a raw pointer to the word containing the start of the element at `index`. It returns `None` if the index is out of bounds.
        *   **Safety**: The method itself is safe because it only returns a pointer. Dereferencing the pointer remains an `unsafe` operation for the caller, which is the correct Rust pattern.

#### **4.2. High-Performance Iterators**

Provide iterator variants that trade safety checks for maximum throughput.

1.  **Forward and Reverse Unchecked Iterators**:
    *   **Concept**: Create iterators that omit all bounds checks within the `next` loop, for use in contexts where the iteration range is externally guaranteed to be valid.
    *   **Technical Specification**:
        *   Define a private `trait UncheckedIterator` with `unsafe fn next_unchecked(&mut self) -> T`.
        *   Create `struct FixedVecUncheckedIter<'a, ...>` and `struct FixedVecReverseUncheckedIter<'a, ...>`.
        *   Implement `UncheckedIterator` for both. The reverse iterator's logic will be more complex, requiring careful management of a bit buffer filled from the end of the `bits` slice.
        *   Expose them via `unsafe` methods on `FixedVec`:
            *   `pub unsafe fn iter_unchecked(&self) -> FixedVecUncheckedIter<...>`
            *   `pub unsafe fn iter_rev_unchecked(&self) -> FixedVecReverseUncheckedIter<...>`

#### **4.3. Chunking API for Parallelism**

Provide primitives for easily dividing a `FixedVec` into independent sections.

1.  **Immutable and Mutable Chunk Iterators**:
    *   **Concept**: Offer iterators that yield `FixedVecView` or `FixedVecViewMut` sub-slices of a given size.
    *   **Technical Specification**:
        *   `pub fn chunks(&self, chunk_size: usize) -> ChunksIter<...>`
        *   `pub fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMutIter<...>`
        *   **Implementation**: These methods will be complex. If `chunk_size * bit_width` is not a multiple of `W::BITS`, the chunks will not align with the underlying `&[W]` or `&mut [W]` slices. The implementation cannot simply wrap `slice::chunks`. It will need to create `FixedVecView` / `FixedVecViewMut` instances manually by calculating the correct `start` and `len` for each chunk. For `chunks_mut`, this is safe as the views are guaranteed not to overlap.

#### **4.4. Ecosystem Integration and Interoperability**

Ensure the `FixedVec` family feels like a native Rust collection.

1.  **Idiomatic Construction (`FromIterator`)**:
    *   **Concept**: Allow `iterator.collect::<FixedVec<...>>()` syntax.
    *   **Technical Specification**:
        *   Implement `impl<T, W, E> FromIterator<T> for FixedVec<T, W, E, Vec<W>>`.
        *   **Logic**: The implementation will first collect all items into a temporary `Vec<T>`. It will then use the `FixedVecBuilder` to perform the two-pass analysis and construction. The documentation will note the memory overhead of this convenience.

2.  **Bracket Access (`Index` and `IndexMut`)**:
    *   **Concept**: Enable `vec[i]` for reading and `vec[i] = value` for writing.
    *   **Technical Specification**:
        *   `impl Index<usize> for FixedVec<...>`: The `index` method will call `self.get(index)` and `panic!` on `None`, mirroring the behavior of `Vec`.
        *   `impl IndexMut<usize> for FixedVec<...>`: This is the most complex part. The `index_mut` method must return a `&mut T`. Since `FixedVec` stores packed bits, we cannot return a direct mutable reference. The solution is to return a **proxy object** that, upon being dropped (`Drop`), writes the modified value back into the `FixedVec`.
            ```rust
            pub struct MutProxy<'a, ...> { vec: &'a mut FixedVec<...>, index: usize, value: T }
            impl Drop for MutProxy<...> { fn drop(&mut self) { self.vec.set(self.index, self.value); } }
            impl DerefMut for MutProxy<...> { /* to allow `*proxy = new_val;` */ }
            ```

3.  **Zero-Copy Deserialization (`epserde`)**:
    *   **Concept**: Enable instantaneous loading of `FixedVec` data structures.
    *   **Technical Specification**:
        *   Under a feature flag `epserde`, `impl Epserde for FixedVec<...>` will be provided.
        *   The serialization schema will write the metadata (`len`, `bit_width`, etc.) followed by the raw `bits` buffer.
        *   The zero-copy deserialization will be implemented for `FixedVec<T, W, E, &'a [W]>`, which will read the metadata and then create a view over the remaining byte slice.

4.  **Inter-Variant Conversions (`From`)**:
    *   **Concept**: Allow seamless conversion between different `FixedVec` forms.
    *   **Technical Specification**: Provide `From` implementations for:
        *   Owned -> Boxed: `FixedVec<..., Vec<W>>` -> `FixedVec<..., Box<[W]>>`.
        *   Non-Atomic -> Atomic: `FixedVec<T, W, ...>` -> `AtomicFixedVec<T, W, ...>` (and the reverse, which may be `unsafe`).
        *   This ensures that the different vector types can be used interchangeably where appropriate.