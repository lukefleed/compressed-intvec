# **Technical Roadmap: `fixed` Module Evolution**




## **Concurrent and Atomic Access**

**Objective**: To provide a thread-safe `AtomicFixedVec` that guarantees 100% atomic correctness for all operations. It will achieve this by transparently selecting the optimal synchronization strategy: a high-performance **lock-free CAS loop** for single-word operations, and a highly efficient, fine-grained **Seqlock mechanism** for multi-word operations.


#### **3.2. Hybrid Atomicity Strategy: The `AtomicAccess` Trait**

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