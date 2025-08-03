# Fixed Length Vector

- [x] **Implement Generic Backend Support**
    -   **Action**: Refactor `FixedVec<E>` to `FixedVec<E, B: AsRef<[u64]> = Vec<u64>>`.
    -   **API Changes**:
        -   Add a public, **safe** constructor: `pub fn from_parts(bits: B, len: usize, num_bits: usize) -> Result<Self, FixedVecError>`. This constructor will perform all necessary validation, including checking for the required padding word.
        -   Maintain an internal, `pub(crate) unsafe fn new_unchecked(...)` for use by the builder and other internal components that can guarantee invariants.
    -   **Rationale**: Enables zero-copy views over existing data buffers (`&[u64]`, memory-mapped files)

- [ ] **Add SIMD-based Access Methods**
    -   **Action**: Introduce methods for reading multiple elements into a SIMD vector
    -   **Target**: Primarily for `num_bits` values that are byte-aligned (8, 16, 32, 64) where implementation is straightforward.


- [ ] **Introduce a Mutable `MutFixedVec` Variant**
    -   **Action**: Design and implement a new `MutFixedVec` struct that supports in-place modifications (`set`) and resizing (`push`, `pop`, `resize`).
    -   **Architecture**: This will be a distinct type from the immutable `FixedVec` to maintain clear compile-time guarantees. It will require `AsMut<[u64]>` on its backend type.

- [ ] **Implement `AtomicFixedVec` for Concurrent Access**
    -   **Action**: Create `AtomicFixedVec` with a backend of atomic types (`Vec<AtomicU64>`).
    -   **Implementation**: Provide atomic `get` and `set` methods. `set` operations spanning word boundaries will require careful implementation using `compare_exchange` loops to ensure atomicity.

- [ ] **Introduce Safe Low-Level API Methods**
    -   **Action**: Evaluate and add safe abstractions for low-level performance tuning.
    -   **Example**: `try_get_unaligned`, which would check if an unaligned read is possible and safe for the current configuration before executing it.
    -   **Rationale**: Provides access to advanced performance features without exposing raw pointers or requiring `unsafe` blocks from the user.

# Variable Length Vector


- [ ] **Implement Generic Backend Support**
    -   **Action**: Refactor `IntVec<E>` to `IntVec<E, B: AsRef<[u64]>>`.
    -   **API Changes**:
        -   Add a public, safe constructor `from_parts` that accepts a data buffer and a samples buffer.
        -   Define a clear memory layout for `bits` and `samples` to enable true zero-copy views from a single memory region.

- [ ] **Implement Slicing**
    -   **Action**: Create `IntVecSlice` and `IntVecSliceIter` structs.
    -   **API Changes**: Add `slice()` and `split_at()` methods to `IntVec`.
    -   **Dependencies**: Requires completion of Generic Backend Support.
