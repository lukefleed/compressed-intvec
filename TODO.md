# Fixed Length Vector

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
s**: Requires completion of Generic Backend Support.
