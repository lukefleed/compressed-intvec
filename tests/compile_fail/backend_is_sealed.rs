//! `Backend` is sealed: everything below resolves, and only the implementation for a foreign
//! type fails, because the unchecked reads of `FixedVec` trust the slice that `as_ref` returns.

use compressed_intvec::Backend;

struct Halves(Vec<u64>);

impl AsRef<[u64]> for Halves {
    fn as_ref(&self) -> &[u64] {
        &self.0[..self.0.len() / 2]
    }
}

impl Backend for Halves {}

fn main() {}
