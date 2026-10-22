//! Parallel iteration over a [`FixedVec`], with the `parallel` feature.

use super::{Backend, FixedVec};
use crate::Storable;
use rayon::prelude::*;

impl<T: Storable + Send, B: Backend> FixedVec<T, B> {
    /// Returns an indexed parallel iterator yielding decoded elements by value.
    ///
    /// The iterator borrows the packed words and decodes each element independently.
    /// Available with the `parallel` feature.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::FixedVec;
    /// use rayon::prelude::*;
    ///
    /// let vec: FixedVec<u32> = (0..1000).collect();
    /// let even = vec.par_iter().filter(|x| x % 2 == 0).count();
    /// assert_eq!(even, 500);
    /// ```
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = T> + '_ {
        let slice = self.as_slice();
        (0..slice.len())
            .into_par_iter()
            // SAFETY: the range yields indices below `slice.len()`.
            .map(move |index| unsafe { slice.element(index) })
    }
}
