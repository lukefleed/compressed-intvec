//! Parallel iteration over a [`SeqVec`], with the `parallel` feature.

use super::{SeqIter, SeqVec};
use crate::{Backend, Storable};
use rayon::prelude::*;

impl<T: Storable, B: Backend> SeqVec<T, B> {
    /// Returns an indexed parallel iterator over sequences in order.
    ///
    /// Requires the `parallel` feature. Each item is a [`SeqIter`] that decodes one sequence
    /// sequentially. Rayon distributes work between sequences, so a sequence's values are
    /// decoded by the consumer of that item.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::SeqVec;
    /// use rayon::prelude::*;
    ///
    /// let vec: SeqVec<u32> = (0..1000u32).map(|i| vec![i; 3]).collect();
    /// let sums: Vec<u32> = vec.par_iter().map(|sequence| sequence.sum()).collect();
    /// assert_eq!(sums[10], 30);
    /// ```
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = SeqIter<'_, T>> + '_ {
        let view = self.view();
        (0..self.len())
            .into_par_iter()
            .map(move |index| SeqIter::new(view, index))
    }
}
