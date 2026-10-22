//! Parallel iteration over a [`VarVec`], with the `parallel` feature.

use super::{VarVec, VarVecIter, View};
use crate::{Backend, Storable};
use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::marker::PhantomData;

impl<T: Storable + Send, B: Backend> VarVec<T, B> {
    /// Returns an indexed parallel iterator yielding elements by value.
    ///
    /// Available with the `parallel` feature. Each Rayon task decodes its range sequentially,
    /// starting at the nearest preceding sample. The producer requests a minimum task length
    /// of [`sampling_rate`](VarVec::sampling_rate) elements to limit the overhead of decoding
    /// before a range starts. The indexed iterator preserves element order when collected.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::VarVec;
    /// use rayon::prelude::*;
    ///
    /// let vec: VarVec<u32> = (0..10_000).collect();
    /// let sum: u64 = vec.par_iter().map(u64::from).sum();
    /// assert_eq!(sum, 49_995_000);
    /// let positions: Vec<usize> = vec.par_iter().positions(|x| x % 5000 == 0).collect();
    /// assert_eq!(positions, [0, 5000]);
    /// ```
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = T> + '_ {
        Elements {
            view: self.view(),
            front: 0,
            back: self.len,
            element: PhantomData,
        }
    }
}

/// The elements `front..back` of a vector, as a parallel iterator and as a rayon producer.
struct Elements<'a, T> {
    view: View<'a>,
    front: usize,
    back: usize,
    element: PhantomData<fn() -> T>,
}

impl<T: Storable + Send> ParallelIterator for Elements<'_, T> {
    type Item = T;

    fn drive_unindexed<C: UnindexedConsumer<T>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.back - self.front)
    }
}

impl<T: Storable + Send> IndexedParallelIterator for Elements<'_, T> {
    fn len(&self) -> usize {
        self.back - self.front
    }

    fn drive<C: Consumer<T>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<T>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }
}

impl<'a, T: Storable + Send> Producer for Elements<'a, T> {
    type Item = T;
    type IntoIter = VarVecIter<'a, T>;

    fn into_iter(self) -> VarVecIter<'a, T> {
        VarVecIter::new(self.view, self.front, self.back)
    }

    fn min_len(&self) -> usize {
        // A shorter range would seek to a sample and decode elements that it then skips.
        self.view.sampling_rate
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.front + index;
        let left = Self {
            back: mid,
            element: PhantomData,
            ..self
        };
        let right = Self {
            front: mid,
            element: PhantomData,
            ..self
        };
        (left, right)
    }
}
