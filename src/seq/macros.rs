//! Convenience macros for creating a [`SeqVec`] with a familiar syntax.

/// Creates a [`crate::seq::LESeqVec`] containing the given sequences.
///
/// This macro constructs a [`crate::seq::SeqVec`] with little-endian encoding and automatic
/// codec selection, using the default `u64` word type for storage.
///
/// # Examples
///
/// ```
/// use compressed_intvec::seq_vec;
///
/// let v = seq_vec![[1u32, 2, 3], [10u32, 20], [100u32]];
/// assert_eq!(v.num_sequences(), 3);
/// assert_eq!(v.get(0).unwrap().collect::<Vec<_>>(), vec![1u32, 2, 3]);
/// assert_eq!(v.get(1).unwrap().collect::<Vec<_>>(), vec![10u32, 20]);
/// assert_eq!(v.get(2).unwrap().collect::<Vec<_>>(), vec![100u32]);
/// ```
///
/// # Notes
///
/// - Empty sequences are supported: `seq_vec![[1, 2], [], [3]]`.
/// - All elements in a sequence must be the same type.
/// - The macro uses `from_slices` internally, which performs codec analysis.
#[macro_export]
macro_rules! seq_vec {
    // Empty case: seq_vec![]
    () => {
        $crate::seq::SeqVec::<u64, dsi_bitstream::prelude::LE>::from_slices::<&[u64]>(&[]).unwrap()
    };
    // One or more sequences: seq_vec![[1, 2], [3, 4]]
    ($([$($elem:expr),* $(,)?]),* $(,)?) => {
        $crate::seq::SeqVec::<_, dsi_bitstream::prelude::LE>::from_slices(&[
            $(&[$($elem),*] as &[_]),*
        ]).unwrap()
    };
}

/// Creates a [`crate::seq::LESeqVec`] of signed integers.
///
/// This macro is similar to [`seq_vec!`](crate::seq_vec), but automatically
/// casts all elements to `i64` before creating the vector. This is useful when
/// working with signed integer data.
///
/// # Examples
///
/// ```
/// use compressed_intvec::seq_vec_signed;
/// use compressed_intvec::seq::LESeqVec;
///
/// let v: LESeqVec<i64> = seq_vec_signed![[-1i32, -2, 3], [10, -20], [-100]];
/// assert_eq!(v.num_sequences(), 3);
/// assert_eq!(v.get(0).unwrap().collect::<Vec<_>>(), vec![-1i64, -2i64, 3i64]);
/// ```
///
/// # Notes
///
/// - All elements are converted to `i64`, so ensure your integer literals fit.
/// - The macro uses `from_slices` internally, which performs codec analysis.
#[macro_export]
macro_rules! seq_vec_signed {
    // Empty case
    () => {
        $crate::seq::SeqVec::<i64, dsi_bitstream::prelude::LE>::from_slices::<&[i64]>(&[]).unwrap()
    };
    // One or more sequences with automatic casting to i64
    ($([$($elem:expr),* $(,)?]),* $(,)?) => {
        $crate::seq::SeqVec::<_, dsi_bitstream::prelude::LE>::from_slices(&[
            $(&[$($elem as i64),*] as &[_]),*
        ]).unwrap()
    };
}
