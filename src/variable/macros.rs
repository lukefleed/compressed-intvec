//! Convenience macros for creating an [`VarVec`] with a `vec!`-like syntax.
//!
//! These macros provide a familiar, ergonomic way to initialize a compressed
//! integer vector. They are shortcuts for the [`VarVec::builder`] and use a set
//! of reasonable defaults for compression and sampling rate.
//!
//! [`VarVec`]: crate::variable::VarVec
//! [`VarVec::builder`]: crate::variable::VarVec::builder

/// Creates a [`LEVarVec`] (an `VarVec` of [`u64`]s) containing the given elements.
///
/// `int_vec!` allows for concise initialization of a [`LEVarVec`], which is an
/// alias for `VarVec<u64, LE>`. It uses a set of reasonable defaults for its
/// build parameters:
///
/// - **Codec**: [`Codec::Auto`] is used to automatically select the
///   most space-efficient codec for the provided data.
/// - **Sampling Rate (`k`)**: A default value of `16` is used, offering a
///   good balance between random access speed and memory overhead.
///
/// # Note on Types
///
/// The macro infers the element type from the input. For explicit control over
/// parameters, use [`VarVec::builder`](crate::variable::VarVec::builder).
///
/// # Examples
///
/// Create an empty [`LEVarVec`]:
/// ```
/// # use compressed_intvec::int_vec;
/// # use compressed_intvec::prelude::LEVarVec;
/// let v: LEVarVec = int_vec![];
/// assert!(v.is_empty());
/// ```
///
/// Create a [`crate::variable::VarVec`] from a list of elements:
/// ```
/// # use compressed_intvec::int_vec;
///   let v = int_vec![100u32, 200, 300, 1024];
/// assert_eq!(v.len(), 4);
/// assert_eq!(v.get(1), Some(200));
/// ```
///
/// Create a [`crate::variable::VarVec`] with a repeated element:
/// ```
/// # use compressed_intvec::int_vec;
///   let v = int_vec![42u8; 100];
/// assert_eq!(v.len(), 100);
/// assert_eq!(v.get(50), Some(42));
/// ```
///
/// [`LEVarVec`]: crate::variable::LEVarVec
/// [`Codec::Auto`]: crate::variable::Codec::Auto
#[macro_export]
macro_rules! int_vec {
    () => {
        $crate::variable::VarVec::<u64, dsi_bitstream::prelude::LE>::builder().build(&[0u64; 0]).unwrap()
    };
    ($($elem:expr),* $(,)?) => {
        $crate::variable::VarVec::<_, dsi_bitstream::prelude::LE>::builder()
            .codec($crate::variable::Codec::Auto)
            .k(16)
            .build(&[$($elem),*])
            .unwrap()
    };
    ($elem:expr; $len:expr) => {
        {
            let mut v = ::std::vec::Vec::with_capacity($len);
            v.resize($len, $elem);
            $crate::variable::VarVec::<_, dsi_bitstream::prelude::LE>::builder()
                .codec($crate::variable::Codec::Auto)
                .k(16)
                .build(&v)
                .unwrap()
        }
    };
}

/// Creates a [`LESVarVec`] (an `VarVec` of [`i64`]s) containing the given elements.
///
/// `sint_vec!` allows for concise initialization of a [`LESVarVec`], which is an
/// alias for `VarVec<i64, LE>`. It uses a set of reasonable defaults:
///
/// - **Codec**: [`Codec::Auto`] is used to automatically select the
///   best codec based on the data's properties (via zig-zag encoding).
/// - **Sampling Rate (`k`)**: A default value of `16` is used.
///
/// # Note on Types
///
/// All input elements are automatically cast to [`i64`].
///
/// For more control over these parameters, or to use a different integer type,
/// please use the [`VarVec::builder`](crate::variable::VarVec::builder).
///
/// # Examples
///
/// Create an empty [`LESVarVec`]:
/// ```
/// # use compressed_intvec::sint_vec;
/// # use compressed_intvec::prelude::LESVarVec;
/// let v: LESVarVec = sint_vec![];
/// assert!(v.is_empty());
/// ```
///
/// Create an [`LESVarVec`] from a list of elements:
/// ```
/// # use compressed_intvec::sint_vec;
///   # use compressed_intvec::prelude::LESVarVec;
///   let v: LESVarVec = sint_vec![-100, 200, -300];
/// assert_eq!(v.len(), 3);
/// assert_eq!(v.get(2), Some(-300));
/// ```
///
/// Create an [`LESVarVec`] with a repeated element:
/// ```
/// # use compressed_intvec::sint_vec;
///   # use compressed_intvec::prelude::LESVarVec;
///   let v: LESVarVec = sint_vec![-42; 100];
/// assert_eq!(v.len(), 100);
/// assert_eq!(v.get(50), Some(-42));
/// ```
///
/// [`LESVarVec`]: crate::variable::LESVarVec
/// [`Codec::Auto`]: crate::variable::Codec::Auto
#[macro_export]
macro_rules! sint_vec {
    () => {
        $crate::variable::VarVec::<i64, dsi_bitstream::prelude::LE>::builder().build(&[0i64; 0]).unwrap()
    };
    ($($elem:expr),* $(,)?) => {
        $crate::variable::VarVec::<i64, dsi_bitstream::prelude::LE>::builder()
            .codec($crate::variable::Codec::Auto)
            .k(16)
            .build(&[$($elem as i64),*])
            .unwrap()
    };
    ($elem:expr; $len:expr) => {
        {
            let mut v = ::std::vec::Vec::with_capacity($len);
            v.resize($len, $elem as i64);
            $crate::variable::VarVec::<i64, dsi_bitstream::prelude::LE>::builder()
                .codec($crate::variable::Codec::Auto)
                .k(16)
                .build(&v)
                .unwrap()
        }
    };
}
