//! Convenience macros for creating an [`IntVec`] with a `vec!`-like syntax.
//!
//! These macros provide a familiar, ergonomic way to initialize a compressed
//! integer vector. They are shortcuts for the [`IntVec::builder`] and use a set
//! of reasonable defaults for compression and sampling rate.
//!
//! [`IntVec`]: crate::variable::IntVec
//! [`IntVec::builder`]: crate::variable::IntVec::builder

/// Creates a [`LEIntVec`] (a vector of `u64`s) containing the given elements.
///
/// `int_vec!` allows for concise initialization of a [`LEIntVec`], which is an
/// alias for `IntVec<u64, LE>`. It uses a set of reasonable defaults for its
/// build parameters:
///
/// - **Codec**: [`VariableCodecSpec::Auto`] is used to automatically select the
///   most space-efficient codec for the provided data.
/// - **Sampling Rate (`k`)**: A default value of `32` is used, offering a
///   good balance between random access speed and memory overhead.
///
/// For more control over these parameters, or to use a different integer type,
/// please use the [`IntVec::builder`](crate::variable::IntVec::builder).
///
/// # Examples
///
/// Create an empty [`LEIntVec`]:
/// ```
/// # use compressed_intvec::int_vec;
/// # use compressed_intvec::prelude::LEIntVec;
/// let v: LEIntVec = int_vec![];
/// assert!(v.is_empty());
/// ```
///
/// Create an [`LEIntVec`] from a list of elements:
/// ```
/// # use compressed_intvec::int_vec;
///   # use compressed_intvec::prelude::LEIntVec;
///   let v: LEIntVec = int_vec![100u64, 200, 300, 1024];
/// assert_eq!(v.len(), 4);
/// assert_eq!(v.get(1), Some(200));
/// ```
///
/// Create an [`LEIntVec`] with a repeated element:
/// ```
/// # use compressed_intvec::int_vec;
///   # use compressed_intvec::prelude::LEIntVec;
///   let v: LEIntVec = int_vec![42u64; 100];
/// assert_eq!(v.len(), 100);
/// assert_eq!(v.get(50), Some(42));
/// ```
/// 
/// [`LEIntVec`]: crate::variable::LEIntVec
/// [`VariableCodecSpec::Auto`]: crate::variable::VariableCodecSpec::Auto
#[macro_export]
macro_rules! int_vec {
    () => {
        $crate::variable::IntVec::builder(&[0u64; 0]).build().unwrap()
    };
    ($($elem:expr),* $(,)?) => {
        $crate::variable::IntVec::builder(&[$($elem as u64),*])
            .codec($crate::variable::VariableCodecSpec::Auto)
            .k(32)
            .build()
            .unwrap()
    };
    ($elem:expr; $len:expr) => {
        {
            let mut v = ::std::vec::Vec::with_capacity($len);
            v.resize($len, $elem as u64);
            $crate::variable::IntVec::builder(&v)
                .codec($crate::variable::VariableCodecSpec::Auto)
                .k(32)
                .build()
                .unwrap()
        }
    };
}

/// Creates a [`LESIntVec`] (a vector of `i64`s) containing the given elements.
///
/// `sint_vec!` allows for concise initialization of a [`LESIntVec`], which is an
/// alias for `IntVec<i64, LE>`. It uses a set of reasonable defaults:
///
/// - **Codec**: [`VariableCodecSpec::Auto`] is used to automatically select the
///   best codec based on the data's properties (via zig-zag encoding).
/// - **Sampling Rate (`k`)**: A default value of `32` is used.
///
/// For more control over these parameters, or to use a different integer type,
/// please use the [`IntVec::builder`](crate::variable::IntVec::builder).
///
/// # Examples
///
/// Create an empty [`LESIntVec`]:
/// ```
/// # use compressed_intvec::sint_vec;
/// # use compressed_intvec::prelude::LESIntVec;
/// let v: LESIntVec = sint_vec![];
/// assert!(v.is_empty());
/// ```
///
/// Create an [`LESIntVec`] from a list of elements:
/// ```
/// # use compressed_intvec::sint_vec;
///   # use compressed_intvec::prelude::LESIntVec;
///   let v: LESIntVec = sint_vec![-100, 200, -300];
/// assert_eq!(v.len(), 3);
/// assert_eq!(v.get(2), Some(-300));
/// ```
///
/// Create an [`LESIntVec`] with a repeated element:
/// ```
/// # use compressed_intvec::sint_vec;
///   # use compressed_intvec::prelude::LESIntVec;
///   let v: LESIntVec = sint_vec![-42; 100];
/// assert_eq!(v.len(), 100);
/// assert_eq!(v.get(50), Some(-42));
/// ```
/// 
/// [`LESIntVec`]: crate::variable::LESIntVec
/// [`VariableCodecSpec::Auto`]: crate::variable::VariableCodecSpec::Auto
#[macro_export]
macro_rules! sint_vec {
    () => {
        $crate::variable::IntVec::builder(&[0i64; 0]).build().unwrap()
    };
    ($($elem:expr),* $(,)?) => {
        $crate::variable::IntVec::builder(&[$($elem as i64),*])
            .codec($crate::variable::VariableCodecSpec::Auto)
            .k(32)
            .build()
            .unwrap()
    };
    ($elem:expr; $len:expr) => {
        {
            let mut v = ::std::vec::Vec::with_capacity($len);
            v.resize($len, $elem as i64);
            $crate::variable::IntVec::builder(&v)
                .codec($crate::variable::VariableCodecSpec::Auto)
                .k(32)
                .build()
                .unwrap()
        }
    };
}
