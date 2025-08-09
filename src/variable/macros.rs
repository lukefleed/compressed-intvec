//! Macros for creating integer vectors with a `vec!`-like syntax.

/// Creates a [`LEIntVec`] (a vector of `u64`s) containing the given elements.
///
/// `int_vec!` allows for concise initialization of a `LEIntVec`. It uses a
/// set of reasonable defaults for its build parameters:
/// - **Codec**: `VariableCodecSpec::Auto` is used to automatically select the
///   most space-efficient codec for the provided data.
/// - **Sampling Rate (`k`)**: A default value of `32` is used, which offers a
///   good balance between random access speed and memory overhead.
///
/// For fine-grained control over these parameters, please use the
/// [`IntVec::builder`].
///
/// # Syntax
///
/// - Create an empty `LEIntVec`:
///   ```
///   # use compressed_intvec::int_vec;
///   # use compressed_intvec::prelude::*;
///   let v: LEIntVec = int_vec![];
///   assert!(v.is_empty());
///   ```
///
/// - Create an `LEIntVec` from a list of elements:
///   ```
///   # use compressed_intvec::int_vec;
///   # use compressed_intvec::prelude::*;
///   let v = int_vec![100u64, 200, 300, 1024];
///   assert_eq!(v.len(), 4);
///   assert_eq!(v.get(1), Some(200));
///   ```
///
/// - Create an `LEIntVec` with a repeated element and a given length:
///   ```
///   # use compressed_intvec::int_vec;
///   # use compressed_intvec::prelude::*;
///   let v = int_vec![42u64; 100];
///   assert_eq!(v.len(), 100);
///   assert_eq!(v.get(50), Some(42));
///   ```
///
#[macro_export]
macro_rules! int_vec {
    () => {
        $crate::prelude::LEIntVec::builder(&[0u64; 0]).build().unwrap()
    };
    ($($elem:expr),* $(,)?) => {
        // Ensure literals are treated as u64 to satisfy trait bounds
        $crate::prelude::LEIntVec::builder(&[$($elem as u64),*])
            // Use reasonable defaults for ergonomic macro usage.
            .codec($crate::prelude::VariableCodecSpec::Auto)
            .k(32)
            .build()
            .unwrap()
    };
    ($elem:expr; $len:expr) => {
        {
            let mut v = ::std::vec::Vec::with_capacity($len);
            // Ensure the element is u64
            v.resize($len, $elem as u64);
            $crate::prelude::LEIntVec::builder(&v)
                .codec($crate::prelude::VariableCodecSpec::Auto)
                .k(32)
                .build()
                .unwrap()
        }
    };
}

/// Creates a [`LESIntVec`] (a vector of `i64`s) containing the given elements.
///
/// `sint_vec!` allows for concise initialization of a `LESIntVec`. It uses a
/// set of reasonable defaults for its build parameters:
/// - **Codec**: `VariableCodecSpec::Auto` is used to automatically select the
///   most space-efficient codec for the provided data.
/// - **Sampling Rate (`k`)**: A default value of `32` is used.
///
/// For fine-grained control over these parameters, please use the
/// [`IntVec::builder`].
///
/// # Syntax
///
/// - Create an empty `LESIntVec`:
///   ```
///   # use compressed_intvec::sint_vec;
///   # use compressed_intvec::prelude::*;
///   let v: LESIntVec = sint_vec![];
///   assert!(v.is_empty());
///   ```
///
/// - Create an `LESIntVec` from a list of elements:
///   ```
///   # use compressed_intvec::sint_vec;
///   # use compressed_intvec::prelude::*;
///   let v = sint_vec![-100, 200, -300];
///   assert_eq!(v.len(), 3);
///   assert_eq!(v.get(2), Some(-300));
///   ```
///
/// - Create an `LESIntVec` with a repeated element and a given length:
///   ```
///   # use compressed_intvec::sint_vec;
///   # use compressed_intvec::prelude::*;
///   let v = sint_vec![-42; 100];
///   assert_eq!(v.len(), 100);
///   assert_eq!(v.get(50), Some(-42));
///   ```
#[macro_export]
macro_rules! sint_vec {
    () => {
        $crate::prelude::LESIntVec::builder(&[0i64; 0]).build().unwrap()
    };
    ($($elem:expr),* $(,)?) => {
        // Ensure literals are treated as i64
        $crate::prelude::LESIntVec::builder(&[$($elem as i64),*])
            // Use reasonable defaults. Auto is now supported.
            .codec($crate::prelude::VariableCodecSpec::Auto)
            .k(32)
            .build()
            .unwrap()
    };
    ($elem:expr; $len:expr) => {
        {
            let mut v = ::std::vec::Vec::with_capacity($len);
            // Ensure the element is i64
            v.resize($len, $elem as i64);
            $crate::prelude::LESIntVec::builder(&v)
                .codec($crate::prelude::VariableCodecSpec::Auto)
                .k(32)
                .build()
                .unwrap()
        }
    };
}
