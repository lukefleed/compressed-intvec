//! Macros for creating `FixedVec` and `SFixedVec` instances with a `vec!`-like syntax.

/// Creates a [`FixedVec`] containing the given elements with a specified bit width.
///
/// `fixed_vec!` allows for concise initialization of a `FixedVec`. It defaults
/// to a Little-Endian (`LE`) memory layout, which is optimal for most modern CPUs.
///
/// # Syntax
///
/// - Create an empty `FixedVec` with a given bit width:
///   ```
///   # use compressed_intvec::prelude::*;
///   # use compressed_intvec::fixed_vec;
///   let v: LEFixedVec = fixed_vec!; // or fixed_vec![8;]
///   assert!(v.is_empty());
///   assert_eq!(v.num_bits(), 8);
///   ```
///
/// - Create a `FixedVec` with a given bit width and a list of elements:
///   ```
///   # use compressed_intvec::prelude::*;
///   # use compressed_intvec::fixed_vec;
///   let v = fixed_vec![10; 100u64, 200, 300];
///   assert_eq!(v.len(), 3);
///   assert_eq!(v.get(1), Some(200));
///   ```
///
/// - Create a `FixedVec` with a given bit width, an element, and a length:
///   ```
///   # use compressed_intvec::prelude::*;
///   # use compressed_intvec::fixed_vec;
///   let v = fixed_vec![16 => 42u64; 10];
///   assert_eq!(v.len(), 10);
///   assert_eq!(v.get(5), Some(42));
///   ```
///
/// The macro uses `BitWidth::Explicit` internally. If you need automatic bit-width
/// detection, please use the [`FixedVec::builder`].
#[macro_export]
macro_rules! fixed_vec {
    ($bit_width:expr) => {
        $crate::prelude::LEFixedVec::builder(&[0u64; 0])
            .bit_width($crate::prelude::BitWidth::Explicit($bit_width))
            .build()
            .unwrap()
    };
    ($bit_width:expr; $($elem:expr),* $(,)?) => {
        // Ensure literals are treated as u64 to satisfy trait bounds
        $crate::prelude::LEFixedVec::builder(&[$($elem as u64),*])
            .bit_width($crate::prelude::BitWidth::Explicit($bit_width))
            .build()
            .unwrap()
    };
    ($bit_width:expr => $elem:expr; $len:expr) => {
        {
            let mut v = ::std::vec::Vec::with_capacity($len);
            // Ensure the element is u64
            v.resize($len, $elem as u64);
            $crate::prelude::LEFixedVec::builder(&v)
                .bit_width($crate::prelude::BitWidth::Explicit($bit_width))
                .build()
                .unwrap()
        }
    };
}

/// Creates an [`SFixedVec`] containing the given elements with a specified bit width.
///
/// `s_fixed_vec!` provides a concise syntax for `SFixedVec` initialization.
/// It defaults to a Little-Endian (`LE`) memory layout.
///
/// # Syntax
///
/// - Create an empty `SFixedVec` with a given bit width:
///
///   ```rust
///   # use compressed_intvec::prelude::*;
///   # use compressed_intvec::s_fixed_vec;
///   let v: LESFixedVec = s_fixed_vec!;
///   assert!(v.is_empty());
///   assert_eq!(v.num_bits(), 8);
///   ```
///
/// - Create an `SFixedVec` with a given bit width and a list of elements:
///   ```rust
///   # use compressed_intvec::prelude::*;
///   # use compressed_intvec::s_fixed_vec;
///   let v = s_fixed_vec![12; -100, 200, -300];
///   assert_eq!(v.len(), 3);
///   assert_eq!(v.get(2), Some(-300));
///   ```
///
/// - Create an `SFixedVec` with a given bit width, an element, and a length:
///   ```rust
///   # use compressed_intvec::prelude::*;
///   # use compressed_intvec::s_fixed_vec;
///   let v = s_fixed_vec![16 => -42; 10];
///   assert_eq!(v.len(), 10);
///   assert_eq!(v.get(5), Some(-42));
///   ```
#[macro_export]
macro_rules! s_fixed_vec {
    ($bit_width:expr) => {
        $crate::prelude::LESFixedVec::builder(&[0i64; 0])
            .bit_width($crate::prelude::BitWidth::Explicit($bit_width))
            .build()
            .unwrap()
    };
    ($bit_width:expr; $($elem:expr),* $(,)?) => {
        // Ensure literals are treated as i64
        $crate::prelude::LESFixedVec::builder(&[$($elem as i64),*])
            .bit_width($crate::prelude::BitWidth::Explicit($bit_width))
            .build()
            .unwrap()
    };
    ($bit_width:expr => $elem:expr; $len:expr) => {
        {
            let mut v = ::std::vec::Vec::with_capacity($len);
            // Ensure the element is i64
            v.resize($len, $elem as i64);
            $crate::prelude::LESFixedVec::builder(&v)
                .bit_width($crate::prelude::BitWidth::Explicit($bit_width))
                .build()
                .unwrap()
        }
    };
}
