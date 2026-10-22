//! The coded bit stream that [`VarVec`](crate::VarVec) and [`SeqVec`](crate::SeqVec) share: the
//! codes, the automatic choice among them, and the encoder and decoder over `u64` words.

mod codec;
#[cfg(feature = "serde")]
pub(crate) mod serde;
mod stream;

pub use codec::{Automatic, Codec, CodecError};
pub(crate) use codec::{ValidCodec, select};
pub(crate) use stream::{CheckedDecoder, CodewordError, Decoder, Encoder};
