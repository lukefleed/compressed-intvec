/// Internal utilities shared across modules.
///
/// This module contains internal implementation details that are not part of
/// the public API. The contents are hidden from documentation and are not
/// accessible to external users.
pub(crate) mod codec_reader;
pub(crate) mod codec_writer;

// Conditionally compile the serde module.
#[cfg(feature = "serde")]
pub(crate) mod serde;
