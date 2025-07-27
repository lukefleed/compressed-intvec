//! Builders for `IntVec`.

use super::{resolve_codec, CodecSpec, Encoding, IntVec, IntVecBitWriter, IntVecError, Samples};
use dsi_bitstream::{
    dispatch::{FuncCodeWriter, StaticCodeWrite},
    impls::MemWordWriterVec,
    prelude::{BitWrite, CodesWrite, Endianness},
};
use std::marker::PhantomData;

/// A builder for creating an [`IntVec`] from a slice (`&[u64]`).
///
/// This builder is obtained by calling [`IntVec::builder`]. It provides a fluent
/// interface for setting parameters like the sampling rate `k` and the
/// compression `CodecSpec`.
///
/// Because it operates on a slice, it can automatically select the best codec
/// parameters by analyzing the data first. This is the recommended way to construct
/// an [`IntVec`] when all data is available in memory.
#[derive(Debug)]
pub struct IntVecBuilder<'a, E: Endianness> {
    pub(super) input: &'a [u64],
    pub(super) k: usize,
    pub(super) codec_spec: CodecSpec,
    pub(super) _endian: PhantomData<E>,
}

impl<'a, E: Endianness> IntVecBuilder<'a, E> {
    /// Creates a new builder for an `IntVec`.
    ///
    /// This constructor is `pub(super)`, meaning it is only visible to the parent
    /// module (`intvec`). The public entry point for creating this builder is
    /// [`IntVec::builder`].
    pub(super) fn new(input: &'a [u64]) -> Self {
        Self {
            input,
            k: 32,                       // Default sampling rate
            codec_spec: CodecSpec::Auto, // Default to auto-selection
            _endian: PhantomData,
        }
    }

    /// Sets the sampling rate `k` for DSI-based codecs.
    ///
    /// The sampling rate determines how frequently a sample of the bitstream's position
    /// is stored. A smaller `k` leads to faster random access but increases memory
    /// overhead. A larger `k` reduces memory usage but slows down access.
    ///
    /// The value must be greater than 0. This parameter is ignored for
    /// `FixedLength` encoding, as random access is already $O(1)$.
    ///
    /// # Arguments
    /// * `k`: The sampling rate.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the codec specification for compression.
    ///
    /// This determines which compression algorithm will be used. See [`CodecSpec`]
    /// for available options. It can be a specific codec (e.g., `Gamma`) or a request
    /// for automatic selection (`Auto`).
    ///
    /// # Arguments
    /// * `codec_spec`: The desired codec specification.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: &[u64] = &[1, 2, 3, 4, 5];
    /// let intvec = LEIntVec::builder(data)
    ///     .codec(CodecSpec::Gamma)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(intvec.encoding(), Encoding::Dsi(dsi_bitstream::prelude::Codes::Gamma));
    /// ```
    pub fn codec(mut self, codec_spec: CodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the `IntVec`, consuming the builder.
    ///
    /// This method performs the main compression logic. It resolves the `CodecSpec`,
    /// automatically selecting parameters if requested (e.g., for `CodecSpec::Auto` or
    /// `CodecSpec::Rice { log2_b: None }`). It then encodes the input data and
    /// builds the final `IntVec` structure.
    ///
    /// # Returns
    ///
    /// A `Result` containing the constructed `IntVec` on success, or an
    /// `IntVecError` if there's a problem, such as:
    /// - `k=0` for a DSI-based codec.
    /// - A value in the input data does not fit within the specified number of bits
    ///   for `FixedLength` encoding.
    ///
    /// # Examples
    ///
    /// ### Automatic Codec Selection
    ///
    /// Using `CodecSpec::Auto` (the default) allows the builder to choose an
    /// optimal codec based on a sample of the data.
    ///
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: Vec<u64> = (0..100).map(|x| x * x).collect();
    ///
    /// // Build with default settings (auto codec, k=32).
    /// let intvec = LEIntVec::builder(&data).build().unwrap();
    ///
    /// assert_eq!(intvec.len(), 100);
    /// assert_eq!(intvec.get(10), Some(100)); // 10*10
    /// ```
    ///
    /// ### Specifying a Codec
    ///
    /// You can explicitly set the codec and sampling rate.
    ///
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// let data: &[u64] = &[1, 2, 3, 4, 5];
    ///
    /// // Use Gamma coding with a small sampling rate.
    /// let intvec = LEIntVec::builder(data)
    ///     .codec(CodecSpec::Gamma)
    ///     .k(2)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(intvec.get(4), Some(5));
    /// assert_eq!(intvec.get_sampling_rate(), Some(2));
    /// ```
    ///
    /// ### Using Fixed-Length Encoding
    ///
    /// For data that is bounded, `FixedLength` can be very efficient.
    ///
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// // All values are less than 256, so they fit in 8 bits.
    /// let data: &[u64] = &[100, 200, 255, 0, 1];
    ///
    /// let intvec = LEIntVec::builder(data)
    ///     .codec(CodecSpec::FixedLength { num_bits: Some(8) })
    ///     .build()
    ///     .unwrap();
    ///
    /// // For FixedLength, sampling rate is not applicable.
    /// assert_eq!(intvec.get_sampling_rate(), None);
    /// assert_eq!(intvec.get(1), Some(200));
    ///
    /// // Building fails if a value does not fit.
    /// let data_too_large: &[u64] = &[256]; // 256 requires 9 bits.
    /// let result = LEIntVec::builder(data_too_large)
    ///     .codec(CodecSpec::FixedLength { num_bits: Some(8) })
    ///     .build();
    /// assert!(matches!(result, Err(IntVecError::InvalidParameters(_))));
    /// ```
    ///
    /// # Implementation Notes
    ///
    /// After writing the compressed bitstream, this method calls `shrink_to_fit`
    /// on the underlying `Vec<u64>` used for storage. While this may incur a
    /// one-time cost of reallocation and copying, it ensures that the final `IntVec`
    /// occupies the minimum necessary memory, which is critical for a data structure
    /// focused on compression.
    pub fn build(self) -> Result<IntVec<E>, IntVecError>
    where
        IntVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        let resolved_encoding = resolve_codec(self.input, self.codec_spec)?;

        if self.input.is_empty() {
            let (k, samples) = if matches!(resolved_encoding, Encoding::Dsi(_)) {
                (Some(self.k), Some(Samples::U32(Vec::new())))
            } else {
                (None, None)
            };
            return Ok(IntVec {
                data: Vec::new(),
                samples,
                k,
                len: 0,
                encoding: resolved_encoding,
                endian: PhantomData,
            });
        }

        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = IntVecBitWriter::<E>::new(word_writer);
        let mut samples = None;
        let mut k = None;

        match resolved_encoding {
            Encoding::Dsi(code) => {
                if self.k == 0 {
                    return Err(IntVecError::InvalidParameters(
                        "Sampling rate k cannot be zero for DSI encodings".to_string(),
                    ));
                }
                k = Some(self.k);
                let code_writer = FuncCodeWriter::new(code)
                    .map_err(|e| IntVecError::CodecDispatch(e.to_string()))?;
                let sample_capacity = self.input.len().div_ceil(self.k);
                let mut temp_samples = Vec::with_capacity(sample_capacity);
                let mut current_bit_offset = 0usize;

                for (i, &value) in self.input.iter().enumerate() {
                    if i % self.k == 0 {
                        temp_samples.push(current_bit_offset as u64);
                    }
                    let bits_written = code_writer.write(&mut writer, value).unwrap();
                    current_bit_offset += bits_written;
                }
                writer.write_bits(u64::MAX, 64).unwrap(); // Stopper

                samples = Some(if current_bit_offset as u64 <= u32::MAX as u64 {
                    Samples::U32(temp_samples.into_iter().map(|s| s as u32).collect())
                } else {
                    Samples::U64(temp_samples)
                });
            }
            Encoding::Fixed { num_bits } => {
                for (i, &value) in self.input.iter().enumerate() {
                    if num_bits > 0 && num_bits < 64 && value >= (1u64 << num_bits) {
                        return Err(IntVecError::InvalidParameters(format!(
                            "Value {} at index {} does not fit in {} bits",
                            value, i, num_bits
                        )));
                    }
                    writer.write_bits(value, num_bits).unwrap();
                }
            }
        }

        writer.flush().unwrap();
        let mut data = writer.into_inner().unwrap().into_inner();
        // Shrink the data to the actual size written
        data.shrink_to_fit();

        Ok(IntVec {
            data,
            samples,
            k,
            len: self.input.len(),
            encoding: resolved_encoding,
            endian: PhantomData,
        })
    }
}

/// A builder for creating an [`IntVec`] from an iterator.
///
/// This builder is obtained by calling [`IntVec::from_iter_builder`]. It is
/// designed for scenarios where data is too large to fit in memory as a
/// `Vec<u64>` or is generated on-the-fly.
///
/// # Limitations
///
/// **This builder does not support automatic parameter selection for codecs.**
/// Because the data is processed in a stream, the builder cannot analyze it beforehand
/// to determine optimal parameters. You must provide a `CodecSpec` with fixed,
/// pre-determined parameters.
#[derive(Debug)]
pub struct IntVecFromIterBuilder<E: Endianness, I: IntoIterator<Item = u64>> {
    iter: I,
    k: usize,
    codec_spec: CodecSpec,
    _endian: PhantomData<E>,
}

impl<E: Endianness, I: IntoIterator<Item = u64>> IntVecFromIterBuilder<E, I> {
    /// Creates a new builder from an iterator.
    ///
    /// This constructor is `pub(super)`, making it accessible only to the parent
    /// module. The public entry point is [`IntVec::from_iter_builder`].
    pub(super) fn new(iter: I) -> Self {
        Self {
            iter,
            k: 32,
            codec_spec: CodecSpec::Gamma, // Default to a safe, parameter-free codec
            _endian: PhantomData,
        }
    }

    /// Sets the sampling rate `k` for DSI-based codecs.
    ///
    /// Refer to [`IntVecBuilder::k`] for more details.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the codec specification.
    ///
    /// The provided [`CodecSpec`] must have all parameters explicitly defined,
    /// as automatic parameter selection is not supported for iterator-based building.
    ///
    /// # Arguments
    /// * `codec_spec`: The desired codec specification with fixed parameters.
    pub fn codec(mut self, codec_spec: CodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the `IntVec` by consuming the iterator.
    ///
    /// This method iterates through the provided data, encodes it according to the
    /// specified codec, and constructs the final `IntVec`.
    ///
    /// # Returns
    ///
    /// A `Result` containing the constructed `IntVec` on success, or an
    /// `IntVecError` on failure. Errors can occur if:
    /// - An automatic or parameter-less codec spec is provided (e.g., `CodecSpec::Auto`).
    /// - `k=0` is used with a DSI-based codec.
    /// - A value from the iterator does not fit within the specified number of bits for
    ///   `FixedLength` encoding.
    ///
    /// # Example
    /// ```rust
    /// use compressed_intvec::prelude::*;
    ///
    /// // Create an IntVec from a range iterator.
    /// let intvec = LEIntVec::from_iter_builder(0..1000u64)
    ///     .codec(CodecSpec::FixedLength{ num_bits: Some(10) }) // 1000 fits in 10 bits
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(intvec.len(), 1000);
    /// assert_eq!(intvec.get(999), Some(999));
    ///
    /// // Building fails if automatic parameter selection is attempted.
    /// let result = LEIntVec::from_iter_builder(0..100u64)
    ///     .codec(CodecSpec::Auto)
    ///     .build();
    /// assert!(matches!(result, Err(IntVecError::InvalidParameters(_))));
    /// ```
    ///
    /// # Implementation Notes
    ///
    /// Similar to the slice-based builder, this method calls `shrink_to_fit`
    /// on the underlying storage vector to ensure minimal memory usage.
    pub fn build(self) -> Result<IntVec<E>, IntVecError>
    where
        IntVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Validate that the codec spec does not require data pre-analysis.
        let resolved_encoding = match self.codec_spec {
            CodecSpec::Auto
            | CodecSpec::FixedLength { num_bits: None }
            | CodecSpec::Rice { log2_b: None }
            | CodecSpec::Zeta { k: None } => {
                return Err(IntVecError::InvalidParameters("Automatic parameter selection is not supported for iterator-based construction. Please provide fixed parameters.".to_string()));
            }
            // For other codecs, we can resolve them with an empty slice as a dummy.
            spec => resolve_codec(&[], spec)?,
        };

        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = IntVecBitWriter::<E>::new(word_writer);
        let mut samples = None;
        let mut k = None;
        let mut len = 0;

        match resolved_encoding {
            Encoding::Dsi(code) => {
                if self.k == 0 {
                    return Err(IntVecError::InvalidParameters(
                        "Sampling rate k cannot be zero for DSI encodings".to_string(),
                    ));
                }
                k = Some(self.k);
                let code_writer = FuncCodeWriter::new(code)
                    .map_err(|e| IntVecError::CodecDispatch(e.to_string()))?;
                let mut temp_samples = Vec::new();
                let mut current_bit_offset = 0usize;

                for (i, value) in self.iter.into_iter().enumerate() {
                    if i % self.k == 0 {
                        temp_samples.push(current_bit_offset as u64);
                    }
                    let bits_written = code_writer.write(&mut writer, value).unwrap();
                    current_bit_offset += bits_written;
                    len += 1;
                }
                writer.write_bits(u64::MAX, 64).unwrap(); // Stopper

                samples = Some(if current_bit_offset as u64 <= u32::MAX as u64 {
                    Samples::U32(temp_samples.into_iter().map(|s| s as u32).collect())
                } else {
                    Samples::U64(temp_samples)
                });
            }
            Encoding::Fixed { num_bits } => {
                for (i, value) in self.iter.into_iter().enumerate() {
                    if num_bits > 0 && num_bits < 64 && value >= (1u64 << num_bits) {
                        return Err(IntVecError::InvalidParameters(format!(
                            "Value {} at index {} does not fit in {} bits",
                            value, i, num_bits
                        )));
                    }
                    writer.write_bits(value, num_bits).unwrap();
                    len += 1;
                }
            }
        }

        writer.flush().unwrap();
        let mut data = writer.into_inner().unwrap().into_inner();
        // Shrink the data to the actual size written
        data.shrink_to_fit();

        Ok(IntVec {
            data,
            samples,
            k,
            len,
            encoding: resolved_encoding,
            endian: PhantomData,
        })
    }
}
