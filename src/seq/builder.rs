//! Builders for constructing a [`SeqVec`].
//!
//! This module provides the builders for creating a [`SeqVec`]:
//!
//! - [`SeqVecBuilder`]: For building from a collection of sequences (slices or vectors).
//!   This builder can analyze the data to automatically select an optimal codec.
//! - [`SeqVecFromIterBuilder`]: For building from an iterator of sequences. This is
//!   suitable for large datasets that are generated on the fly, but requires
//!   manual codec specification.
//!
//! [`SeqVec`]: crate::seq::SeqVec

use super::{SeqVec, SeqVecError};
use crate::fixed::{BitWidth, FixedVec};
use crate::variable::{codec, traits::Storable, VariableCodecSpec};
use dsi_bitstream::{
    codes::{
        DeltaWrite, ExpGolombWrite, GammaWrite, GolombWrite, OmegaWrite, PiWrite, RiceWrite,
        VByteBeWrite, VByteLeWrite, ZetaWrite,
    },
    impls::{BufBitWriter, MemWordWriterVec},
    prelude::{BitWrite, Codes, CodesWrite, Endianness, LE},
};
use std::marker::PhantomData;

/// Type alias for the bit writer used internally during construction.
pub(crate) type SeqVecBitWriter<E> = BufBitWriter<E, MemWordWriterVec<u64, Vec<u64>>>;

/// A builder for creating a [`SeqVec`] from a collection of sequences.
///
/// This builder allows customization of the compression codec. It analyzes
/// all input data when [`VariableCodecSpec::Auto`] is used to select the
/// optimal codec.
///
/// # Examples
///
/// ```ignore
/// use compressed_intvec::seq::{SeqVec, LESeqVec};
/// use compressed_intvec::variable::VariableCodecSpec;
///
/// let sequences = vec![vec![1u32, 2, 3], vec![10, 20]];
///
/// let vec: LESeqVec<u32> = SeqVec::builder()
///     .codec(VariableCodecSpec::Delta)
///     .build(&sequences)
///     .unwrap();
/// ```
#[derive(Debug)]
pub struct SeqVecBuilder<T: Storable, E: Endianness> {
    codec_spec: VariableCodecSpec,
    _markers: PhantomData<(T, E)>,
}

impl<T: Storable, E: Endianness> Default for SeqVecBuilder<T, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Storable, E: Endianness> SeqVecBuilder<T, E> {
    /// Creates a new builder with default settings.
    ///
    /// The default codec is [`VariableCodecSpec::Auto`], which analyzes the
    /// input data to select the best compression codec.
    #[inline]
    pub fn new() -> Self {
        Self {
            codec_spec: VariableCodecSpec::Auto,
            _markers: PhantomData,
        }
    }

    /// Sets the compression codec to use.
    ///
    /// The choice of codec can significantly impact compression ratio and
    /// decoding speed. Use [`VariableCodecSpec::Auto`] (the default) to let
    /// the builder analyze the data and select the best codec.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    /// use compressed_intvec::variable::VariableCodecSpec;
    ///
    /// let sequences = vec![vec![1u64, 2, 3]];
    ///
    /// // Use Zeta coding with k=3
    /// let vec: LESeqVec<u64> = SeqVec::builder()
    ///     .codec(VariableCodecSpec::Zeta { k: Some(3) })
    ///     .build(&sequences)
    ///     .unwrap();
    /// ```
    #[inline]
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the [`SeqVec`] from a collection of sequences.
    ///
    /// The input can be any type that can be iterated to yield sequences,
    /// where each sequence can be referenced as a slice of `T`.
    ///
    /// # Errors
    ///
    /// Returns a [`SeqVecError`] if compression fails or if the codec
    /// parameters are invalid.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// // From a slice of slices
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::builder().build(sequences).unwrap();
    ///
    /// // From a Vec of Vecs
    /// let sequences: Vec<Vec<u32>> = vec![vec![1, 2], vec![3, 4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::builder().build(&sequences).unwrap();
    /// ```
    pub fn build<I, S>(self, sequences: I) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[T]>,
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Collect sequences for analysis and encoding.
        // We need to iterate twice: once for codec analysis, once for encoding.
        let sequences: Vec<Vec<T>> = sequences.into_iter().map(|s| s.as_ref().to_vec()).collect();

        let num_sequences = sequences.len();

        // Handle empty case.
        if num_sequences == 0 {
            let empty_offsets = FixedVec::<u64, u64, LE>::builder()
                .bit_width(BitWidth::Explicit(1))
                .build(&[0u64])
                .unwrap();
            return Ok(unsafe { SeqVec::new_unchecked(Vec::new(), empty_offsets, Codes::Gamma) });
        }

        // Flatten all elements for codec analysis.
        let all_words: Vec<u64> = sequences
            .iter()
            .flat_map(|seq| seq.iter().map(|&x| x.to_word()))
            .collect();

        // Resolve the codec. If all sequences are empty, use Gamma as default.
        let resolved_code = if all_words.is_empty() {
            Codes::Gamma
        } else {
            codec::resolve_codec(&all_words, self.codec_spec).map_err(|e| {
                SeqVecError::InvalidParameters(format!("Failed to resolve codec: {}", e))
            })?
        };

        // Prepare the bit writer.
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = SeqVecBitWriter::<E>::new(word_writer);

        // Track bit offsets for each sequence.
        // We store N+1 offsets: the start of each sequence plus the final position.
        let mut bit_offsets: Vec<u64> = Vec::with_capacity(num_sequences + 1);
        let mut current_bit_offset: u64 = 0;

        // Encode each sequence and record its starting bit offset.
        for seq in &sequences {
            bit_offsets.push(current_bit_offset);

            for &elem in seq.iter() {
                let word = elem.to_word();
                let bits_written = write_code(&mut writer, word, resolved_code)?;
                current_bit_offset += bits_written;
            }
        }

        // Push the sentinel offset (total bits).
        bit_offsets.push(current_bit_offset);

        // Finalize the bitstream.
        writer.flush().map_err(SeqVecError::Io)?;
        let mut data = writer.into_inner().unwrap().into_inner();
        data.shrink_to_fit();

        // Build the bit offsets FixedVec with minimal bit width.
        let bit_offsets_vec = FixedVec::<u64, u64, LE>::builder()
            .bit_width(BitWidth::Minimal)
            .build(&bit_offsets)?;

        Ok(unsafe { SeqVec::new_unchecked(data, bit_offsets_vec, resolved_code) })
    }
}

/// A builder for creating a [`SeqVec`] from an iterator of sequences.
///
/// This builder is suitable for large datasets that are generated on the fly
/// or cannot fit entirely in memory. Unlike [`SeqVecBuilder`], it requires
/// manual codec specification because the data cannot be analyzed in advance.
///
/// # Examples
///
/// ```ignore
/// use compressed_intvec::seq::{SeqVec, LESeqVec};
/// use compressed_intvec::variable::VariableCodecSpec;
///
/// // Generate sequences on the fly
/// let sequences = (0..100).map(|i| vec![i as u32, (i + 1) as u32]);
///
/// let vec: LESeqVec<u32> = SeqVec::from_iter_builder(sequences)
///     .codec(VariableCodecSpec::Gamma)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug)]
pub struct SeqVecFromIterBuilder<T: Storable, E: Endianness, I> {
    iter: I,
    codec_spec: VariableCodecSpec,
    _markers: PhantomData<(T, E)>,
}

impl<T: Storable + 'static, E: Endianness> SeqVec<T, E, Vec<u64>> {
    /// Creates a builder for constructing a [`SeqVec`] from an iterator.
    ///
    /// This method is useful for streaming construction when sequences are
    /// generated on the fly.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator yielding sequences (each sequence must be
    ///   convertible to a slice of `T`).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    /// use compressed_intvec::variable::VariableCodecSpec;
    ///
    /// let sequences = vec![vec![1u32, 2], vec![3, 4, 5]];
    ///
    /// let vec: LESeqVec<u32> = SeqVec::from_iter_builder(sequences.into_iter())
    ///     .codec(VariableCodecSpec::Delta)
    ///     .build()
    ///     .unwrap();
    /// ```
    #[inline]
    pub fn from_iter_builder<I, S>(iter: I) -> SeqVecFromIterBuilder<T, E, I>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[T]>,
    {
        SeqVecFromIterBuilder::new(iter)
    }
}

impl<T: Storable, E: Endianness, I> SeqVecFromIterBuilder<T, E, I> {
    /// Creates a new iterator builder.
    #[inline]
    pub(crate) fn new(iter: I) -> Self {
        Self {
            iter,
            codec_spec: VariableCodecSpec::Gamma, // Default for streaming
            _markers: PhantomData,
        }
    }

    /// Sets the compression codec to use.
    ///
    /// Unlike [`SeqVecBuilder`], this builder cannot use [`VariableCodecSpec::Auto`]
    /// because the data is streamed and cannot be analyzed in advance.
    ///
    /// # Panics
    ///
    /// The [`build`](Self::build) method will return an error if
    /// [`VariableCodecSpec::Auto`] is used.
    #[inline]
    pub fn codec(mut self, codec_spec: VariableCodecSpec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Builds the [`SeqVec`] by consuming the iterator.
    ///
    /// # Errors
    ///
    /// Returns a [`SeqVecError`] if:
    /// - [`VariableCodecSpec::Auto`] is used (not supported for streaming).
    /// - Compression fails.
    pub fn build<S>(self) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[T]>,
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Auto codec is not supported for streaming.
        if matches!(self.codec_spec, VariableCodecSpec::Auto) {
            return Err(SeqVecError::InvalidParameters(
                "VariableCodecSpec::Auto is not supported for from_iter_builder. \
                 Specify an explicit codec."
                    .to_string(),
            ));
        }

        // Resolve the codec from the spec.
        // Since we don't have data to analyze, we pass an empty slice.
        let resolved_code = codec::resolve_codec(&[], self.codec_spec).map_err(|e| {
            SeqVecError::InvalidParameters(format!("Failed to resolve codec: {}", e))
        })?;

        // Prepare the bit writer.
        let word_writer = MemWordWriterVec::new(Vec::new());
        let mut writer = SeqVecBitWriter::<E>::new(word_writer);

        let mut bit_offsets: Vec<u64> = Vec::new();
        let mut current_bit_offset: u64 = 0;

        // Encode each sequence.
        for seq in self.iter {
            bit_offsets.push(current_bit_offset);

            for &elem in seq.as_ref().iter() {
                let word = elem.to_word();
                let bits_written = write_code(&mut writer, word, resolved_code)?;
                current_bit_offset += bits_written;
            }
        }

        // Push the sentinel.
        bit_offsets.push(current_bit_offset);

        // Handle the edge case of zero sequences.
        if bit_offsets.len() == 1 {
            bit_offsets.push(0);
        }

        // Finalize.
        writer.flush().map_err(SeqVecError::Io)?;
        let mut data = writer.into_inner().unwrap().into_inner();
        data.shrink_to_fit();

        let bit_offsets_vec = FixedVec::<u64, u64, LE>::builder()
            .bit_width(BitWidth::Minimal)
            .build(&bit_offsets)?;

        Ok(unsafe { SeqVec::new_unchecked(data, bit_offsets_vec, resolved_code) })
    }
}

/// Writes a single value using the specified codec and returns the number of
/// bits written.
///
/// This function dispatches to the appropriate codec write method based on
/// the [`Codes`] enum variant.
#[inline]
fn write_code<E: Endianness, W>(writer: &mut W, value: u64, code: Codes) -> Result<u64, SeqVecError>
where
    W: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
{
    let bits = match code {
        Codes::Unary => writer.write_unary(value).unwrap(),
        Codes::Gamma => writer.write_gamma(value).unwrap(),
        Codes::Delta => writer.write_delta(value).unwrap(),
        Codes::Omega => writer.write_omega(value).unwrap(),
        Codes::VByte => {
            // Dispatch based on endianness at runtime.
            // This is not ideal but necessary given the current dsi-bitstream API.
            if E::VALUE == dsi_bitstream::prelude::LE::VALUE {
                VByteLeWrite::write_vbyte_le(writer, value).unwrap()
            } else {
                VByteBeWrite::write_vbyte_be(writer, value).unwrap()
            }
        }
        Codes::Zeta { k } => writer.write_zeta(value, k).unwrap(),
        Codes::Pi { k } => writer.write_pi(value, k).unwrap(),
        Codes::PiWeb { k } => writer.write_pi_web(value, k).unwrap(),
        Codes::Golomb { b } => writer.write_golomb(value, b).unwrap(),
        Codes::Rice { log2_b } => writer.write_rice(value, log2_b).unwrap(),
        Codes::ExpGolomb { k } => writer.write_exp_golomb(value, k).unwrap(),
    };
    Ok(bits as u64)
}
