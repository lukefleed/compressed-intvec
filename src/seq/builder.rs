//! Builders for constructing a [`SeqVec`].
//!
//! This module provides two builders for creating a [`SeqVec`]:
//!
//! - [`SeqVecBuilder`]: For building from a collection of sequences already in
//!   memory. This builder can analyze the data to automatically select an
//!   optimal compression codec.
//!
//! - [`SeqVecFromIterBuilder`]: For building from an iterator of sequences.
//!   This is suitable for large datasets generated on the fly, but requires
//!   the codec to be specified manually since single-pass construction cannot
//!   perform data analysis.
//!
//! [`SeqVec`]: crate::seq::SeqVec

use super::{SeqVec, SeqVecBitWriter, SeqVecError};
use crate::variable::codec::{self, Codec};
use crate::variable::traits::Storable;
use crate::fixed::{BitWidth, FixedVec};
use crate::common::codec_writer::CodecWriter;
use dsi_bitstream::{
    dispatch::StaticCodeWrite,
    impls::MemWordWriterVec,
    prelude::{BitWrite, Codes, CodesWrite, Endianness},
};
use std::marker::PhantomData;

/// A builder for creating a [`SeqVec`] from a collection of sequences.
///
/// This builder is the primary entry point for constructing a compressed
/// sequence vector when the sequences are already available in memory. It
/// allows configuration of the compression codec.
///
/// The builder always produces an owned `SeqVec<T, E, Vec<u64>>`.
///
/// ## Construction Strategy
///
/// When the codec is [`Codec::Auto`] or requires parameter
/// estimation (e.g., `Rice { log2_b: None }`), the builder performs a two-pass
/// construction:
///
/// 1. **Analysis pass**: Collects all elements to determine the optimal codec.
/// 2. **Encoding pass**: Compresses the data using the selected codec.
///
/// When a fully-specified codec is provided (e.g., `Gamma`, `Delta`,
/// `Zeta { k: Some(3) }`), the builder performs **single-pass construction**,
/// avoiding the temporary allocation of all elements.
///
/// ## Examples
///
/// ```
/// use compressed_intvec::seq::{SeqVec, LESeqVec, Codec};
///
/// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20], &[100]];
///
/// // Automatic codec selection (two-pass)
/// let vec_auto: LESeqVec<u32> = SeqVec::builder()
///     .codec(Codec::Auto)
///     .build(sequences)
///     .unwrap();
///
/// // Explicit codec (single-pass, more efficient)
/// let vec_gamma: LESeqVec<u32> = SeqVec::builder()
///     .codec(Codec::Gamma)
///     .build(sequences)
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SeqVecBuilder<T: Storable, E: Endianness> {
    codec_spec: Codec,
    store_lengths: bool,
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
    /// The default codec is [`Codec::Auto`], which analyzes the
    /// data to select the best codec.
    #[inline]
    pub fn new() -> Self {
        Self {
            codec_spec: Codec::Auto,
            store_lengths: false,
            _markers: PhantomData,
        }
    }

    /// Sets the compression codec to use.
    ///
    /// For the available codecs, see [`Codec`].
    #[inline]
    pub fn codec(mut self, codec_spec: Codec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Enables or disables storing explicit sequence lengths.
    ///
    /// When enabled, the builder stores a compact [`FixedVec`] of per-sequence
    /// lengths. This allows O(1) length queries and enables faster decoding
    /// paths that avoid end-bit checks.
    ///
    /// The default is `false` to minimize memory usage.
    #[inline]
    pub fn store_lengths(mut self, store: bool) -> Self {
        self.store_lengths = store;
        self
    }

    /// Builds the [`SeqVec`] from a slice of sequences.
    ///
    /// Each element represents a sequence to compress and store. Empty sequences
    /// are supported.
    ///
    /// ## Type Requirements
    ///
    /// The sequences can be any type that implements [`AsRef<[T]>`], such as
    /// `&[T]`, `Vec<T>`, or `Box<[T]>`.
    ///
    /// # Arguments
    ///
    /// * `sequences` - A slice of sequences to compress. Each sequence is accessed
    ///   via [`AsRef<[T]>`].
    ///
    /// # Errors
    ///
    /// Returns a [`SeqVecError`] if:
    /// - Codec resolution fails.
    /// - An I/O error occurs during encoding.
    ///
    /// [`AsRef<[T]>`]: core::convert::AsRef
    ///
    /// ## Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec};
    ///
    /// // From slice of slices
    /// let data: &[&[u32]] = &[&[1, 2], &[3, 4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::builder().build(data).unwrap();
    ///
    /// // From Vec of Vecs
    /// let data: Vec<Vec<u32>> = vec![vec![1, 2], vec![3, 4, 5]];
    /// let vec: LESeqVec<u32> = SeqVec::builder().build(&data).unwrap();
    /// ```
    pub fn build<S: AsRef<[T]>>(
        self,
        sequences: &[S],
    ) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        T: 'static,
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Determine if we can use single-pass construction.
        // Single-pass is possible only when the codec is fully specified.
        if self.codec_spec.requires_analysis() {
            self.build_two_pass(sequences)
        } else {
            self.build_single_pass(sequences)
        }
    }

    /// Two-pass construction: analyze data first, then encode.
    ///
    /// This method is used internally when the codec requires data analysis to
    /// determine optimal parameters. It collects all elements in the first pass,
    /// analyzes their distribution, then encodes them in a second pass using the
    /// selected codec. This avoids unnecessary analysis for pre-specified codecs.
    fn build_two_pass<S: AsRef<[T]>>(
        self,
        sequences: &[S],
    ) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        T: 'static,
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Resolve codec from iterator without intermediate allocation.
        // This avoids materializing all elements to a vector when analyzing data.
        let resolved_codec = codec::resolve_codec_from_iter(
            sequences
                .iter()
                .flat_map(|seq| seq.as_ref().iter().map(|x| x.to_word())),
            self.codec_spec,
        )
        .map_err(|e| SeqVecError::CodecDispatch(e.to_string()))?;

        // Pass 2: Encode with the selected codec.
        self.encode_sequences(sequences, resolved_codec)
    }

    /// Single-pass construction: encode directly without data analysis.
    ///
    /// This method is used when the codec is fully specified and requires no
    /// data analysis. It streams sequences directly to the encoder without
    /// collecting them, making it more memory-efficient for large datasets.
    fn build_single_pass<S: AsRef<[T]>>(
        self,
        sequences: &[S],
    ) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        T: 'static,
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Resolve the codec without data analysis. Pass an empty u64 slice
        // since fully-specified codecs do not require data.
        let resolved_codec = codec::resolve_codec::<u64>(&[], self.codec_spec)
            .map_err(|e| SeqVecError::CodecDispatch(e.to_string()))?;

        self.encode_sequences(sequences, resolved_codec)
    }

    /// Core encoding logic shared by both construction paths.
    ///
    /// This method handles the actual compression of sequences, including:
    /// - Iterating over all sequences and their elements
    /// - Writing compressed data to the bit writer
    /// - Tracking bit offsets for each sequence boundary
    /// - Optionally storing per-sequence lengths
    /// - Building the final [`SeqVec`] structure
    ///
    /// # Arguments
    ///
    /// * `sequences` - The sequences to compress.
    /// * `resolved_codec` - The codec to use for encoding.
    fn encode_sequences<S: AsRef<[T]>>(
        self,
        sequences: &[S],
        resolved_codec: Codes,
    ) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        let num_sequences = sequences.len();

        // Handle empty input.
        if num_sequences == 0 {
            let empty_offsets = FixedVec::<u64, u64, E>::builder()
                .bit_width(BitWidth::Minimal)
                .build(&[0u64])?;
            let seq_lengths = if self.store_lengths {
                Some(
                    FixedVec::<u64, u64, E>::builder()
                        .bit_width(BitWidth::Minimal)
                        .build(&[])?,
                )
            } else {
                None
            };
            return Ok(SeqVec {
                data: Vec::new(),
                bit_offsets: empty_offsets,
                seq_lengths,
                encoding: resolved_codec,
                _markers: PhantomData,
            });
        }

        let (data, offsets, lengths) = encode_sequences_impl(
            sequences.iter(),
            resolved_codec,
            Vec::with_capacity(num_sequences + 1),
            self.store_lengths,
            num_sequences,
        )?;

        // Build the bit offsets index with minimal bit width.
        let bit_offsets = FixedVec::<u64, u64, E>::builder()
            .bit_width(BitWidth::Minimal)
            .build(&offsets)?;

        let seq_lengths = if let Some(lengths) = lengths {
            Some(
                FixedVec::<u64, u64, E>::builder()
                    .bit_width(BitWidth::Minimal)
                    .build(&lengths)?,
            )
        } else {
            None
        };

        Ok(SeqVec {
            data,
            bit_offsets,
            seq_lengths,
            encoding: resolved_codec,
            _markers: PhantomData,
        })
    }
}

/// A builder for creating a [`SeqVec`] from an iterator of sequences.
///
/// This builder is designed for constructing a [`SeqVec`] from a data source
/// that produces sequences on the fly. It consumes the iterator in a single
/// pass, compressing sequences as they arrive.
///
/// ## Limitations
///
/// This builder does **not** support:
/// - [`Codec::Auto`]: Automatic codec selection requires analyzing
///   all data, which is impossible in a single pass.
/// - Parameter estimation for codecs like `Rice { log2_b: None }` or
///   `Zeta { k: None }`.
///
/// The codec must be fully specified. If an unsupported codec is provided,
/// the [`build`](Self::build) method will return an error.
///
/// ## Examples
///
/// ```
/// use compressed_intvec::seq::{SeqVec, LESeqVec, Codec};
///
/// // Generate sequences on the fly
/// let sequences_iter = (0..100).map(|i| vec![i as u32, i as u32 + 1]);
///
/// let vec: LESeqVec<u32> = SeqVec::from_iter_builder(sequences_iter)
///     .codec(Codec::Gamma) // Must be specified
///     .build()
///     .unwrap();
///
/// assert_eq!(vec.num_sequences(), 100);
/// ```
#[derive(Debug)]
pub struct SeqVecFromIterBuilder<T: Storable, E: Endianness, I> {
    iter: I,
    codec_spec: Codec,
    store_lengths: bool,
    _markers: PhantomData<(T, E)>,
}

impl<T, E, I, S> SeqVecFromIterBuilder<T, E, I>
where
    T: Storable,
    E: Endianness,
    I: IntoIterator<Item = S>,
    S: AsRef<[T]>,
{
    /// Creates a new builder from an iterator with default settings.
    ///
    /// The default codec is [`Codec::Gamma`], as automatic
    /// selection is not possible in single-pass construction.
    #[inline]
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            codec_spec: Codec::Gamma,
            store_lengths: false,
            _markers: PhantomData,
        }
    }

    /// Sets the compression codec to use.
    ///
    /// The codec must be fully specified (no `Auto`, no `None` parameters).
    ///
    /// # Arguments
    ///
    /// * `codec_spec` - The fully-specified codec to use for encoding.
    ///
    /// # Errors
    ///
    /// The [`build`](Self::build) method will return an error if a codec
    /// requiring data analysis is provided.
    #[inline]
    pub fn codec(mut self, codec_spec: Codec) -> Self {
        self.codec_spec = codec_spec;
        self
    }

    /// Enables or disables storing explicit sequence lengths.
    ///
    /// When enabled, the builder stores a compact [`FixedVec`] of per-sequence
    /// lengths. This allows O(1) length queries and enables faster decoding
    /// paths that avoid end-bit checks.
    ///
    /// The default is `false` to minimize memory usage.
    #[inline]
    pub fn store_lengths(mut self, store: bool) -> Self {
        self.store_lengths = store;
        self
    }

    /// Builds the [`SeqVec`] by consuming the iterator.
    ///
    /// This method streams sequences directly from the iterator without
    /// materializing them all in memory. Single-pass construction avoids
    /// temporary allocations but requires the codec to be fully specified.
    ///
    /// # Errors
    ///
    /// Returns a [`SeqVecError`] if:
    /// - An automatic or parameter-estimating codec spec is used.
    /// - An I/O error occurs during encoding.
    ///
    /// ## Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec, Codec};
    ///
    /// let sequences: Vec<Vec<u32>> = vec![vec![1, 2], vec![3, 4, 5]];
    ///
    /// let vec: LESeqVec<u32> = SeqVec::from_iter_builder(sequences.into_iter())
    ///     .codec(Codec::Delta)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn build(self) -> Result<SeqVec<T, E, Vec<u64>>, SeqVecError>
    where
        T: 'static,
        SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
    {
        // Reject codecs that require data analysis.
        if self.codec_spec.requires_analysis() {
            return Err(SeqVecError::InvalidParameters(
                "Automatic codec selection is not supported for iterator-based construction. \
                 Please provide a fully-specified codec"
                    .to_string(),
            ));
        }

        // Resolve the codec without data analysis. Pass an empty u64 slice
        // since fully-specified codecs do not require data.
        let resolved_codec = codec::resolve_codec::<u64>(&[], self.codec_spec)
            .map_err(|e| SeqVecError::CodecDispatch(e.to_string()))?;

        let iter = self.iter.into_iter();
        // Use size_hint to pre-allocate offsets for efficiency.
        let (lower, _) = iter.size_hint();
        let offsets = Vec::with_capacity(lower.saturating_add(1));

        let (data, offsets, lengths) =
            encode_sequences_impl(iter, resolved_codec, offsets, self.store_lengths, lower)?;

        // Handle empty iterator case.
        if offsets.is_empty() {
            let empty_offsets = FixedVec::<u64, u64, E>::builder()
                .bit_width(BitWidth::Minimal)
                .build(&[0u64])?;
            let seq_lengths = if self.store_lengths {
                Some(
                    FixedVec::<u64, u64, E>::builder()
                        .bit_width(BitWidth::Minimal)
                        .build(&[])?,
                )
            } else {
                None
            };
            return Ok(SeqVec {
                data: Vec::new(),
                bit_offsets: empty_offsets,
                seq_lengths,
                encoding: resolved_codec,
                _markers: PhantomData,
            });
        }

        // Build the bit offsets index.
        let bit_offsets = FixedVec::<u64, u64, E>::builder()
            .bit_width(BitWidth::Minimal)
            .build(&offsets)?;

        let seq_lengths = if let Some(lengths) = lengths {
            Some(
                FixedVec::<u64, u64, E>::builder()
                    .bit_width(BitWidth::Minimal)
                    .build(&lengths)?,
            )
        } else {
            None
        };

        Ok(SeqVec {
            data,
            bit_offsets,
            seq_lengths,
            encoding: resolved_codec,
            _markers: PhantomData,
        })
    }
}

/// Extension trait for `Codec` to check if analysis is required.
#[allow(dead_code)]
trait CodecSpecExt {
    /// Returns `true` if this codec spec requires data analysis to resolve.
    fn requires_analysis(&self) -> bool;
}

impl CodecSpecExt for Codec {
    #[inline]
    fn requires_analysis(&self) -> bool {
        matches!(
            self,
            Codec::Auto
                | Codec::Rice { log2_b: None }
                | Codec::Zeta { k: None }
                | Codec::Golomb { b: None }
                | Codec::Pi { k: None }
                | Codec::ExpGolomb { k: None }
        )
    }
}

/// Type alias for the return value of `encode_sequences_impl`.
///
/// Contains the compressed word data, bit offset boundaries, and optional
/// per-sequence lengths (stored as `u64` for architecture independence).
type EncodeSequencesResult = (Vec<u64>, Vec<u64>, Option<Vec<u64>>);

/// Shared encoding implementation for sequences from an iterator.
///
/// This function encodes all sequences using a single resolved codec and
/// pre-allocated offsets vector. The codec dispatch is resolved once at the
/// beginning via [`CodecWriter`] rather than per-element, avoiding repeated
/// dispatch overhead and improving throughput.
///
/// # Arguments
///
/// * `sequences` - Iterator of sequences to encode. Each sequence is accessed
///   via [`AsRef<[T]>`].
/// * `resolved_codec` - The codec specification to use for all elements.
/// * `offsets` - Pre-allocated vector to store bit offset boundaries. This vector
///   is populated with one offset per sequence plus a final sentinel offset.
/// * `store_lengths` - Whether to compute and store per-sequence lengths.
/// * `lengths_capacity_hint` - Capacity hint for the lengths vector when
///   `store_lengths` is true.
///
/// # Returns
///
/// A tuple containing:
/// - Encoded word data (`Vec<u64>`)
/// - Bit offset boundaries (`Vec<u64>`), with length = num_sequences + 1
/// - Optional per-sequence lengths (`Vec<u64>`), if `store_lengths` is true
///
/// [`AsRef<[T]>`]: core::convert::AsRef
fn encode_sequences_impl<T: Storable, E: Endianness, I, S>(
    sequences: I,
    resolved_codec: Codes,
    mut offsets: Vec<u64>,
    store_lengths: bool,
    lengths_capacity_hint: usize,
) -> Result<EncodeSequencesResult, SeqVecError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<[T]>,
    SeqVecBitWriter<E>: BitWrite<E, Error = core::convert::Infallible> + CodesWrite<E>,
{
    // Initialize the bit writer.
    let word_writer = MemWordWriterVec::new(Vec::new());
    let mut writer = SeqVecBitWriter::<E>::new(word_writer);
    let mut current_bit_offset: u64 = 0;

    // Resolve the codec dispatch ONCE at the beginning.
    // This eliminates per-element match overhead for common codecs.
    let code_writer = CodecWriter::new(resolved_codec);

    // Prepare optional length storage.
    let mut lengths = if store_lengths {
        Some(Vec::with_capacity(lengths_capacity_hint))
    } else {
        None
    };

    // Process each sequence, recording bit offsets at boundaries.
    for seq in sequences {
        let seq_ref = seq.as_ref();
        offsets.push(current_bit_offset);

        if let Some(ref mut lengths) = lengths {
            lengths.push(seq_ref.len() as u64);
        }

        for elem in seq_ref {
            let bits_written = code_writer.write(&mut writer, elem.to_word())?;
            current_bit_offset += bits_written as u64;
        }
    }

    // Sentinel: total bit length.
    offsets.push(current_bit_offset);

    // Finalize the writer.
    writer.flush()?;
    let mut data = writer.into_inner()?.into_inner();
    data.shrink_to_fit();

    Ok((data, offsets, lengths))
}

// --- Integration with SeqVec ---

impl<T: Storable + 'static, E: Endianness> SeqVec<T, E, Vec<u64>> {
    /// Creates a builder for constructing a [`SeqVec`] with custom settings.
    ///
    /// This is the most flexible way to create a [`SeqVec`], allowing
    /// customization of the compression codec and other parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec, Codec};
    ///
    /// let sequences: &[&[u32]] = &[&[1, 2, 3], &[10, 20]];
    ///
    /// let vec: LESeqVec<u32> = SeqVec::builder()
    ///     .codec(Codec::Zeta { k: Some(3) })
    ///     .build(sequences)
    ///     .unwrap();
    /// ```
    #[inline]
    pub fn builder() -> SeqVecBuilder<T, E> {
        SeqVecBuilder::new()
    }

    /// Creates a builder for constructing a [`SeqVec`] from an iterator.
    ///
    /// This is useful for large datasets that are generated on the fly.
    /// The codec must be specified explicitly since single-pass construction
    /// cannot perform data analysis.
    ///
    /// # Examples
    ///
    /// ```
    /// use compressed_intvec::seq::{SeqVec, LESeqVec, Codec};
    ///
    /// // Generate sequences programmatically
    /// let sequences = (0..50).map(|i| vec![i as u32; i % 5 + 1]);
    ///
    /// let vec: LESeqVec<u32> = SeqVec::from_iter_builder(sequences)
    ///     .codec(Codec::Gamma)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(vec.num_sequences(), 50);
    /// ```
    #[inline]
    pub fn from_iter_builder<I, S>(iter: I) -> SeqVecFromIterBuilder<T, E, I>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[T]>,
    {
        SeqVecFromIterBuilder::new(iter)
    }

    /// Creates a `SeqVec` from raw parts without validation.
    ///
    /// # Arguments
    ///
    /// * `data` - The compressed data buffer containing encoded words.
    /// * `bit_offsets` - Bit offset index where each offset points to the start
    ///   of a sequence. Must contain at least 2 elements: the start offset and
    ///   end sentinel.
    /// * `encoding` - The codec specification used to encode the data.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it does not validate that the `data` and
    /// `bit_offsets` are consistent with each other. The caller must ensure:
    /// - The `bit_offsets` array has at least 2 elements (start and end sentinel).
    /// - All offsets are valid bit positions within the `data` buffer.
    /// - The last offset equals the total number of bits in the compressed data.
    #[inline]
    pub unsafe fn from_raw_parts(
        data: Vec<u64>,
        bit_offsets: crate::fixed::FixedVec<u64, u64, E, Vec<u64>>,
        encoding: dsi_bitstream::prelude::Codes,
    ) -> Self {
        SeqVec {
            data,
            bit_offsets,
            seq_lengths: None,
            encoding,
            _markers: PhantomData,
        }
    }

    /// Creates a `SeqVec` from raw parts with optional stored sequence lengths.
    ///
    /// This method is identical to [`from_raw_parts`](Self::from_raw_parts) but
    /// allows providing pre-computed per-sequence lengths for faster random access.
    ///
    /// # Arguments
    ///
    /// * `data` - The compressed data buffer.
    /// * `bit_offsets` - Bit offset boundaries for each sequence.
    /// * `seq_lengths` - Optional per-sequence lengths. If provided, must have
    ///   length equal to the number of sequences (offsets.len() - 1).
    /// * `encoding` - The codec specification used to encode the data.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data`, `bit_offsets`, and `seq_lengths`
    /// (if present) are consistent with each other and the codec.
    #[inline]
    pub unsafe fn from_raw_parts_with_lengths(
        data: Vec<u64>,
        bit_offsets: crate::fixed::FixedVec<u64, u64, E, Vec<u64>>,
        seq_lengths: Option<crate::fixed::FixedVec<u64, u64, E, Vec<u64>>>,
        encoding: dsi_bitstream::prelude::Codes,
    ) -> Self {
        SeqVec {
            data,
            bit_offsets,
            seq_lengths,
            encoding,
            _markers: PhantomData,
        }
    }
}
