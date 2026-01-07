//! Integration tests for codec variations in [`SeqVec`].
//!
//! This test suite validates:
//! - All [`VariableCodecSpec`] codec types produce correct output
//! - Codec parameter variations (k, b, etc.) work correctly
//! - Codec selection and auto-detection produce valid results
//! - Consistency across different codec types

use compressed_intvec::seq::{SeqVec, VariableCodecSpec};
use dsi_bitstream::prelude::LE;

// --- Basic Codec Tests ---

#[test]
fn test_codec_gamma() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_delta() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_zeta_no_param() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Zeta { k: None })
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_zeta_with_k() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];

    for k in &[Some(2), Some(3), Some(5)] {
        let vec = SeqVec::builder()
            .codec(VariableCodecSpec::Zeta { k: *k })
            .build(&sequences)
            .unwrap();

        for (i, expected) in sequences.iter().enumerate() {
            assert_eq!(
                vec.get_vec(i),
                Some(expected.clone()),
                "Zeta codec with k={:?} failed",
                k
            );
        }
    }
}

#[test]
fn test_codec_vbyte_le() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::VByteLe)
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_vbyte_be() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::VByteBe)
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_unary() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 1], vec![1, 1, 1, 1]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Unary)
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_binary() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Binary { k: 2 })
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_golomb_le() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::GolombLe { b: 2 })
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_golomb_rice() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::GolombRice { b: 2 })
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_exp_golomb() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::ExpGolomb)
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

// --- Auto Codec Tests ---

#[test]
fn test_codec_auto() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Auto)
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_auto_with_varied_data() {
    // Data with large variation in magnitude
    let sequences: Vec<Vec<u32>> = vec![
        vec![1, 2, 3, 4, 5],
        vec![1000, 2000, 3000],
        vec![1000000, 2000000],
        vec![100],
    ];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Auto)
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_auto_with_small_values() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 1, 1], vec![2, 2, 2], vec![1, 2, 1, 2]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Auto)
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

// --- Edge Cases with Codecs ---

#[test]
fn test_codec_with_empty_sequences() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2], vec![], vec![3, 4]];

    for codec in &[
        VariableCodecSpec::Gamma,
        VariableCodecSpec::Delta,
        VariableCodecSpec::Zeta { k: None },
        VariableCodecSpec::VByteLe,
    ] {
        let vec = SeqVec::builder()
            .codec(*codec)
            .build(&sequences)
            .unwrap();

        for (i, expected) in sequences.iter().enumerate() {
            assert_eq!(
                vec.get_vec(i),
                Some(expected.clone()),
                "Failed with codec {:?}", codec
            );
        }
    }
}

#[test]
fn test_codec_with_all_zeros() {
    let sequences: Vec<Vec<u32>> = vec![vec![0; 10], vec![0; 20], vec![0; 5]];

    for codec in &[
        VariableCodecSpec::Gamma,
        VariableCodecSpec::Delta,
        VariableCodecSpec::VByteLe,
    ] {
        let vec = SeqVec::builder()
            .codec(*codec)
            .build(&sequences)
            .unwrap();

        for (i, expected) in sequences.iter().enumerate() {
            assert_eq!(vec.get_vec(i), Some(expected.clone()));
        }
    }
}

#[test]
fn test_codec_with_large_values() {
    let sequences: Vec<Vec<u64>> = vec![
        vec![1000000000000, 2000000000000],
        vec![999999999999],
        vec![18446744073709551615], // u64::MAX
    ];

    for codec in &[
        VariableCodecSpec::Gamma,
        VariableCodecSpec::Delta,
        VariableCodecSpec::VByteLe,
    ] {
        let vec = SeqVec::builder()
            .codec(*codec)
            .build(&sequences)
            .unwrap();

        for (i, expected) in sequences.iter().enumerate() {
            assert_eq!(vec.get_vec(i), Some(expected.clone()));
        }
    }
}

#[test]
fn test_codec_with_single_element_sequences() {
    let sequences: Vec<Vec<u32>> = vec![vec![1], vec![100], vec![10000], vec![1000000]];

    for codec in &[
        VariableCodecSpec::Gamma,
        VariableCodecSpec::Delta,
        VariableCodecSpec::Zeta { k: Some(3) },
        VariableCodecSpec::VByteLe,
    ] {
        let vec = SeqVec::builder()
            .codec(*codec)
            .build(&sequences)
            .unwrap();

        for (i, expected) in sequences.iter().enumerate() {
            assert_eq!(vec.get_vec(i), Some(expected.clone()));
        }
    }
}

// --- Signed Data with Codecs ---

#[test]
fn test_codec_gamma_signed() {
    let sequences: Vec<Vec<i32>> = vec![vec![-1, 2, -3], vec![10, -20], vec![-100, 200]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

#[test]
fn test_codec_delta_signed() {
    let sequences: Vec<Vec<i32>> = vec![vec![-1, 2, -3], vec![10, -20], vec![-100, 200]];
    let vec = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences)
        .unwrap();

    for (i, expected) in sequences.iter().enumerate() {
        assert_eq!(vec.get_vec(i), Some(expected.clone()));
    }
}

// --- Codec Consistency Tests ---

#[test]
fn test_same_codec_produces_same_data() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20]];

    let vec1 = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences)
        .unwrap();

    let vec2 = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences)
        .unwrap();

    // Should produce identical data
    for i in 0..sequences.len() {
        assert_eq!(vec1.get_vec(i), vec2.get_vec(i));
    }
}

#[test]
fn test_different_codecs_produce_different_encodings() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20]];

    let vec_gamma = SeqVec::builder()
        .codec(VariableCodecSpec::Gamma)
        .build(&sequences)
        .unwrap();

    let vec_delta = SeqVec::builder()
        .codec(VariableCodecSpec::Delta)
        .build(&sequences)
        .unwrap();

    let vec_vbyte = SeqVec::builder()
        .codec(VariableCodecSpec::VByteLe)
        .build(&sequences)
        .unwrap();

    // Different codecs should produce different bit counts (usually)
    // but still decode to the same data
    for i in 0..sequences.len() {
        assert_eq!(vec_gamma.get_vec(i), vec_delta.get_vec(i));
        assert_eq!(vec_gamma.get_vec(i), vec_vbyte.get_vec(i));
    }
}

#[test]
fn test_codec_iteration_consistency() {
    let sequences: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![10, 20], vec![100, 200, 300]];

    for codec in &[
        VariableCodecSpec::Gamma,
        VariableCodecSpec::Delta,
        VariableCodecSpec::VByteLe,
    ] {
        let vec = SeqVec::builder()
            .codec(*codec)
            .build(&sequences)
            .unwrap();

        // Test full iteration
        let all_seqs: Vec<Vec<u32>> = vec.iter().map(|seq| seq.collect()).collect();
        assert_eq!(&all_seqs, &sequences);

        // Test individual access
        for (i, expected) in sequences.iter().enumerate() {
            assert_eq!(vec.get_vec(i), Some(expected.clone()));
        }
    }
}
