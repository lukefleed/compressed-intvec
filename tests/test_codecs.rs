extern crate compressed_intvec;
use compressed_intvec::{BEIntVec, Codec, DeltaCodec, ExpGolombCodec, GammaCodec, LEIntVec};

#[cfg(test)]
mod tests {
    use super::*;
    use compressed_intvec::{ParamDeltaCodec, ParamGammaCodec, RiceCodec};
    use dsi_bitstream::{
        impls::{BufBitWriter, MemWordWriterVec},
        traits::{BE, LE},
    };
    use rand::{Rng, SeedableRng};

    /// Generate a random vector of `u64` values using a fixed seed for reproducibility.
    fn generate_random_vec(size: usize) -> Vec<u64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        (0..size).map(|_| rng.random_range(0..1000)).collect()
    }

    /// Helper for testing LE codecs.
    ///
    /// Constructs an `LEIntVec` with the given codec and codec parameter,
    /// then verifies that every index returns the original value.
    /// Also confirms that out-of-bound accesses return `None`.
    fn test_codec_le<C>(input: &[u64], k: usize, codec_param: C::Params)
    where
        C: Codec<LE, BufBitWriter<LE, MemWordWriterVec<u64, Vec<u64>>>>,
        C::Params: Copy + Clone + PartialEq + std::fmt::Debug,
    {
        let vec_le = LEIntVec::<C>::from_with_param(input.to_vec(), k, codec_param.clone())
            .expect("Failed to create LEIntVec");
        assert_eq!(vec_le.len(), input.len());
        for (i, &val) in input.iter().enumerate() {
            assert_eq!(vec_le.get(i).unwrap(), val, "Mismatch at index {}", i);
        }
        assert!(vec_le.get(input.len()).is_none());
    }

    /// Helper for testing BE codecs.
    ///
    /// Since `BEIntVec` does not provide a `get` method, this function validates
    /// internal fields like `len`, `k`, and the computed sample positions.
    fn test_codec_be<C>(input: &[u64], k: usize, codec_param: C::Params)
    where
        C: Codec<BE, BufBitWriter<BE, MemWordWriterVec<u64, Vec<u64>>>>,
        C::Params: Clone + Copy + std::fmt::Debug,
    {
        let vec_be = BEIntVec::<C>::from_with_param(input.to_vec(), k, codec_param.clone())
            .expect("Failed to create BEIntVec");
        assert_eq!(vec_be.len(), input.len());
        assert_eq!(vec_be.k, k);
        let expected_samples = if input.is_empty() {
            0
        } else {
            (input.len() + k - 1) / k
        };
        assert_eq!(vec_be.samples.len(), expected_samples);
    }

    // --- GammaCodec Tests ---
    mod gamma_tests {
        use super::*;
        #[test]
        fn le() {
            let input = generate_random_vec(100);
            let k = 3;
            test_codec_le::<GammaCodec>(&input, k, ());
        }

        #[test]
        fn be() {
            let input = generate_random_vec(100);
            let k = 2;
            test_codec_be::<GammaCodec>(&input, k, ());
        }
    }

    // --- DeltaCodec Tests ---
    mod delta_tests {
        use super::*;
        #[test]
        fn le() {
            let input = generate_random_vec(100);
            let k = 2;
            test_codec_le::<DeltaCodec>(&input, k, ());
        }

        #[test]
        fn be() {
            let input = generate_random_vec(100);
            let k = 2;
            test_codec_be::<DeltaCodec>(&input, k, ());
        }
    }

    // --- Exp-Golomb Codec Tests ---
    mod exp_golomb_tests {
        use super::*;
        #[test]
        fn le() {
            let input = generate_random_vec(100);
            let k = 2;
            test_codec_le::<ExpGolombCodec>(&input, k, 3);
        }

        #[test]
        fn be() {
            let input = generate_random_vec(100);
            let k = 1;
            test_codec_be::<ExpGolombCodec>(&input, k, 3);
        }
    }

    // --- RiceCodec Tests ---
    mod rice_tests {
        use super::*;
        #[test]
        fn le() {
            let input = generate_random_vec(100);
            let k = 2;
            test_codec_le::<RiceCodec>(&input, k, 3);
        }

        #[test]
        fn be() {
            let input = generate_random_vec(100);
            let k = 2;
            test_codec_be::<RiceCodec>(&input, k, 3);
        }
    }

    // --- ParamDeltaCodec Tests ---
    mod param_delta_tests {
        use super::*;
        #[test]
        fn le() {
            let input = generate_random_vec(100);
            let k = 2;
            test_codec_le::<ParamDeltaCodec<true, true>>(&input, k, ());
            test_codec_le::<ParamDeltaCodec<false, true>>(&input, k, ());
            test_codec_le::<ParamDeltaCodec<true, false>>(&input, k, ());
            test_codec_le::<ParamDeltaCodec<false, false>>(&input, k, ());
        }

        #[test]
        fn be() {
            let input = generate_random_vec(100);
            let k = 2;
            test_codec_be::<ParamDeltaCodec<true, true>>(&input, k, ());
            test_codec_be::<ParamDeltaCodec<false, true>>(&input, k, ());
            test_codec_be::<ParamDeltaCodec<true, false>>(&input, k, ());
            test_codec_be::<ParamDeltaCodec<false, false>>(&input, k, ());
        }
    }

    // --- ParamGammaCodec Tests ---
    mod param_gamma_tests {
        use super::*;
        #[test]
        fn le() {
            let input = generate_random_vec(100);
            let k = 2;
            test_codec_le::<ParamGammaCodec<true>>(&input, k, ());
            test_codec_le::<ParamGammaCodec<false>>(&input, k, ());
        }

        #[test]
        fn be() {
            let input = generate_random_vec(100);
            let k = 2;
            test_codec_be::<ParamGammaCodec<true>>(&input, k, ());
            test_codec_be::<ParamGammaCodec<false>>(&input, k, ());
        }
    }

    // --- Edge Case Tests ---
    mod edge_cases {
        use super::*;
        #[test]
        fn empty_input_le() {
            let input: Vec<u64> = vec![];
            let k = 3;
            let vec_le = LEIntVec::<GammaCodec>::from_with_param(input.clone(), k, ())
                .expect("Failed on empty input");
            assert_eq!(vec_le.len, 0);
            assert!(vec_le.get(0).is_none());
        }

        #[test]
        fn empty_input_be() {
            let input: Vec<u64> = vec![];
            let vec_be = BEIntVec::<DeltaCodec>::from_with_param(input.clone(), 2, ())
                .expect("Failed on empty input");
            assert_eq!(vec_be.len, 0);
            assert_eq!(vec_be.samples.len(), 0);
        }

        #[test]
        fn single_element_le() {
            let input = vec![42];
            let k = 3;
            test_codec_le::<GammaCodec>(&input, k, ());
        }

        #[test]
        fn single_element_be() {
            let input = vec![42];
            let k = 3;
            test_codec_be::<DeltaCodec>(&input, k, ());
        }

        // #[test]
        // fn test_in_order_iter() {
        //     let input = generate_random_vec(100);
        //     let k = 3;
        //     let vec_le = LEIntVec::<GammaCodec>::from_with_param(input.clone(), k, ())
        //         .expect("Failed to create LEIntVec");

        //     for (i, val) in vec_le.iter().enumerate() {
        //         assert_eq!(val, input[i]);
        //     }

        //     assert_eq!(vec_le.into_vec(), input);
        // }
    }
}
