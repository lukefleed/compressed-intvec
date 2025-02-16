extern crate compressed_intvec;
use compressed_intvec::{
    BEIntVec, Codec, DeltaCodec, ExpGolombCodec, GammaCodec, LEIntVec, MyBitRead, MyBitWrite,
};

#[cfg(test)]
mod tests {
    use compressed_intvec::{
        ParamDeltaCodec, ParamGammaCodec, ParamZetaCodec, RiceCodec, ZetaCodec,
    };
    use dsi_bitstream::traits::{BE, LE};
    use rand::{Rng, SeedableRng};

    use super::*;

    // Helper function to generate a random vector of u64 values with seed 42
    fn generate_random_vec(size: usize) -> Vec<u64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        (0..size).map(|_| rng.random::<u64>()).collect()
    }

    // Helper for testing LE vectors.
    //
    // This function constructs a LEIntVec using the provided codec and codec parameter,
    // then verifies that every index returns the original input value and that out‐of‐bounds
    // accesses return None.
    fn test_codec_le<C>(input: &[u64], k: usize, codec_param: C::Params)
    where
        C: Codec<LE, MyBitWrite<LE>, MyBitRead<LE>>,
        C::Params: Clone + PartialEq + std::fmt::Debug,
    {
        let vec_le = LEIntVec::<C>::from_with_param(input.to_vec(), k, codec_param.clone())
            .expect("Failed to create LEIntVec");
        // Check that the length is correct.
        assert_eq!(vec_le.len(), input.len());
        // Validate each element
        for (i, &val) in input.iter().enumerate() {
            assert_eq!(vec_le.get(i).unwrap(), val, "Mismatch at index {}", i);
        }
        // Out-of-bound access should return None.
        assert!(vec_le.get(input.len()).is_none());
    }

    // Helper for testing BE vectors.
    //
    // As BEIntVec does not provide a get method, we test internal fields such as len,
    // k, and samples.
    fn test_codec_be<C>(input: &[u64], k: usize, codec_param: C::Params)
    where
        C: Codec<BE, MyBitWrite<BE>, MyBitRead<BE>>,
        C::Params: Clone + std::fmt::Debug,
    {
        let vec_be = BEIntVec::<C>::from_with_param(input.to_vec(), k, codec_param.clone())
            .expect("Failed to create BEIntVec");
        // Verify the total number of values.
        assert_eq!(vec_be.len(), input.len());
        // Verify the sampling parameter.
        assert_eq!(vec_be.k, k);
        // Verify that the number of computed sample positions is correct.
        let expected_samples = if input.is_empty() {
            0
        } else {
            (input.len() + k - 1) / k
        };
        assert_eq!(vec_be.samples.len(), expected_samples);
    }

    // --- GammaCodec Tests ---

    #[test]
    fn test_gamma_codec_le() {
        // GammaCodec requires no extra runtime parameter (use unit type ()).
        let input = generate_random_vec(100);
        let k = 3;
        test_codec_le::<GammaCodec>(&input, k, ());
    }

    #[test]
    fn test_gamma_codec_be() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_be::<GammaCodec>(&input, k, ());
    }

    // --- DeltaCodec Tests ---

    #[test]
    fn test_delta_codec_le() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_le::<DeltaCodec>(&input, k, ());
    }

    #[test]
    fn test_delta_codec_be() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_be::<DeltaCodec>(&input, k, ());
    }

    // --- Exp‑Golomb Codec Tests ---
    //
    // For Exp‑Golomb, we assume the codec parameter is an integer (e.g. k = 3).
    #[test]
    fn test_exp_golomb_codec_le() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_le::<ExpGolombCodec>(&input, k, 3);
    }

    #[test]
    fn test_exp_golomb_codec_be() {
        let input = generate_random_vec(100);
        let k = 1;
        test_codec_be::<ExpGolombCodec>(&input, k, 3);
    }

    // --- Test ZetaCodec ---
    //
    // For ZetaCodec, we assume the codec parameter is an integer (e.g. k = 3).
    #[test]
    fn test_zeta_codec_le() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_le::<ZetaCodec>(&input, k, 3);
    }

    #[test]
    fn test_zeta_codec_be() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_be::<ZetaCodec>(&input, k, 3);
    }

    // --- Test RiceCodec ---
    //
    // For RiceCodec, we assume the codec parameter is an integer (e.g. k = 3).
    #[test]
    fn test_rice_codec_le() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_le::<RiceCodec>(&input, k, 3);
    }

    #[test]
    fn test_rice_codec_be() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_be::<RiceCodec>(&input, k, 3);
    }

    // --- Test ParamZetaCodec ---
    //
    // For ParamZetaCodec, we assume the codec parameter is an integer (e.g. k = 3) and that the USE_TABLE is set to true and then false.
    #[test]
    fn test_param_zeta_codec_le() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_le::<ParamZetaCodec<true>>(&input, k, ());
        test_codec_le::<ParamZetaCodec<false>>(&input, k, ());
    }

    #[test]
    fn test_param_zeta_codec_be() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_be::<ParamZetaCodec<true>>(&input, k, ());
        test_codec_be::<ParamZetaCodec<false>>(&input, k, ());
    }

    // --- Test ParamDeltaCodec ---
    //
    // For ParamDeltaCodec, we assume the codec parameter is an integer (e.g. k = 3) and that the USE_TABLE is set to true and then false.
    #[test]
    fn test_param_delta_codec_le() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_le::<ParamDeltaCodec<true, true>>(&input, k, ());
        test_codec_le::<ParamDeltaCodec<false, true>>(&input, k, ());
        test_codec_le::<ParamDeltaCodec<true, false>>(&input, k, ());
        test_codec_le::<ParamDeltaCodec<false, false>>(&input, k, ());
    }

    #[test]
    fn test_param_delta_codec_be() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_be::<ParamDeltaCodec<true, true>>(&input, k, ());
        test_codec_be::<ParamDeltaCodec<false, true>>(&input, k, ());
        test_codec_be::<ParamDeltaCodec<true, false>>(&input, k, ());
        test_codec_be::<ParamDeltaCodec<false, false>>(&input, k, ());
    }

    // --- Test ParamGammaCodec ---
    //
    // For ParamGammaCodec, we assume the codec parameter is an integer (e.g. k = 3) and that the USE_TABLE is set to true and then false.

    #[test]
    fn test_param_gamma_codec_le() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_le::<ParamGammaCodec<true>>(&input, k, ());
        test_codec_le::<ParamGammaCodec<false>>(&input, k, ());
    }

    #[test]
    fn test_param_gamma_codec_be() {
        let input = generate_random_vec(100);
        let k = 2;
        test_codec_be::<ParamGammaCodec<true>>(&input, k, ());
        test_codec_be::<ParamGammaCodec<false>>(&input, k, ());
    }

    // --- Edge cases ---
    #[test]
    fn test_empty_input_le() {
        let input: Vec<u64> = vec![];
        let k = 3;
        let vec_le = LEIntVec::<GammaCodec>::from_with_param(input.clone(), k, ())
            .expect("Failed on empty input");
        assert_eq!(vec_le.len, 0);
        assert!(vec_le.get(0).is_none());
    }

    #[test]
    fn test_empty_input_be() {
        let input: Vec<u64> = vec![];
        let vec_be = BEIntVec::<DeltaCodec>::from_with_param(input.clone(), 2, ())
            .expect("Failed on empty input");
        assert_eq!(vec_be.len, 0);
        assert_eq!(vec_be.samples.len(), 0);
    }

    #[test]
    fn test_single_element_le() {
        let input = vec![42];
        let k = 3;
        test_codec_le::<GammaCodec>(&input, k, ());
    }

    #[test]
    fn test_single_element_be() {
        let input = vec![42];
        let k = 3;
        test_codec_be::<DeltaCodec>(&input, k, ());
    }
}
