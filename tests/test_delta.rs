#[cfg(test)]
mod tests {
    use compressed_intvec::{DeltaCodec, IntVec};

    #[test]
    fn test_delta_codec() {
        let input: Vec<u64> = (0..1000).map(|_| rand::random::<u64>() % 10000).collect();
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64).unwrap();

        for i in 0..input.len() {
            assert_eq!(input[i], compressed_input.get(i).unwrap());
        }
    }

    #[test]
    fn test_delta_codec_empty() {
        let input: Vec<u64> = Vec::new();
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64).unwrap();

        assert_eq!(compressed_input.len(), 0);
        assert_eq!(compressed_input.is_empty(), true);
    }

    #[test]
    fn test_delta_codec_single_element() {
        let input: Vec<u64> = vec![42];
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64).unwrap();

        assert_eq!(compressed_input.len(), 1);
        assert_eq!(compressed_input.get(0).unwrap(), 42);
    }

    #[test]
    fn test_delta_codec_sequential() {
        let input: Vec<u64> = (0..100).collect();
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64).unwrap();

        for i in 0..input.len() {
            assert_eq!(input[i], compressed_input.get(i).unwrap());
        }
    }

    #[test]
    fn test_delta_codec_descending() {
        let input: Vec<u64> = (0..100).rev().collect();
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64).unwrap();

        for i in 0..input.len() {
            assert_eq!(input[i], compressed_input.get(i).unwrap());
        }
    }

    #[test]
    fn test_delta_codec_large_gaps() {
        let input: Vec<u64> = (0..100).map(|x| x * 100).collect();
        let compressed_input = IntVec::<DeltaCodec>::from(input.clone(), 64).unwrap();

        for i in 0..input.len() {
            assert_eq!(input[i], compressed_input.get(i).unwrap());
        }
    }
}
