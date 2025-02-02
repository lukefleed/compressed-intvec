use compressed_intvec::{GammaCodec, IntVec};

fn main() {
    let input: Vec<u64> = (0..1000).map(|_| rand::random::<u64>() % 10000).collect();
    let compressed_input = IntVec::<GammaCodec>::from(input.clone(), GammaCodec, 64);

    for i in 0..compressed_input.len() {
        println!("{} {} {}", i, compressed_input.get(i).unwrap(), input[i]);
    }
}
