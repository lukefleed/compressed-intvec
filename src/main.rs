use compressed_intvec::{DeltaCodec, ExpGolombCodec, GammaCodec, IntVec};
use mem_dbg::{DbgFlags, MemDbg};

fn main() {
    let input: Vec<u64> = (0..100000).map(|_| rand::random::<u64>() % 10000).collect();

    println!("\nSpace used by the original vector:");
    input
        .mem_dbg(DbgFlags::default() | DbgFlags::CAPACITY | DbgFlags::HUMANIZE)
        .unwrap();

    let compressed_expgolomb = IntVec::<ExpGolombCodec>::from(input.clone(), 64).unwrap();

    println!("\nSpace used by the Compressed Integer Vector with Exp-Golomb codec:");
    compressed_expgolomb
        .mem_dbg(DbgFlags::default() | DbgFlags::CAPACITY | DbgFlags::HUMANIZE)
        .unwrap();

    let compressed_delta = IntVec::<DeltaCodec>::from(input.clone(), 64).unwrap();
    println!("\nSpace used by the Compressed Integer Vector with Delta codec:");
    compressed_delta
        .mem_dbg(DbgFlags::default() | DbgFlags::CAPACITY | DbgFlags::HUMANIZE)
        .unwrap();

    let compressed_gamma = IntVec::<GammaCodec>::from(input.clone(), 64).unwrap();
    println!("\nSpace used by the Compressed Integer Vector with Gamma codec:");
    compressed_gamma
        .mem_dbg(DbgFlags::default() | DbgFlags::CAPACITY | DbgFlags::HUMANIZE)
        .unwrap();
}
