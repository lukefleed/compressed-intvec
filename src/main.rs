use compressed_intvec::{DeltaCodec, ExpGolombCodec, GammaCodec, IntVec};
use mem_dbg::{DbgFlags, MemDbg};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn generate_vec(size: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..size).map(|_| rng.random_range(0..10000)).collect()
}

fn main() {
    let random_vec = generate_vec(100000);

    // add the space of the standard vector
    println!("=== Standard Vec ===");
    random_vec
        .mem_dbg(DbgFlags::default() | DbgFlags::CAPACITY | DbgFlags::HUMANIZE)
        .unwrap();

    println!("\n=== Delta Codec ===");
    let delta_vec = IntVec::<DeltaCodec>::from(random_vec.clone(), 32).unwrap();
    delta_vec
        .mem_dbg(DbgFlags::default() | DbgFlags::CAPACITY | DbgFlags::HUMANIZE)
        .unwrap();

    println!("\n=== Gamma Codec ===");
    let gamma_vec = IntVec::<GammaCodec>::from(random_vec.clone(), 32).unwrap();
    gamma_vec
        .mem_dbg(DbgFlags::default() | DbgFlags::CAPACITY | DbgFlags::HUMANIZE)
        .unwrap();

    println!("\n=== ExpGolomb Codec ===");
    let exp_golomb_vec =
        IntVec::<ExpGolombCodec>::from_with_param(random_vec.clone(), 32, 3).unwrap();
    exp_golomb_vec
        .mem_dbg(DbgFlags::default() | DbgFlags::CAPACITY | DbgFlags::HUMANIZE)
        .unwrap();
}
