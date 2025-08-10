use compressed_intvec::prelude::*;
use mem_dbg::{DbgFlags, MemDbg};
use rand::{rngs::SmallRng, Rng, SeedableRng};

// Generates a vector with uniformly random values.
fn generate_random_vec(size: usize, max: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..size).map(|_| rng.random_range(0..max)).collect()
}

fn main() {
    let data = generate_random_vec(10_000, 1 << 20);

    println!("Size of the uncompressed Vec<u64>:");
    data.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE);

    // Create an IntVec with a generic Gamma encoding.
    let gamma_intvec = LEIntVec::builder(&data)
        .codec(VariableCodecSpec::Delta)
        .build()
        .unwrap();

    println!("\nSize of the IntVec with Gamma encoding:");
    gamma_intvec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE);

    // Let the library analyze the data and choose the best codec.
    let auto_intvec = LEIntVec::builder(&data)
        .codec(VariableCodecSpec::Auto)
        .build()
        .unwrap();

    println!("\nSize of the IntVec with Auto encoding:");
    auto_intvec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::PERCENTAGE);
    println!("\nCodec selected by Auto: {:?}", auto_intvec.encoding());
}
