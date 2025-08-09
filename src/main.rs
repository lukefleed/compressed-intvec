use compressed_intvec::{fixed::{BitWidth, FixedVec, UFixedVec}, prelude::{IntVec, UIntVec, VariableCodecSpec}};
use mem_dbg::{DbgFlags, MemDbg};
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn generate_random_vec(size: usize, max_val_exclusive: u64) -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(42);
    if max_val_exclusive == 0 {
        // This case occurs if the requested bit width is 64.
        // We generate full-range u64 values.
        return (0..size).map(|_| rng.random::<u64>()).collect();
    }
    (0..size)
        .map(|_| rng.random_range(0..max_val_exclusive))
        .collect()
}

fn main() {
    let data = generate_random_vec(1_000_000, 1 << 36);

    println!("Size of the standard vector");
    let _ = data.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::COLOR | DbgFlags::PERCENTAGE);

    // The values require 6 bits, but we can force it to use 8 (a power of two).
    let vec: UFixedVec<u64> = FixedVec::builder()
        .bit_width(BitWidth::Minimal)
        .build(&data)
        .unwrap();

    println!("\nSize of the fixed vector with power of two bit width");
    let _ = vec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::COLOR | DbgFlags::PERCENTAGE);

    let var_vec: UIntVec<u64> = IntVec::builder(&data)
    .k(8) // Set sampling rate
    .codec(VariableCodecSpec::Gamma) // Set compression codec
    .build()
    .unwrap();

    println!("\nSize of the variable vector");
    let _ = var_vec.mem_dbg(DbgFlags::HUMANIZE | DbgFlags::COLOR | DbgFlags::PERCENTAGE);

}