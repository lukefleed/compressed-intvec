// use compressed_intvec::{
//     codecs::{DeltaCodec, MinimalBinaryCodec},
//     intvec::{BEIntVec, LEIntVec},
// };
// use mem_dbg::{DbgFlags, MemDbg};
// use rand::SeedableRng;
// use rand_distr::{Distribution, Uniform};

// fn generate_uniform_vec(size: usize, max: u64) -> Vec<u64> {
//     let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//     let uniform = Uniform::new(0, max).unwrap();
//     (0..size).map(|_| uniform.sample(&mut rng)).collect()
// }

fn main() {
    // let input_vec = generate_uniform_vec(50, 50000);

    // let minimal_intvec = BEIntVec::<DeltaCodec>::from(&input_vec, 64);

    // for (i, val) in minimal_intvec.iter().enumerate() {
    //     assert_eq!(val, input_vec[i]);
    // }

    // println!("Size of the standard Vec<u64>");
    // input_vec.mem_dbg(DbgFlags::empty());

    // println!("\nSize of the compressed IntVec with MinimalBinaryCodec");
    // minimal_intvec.mem_dbg(DbgFlags::empty());

    // let delta_intvec = BEIntVec::<DeltaCodec>::from(&input_vec, 64);

    // println!("\nSize of the compressed IntVec with DeltaCodec");
    // delta_intvec.mem_dbg(DbgFlags::empty());
}
