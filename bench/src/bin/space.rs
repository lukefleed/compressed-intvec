//! Prints, as CSV on standard output, the space that each vector of the benchmark data takes.
//!
//! `cargo run -p compressed-intvec-bench --release --bin space > space.csv`
//!
//! Space is reported here, outside the timing harness. `payload_bytes` counts the words that
//! the vector uses, `allocated_bytes` the capacity of its allocations; neither counts the
//! fixed-size struct itself or the allocator's overhead. For `Vec<Vec<u32>>`,
//! `allocated_bytes` includes the 24-byte header of each inner `Vec`.

use compressed_intvec::{FixedVec, SeqVec, SeqVecParts, VarVec, VarVecParts};
use compressed_intvec_bench::{Code, Distribution, FlatSequences, Lengths, Shape, VALUE_SEED};

/// The elements of every `VarVec`, and the sequences of every `SeqVec`.
const LEN: usize = 1 << 20;

/// The payload and allocated bytes of words.
fn words(words: &Vec<u64>) -> (usize, usize) {
    (words.len() * 8, words.capacity() * 8)
}

fn add((a, b): (usize, usize), (c, d): (usize, usize)) -> (usize, usize) {
    (a + c, b + d)
}

fn fixed_bytes<T>(vec: FixedVec<T>) -> (usize, usize) {
    words(&vec.into_parts().0)
}

fn var_bytes(vec: VarVec<u32>) -> (usize, usize) {
    let VarVecParts { data, samples, .. } = vec.into_parts();
    add(words(&data), fixed_bytes(samples))
}

fn seq_bytes(vec: SeqVec<u32>) -> (usize, usize) {
    let SeqVecParts {
        data,
        offsets,
        lengths,
        ..
    } = vec.into_parts();
    let lengths = lengths.map_or((0, 0), fixed_bytes);
    add(add(words(&data), fixed_bytes(offsets)), lengths)
}

fn row(
    structure: &str,
    data: &str,
    code: &str,
    sampling_rate: &str,
    elements: usize,
    (payload, allocated): (usize, usize),
) {
    let bits = payload as f64 * 8.0 / elements as f64;
    println!(
        "{structure},{data},{code},{sampling_rate},{elements},{payload},{allocated},{bits:.3}"
    );
}

fn main() {
    println!(
        "structure,data,code,sampling_rate,elements,payload_bytes,allocated_bytes,bits_per_element"
    );
    for distribution in Distribution::ALL {
        let values = distribution.values(VALUE_SEED, LEN);
        let data = distribution.name();
        for code in Code::ALL {
            for rate in [4, 8, 16, 32, 64, 128, 256] {
                let vec = code.var_vec(&values, rate);
                // The code that the automatic choice picked, without the commas of `Debug`.
                let name = match code {
                    Code::Automatic => format!("automatic: {:?}", vec.codec()).replace(',', ";"),
                    _ => code.name().to_owned(),
                };
                row(
                    "VarVec<u32>",
                    data,
                    &name,
                    &rate.to_string(),
                    LEN,
                    var_bytes(vec),
                );
            }
        }
        let fixed = FixedVec::from_slice(&values);
        let width = format!("width {}", fixed.bit_width());
        row("FixedVec<u32>", data, &width, "-", LEN, fixed_bytes(fixed));
        row(
            "Vec<u32>",
            data,
            "-",
            "-",
            LEN,
            (LEN * 4, values.capacity() * 4),
        );
    }
    for shape in Shape::ALL {
        let sequences = shape.sequences(Distribution::PowerLaw, LEN);
        let elements = sequences.iter().map(Vec::len).sum();
        let data = format!("{} sequences of power_law values", shape.name());
        for lengths in Lengths::ALL {
            for code in Code::SEQ {
                let vec = code.seq_vec(&sequences, lengths);
                let name = match code {
                    Code::Automatic => format!("automatic: {:?}", vec.codec()).replace(',', ";"),
                    _ => code.name().to_owned(),
                };
                let structure = format!("SeqVec<u32> ({})", lengths.name());
                row(&structure, &data, &name, "-", elements, seq_bytes(vec));
            }
        }
        let nested = sequences.iter().fold((0, 0), |(payload, allocated), s| {
            (payload + s.len() * 4, allocated + 24 + s.capacity() * 4)
        });
        row("Vec<Vec<u32>>", &data, "-", "-", elements, nested);
        let flat = FlatSequences::new(&sequences);
        let payload = elements * 4 + (flat.len() + 1) * 8;
        row(
            "FlatSequences",
            &data,
            "-",
            "-",
            elements,
            (payload, flat.allocated_bytes()),
        );
    }
}
