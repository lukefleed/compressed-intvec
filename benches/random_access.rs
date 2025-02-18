use compressed_intvec::codecs::{
    DeltaCodec, ExpGolombCodec, GammaCodec, ParamDeltaCodec, ParamGammaCodec, RiceCodec,
};
use compressed_intvec::intvec::{BEIntVec, LEIntVec};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use std::u64;

/// Genera un vettore di `size` valori di tipo `u64` campionati dalla distribuzione `dist`
/// che produce valori di tipo `T`, convertendoli in u64 tramite la funzione `convert`.
fn generate_vec_with_distribution<D, T>(
    size: usize,
    dist: D,
    convert: impl Fn(T) -> u64,
) -> Vec<u64>
where
    D: Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(42);
    (0..size).map(|_| convert(dist.sample(&mut rng))).collect()
}

/// Genera una lista di indici casuali nell'intervallo [0, max)
fn generate_random_indexes(n: usize, max: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.random_range(0..max)).collect()
}

/// Benchmark per l'accesso casuale, che raccoglie il tempo di esecuzione in `results`.
///
/// - `results`: vettore per salvare (nome benchmark, valore di k, tempo in sec)
/// - `input`: vettore di input su cui effettuare il benchmark
/// - `k`: parametro usato per la costruzione della struttura
/// - `param`: parametro specifico della codec
/// - `build_vec`: closure per costruire la struttura compressa a partire dai dati
/// - `get`: funzione per effettuare l'accesso ad un elemento
fn benchmark_random_access<T, C: Copy>(
    results: &mut Vec<(String, usize, f64)>,
    c: &mut Criterion,
    name: &str,
    input: Vec<u64>,
    k: usize,
    param: C,
    build_vec: impl Fn(Vec<u64>, usize, C) -> T,
    get: impl Fn(&T, usize) -> Option<u64>,
) {
    // Benchmark tramite Criterion
    c.bench_function(name, |b| {
        b.iter(|| {
            let vec = build_vec(input.clone(), k, param);
            let indexes = generate_random_indexes(input.len(), input.len());
            for &i in &indexes {
                black_box(get(&vec, i).unwrap());
            }
        });
    });

    // Misurazione diretta per registrare il tempo in un CSV
    let vec = build_vec(input.clone(), k, param);
    let indexes = generate_random_indexes(input.len(), input.len());
    let start = Instant::now();
    for &i in &indexes {
        black_box(get(&vec, i).unwrap());
    }
    let elapsed = start.elapsed().as_secs_f64();
    results.push((name.to_string(), k, elapsed));
}

/// Funzione principale dei benchmark.
fn bench_all(c: &mut Criterion) {
    let input_size = 10_000;
    let max_value = u64::MAX;
    let ks = vec![4, 8, 16, 32, 64, 128];

    // Vettore per salvare i risultati
    let mut results: Vec<(String, usize, f64)> = Vec::new();

    // Esempio 1: utilizzo di una distribuzione che restituisce già u64 (Uniform)
    let dist_vec = Uniform::new(0, max_value).unwrap();
    let uniform = generate_vec_with_distribution(input_size, dist_vec, |x| x);

    // Benchmark di riferimento su Vec<u64>
    let indexes = generate_random_indexes(input_size, input_size);
    c.bench_function("Vec<u64> random access (uniform)", |b| {
        b.iter(|| {
            for &i in &indexes {
                black_box(uniform[i]);
            }
        });
    });

    // Esecuzione dei benchmark per diverse strutture e codec
    for &k in &ks {
        // LEIntVec benchmarks
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec GammaCodec"),
            uniform.clone(),
            k,
            &GammaCodec,
            |data: Vec<u64>, k: usize, _codec: &GammaCodec| {
                LEIntVec::<GammaCodec>::from_with_param(&data, k, ())
            },
            LEIntVec::<_>::get,
        );
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec DeltaCodec"),
            uniform.clone(),
            k,
            &DeltaCodec,
            |data: Vec<u64>, k: usize, _codec: &DeltaCodec| {
                LEIntVec::<DeltaCodec>::from_with_param(&data, k, ())
            },
            LEIntVec::<_>::get,
        );
        let exp_k = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec ExpGolombCodec"),
            uniform.clone(),
            k,
            exp_k,
            |data: Vec<u64>, k: usize, param: usize| {
                LEIntVec::<ExpGolombCodec>::from_with_param(&data, k, param)
            },
            LEIntVec::<_>::get,
        );
        let rice_k = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec RiceCodec"),
            uniform.clone(),
            k,
            rice_k,
            |data: Vec<u64>, k: usize, param: usize| {
                LEIntVec::<RiceCodec>::from_with_param(&data, k, param)
            },
            LEIntVec::<_>::get,
        );
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec ParamDeltaCodec"),
            uniform.clone(),
            k,
            &ParamDeltaCodec::<true, true>,
            |data: Vec<u64>, k: usize, _codec: &ParamDeltaCodec<true, true>| {
                LEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(&data, k, ())
            },
            LEIntVec::<_>::get,
        );
        benchmark_random_access(
            &mut results,
            c,
            &format!("LEIntVec ParamGammaCodec"),
            uniform.clone(),
            k,
            &ParamGammaCodec::<true>,
            |data: Vec<u64>, k: usize, _codec: &ParamGammaCodec<true>| {
                LEIntVec::<ParamGammaCodec<true>>::from_with_param(&data, k, ())
            },
            LEIntVec::<_>::get,
        );

        // BEIntVec benchmarks
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec GammaCodec"),
            uniform.clone(),
            k,
            &GammaCodec,
            |data: Vec<u64>, k: usize, _codec: &GammaCodec| {
                BEIntVec::<GammaCodec>::from_with_param(&data, k, ())
            },
            BEIntVec::<_>::get,
        );
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec DeltaCodec"),
            uniform.clone(),
            k,
            &DeltaCodec,
            |data: Vec<u64>, k: usize, _codec: &DeltaCodec| {
                BEIntVec::<DeltaCodec>::from_with_param(&data, k, ())
            },
            BEIntVec::<_>::get,
        );
        let exp_k = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec ExpGolombCodec"),
            uniform.clone(),
            k,
            exp_k,
            |data: Vec<u64>, k: usize, param: usize| {
                BEIntVec::<ExpGolombCodec>::from_with_param(&data, k, param)
            },
            BEIntVec::<_>::get,
        );
        let rice_k = (uniform.iter().sum::<u64>() as f64 / uniform.len() as f64)
            .log2()
            .floor() as usize;
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec RiceCodec"),
            uniform.clone(),
            k,
            rice_k,
            |data: Vec<u64>, k: usize, param: usize| {
                BEIntVec::<RiceCodec>::from_with_param(&data, k, param)
            },
            BEIntVec::<_>::get,
        );
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec ParamDeltaCodec"),
            uniform.clone(),
            k,
            &ParamDeltaCodec::<true, true>,
            |data: Vec<u64>, k: usize, _codec: &ParamDeltaCodec<true, true>| {
                BEIntVec::<ParamDeltaCodec<true, true>>::from_with_param(&data, k, ())
            },
            BEIntVec::<ParamDeltaCodec<true, true>>::get,
        );
        benchmark_random_access(
            &mut results,
            c,
            &format!("BEIntVec ParamGammaCodec"),
            uniform.clone(),
            k,
            &ParamGammaCodec::<true>,
            |data: Vec<u64>, k: usize, _codec: &ParamGammaCodec<true>| {
                BEIntVec::<ParamGammaCodec<true>>::from_with_param(&data, k, ())
            },
            BEIntVec::<ParamGammaCodec<true>>::get,
        );
    }

    // Scrittura dei risultati in un file CSV
    let mut file =
        File::create("benchmark_random_access.csv").expect("Impossibile creare il file CSV");
    writeln!(file, "name,k,elapsed").expect("Errore di scrittura header CSV");
    for (name, k, elapsed) in results {
        writeln!(file, "{},{},{}", name, k, elapsed).expect("Errore di scrittura nel CSV");
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
