use compressed_intvec::codecs::MinimalBinaryCodec;
use compressed_intvec::intvec::LEIntVec;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use qwt::{AccessUnsigned, QWT512, HQWT512};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use std::time::Duration;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use serde_json::Value;
use plotters::prelude::*;

/// Generates a vector of `size` u64 values uniformly distributed in range [0, alphabet_size)
fn generate_uniform_vec(size: usize, alphabet_size: u64) -> Vec<u64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let uniform = Uniform::new(0, alphabet_size).unwrap();
    (0..size).map(|_| uniform.sample(&mut rng)).collect()
}

/// Generates a list of random indices in the interval [0, max).
fn generate_random_indexes(n: usize, max: usize) -> Vec<usize> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.random_range(0..max)).collect()
}

/// Parse Criterion JSON results and extract mean times
fn parse_criterion_results(alphabet_sizes: &[u64]) -> HashMap<u64, (f64, f64, f64)> {
    let mut results = HashMap::new();
    
    for &alphabet_size in alphabet_sizes {
        let group_dir = format!("target/criterion/alphabet_size_{}", alphabet_size);
        
        // Paths to the three benchmark results - FIXED PATHS
        let intvec_path = Path::new(&group_dir)
            .join("LEIntVec_MinimalBinary")
            .join(alphabet_size.to_string())  // Fixed: Use correct directory structure
            .join("new/estimates.json");
            
        let qwt512_path = Path::new(&group_dir)
            .join("QWT512")
            .join(alphabet_size.to_string())  // Fixed: Use correct directory structure
            .join("new/estimates.json");
            
        let hqwt512_path = Path::new(&group_dir)
            .join("HQWT512")
            .join(alphabet_size.to_string())  // Fixed: Use correct directory structure
            .join("new/estimates.json");
            
        // Parse each JSON file and extract mean time (in nanoseconds)
        let intvec_time = parse_mean_time_ns(&intvec_path);
        let qwt512_time = parse_mean_time_ns(&qwt512_path);
        let hwqt512_time = parse_mean_time_ns(&hqwt512_path);
        
        results.insert(alphabet_size, (intvec_time, qwt512_time, hwqt512_time));
    }
    
    results
}

/// Parse a single Criterion JSON file and extract mean time in nanoseconds
fn parse_mean_time_ns(path: &Path) -> f64 {
    if !path.exists() {
        eprintln!("Warning: Path does not exist: {}", path.display());
        return 0.0;
    }
    
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error opening {}: {}", path.display(), e);
            return 0.0;
        }
    };
    
    let mut contents = String::new();
    if let Err(e) = file.read_to_string(&mut contents) {
        eprintln!("Error reading {}: {}", path.display(), e);
        return 0.0;
    }
    
    match serde_json::from_str::<Value>(&contents) {
        Ok(json) => {
            // Extract mean time in nanoseconds
            json["mean"]["point_estimate"]
                .as_f64()
                .unwrap_or(0.0)
        },
        Err(e) => {
            eprintln!("Error parsing JSON from {}: {}", path.display(), e);
            0.0
        }
    }
}

/// Plot benchmark results as a multiline chart
fn plot_results(results: &HashMap<u64, (f64, f64, f64)>) -> Result<(), Box<dyn std::error::Error>> {
    // Create output directory if it doesn't exist
    fs::create_dir_all("images/qwt")?;
    
    // Prepare the drawing area using the SVG backend
    let root = SVGBackend::new("images/qwt/benchmark_plot.svg", (800, 600))
        .into_drawing_area();
    
    root.fill(&WHITE)?;
    
    // Get the min and max values for x and y axes
    let mut sorted_keys: Vec<_> = results.keys().copied().collect();
    sorted_keys.sort_unstable();
    
    let min_x = *sorted_keys.first().unwrap_or(&1) as f64;
    let max_x = *sorted_keys.last().unwrap_or(&1000) as f64;
    
    let mut max_y: f64 = 0.0;
    for &(a, b, c) in results.values() {
        max_y = max_y.max(a).max(b).max(c);
    }
    max_y *= 1.1; // Add some margin
    
    // Create the chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Benchmark Results: Random Access Time vs Alphabet Size", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(
            (min_x..max_x).log_scale(), // X-axis logarithmic scale
            0.0..max_y
        )?;
    
    chart.configure_mesh()
        .x_desc("Alphabet Size")
        .y_desc("Time (nanoseconds)")
        .draw()?;
    
    // Prepare data series
    let intvec_data: Vec<_> = sorted_keys.iter()
        .filter_map(|&k| results.get(&k).map(|&(time, _, _)| (k as f64, time)))
        .collect();
        
    let qwt512_data: Vec<_> = sorted_keys.iter()
        .filter_map(|&k| results.get(&k).map(|&(_, time, _)| (k as f64, time)))
        .collect();
        
    let hqwt512_data: Vec<_> = sorted_keys.iter()
        .filter_map(|&k| results.get(&k).map(|&(_, _, time)| (k as f64, time)))
        .collect();
    
    // Plot the lines
    chart.draw_series(LineSeries::new(
        intvec_data,
        &RED,
    ))?
    .label("LEIntVec_MinimalBinary")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    
    chart.draw_series(LineSeries::new(
        qwt512_data,
        &BLUE,
    ))?
    .label("QWT512")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    
    chart.draw_series(LineSeries::new(
        hqwt512_data,
        &GREEN,
    ))?
    .label("HQWT512")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
    
    // Draw the legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    
    // Save the result
    root.present()?;
    
    println!("Plot saved to images/qwt/benchmark_plot.svg");
    Ok(())
}

/// Benchmarks random access performance with different alphabet sizes
fn bench_alphabet_comparison(c: &mut Criterion) {
    let vector_size = 100_000;
    let query_count = 10_000;
    let alphabet_sizes = vec![16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576];
    
    for &alphabet_size in &alphabet_sizes {
        let input = generate_uniform_vec(vector_size, alphabet_size);
        let indexes = generate_random_indexes(query_count, vector_size);
        
        // // MinimalBinaryCodec parameter (bits needed to represent alphabet)
        // let b = (alphabet_size as f64).log2().ceil() as u64;
        
        // Format for consistent benchmark naming
        let benchmark_group_name = format!("alphabet_size_{}", alphabet_size);
        let mut group = c.benchmark_group(&benchmark_group_name);
        
        // LEIntVec with MinimalBinaryCodec
        let intvec = LEIntVec::<MinimalBinaryCodec>::from_with_param(&input, 16, 16).unwrap();
        let id_intvec = BenchmarkId::new("LEIntVec_MinimalBinary", alphabet_size);
        
        // Benchmark LEIntVec
        group.bench_function(id_intvec, |b| {
            b.iter(|| {
                for &i in &indexes {
                    black_box(intvec.get(i));
                }
            });
        });
        
        // QWT512
        let qwt = QWT512::from(input.clone());
        let id_qwt512 = BenchmarkId::new("QWT512", alphabet_size);
        
        group.bench_function(id_qwt512, |b| {
            b.iter(|| {
                for &i in &indexes {
                    black_box(qwt.get(i));
                }
            });
        });
        
        // HQWT512
        let qwt_pfs = HQWT512::from(input);
        let id_hqwt512 = BenchmarkId::new("HQWT512", alphabet_size);
        
        group.bench_function(id_hqwt512, |b| {
            b.iter(|| {
                for &i in &indexes {
                    black_box(qwt_pfs.get(i));
                }
            });
        });
        
        group.finish();
    }
    
    // Print info for the user
    println!("\nBenchmark complete!");
    println!("Results are available in the 'target/criterion/' directory");
    println!("A detailed HTML report can be viewed by opening 'target/criterion/report/index.html'");
    
    // Parse results and create plots (only after all benchmarks are complete)
    println!("\nGenerating plots from benchmark results...");
    let results = parse_criterion_results(&alphabet_sizes);
    if !results.is_empty() {
        if let Err(e) = plot_results(&results) {
            eprintln!("Error creating plot: {}", e);
        }
    } else {
        eprintln!("No benchmark results found to plot!");
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(1));
    targets = bench_alphabet_comparison
}
criterion_main!(benches);