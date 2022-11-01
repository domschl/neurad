extern crate ndarray;

use ndarray::prelude::*;
use ndarray::Array;
use std::time::Duration;

fn main() {
    let dims = vec![
        2, 3, 4, 5, 6, 8, 10, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000, 12000, 14000,
        20000,
    ];
    for dim in dims {
        let mut reps = 0;
        let a = Array::<f32, _>::zeros((dim, dim).f()) + 0.1;
        let b = Array::<f32, _>::zeros((dim, dim).f()) + 0.01;
        let mut min = Duration::new(0, 0);
        if dim > 2048 {
            reps = 10;
        } else {
            if dim > 64 {
                reps = 50;
            } else {
                reps = 500;
            }
        }
        for n in 0..reps {
            use std::time::Instant;
            let now = Instant::now();
            let c = a.dot(&b);
            let elapsed = now.elapsed();
            if n == 0 || elapsed < min {
                min = elapsed;
            }
        }
        println!("Dim: {}x{}, elapsed: {:.2?}", dim, dim, min);
    }
    //println!("{}", c);
}
