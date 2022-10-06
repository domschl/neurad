extern crate ndarray;

use ndarray::prelude::*;
use ndarray::Array;

fn main() {
    let dim=1024;
    let a = Array::<f32, _>::zeros((dim,dim).f())+0.1;
    let b = Array::<f32, _>::zeros((dim,dim).f())+0.01;
    use std::time::Instant;
    let now = Instant::now();
    let c = a.dot(&b);
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    println!("{}",c);
}
