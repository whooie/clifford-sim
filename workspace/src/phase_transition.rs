#![allow(dead_code, non_snake_case, non_upper_case_globals)]

use std::path::PathBuf;
use clifford_sim::circuit::StabCircuit;
use ndarray as nd;
use whooie::{ loop_call, mkdir, write_npz };

fn eval_entropy(nqubits: usize, p_meas: f32, depth: usize, avg: usize) -> f32 {
    let mut circuit: StabCircuit;
    let mut s: Vec<f32>;
    let mut s0: f32 = 0.0;
    for _ in 0..avg {
        circuit = StabCircuit::new(nqubits, p_meas, None, None);
        s = circuit.run_simple(depth, false);
        s0 += s.into_iter().skip(depth / 3).sum::<f32>() / (depth / 3) as f32;
    }
    s0 / avg as f32
}

fn main() {
    const DEPTH: usize = 1000;
    const AVG: usize = 20;

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let p_meas: nd::Array1<f32> = nd::Array1::linspace(0.02, 0.2, 10);
    let size: nd::Array1<u32> = (1..=9_u32).map(|n| 2_u32.pow(n)).collect();
    let caller = |q: Vec<usize>| -> (f32,) {
        (eval_entropy(size[q[1]] as usize, p_meas[q[0]], DEPTH, AVG),)
    };
    let (entropy,): (nd::ArrayD<f32>,)
        = loop_call!(
            caller => (s: f32,),
            vars: { p_meas, size }
        );

    write_npz!(
        outdir.join("phase_transition.npz"),
        arrays: {
            "p_meas" => &p_meas,
            "size" => &size,
            "entropy" => &entropy,
        }
    );
}

