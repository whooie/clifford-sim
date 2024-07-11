#![allow(dead_code, non_snake_case, non_upper_case_globals)]

use std::path::PathBuf;
use clifford_sim::circuit::*;
use ndarray as nd;
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use whooie::{ loop_call, mkdir, write_npz };

fn eval_entropy(n: usize, p: f32, avg: usize) -> f32 {
    // let mut circuit: StabCircuit;
    // let mut s: Vec<f32>;
    // let mut s0: f32 = 0.0;
    // for _ in 0..avg {
    //     circuit = StabCircuit::new(n, None, None);
    //     let config = CircuitConfig {
    //         depth: DepthConfig::Const(depth),
    //         gates: GateConfig::Simple,
    //         // gates: GateConfig::GateSet(G1Set::HS, G2Set::CZ),
    //         boundaries: BoundaryConfig::Periodic,
    //         measurement: MeasureConfig {
    //             layer: MeasLayerConfig::Every,
    //             prob: MeasProbConfig::Random(p),
    //         },
    //     };
    //     s = circuit.run_entropy(config, None);
    //     s0 += s.into_iter().skip(2 * depth / 3).sum::<f32>()
    //         / (depth / 3) as f32;
    // }
    // s0 / avg as f32

    (0..avg).into_par_iter()
        .map(|_| {
            let mut circuit = StabCircuit::new(n, None, None);
            let config = CircuitConfig {
                depth: DepthConfig::Const(4 * n),
                // gates: GateConfig::Simple,
                gates: GateConfig::GateSet(G1Set::HS, G2Set::CZ),
                boundaries: BoundaryConfig::Open,
                measurement: MeasureConfig {
                    layer: MeasLayerConfig::Every,
                    // layer: MeasLayerConfig::Period(8),
                    // prob: MeasProbConfig::Random(p),
                    prob: MeasProbConfig::cycling_prob(p),
                    // prob: MeasProbConfig::Block((p * n as f32).round() as usize),
                },
            };
            circuit.run_entropy(config, None)
                .into_iter()
                .skip(3 * n)
                .sum::<f32>() / (n as f32)
        })
        .sum::<f32>() / avg as f32
}

fn main() {
    const AVG: usize = 2500;

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    // let p_meas: nd::Array1<f32> = nd::Array1::linspace(0.01, 0.20, 20);
    // let size: nd::Array1<u32> = (2..=8_u32).map(|n| 2_u32.pow(n)).collect();

    let p_meas: nd::Array1<f32> = nd::Array1::linspace(0.05, 0.14, 10);
    let size: nd::Array1<u32> = (4..=20).step_by(2).collect();
    // let size: nd::Array1<u32> = (2..=7_u32).map(|n| 2_u32.pow(n)).collect();
    let caller = |q: &[usize]| -> (f32,) {
        (eval_entropy(size[q[1]] as usize, p_meas[q[0]], AVG),)
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

