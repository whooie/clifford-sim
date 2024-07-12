#![allow(dead_code, non_snake_case, non_upper_case_globals)]

use std::path::PathBuf;
#[allow(unused_imports)]
use clifford_sim::{ circuit::*, stab::* };
#[allow(unused_imports)]
use itertools::Itertools;
use ndarray as nd;
#[allow(unused_imports)]
use rayon::iter::{ IntoParallelIterator, ParallelBridge, ParallelIterator };
#[allow(unused_imports)]
use whooie::{
    loop_call,
    mkdir,
    print_flush,
    write_npz,
};

// fn eval_entropy(n: usize, p: f32, depth: usize, part: &Partition, avg: usize)
//     -> f32
// {
//     let mut circuit: StabCircuit;
//     let mut s: f32 = 0.0;
//     for _ in 0..avg {
//         circuit = StabCircuit::new(n, Some(part.clone()), None);
//         let config = CircuitConfig {
//             depth: DepthConfig::Const(depth),
//             gates: GateConfig::Simple,
//             // gates: GateConfig::GateSet(G1Set::HS, G2Set::CZ),
//             boundaries: BoundaryConfig::Periodic,
//             measurement: MeasureConfig {
//                 layer: MeasLayerConfig::Every,
//                 prob: MeasProbConfig::Random(p),
//                 reset: false,
//             },
//         };
//         s += circuit.run_entropy(config, None)
//             .into_iter()
//             .skip(3 * depth / 4)
//             .sum::<f32>() / avg as f32;
//     }
//     s
// }
//
// fn eval_mutinf(n: usize, p: f32, depth: usize, avg: usize) -> f32 {
//     let a = Partition::Left(n / 2 - 2);
//     let s_a = eval_entropy(n, p, depth, &a, avg);
//     let b = Partition::Range(n / 2, n - 2);
//     let s_b = eval_entropy(n, p, depth, &b, avg);
//     let ab = Partition::Union(a.into(), b.into());
//     let s_ab = eval_entropy(n, p, depth, &ab, avg);
//     s_a + s_b - s_ab
// }

fn eval_mutinf(n: usize, p: f32, depth: usize, avg: usize) -> f32 {
    (0..avg).into_par_iter()
        .map(|_| {
            let mut circuit = StabCircuit::new(n, None, None);
            let config = CircuitConfig {
                depth: DepthConfig::Const(depth),
                // gates: GateConfig::Simple,
                gates: GateConfig::GateSet(G1Set::HS, G2Set::CZ),
                boundaries: BoundaryConfig::Periodic,
                measurement: MeasureConfig {
                    // layer: MeasLayerConfig::Every,
                    layer: MeasLayerConfig::Period(8),
                    // prob: MeasProbConfig::Random(p),
                    // prob: MeasProbConfig::Cycling(p.recip().floor() as usize),
                    prob: MeasProbConfig::Block((p * n as f32).round() as usize),
                    reset: false,
                },
            };
            circuit.run_mutinf(config, Some(n / 2 - 2), None)
                .into_iter()
                .skip(3 * depth / 4)
                .sum::<f32>() / (depth as f32 / 4.0)
        })
        .sum::<f32>() / avg as f32

    // let mut circuit: StabCircuit;
    // let mut inf: f32 = 0.0;
    // for _ in 0..avg {
    //     circuit = StabCircuit::new(n, None, None);
    //     let config = CircuitConfig {
    //         depth: DepthConfig::Const(depth),
    //         gates: GateConfig::Simple,
    //         boundaries: BoundaryConfig::Periodic,
    //         measurement: MeasureConfig {
    //             layer: MeasLayerConfig::Every,
    //             prob: MeasProbConfig::Random(p),
    //             reset: false,
    //         },
    //     };
    //     inf += circuit.run_mutinf(config, Some(2 * n / 8), None)
    //         .into_iter()
    //         .skip(3 * depth / 4)
    //         .sum::<f32>() / (depth as f32 / 4.0);
    // }
    // inf / avg as f32
}

fn main() {
    const DEPTH: usize = 1000;
    const AVG: usize = 100;

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let size: nd::Array1<u32> = (7..=8_u32).map(|n| 2_u32.pow(n)).collect();
    let p_meas: nd::Array1<f32> = nd::Array1::linspace(0.50, 0.8, 20);

    // static mut COUNT: usize = 0;
    // let total: usize = size.len() * p_meas.len();
    // print_flush!("  0 / {} ", total);
    // let mut mutinf: Vec<(usize, f32)>
    //     = size.iter()
    //     .cartesian_product(p_meas.iter())
    //     .enumerate()
    //     .par_bridge()
    //     .map(|(k, (&s, &p))| {
    //         let res = eval_mutinf(s as usize, p, 4 * s as usize, AVG);
    //         unsafe {
    //             COUNT += 1;
    //             print_flush!("\r  {} / {} ", COUNT, total);
    //         }
    //         (k, res)
    //     })
    //     .collect();
    // println!();
    // mutinf.sort_by_key(|(k, _)| *k);
    // let mutinf: nd::Array2<f32>
    //     = mutinf.into_iter()
    //     .map(|(_, mi)| mi)
    //     .collect::<nd::Array1<f32>>()
    //     .into_shape((size.len(), p_meas.len()))
    //     .unwrap();

    let caller = |q: &[usize]| -> (f32,) {
        (eval_mutinf(
            size[q[0]] as usize,
            p_meas[q[1]],
            4 * size[q[0]] as usize,
            AVG,
        ),)
    };
    let (mutinf,): (nd::ArrayD<f32>,)
        = loop_call!(
            caller => (s: f32,),
            vars: { size, p_meas }
        );

    write_npz!(
        outdir.join("mutual_information.npz"),
        arrays: {
            "size" => &size,
            "p_meas" => &p_meas,
            "mutinf" => &mutinf,
        }
    );
}

