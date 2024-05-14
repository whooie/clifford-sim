use std::path::PathBuf;
use clifford_sim::circuit::StabCircuit;
use ndarray as nd;
use whooie::{ mkdir, print_flush, write_npz };

const N: usize = 256; // number of qubits

fn main() {
    const P_MEAS: f32 = 0.08;
    const MC: usize = 250;
    const TOL: f32 = 1e-6;

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let mut depth_acc: Vec<usize> = Vec::with_capacity(MC);
    let mut entropy_acc: Vec<f32> = Vec::with_capacity(MC);

    for k in 0..MC {
        print_flush!("\r {} ", k);
        let mut circuit = StabCircuit::new(N, P_MEAS, None, None);
        let s = circuit.run_simple_until_converged(Some(TOL));
        let n = s.len();
        depth_acc.push(n);
        entropy_acc.push(
            s.into_iter().skip(n / 2).sum::<f32>() / (n / 2) as f32);
    }
    println!();

    let depth_mean: f32
        = depth_acc.iter().copied()
        .map(|d| d as f32)
        .sum::<f32>() / MC as f32;
    let depth_std: f32
        = depth_acc.iter().copied()
        .map(|d| (d as f32 - depth_mean).powi(2))
        .sum::<f32>()
        .sqrt() / (MC as f32).sqrt();

    let entropy_mean: f32
        = entropy_acc.iter().copied()
        .sum::<f32>() / MC as f32;
    let entropy_std: f32
        = entropy_acc.iter().copied()
        .map(|s| (s - entropy_mean).powi(2))
        .sum::<f32>()
        .sqrt() / (MC as f32).sqrt();

    println!("depth = {:.3} ± {:.3}", depth_mean, depth_std);
    println!("entropy = {:.3} ± {:.3}", entropy_mean, entropy_std);

    let depth_acc: nd::Array1<u32>
        = depth_acc.into_iter().map(|d| d as u32).collect();
    let entropy_acc: nd::Array1<f32>
        = entropy_acc.into_iter().collect();

    write_npz!(
        outdir.join("convergence_test.npz"),
        arrays: {
            "size" => &nd::array![N as u32],
            "mc" => &nd::array![MC as u32],
            "p_meas" => &nd::array![P_MEAS],
            "tol" => &nd::array![TOL],
            "depth" => &depth_acc,
            "entropy" => &entropy_acc,
        }
    );
}
