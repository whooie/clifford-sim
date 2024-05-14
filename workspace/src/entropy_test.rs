use std::path::PathBuf;
use clifford_sim::circuit::StabCircuit;
use ndarray as nd;
use whooie::{ mkdir, print_flush, write_npz };

const N: usize = 256; // number of qubits

fn main() {
    const MC: usize = 20;
    const DEPTH: usize = 1000;
    const P_MEAS: f32 = 0.08;

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let mut s_acc: nd::Array1<f32> = nd::Array1::zeros(DEPTH + 1);
    for k in 0..MC {
        print_flush!("\r {} ", k);
        let mut circuit = StabCircuit::new(N, P_MEAS, None, None);
        s_acc += &nd::Array1::from(circuit.run_simple(DEPTH, false));
    }
    s_acc /= MC as f32;
    println!();

    write_npz!(
        outdir.join("entropy_test.npz"),
        arrays: {
            "size" => &nd::array![N as u32],
            "mc" => &nd::array![MC as u32],
            "p_meas" => &nd::array![P_MEAS],
            "entropy" => &s_acc,
        }
    );
}

