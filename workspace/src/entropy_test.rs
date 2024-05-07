use std::path::PathBuf;
use clifford_sim::circuit::StabCircuit;
use ndarray as nd;
use whooie::{ mkdir, print_flush, write_npz };

const N: usize = 512; // number of qubits

fn main() {
    const MC: usize = 50;
    const DEPTH: usize = 250;
    const P_MEAS: f32 = 0.08;

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let mut s_acc: nd::Array1<f32> = nd::Array1::zeros(DEPTH + 1);
    for k in 0..MC {
        print_flush!("\r {} ", k);
        let mut circuit = StabCircuit::<N>::new(P_MEAS, None, None);
        s_acc += &nd::Array1::from(circuit.run_simple(DEPTH));
    }
    s_acc /= MC as f32;
    println!();

    write_npz!(
        outdir.join("entropy_test.npz"),
        arrays: { "entropy" => &s_acc }
    );
}
