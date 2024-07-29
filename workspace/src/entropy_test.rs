use std::path::PathBuf;
use clifford_sim::circuit::*;
use ndarray as nd;
use whooie::{ mkdir, write_npz };

const N: usize = 256; // number of qubits
const T: usize = 30; // number of gate layers between measurement layers

fn main() {
    const MC: usize = 100;
    // const DEPTH: usize = 4 * N;
    const P_MEAS: f32 = 0.90;

    let depth: usize
        = if N < T * 5 {
            20 * T
        } else {
            4 * N
        };

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let mut s_acc: nd::Array1<f32> = nd::Array1::zeros(depth + 1);
    for k in 0..MC {
        eprint!("\r {} ", k);
        let mut circuit = StabCircuit::new(N, None, None);
        let config = CircuitConfig {
            depth: DepthConfig::Const(depth),
            // gates: GateConfig::Simple,
            gates: GateConfig::GateSet(G1Set::H, G2Set::CZ),
            // boundaries: BoundaryConfig::Periodic,
            boundaries: BoundaryConfig::Open,
            measurement: MeasureConfig {
                layer: MeasLayerConfig::Period(T),
                prob: MeasProbConfig::Random(P_MEAS),
                // prob: MeasProbConfig::cycling_prob(P_MEAS),
                reset: false,
            },
        };
        s_acc += &nd::Array1::from(circuit.run_entropy(config, None));
    }
    s_acc /= MC as f32;
    eprintln!();

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

