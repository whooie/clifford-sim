#![allow(dead_code, non_snake_case, non_upper_case_globals)]

use std::path::PathBuf;
use clifford_sim::circuit::*;
use ndarray as nd;
use whooie::{ loop_call, mkdir, write_npz };

#[derive(Copy, Clone, Debug)]
enum MeasLayer {
    Every,
    Period(usize),
}

impl MeasLayer {
    fn period(&self) -> usize {
        match *self {
            Self::Every => 1,
            Self::Period(p) => p,
        }
    }

    fn as_conf(&self) -> MeasLayerConfig {
        match *self {
            Self::Every => MeasLayerConfig::Every,
            Self::Period(p) => MeasLayerConfig::Period(p),
        }
    }

    fn as_string(&self) -> String {
        match *self {
            Self::Every => "1".to_string(),
            Self::Period(p) => p.to_string(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum MeasProb {
    Random,
    Cycling,
}

impl MeasProb {
    fn as_conf(&self, p: f32) -> MeasProbConfig {
        match *self {
            Self::Random => MeasProbConfig::Random(p),
            Self::Cycling => MeasProbConfig::cycling_prob(p),
        }
    }

    fn as_string(&self) -> String {
        match *self {
            Self::Random => "prob".to_string(),
            Self::Cycling => "cyc".to_string(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Entropy {
    mean: f32,
    std_p: f32,
    std_m: f32,
}

fn eval_entropy(n: usize, p: f32, avg: usize) -> Entropy {
    let (d, skip) = if n < MEAS_LAYER.period() * 5 {
        (MEAS_LAYER.period() * 20, MEAS_LAYER.period() * 15)
    } else {
        (4 * n, 3 * n)
    };
    // let d = MEAS_LAYER.period().max(4) * n;
    // let skip = (3 * d) / 4;
    let mut entropy: nd::Array1<f32> = nd::Array::zeros(avg);
    nd::Zip::from(entropy.view_mut())
        .par_for_each(move |s| {
            let mut circuit = StabCircuit::new(n, None, None);
            let config = CircuitConfig {
                depth: DepthConfig::Const(d),
                // gates: Config::Simple,
                gates: GateConfig::GateSet(G1SET, G2SET),
                boundaries: BoundaryConfig::Open,
                measurement: MeasureConfig {
                    // layer: MeasLayerConfig::Every,
                    // layer: MeasLayerConfig::Period(4),
                    layer: MEAS_LAYER.as_conf(),
                    // prob: MeasProbConfig::Random(p),
                    // prob: MeasProbConfig::cycling_prob(p),
                    prob: MEAS_PROB.as_conf(p),
                    // reset: false,
                    // reset: true,
                    reset: RESET,
                },
            };
            *s
                = circuit.run_entropy(config, None)
                .into_iter()
                .skip(skip)
                .sum::<f32>() / ((d - skip) as f32)
        });
    let mean = entropy.mean().unwrap();
    let mut n: f32;
    n = 0.0;
    let std_p
        = entropy.iter().copied()
        .filter(|sk| *sk > mean)
        .map(|sk| { n += 1.0; (sk - mean).powi(2) })
        .sum::<f32>()
        .sqrt() / n.sqrt();
    n = 0.0;
    let std_m
        = entropy.iter().copied()
        .filter(|sk| *sk < mean)
        .map(|sk| { n += 1.0; (sk - mean).powi(2) })
        .sum::<f32>()
        .sqrt() / n.sqrt();
    Entropy { mean, std_p, std_m }
}

const G1SET: G1Set = G1Set::H;
const G2SET: G2Set = G2Set::CZ;

// const MEAS_LAYER: MeasLayer = MeasLayer::Every;
const MEAS_LAYER: MeasLayer = MeasLayer::Period(9);

const MEAS_PROB: MeasProb = MeasProb::Random;
// const MEAS_PROB: MeasProb = MeasProb::Cycling;

const RESET: bool = false;
// const RESET: bool = true;

fn main() {
    const AVG: usize = 100;

    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    // let p_meas: nd::Array1<f32> = nd::Array1::linspace(0.01, 0.20, 20);
    // let size: nd::Array1<u32> = (2..=8_u32).map(|n| 2_u32.pow(n)).collect();

    let p_min: f32 = 0.00;
    let p_max: f32 = 1.00;
    let p_meas: nd::Array1<f32> = nd::Array1::linspace(p_min, p_max, 41);
    // let size: nd::Array1<u32> = (4..=20).step_by(2).collect();
    let logn_min: u32 = 2;
    let logn_max: u32 = 9;
    let size: nd::Array1<u32>
        = (logn_min..=logn_max).map(|n| 2_u32.pow(n)).collect();
    let caller = |q: &[usize]| -> (f32, f32, f32) {
        let n = size[q[1]] as usize;
        let p = p_meas[q[0]];
        let Entropy { mean, std_p, std_m } = eval_entropy(n, p, AVG);
        (mean, std_p, std_m)
    };
    let (s_mean, s_std_p, s_std_m)
        = loop_call!(
            caller => (s_mean: f32, s_std_p: f32, s_std_m: f32),
            vars: { p_meas, size }
        );

    let fname: String
        = format!(
            "phase_transition\
            _{}-{}\
            _r:{}\
            _P:{}\
            _p:{}={:.2}..{:.2}\
            _n:log={}..{}\
            .npz",
            format!("{:?}", G1SET).to_lowercase(),
            format!("{:?}", G2SET).to_lowercase(),
            RESET,
            MEAS_LAYER.as_string(),
            MEAS_PROB.as_string(),
            p_min,
            p_max,
            logn_min,
            logn_max,
        );

    write_npz!(
        outdir.join(&fname),
        arrays: {
            "p_meas" => &p_meas,
            "size" => &size,
            "s_mean" => &s_mean,
            "s_std_p" => &s_std_p,
            "s_std_m" => &s_std_m,
            "avg" => &nd::array![AVG as u32],
        }
    );

    println!("output/{}", fname);
}

