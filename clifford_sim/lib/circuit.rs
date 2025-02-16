//! Abstractions for driving randomized Clifford circuits and measuring
//! measurement-induced phase transitions (MIPTs).

use rand::{ rngs::StdRng, Rng, SeedableRng };
use rustc_hash::FxHashSet as HashSet;
use thiserror::Error;
use crate::{
    gate::{ Clifford, Gate, G1, G2 },
    stab::{ Outcome, Partition, Stab },
};

/// Type alias for [`HashSet`].
pub type Set<T> = HashSet<T>;

/// Main driver for running circuits of alternating unitary evolution and
/// measurement.
#[derive(Clone, Debug)]
pub struct StabCircuit {
    pub state: Stab,
    pub part: Option<Partition>,
    n: usize,
    rng: StdRng,
}

/// The outcomes associated with a single measurement layer, ordered by qubit
/// index.
pub type MeasLayer = Vec<Option<Outcome>>;

/// The outcomes from each measurement layer in a full circuit.
pub type MeasRecord = Vec<MeasLayer>;

impl StabCircuit {
    /// Create a new `StabCircuit` for a 1D chain of `n` qubits, with state
    /// initialized to ∣0...0⟩.
    ///
    /// `part` is the partition in which to calculate the entanglement entropy,
    /// defaulting to a single cut at `floor(n / 2)`. Optionally also seed the
    /// internal random number generator.
    pub fn new(
        n: usize,
        part: Option<Partition>,
        seed: Option<u64>,
    ) -> Self
    {
        let rng =
            seed.map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_entropy);
        Self { state: Stab::new(n), part, n, rng }
    }

    /// Return the number of qubits.
    pub fn n(&self) -> usize { self.n }

    fn sample_simple(
        &mut self,
        offs: bool,
        bounds: BoundaryConfig,
        buf: &mut Vec<Gate>,
    ) {
        (0..self.n).for_each(|k| {
            buf.push(Gate::sample_single(k, &mut self.rng));
        });
        match bounds {
            BoundaryConfig::Open => {
                Pairs::new(offs, self.n).for_each(|(a, b)| {
                    buf.push(Gate::CX(a, b));
                });
            },
            BoundaryConfig::Periodic => {
                PairsPeriodic::new(offs, self.n).for_each(|(a, b)| {
                    buf.push(Gate::CX(a, b));
                });
            },
        }
    }

    fn sample_cliffs(
        &mut self,
        offs: bool,
        bounds: BoundaryConfig,
        buf: &mut Vec<Gate>,
    ) {
        match bounds {
            BoundaryConfig::Open => {
                Pairs::new(offs, self.n).for_each(|(a, _b)| {
                    let mut gates = Clifford::gen(2, &mut self.rng).unpack().0;
                    gates.iter_mut().for_each(|g| { g.shift(a); });
                    buf.append(&mut gates);
                });
            },
            BoundaryConfig::Periodic => {
                PairsPeriodic::new(offs, self.n).for_each(|(a, _b)| {
                    let mut gates = Clifford::gen(2, &mut self.rng).unpack().0;
                    gates.iter_mut().for_each(|g| { g.shift_mod(a, self.n); });
                    buf.append(&mut gates);
                });
            },
        }
    }

    fn sample_gateset(
        &mut self,
        g1: &G1Set,
        g2: &G2Set,
        offs: bool,
        bounds: BoundaryConfig,
        buf: &mut Vec<Gate>,
    ) {
        (0..self.n).for_each(|k| {
            buf.push(g1.sample(k, &mut self.rng));
        });
        match bounds {
            BoundaryConfig::Open => {
                Pairs::new(offs, self.n).for_each(|(a, b)| {
                    buf.push(g2.sample(a, b, &mut self.rng));
                })
            },
            BoundaryConfig::Periodic => {
                PairsPeriodic::new(offs, self.n).for_each(|(a, b)| {
                    buf.push(g2.sample(a, b, &mut self.rng));
                })
            },
        }
    }

    fn measure(
        &mut self,
        d: usize,
        config: MeasureConfig,
        buf: &mut [Option<Outcome>]
    ) -> bool
    {
        use MeasLayerConfig::*;
        use MeasProbConfig::*;

        enum Pred<'a> {
            Never,
            Always,
            Prob(f32),
            Func(Box<dyn Fn(usize) -> bool + 'a>),
        }

        fn do_measure(
            circ: &mut StabCircuit,
            pred: Pred,
            reset: bool,
            buf: &mut [Option<Outcome>],
        ) {
            buf.iter_mut()
                .enumerate()
                .for_each(|(k, outk)| {
                    match &pred {
                        Pred::Never => {
                            *outk = None;
                        },
                        Pred::Always => {
                            *outk = if reset {
                                Some(circ.state.measure_reset(k, &mut circ.rng))
                            } else {
                                Some(circ.state.measure(k, &mut circ.rng))
                            };
                        },
                        Pred::Prob(p) => {
                            *outk = (circ.rng.gen::<f32>() < *p)
                                .then(|| if reset {
                                    circ.state.measure_reset(k, &mut circ.rng)
                                } else {
                                    circ.state.measure(k, &mut circ.rng)
                                });
                        },
                        Pred::Func(f) => {
                            *outk = f(k)
                                .then(|| if reset {
                                    circ.state.measure_reset(k, &mut circ.rng)
                                } else {
                                    circ.state.measure(k, &mut circ.rng)
                                });
                        },
                    }
                });
        }

        let MeasureConfig { layer, prob, reset } = config;
        match layer {
            Every | Period(1) => {
                let pred =
                    match prob {
                        Random(p) => Pred::Prob(p),
                        Cycling(0) => Pred::Never,
                        Cycling(1) => Pred::Always,
                        Cycling(n) => Pred::Func(
                            Box::new(move |k| k % n == d % n)),
                        CyclingInv(0) => Pred::Always,
                        CyclingInv(1) => Pred::Never,
                        CyclingInv(n) => Pred::Func(
                            Box::new(move |k| k % n != d % n)),
                        Block(0) => Pred::Never,
                        Block(b) => Pred::Func(
                            Box::new(move |k| k / b == d % b)),
                        Window(0) => Pred::Never,
                        Window(w) => {
                            let d = d as isize;
                            let w = w as isize;
                            let n = self.n as isize;
                            Pred::Func(
                                Box::new(move |k| {
                                    (k as isize - d).rem_euclid(n) / w == 0
                                })
                            )
                        }
                    };
                do_measure(self, pred, reset, buf);
                true
            },
            Period(m) if d % m == 0 => {
                let pred =
                    match prob {
                        Random(p) => Pred::Prob(p),
                        Cycling(0) => Pred::Never,
                        Cycling(1) => Pred::Always,
                        Cycling(n) => Pred::Func(
                            Box::new(move |k| k % n == (d / m) % n)),
                        CyclingInv(0) => Pred::Always,
                        CyclingInv(1) => Pred::Never,
                        CyclingInv(n) => Pred::Func(
                            Box::new(move |k| k % n != (d / m) % n)),
                        Block(0) => Pred::Never,
                        Block(b) => Pred::Func(
                            Box::new(move |k| k / b == (d / m) % b)),
                        Window(0) => Pred::Never,
                        Window(w) => {
                            let d = d as isize;
                            let w = w as isize;
                            let n = self.n as isize;
                            let m = m as isize;
                            Pred::Func(
                                Box::new(move |k| {
                                    (k as isize - d / m).rem_euclid(n) / w == 0
                                })
                            )
                        },
                    };
                do_measure(self, pred, reset, buf);
                true
            },
            _ => {
                buf.iter_mut().for_each(|outk| { *outk = None; });
                false
            },
        }
    }

    fn entropy(&self) -> f32 {
        self.state.entanglement_entropy(self.part.clone())
    }

    fn do_run(
        &mut self,
        config: &mut CircuitConfig,
        mut meas: Option<&mut MeasRecord>,
        mut entropy: Option<&mut Vec<f32>>,
        mut mutinf: Option<(&mut Vec<f32>, Option<usize>)>,
    ) {
        let CircuitConfig {
            depth: depth_conf,
            gates: ref mut gate_conf,
            boundaries: bound_conf,
            measurement: meas_conf,
        } = config;
        let depth_conf = *depth_conf;
        let bound_conf = *bound_conf;
        let meas_conf = *meas_conf;

        let mut outcomes: MeasLayer = vec![None; self.n];
        let mut s: f32 = self.entropy();
        if let Some(rcd) = entropy.as_mut() { rcd.push(s); }
        let mut sbar: f32 = 0.0;
        let mut check: f32;
        if let Some((rcd, part_size)) = mutinf.as_mut() {
            rcd.push(self.state.mutual_information(*part_size));
        }

        let mut gates: Vec<Gate> = Vec::new();
        let mut d: usize = 1;
        loop {
            gates.clear();
            match gate_conf {
                GateConfig::Simple => {
                    self.sample_simple(d % 2 == 0, bound_conf, &mut gates);
                    self.state.apply_circuit(&gates);
                },
                GateConfig::Clifford2 => {
                    self.sample_cliffs(d % 2 == 0, bound_conf, &mut gates);
                    self.state.apply_circuit(&gates);
                },
                GateConfig::GateSet(ref g1, ref g2) => {
                    self.sample_gateset(
                        g1, g2, d % 2 == 0, bound_conf, &mut gates);
                    self.state.apply_circuit(&gates);
                },
                GateConfig::Circuit(ref circ) => {
                    self.state.apply_circuit(circ);
                },
                GateConfig::Feedback(ref mut f) => {
                    match f(d, s, &outcomes) {
                        Feedback::Halt => { break; },
                        Feedback::Simple => {
                            self.sample_simple(
                                d % 2 == 0, bound_conf, &mut gates);
                            self.state.apply_circuit(&gates);
                        },
                        Feedback::Clifford2 => {
                            self.sample_cliffs(
                                d % 2 == 0, bound_conf, &mut gates);
                            self.state.apply_circuit(&gates);
                        },
                        Feedback::GateSet(g1, g2) => {
                            self.sample_gateset(
                                &g1, &g2, d % 2 == 0, bound_conf, &mut gates);
                            self.state.apply_circuit(&gates);
                        },
                        Feedback::Circuit(circ) => {
                            self.state.apply_circuit(&circ);
                        },
                    }
                },
            }

            self.measure(d, meas_conf, &mut outcomes);
            if let Some(rcd) = meas.as_mut() { rcd.push(outcomes.clone()); }

            s = self.entropy();
            if let Some(rcd) = entropy.as_mut() { rcd.push(s); }

            if let Some((rcd, part_size)) = mutinf.as_mut() {
                rcd.push(self.state.mutual_information(*part_size));
            }

            match depth_conf {
                DepthConfig::Unlimited => { },
                DepthConfig::Converge(tol) => {
                    check = (
                        2.0 * (sbar - s) / ((2 * d + 3) as f32 * sbar + s)
                    ).abs();
                    if check < tol.unwrap_or(1e-6) { break; }
                    sbar = (sbar + (d + 1) as f32 + s) / (d + 2) as f32;
                },
                DepthConfig::Const(d0) => {
                    if d >= d0 { break; }
                },
            }
            d += 1;
        }
    }

    /// Run the MIPT procedure for a general config.
    ///
    /// Returns the entanglement entropy measured once before the first layer,
    /// and then after each round of measurements.
    pub fn run_entropy(
        &mut self,
        mut config: CircuitConfig,
        meas: Option<&mut MeasRecord>,
    ) -> Vec<f32> {
        let mut entropy: Vec<f32> = Vec::new();
        self.do_run(&mut config, meas, Some(&mut entropy), None);
        entropy
    }

    /// Run the MIPT procedure for a general config.
    ///
    /// Returns the mutual information measured once before the first layer, and
    /// then after each round of measurements.
    pub fn run_mutinf(
        &mut self,
        mut config: CircuitConfig,
        part_size: Option<usize>,
        meas: Option<&mut MeasRecord>,
    ) -> Vec<f32> {
        let mut mutinf: Vec<f32> = Vec::new();
        self.do_run(&mut config, meas, None, Some((&mut mutinf, part_size)));
        mutinf
    }

    /// Run the MIPT procedure for a general config without recording any
    /// time-evolution data.
    pub fn run(&mut self, mut config: CircuitConfig) {
        self.do_run(&mut config, None, None, None)
    }

    fn do_run_fixed(
        &mut self,
        circ: &Circuit,
        mut meas: Option<&mut MeasRecord>,
        mut entropy: Option<&mut Vec<f32>>,
        mut mutinf: Option<(&mut Vec<f32>, Option<usize>)>,
    ) {
        let Circuit { n: _, ops, reset } = circ;
        let reset = *reset;
        let mut outcomes: MeasLayer = vec![None; self.n];
        if let Some(rcd) = entropy.as_mut() { rcd.push(self.entropy()); }
        for layer in ops.iter() {
            for gate in layer.gates.iter() {
                self.state.apply_gate(*gate);
            }

            if reset {
                for &k in layer.meas.iter() {
                    if let Some(outk) = outcomes.get_mut(k) {
                        *outk
                            = Some(self.state.measure_reset(k, &mut self.rng));
                    }
                }
            } else {
                for &k in layer.meas.iter() {
                    if let Some(outk) = outcomes.get_mut(k) {
                        *outk
                            = Some(self.state.measure(k, &mut self.rng));
                    }
                }
            }
            if let Some(rcd) = meas.as_mut() {
                let mut tmp: MeasLayer = vec![None; self.n];
                std::mem::swap(&mut tmp, &mut outcomes);
                rcd.push(tmp);
            }

            if let Some(rcd) = entropy.as_mut() { rcd.push(self.entropy()); }

            if let Some((rcd, part_size)) = mutinf.as_mut() {
                rcd.push(self.state.mutual_information(*part_size));
            }
        }
    }

    /// Run the MIPT procedure for a completely fixed circuit.
    ///
    /// Returns the entanglement entropy measured once before the first layer,
    /// and then after each round of measurements.
    pub fn run_entropy_fixed(
        &mut self,
        circ: &Circuit,
        meas: Option<&mut MeasRecord>,
    ) -> Vec<f32> {
        let mut entropy: Vec<f32> = Vec::new();
        self.do_run_fixed(circ, meas, Some(&mut entropy), None);
        entropy
    }

    /// Run the MIPT procedure for a completely fixed circuit.
    ///
    /// Returns the mutual information measured once before the first layer, and
    /// then after each round of measurements.
    pub fn run_mutinf_fixed(
        &mut self,
        circ: &Circuit,
        part_size: Option<usize>,
        meas: Option<&mut MeasRecord>,
    ) -> Vec<f32> {
        let mut mutinf: Vec<f32> = Vec::new();
        self.do_run_fixed(circ, meas, None, Some((&mut mutinf, part_size)));
        mutinf
    }

    /// Run the MIPT procedure for a general config without recording any
    /// time-evolution data.
    pub fn run_fixed(&mut self, circ: &Circuit) {
        self.do_run_fixed(circ, None, None, None)
    }
}


struct Pairs {
    iter: std::ops::Range<usize>
}

impl Pairs {
    fn new(offs: bool, stop: usize) -> Self {
        Self { iter: if offs { 1 } else { 0 } .. stop }
    }
}

impl Iterator for Pairs {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().zip(self.iter.next())
    }
}

struct PairsPeriodic {
    iter: std::ops::Range<usize>
}

impl PairsPeriodic {
    fn new(offs: bool, stop: usize) -> Self {
        Self { iter: if offs { 1 } else { 0 } .. stop }
    }
}

impl Iterator for PairsPeriodic {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(a) = self.iter.next() {
            if let Some(b) = self.iter.next() {
                Some((a, b))
            } else if self.iter.end % 2 == 0 {
                Some((a, 0))
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Post-measurement determination of a mid-circuit action.
#[derive(Clone, Debug)]
pub enum Feedback {
    /// Immediately halt the circuit.
    Halt,
    /// Draw the next layer of gates from the "simple" set (all single-qubit
    /// gates and tiling CXs).
    Simple,
    /// Replace distinct, overlapped one- and two-qubit unitaries with tiled,
    /// uniformly sampled two-qubit Clifford gates.
    Clifford2,
    /// Draw the next layer of gates uniformly from gate sets (two-qubit gates
    /// will still alternately tile).
    GateSet(G1Set, G2Set),
    /// Apply a specific sequence of gates.
    Circuit(Vec<Gate>),
}

/// A feedback function.
type FeedbackFn<'a> = Box<dyn FnMut(usize, f32, &[Option<Outcome>]) -> Feedback + 'a>;

/// Set the termination condition for a circuit.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DepthConfig {
    /// Run indefinitely. Only makes sense if using
    /// [feedback][GateConfig::Feedback].
    Unlimited,
    /// Run until the entanglement entropy converges on a steady-state value to
    /// within some tolerance (defaults to 10<sup>-6</sup>).
    ///
    /// The criterion for convergence is
    ///
    /// 2|(*μ*<sub>*k*+1</sub> - *μ*<sub>*k*</sub>)
    /// / (*μ*<sub>*k*+1</sub> + *μ*<sub>*k*</sub>)| < *tol*
    ///
    /// where *μ*<sub>*k*</sub> is the running average of the first *k* entropy
    /// measurements, including the first before the circuit has begun; i.e. the
    /// entropy has reached a steady state when the absolute difference between
    /// consecutive values of the running average divided by their mean is less
    /// than *tol*.
    Converge(Option<f32>),
    /// Run for a constant depth.
    Const(usize),
}

/// One-qubit gate set to draw from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum G1Set {
    /// All available single-qubit gates (see below).
    All,
    /// Only Hadamards.
    H,
    /// Only X.
    X,
    /// Only Y.
    Y,
    /// Only Z.
    Z,
    /// Only S.
    S,
    /// Only S<sup>†</sup>.
    SInv,
    /// Only Hadamards and S.
    HS,
    /// A particular gate set.
    Set(Set<G1>),
}

impl G1Set {
    /// Sample a single gate.
    pub fn sample<R>(&self, k: usize, rng: &mut R) -> Gate
    where R: Rng + ?Sized
    {
        match self {
            Self::All => match rng.gen_range(0..6_u8) {
                0 => Gate::H(k),
                1 => Gate::X(k),
                2 => Gate::Y(k),
                3 => Gate::Z(k),
                4 => Gate::S(k),
                5 => Gate::SInv(k),
                _ => unreachable!(),
            },
            Self::H => Gate::H(k),
            Self::X => Gate::X(k),
            Self::Y => Gate::Y(k),
            Self::Z => Gate::Z(k),
            Self::S => Gate::S(k),
            Self::SInv => Gate::SInv(k),
            Self::HS => if rng.gen::<bool>() {
                Gate::H(k)
            } else {
                Gate::S(k)
            },
            Self::Set(set) => Gate::sample(set, k, rng),
        }
    }
}

/// Two-qubit gate set to draw from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum G2Set {
    /// All available two-qubit gates (see below).
    All,
    /// Only CXs.
    CX,
    /// Only CZs.
    CZ,
    /// Only Swaps.
    Swap,
    /// Only CXs and CZs.
    CXCZ,
    /// A particular gate set.
    Set(Set<G2>),
}

impl G2Set {
    /// Sample a single gate.
    pub fn sample<R>(&self, a: usize, b: usize, rng: &mut R) -> Gate
    where R: Rng + ?Sized
    {
        match self {
            Self::All => match rng.gen_range(0..3_u8) {
                0 => Gate::CX(a, b),
                1 => Gate::CZ(a, b),
                2 => Gate::Swap(a, b),
                _ => unreachable!(),
            },
            Self::CX => Gate::CX(a, b),
            Self::CZ => Gate::CZ(a, b),
            Self::Swap => Gate::Swap(a, b),
            Self::CXCZ => if rng.gen::<bool>() {
                Gate::CX(a, b)
            } else {
                Gate::CZ(a, b)
            },
            Self::Set(set) => Gate::sample(set, (a, b), rng),
        }
    }
}

/// Define one- and two-qubit gate sets to draw from.
pub enum GateConfig<'a> {
    /// The "simple" set (all single-qubit gates and tiling CXs).
    Simple,
    /// Replace distinct, overlapped one- and two-qubit unitaries with tiled,
    /// uniformly sampled two-qubit Clifford gates.
    Clifford2,
    /// A particular gate set.
    GateSet(G1Set, G2Set),
    /// A particular sequence of gates.
    Circuit(Vec<Gate>),
    /// Gates based on a feedback function on measurement outcomes.
    Feedback(FeedbackFn<'a>),
}

/// Boundary conditions, relevant to two-qubit gate tilings.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BoundaryConfig {
    /// Open boundaries.
    Open,
    /// Periodic boundaries.
    Periodic,
}

/// Define the conditions for when measurements are applied.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MeasureConfig {
    /// Application of measurement layers.
    pub layer: MeasLayerConfig,
    /// Application of measurements within a single layer.
    pub prob: MeasProbConfig,
    /// Apply a deterministic reset back to ∣0⟩ after each measurement.
    pub reset: bool,
}

/// Define the conditions for when measurement layers are applied.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MeasLayerConfig {
    /// Perform measurements on every layer.
    Every,
    /// Perform measurements every `n` layers.
    ///
    /// `Period(1)` is equivalent to `Every`, and `Period(0)` applies no
    /// measurements.
    Period(usize),
}

/// Define the conditions for when measurements are applied within a single
/// layer.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MeasProbConfig {
    /// Perform measurements randomly, as normal, with fixed probability.
    Random(f32),
    /// Perform a measurement on every `n`-th qubit, shifting by 1 on every
    /// measurement layer. `Cycling(0)` means to never measure any qubit and
    /// `Cycling(1)` means to always measure every qubit.
    Cycling(usize),
    /// Inverse of `Cycling`: Perform a measurement on every qubit *except*
    /// every `n`-th qubit, shifting by 1 on every measurement layer.
    /// `Cycling(0)` means to always measure every qubit and `Cycling(1)` means
    /// to never measure any qubit.
    CyclingInv(usize),
    /// Perform measurements in blocks of `n` qubits that slide without overlap
    /// across the array.
    Block(usize),
    /// Perform measurements in sliding windows of `n` qubits.
    Window(usize),
}

impl MeasProbConfig {
    /// Convert a measurement probability into a `Cycling` pattern.
    ///
    /// The probability `p` is converted to `Cycling(round(1 / p))` if `p <
    /// 0.5`, otherwise `CyclingInv(round(1 / (1 - p)))`.
    pub fn cycling_prob(p: f32) -> Self {
        if p.abs() < f32::EPSILON {
            Self::Cycling(0)
        } else if (1.0 - p).abs() < f32::EPSILON {
            Self::Cycling(1)
        } else if p.abs() < 0.5 {
            Self::Cycling(p.recip().round() as usize)
        } else {
            Self::CyclingInv((1.0 - p).recip().round() as usize)
        }
    }
}

/// Top-level config for a circuit.
pub struct CircuitConfig<'a> {
    /// Set the depth of the circuit.
    pub depth: DepthConfig,
    /// Set available gates to draw from.
    pub gates: GateConfig<'a>,
    /// Set boundary conditions, relevant to two-qubit gate tilings.
    pub boundaries: BoundaryConfig,
    /// Set conditions for measurements.
    pub measurement: MeasureConfig,
}

/// The operations in a single layer of a [`Circuit`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct OpLayer {
    /// All unitary operations.
    pub gates: Vec<Gate>,
    /// The indices of the qubits to be measured.
    pub meas: Vec<usize>,
}

#[derive(Debug, Error)]
pub enum CircuitError {
    #[error("non-constant depths are not supported by Circuit")]
    NoDynDepth,

    #[error("active-feedback circuits are not supported by Circuit")]
    NoFeedback,
}
use CircuitError::*;
pub type CircuitResult<T> = Result<T, CircuitError>;

/// A fixed set of circuit operations.
#[derive(Clone, Debug, PartialEq)]
pub struct Circuit {
    n: usize,
    ops: Vec<OpLayer>,
    reset: bool,
}

impl Circuit {
    fn sample_simple<R>(
        n: usize,
        offs: bool,
        bounds: BoundaryConfig,
        ops: &mut OpLayer,
        rng: &mut R,
    )
    where R: Rng + ?Sized
    {
        (0..n).for_each(|k| {
            ops.gates.push(Gate::sample_single(k, rng));
        });
        match bounds {
            BoundaryConfig::Open => {
                Pairs::new(offs, n).for_each(|(a, b)| {
                    ops.gates.push(Gate::CX(a, b));
                });
            },
            BoundaryConfig::Periodic => {
                PairsPeriodic::new(offs, n).for_each(|(a, b)| {
                    ops.gates.push(Gate::CX(a, b));
                });
            },
        }
    }

    fn sample_cliffs<R>(
        n: usize,
        offs: bool,
        bounds: BoundaryConfig,
        ops: &mut OpLayer,
        rng: &mut R,
    )
    where R: Rng + ?Sized
    {
        match bounds {
            BoundaryConfig::Open => {
                Pairs::new(offs, n).for_each(|(a, _b)| {
                    let mut gates = Clifford::gen(2, rng).unpack().0;
                    gates.iter_mut().for_each(|g| { g.shift(a); });
                    ops.gates.append(&mut gates);
                });
            },
            BoundaryConfig::Periodic => {
                PairsPeriodic::new(offs, n).for_each(|(a, _b)| {
                    let mut gates = Clifford::gen(2, rng).unpack().0;
                    gates.iter_mut().for_each(|g| { g.shift_mod(a, n); });
                    ops.gates.append(&mut gates);
                });
            },
        }
    }

    fn sample_gateset<R>(
        n: usize,
        g1: &G1Set,
        g2: &G2Set,
        offs: bool,
        bounds: BoundaryConfig,
        ops: &mut OpLayer,
        rng: &mut R,
    )
    where R: Rng + ?Sized
    {
        (0..n).for_each(|k| { ops.gates.push(g1.sample(k, rng)); });
        match bounds {
            BoundaryConfig::Open => {
                Pairs::new(offs, n).for_each(|(a, b)| {
                    ops.gates.push(g2.sample(a, b, rng));
                })
            },
            BoundaryConfig::Periodic => {
                PairsPeriodic::new(offs, n).for_each(|(a, b)| {
                    ops.gates.push(g2.sample(a, b, rng));
                })
            },
        }
    }

    fn sample_measurements<R>(
        n: usize,
        d: usize,
        config: MeasureConfig,
        ops: &mut OpLayer,
        rng: &mut R,
    )
    where R: Rng + ?Sized
    {
        use MeasLayerConfig::*;
        use MeasProbConfig::*;

        enum Pred<'a> {
            Never,
            Always,
            Prob(f32),
            Func(Box<dyn Fn(usize) -> bool + 'a>),
        }

        fn do_measure<R>(
            n: usize,
            pred: Pred,
            ops: &mut OpLayer,
            rng: &mut R,
        )
        where R: Rng + ?Sized
        {
            (0..n)
                .filter(|k| {
                    match &pred {
                        Pred::Never => false,
                        Pred::Always => true,
                        Pred::Prob(p) => rng.gen::<f32>() < *p,
                        Pred::Func(f) => f(*k),
                    }
                })
                .for_each(|k| { ops.meas.push(k); });
        }

        let MeasureConfig { layer, prob, reset: _ } = config;
        match layer {
            Every | Period(1) => {
                let pred =
                    match prob {
                        Random(p) => Pred::Prob(p),
                        Cycling(0) => Pred::Never,
                        Cycling(1) => Pred::Always,
                        Cycling(n) => Pred::Func(
                            Box::new(move |k| k % n == d % n)),
                        CyclingInv(0) => Pred::Always,
                        CyclingInv(1) => Pred::Never,
                        CyclingInv(n) => Pred::Func(
                            Box::new(move |k| k % n != d % n)),
                        Block(0) => Pred::Never,
                        Block(b) => Pred::Func(
                            Box::new(move |k| k / b == d % b)),
                        Window(0) => Pred::Never,
                        Window(w) => {
                            let d = d as isize;
                            let w = w as isize;
                            let n = n as isize;
                            Pred::Func(
                                Box::new(move |k| {
                                    (k as isize - d).rem_euclid(n) / w == 0
                                })
                            )
                        }
                    };
                do_measure(n, pred, ops, rng);
            },
            Period(m) if d % m == 0 => {
                let pred =
                    match prob {
                        Random(p) => Pred::Prob(p),
                        Cycling(0) => Pred::Never,
                        Cycling(1) => Pred::Always,
                        Cycling(n) => Pred::Func(
                            Box::new(move |k| k % n == (d / m) % n)),
                        CyclingInv(0) => Pred::Always,
                        CyclingInv(1) => Pred::Never,
                        CyclingInv(n) => Pred::Func(
                            Box::new(move |k| k % n != (d / m) % n)),
                        Block(0) => Pred::Never,
                        Block(b) => Pred::Func(
                            Box::new(move |k| k / b == (d / m) % b)),
                        Window(0) => Pred::Never,
                        Window(w) => {
                            let d = d as isize;
                            let w = w as isize;
                            let n = n as isize;
                            let m = m as isize;
                            Pred::Func(
                                Box::new(move |k| {
                                    (k as isize - d / m).rem_euclid(n) / w == 0
                                })
                            )
                        },
                    };
                do_measure(n, pred, ops, rng);
            },
            _ => { },
        }
    }

    /// Generate a fixed circuit description for `n` qubits from a
    /// [`CircuitConfig`].
    pub fn gen<R>(n: usize, config: CircuitConfig, rng: &mut R)
        -> CircuitResult<Self>
    where R: Rng + ?Sized
    {
        let CircuitConfig {
            depth: depth_conf,
            gates: gate_conf,
            boundaries: bounds,
            measurement: meas_conf,
        } = config;
        let DepthConfig::Const(depth) = depth_conf
            else { return Err(NoDynDepth); };
        if let GateConfig::Feedback(..) = &gate_conf { return Err(NoFeedback); }

        let mut ops: Vec<OpLayer> = Vec::new();
        let mut layer = OpLayer::default();
        for d in 0..depth {
            match &gate_conf {
                GateConfig::Simple => {
                    Self::sample_simple(n, d % 2 == 1, bounds, &mut layer, rng);
                },
                GateConfig::Clifford2 => {
                    Self::sample_cliffs(n, d % 2 == 1, bounds, &mut layer, rng);
                },
                GateConfig::GateSet(ref g1, ref g2) => {
                    Self::sample_gateset(
                        n, g1, g2, d % 2 == 1, bounds, &mut layer, rng);
                },
                GateConfig::Circuit(circ) => {
                    circ.iter().copied()
                        .for_each(|gate| { layer.gates.push(gate); });
                },
                GateConfig::Feedback(..) => unreachable!(),
            }
            Self::sample_measurements(n, d, meas_conf, &mut layer, rng);
            ops.push(std::mem::take(&mut layer));
        }
        Ok(Self { n, ops, reset: meas_conf.reset })
    }
}


