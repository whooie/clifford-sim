//! Abstractions for driving randomized Clifford circuits and measuring
//! measurement-induced phase transitions (MIPTs).

use rand::{ rngs::StdRng, Rng, SeedableRng };
use rustc_hash::FxHashSet as HashSet;
use crate::{
    gate::{ Clifford, Gate, G1, G2 },
    stab::{ Stab, Outcome },
};

/// Type alias for [`HashSet`].
pub type Set<T> = HashSet<T>;

/// Main driver for running circuits of alternating unitary evolution and
/// measurement.
#[derive(Clone, Debug)]
pub struct StabCircuit {
    pub state: Stab,
    pub p_meas: f32,
    pub cut: Option<EntropyCut>,
    n: usize,
    rng: StdRng,
}

/// The outcomes associated with a single measurement layer, ordered by qubit
/// index.
pub type MeasLayer = Vec<Option<Outcome>>;

/// The outcomes from each measurement layer in a full circuit.
pub type MeasRecord = Vec<MeasLayer>;

impl StabCircuit {
    /// Create a new `StabCircuit` for a 1 chain of `n` qubits, with state
    /// initialized to ∣0...0⟩ and no recorded outcomes.
    ///
    /// `p_meas` is the probability with which qubits are measured in the
    /// Z-basis at each layer in the circuit. `cut` is the location in the chain
    /// at which to place the bipartition for calculation of the entanglement
    /// entropy, defaulting to a single cut at `floor(n / 2)`. Optionally also
    /// seed the internal random number generator.
    ///
    /// *Panics if `meas_rate` is not a valid probability.*
    pub fn new(
        n: usize,
        p_meas: f32,
        cut: Option<EntropyCut>,
        seed: Option<u64>,
    ) -> Self
    {
        if !(0.0..=1.0).contains(&p_meas) {
            panic!("StabCircuit: measurement rate must be a valid probability");
        }
        let rng
            = seed.map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_entropy);
        Self { state: Stab::new(n), p_meas, cut, n, rng }
    }

    /// Return the number of qubits.
    pub fn n(&self) -> usize { self.n }

    fn sample_simple(&mut self, offs: bool, buf: &mut Vec<Gate>) {
        (0..self.n).for_each(|k| {
            buf.push(Gate::sample_single(k, &mut self.rng));
        });
        Pairs::new(offs, self.n).for_each(|(a, b)| {
            buf.push(Gate::CX(a, b));
        });
    }

    fn sample_gateset(
        &mut self,
        g1: &G1Set,
        g2: &G2Set,
        offs: bool,
        buf: &mut Vec<Gate>,
    ) {
        (0..self.n).for_each(|k| {
            buf.push(g1.sample(k, &mut self.rng));
        });
        Pairs::new(offs, self.n).for_each(|(a, b)| {
            buf.push(g2.sample(a, b, &mut self.rng));
        });
    }

    fn measure(&mut self, buf: &mut [Option<Outcome>]) {
        buf.iter_mut()
            .enumerate()
            .for_each(|(k, outk)| {
                if self.rng.gen::<f32>() < self.p_meas {
                    *outk = Some(self.state.measure(k, &mut self.rng));
                } else {
                    *outk = None;
                }
            });
    }

    fn entropy(&self) -> f32 {
        match self.cut {
            None
                => self.state.entanglement_entropy_single(None),
            Some(EntropyCut::Average)
                => self.state.entanglement_entropy(),
            Some(EntropyCut::Single(c))
                => self.state.entanglement_entropy_single(Some(c)),
        }
    }

    /// Run `depth` layers of a simple MIPT procedure.
    ///
    /// This procedure consists of the following at each layer in the circuit:
    /// 1. Apply a random single-qubit rotation (*H*, *X*, *Y*, *Z*, *S*) to
    /// each qubit
    /// 1. Apply a CNOT to adjacent pairs of qubits, alternating between left
    /// and right neighbors on each layer
    /// 1. Perform a projective measurement on qubits with probability
    /// `self.p_meas`
    ///
    /// Returns the entanglement entropy measured once before any operations
    /// have been performed, and then after each round of measurements.
    /// Optionally provide a mutable reference to a list of measurements to
    /// record the outcomes from the circuit.
    pub fn run_simple(
        &mut self,
        depth: usize,
        mut meas: Option<&mut MeasRecord>,
    ) -> Vec<f32>
    {
        let mut gates: Vec<Gate> = Vec::new();
        let mut outcomes: Vec<Option<Outcome>> = vec![None; self.n];
        let mut entropy: Vec<f32> = Vec::with_capacity(depth + 1);
        entropy.push(self.entropy());
        for d in 0..depth {
            gates.clear();
            self.sample_simple(d % 2 == 1, &mut gates);
            self.state.apply_circuit(&gates);

            self.measure(&mut outcomes);
            if let Some(record) = meas.as_mut() {
                record.push(outcomes.clone());
            }

            entropy.push(self.entropy());
        }
        entropy
    }

    /// Lazy iterator version of [`Self::run_simple`].
    pub fn run_simple_lazy<'a>(
        &'a mut self,
        depth: usize,
        mut meas: Option<&'a mut MeasRecord>
    ) -> impl Iterator<Item = f32> + '_
    {
        let mut gates: Vec<Gate> = Vec::new();
        let mut outcomes: MeasLayer = vec![None; self.n];
        let entropy_init = [self.entropy()].into_iter();
        let entropy_iter
            = (0..depth)
            .map(move |d| {
                gates.clear();
                self.sample_simple(d % 2 == 1, &mut gates);
                self.state.apply_circuit(&gates);

                self.measure(&mut outcomes);
                if let Some(record) = meas.as_mut() {
                    record.push(outcomes.clone());
                }
                self.entropy()
            });
        entropy_init.chain(entropy_iter)
    }

    /// Like [`Self::run_simple`], but run the circuit until the entropy in the
    /// system has converged to within `tol`, which defaults to 10<sup>-4</sup>.
    ///
    /// The criterion for convergence is
    /// ```math
    /// 2 \left| \frac{\mu_{k+1} - \mu_k}{\mu_{k+1} + \mu_k} \right| < \text{tol}
    /// ```
    /// where `$\mu_k$` is the running average of the first `$k$` entropy
    /// measurement, including the first before the circuit has begun; i.e. the
    /// entropy has reached a steady state when the absolute difference between
    /// consecutive values of the running average divided by their mean is less
    /// than `tol`.
    pub fn run_simple_until_converged(
        &mut self,
        tol: Option<f32>,
        mut meas: Option<&mut MeasRecord>,
    ) -> Vec<f32>
    {
        let tol = tol.unwrap_or(1e-4);
        let mut gates: Vec<Gate> = Vec::new();
        let mut outcomes: Vec<Option<Outcome>> = vec![None; self.n];
        let mut entropy: Vec<f32> = Vec::new();
        entropy.push(self.entropy());
        let mut d: usize = 0;
        let mut sbar: f32 = 0.0;
        let mut s: f32;
        let mut check: f32;
        loop {
            gates.clear();
            self.sample_simple(d % 2 == 1, &mut gates);
            self.state.apply_circuit(&gates);

            self.measure(&mut outcomes);
            if let Some(record) = meas.as_mut() {
                record.push(outcomes.clone());
            }

            s = self.entropy();
            entropy.push(s);
            check = (2.0 * (sbar - s) / ((2 * d + 3) as f32 * sbar + s)).abs();
            if check < tol {
                break entropy;
            }
            sbar = (sbar * (d + 1) as f32 + s) / (d + 2) as f32;
            d += 1;
        }
    }

    /// Run a simple MIPT procedure with feedback on measurement outcomes.
    ///
    /// At each layer in the circuit, the previous layer's measurement outcomes
    /// and state entropy are passed to the supplied closure along with the
    /// current circuit depth to determine when to halt or what gates to apply.
    /// For the first layer, no measurements have been performed and hence an
    /// array of `None`s is passed to the closure. After gates are applied,
    /// measurements are performed on each qubit with probability `self.p_meas`,
    /// whose outcomes are then passed back to the closure.
    ///
    /// Returns the entanglement entropy measured once before the first layer,
    /// and then after each round of measurements.
    pub fn run_feedback<F>(
        &mut self,
        mut feedback: F,
        mut meas: Option<&mut MeasRecord>,
    ) -> Vec<f32>
    where F: FnMut(usize, f32, &[Option<Outcome>]) -> Feedback
    {
        let mut outcomes: Vec<Option<Outcome>> = vec![None; self.n];
        let mut s: f32 = self.entropy();
        let mut entropy: Vec<f32> = vec![s];
        let mut gates: Vec<Gate> = Vec::new();
        let mut d: usize = 0;
        loop {
            gates.clear();
            match feedback(d, s, &outcomes) {
                Feedback::Halt => { break entropy; },
                Feedback::Simple => {
                    self.sample_simple(d % 2 == 1, &mut gates);
                    self.state.apply_circuit(&gates);
                },
                Feedback::Clifford => {
                    let cliff = Clifford::gen(self.n, &mut self.rng);
                    gates.append(&mut cliff.unpack().0);
                    self.state.apply_circuit(&gates);
                },
                Feedback::GateSet(g1, g2) => {
                    self.sample_gateset(&g1, &g2, d % 2 == 1, &mut gates);
                    self.state.apply_circuit(&gates);
                },
                Feedback::Circuit(circ) => {
                    self.state.apply_circuit(&circ);
                },
            }

            self.measure(&mut outcomes);
            if let Some(record) = meas.as_mut() {
                record.push(outcomes.clone());
            }

            s = self.entropy();
            entropy.push(s);
            d += 1;
        }
    }

    /// Run the MIPT procedure, replacing the unitary evolution between
    /// measurements in the "simple" procedure with a random `n`-qubit Clifford
    /// gate.
    ///
    /// Returns the entanglement entropy measured once before the first layer,
    /// and then after each round of measurements.
    pub fn run_clifford(
        &mut self,
        depth: usize,
        mut meas: Option<&mut MeasRecord>,
    ) -> Vec<f32>
    {
        let mut gates: Clifford;
        let mut outcomes: Vec<Option<Outcome>> = vec![None; self.n];
        let mut entropy: Vec<f32> = Vec::with_capacity(depth + 1);
        entropy.push(self.entropy());
        for _ in 0..depth {
            gates = Clifford::gen(self.n, &mut self.rng);
            self.state.apply_circuit(&gates);

            self.measure(&mut outcomes);
            if let Some(record) = meas.as_mut() {
                record.push(outcomes.clone());
            }

            entropy.push(self.entropy());
        }
        entropy
    }

    /// Run the MIPT procedure for a general config.
    ///
    /// Returns the entanglement entropy measured once before the first layer,
    /// and then after each round of measurements.
    pub fn run<'a>(
        &mut self,
        config: CircuitConfig<'a>,
        mut meas: Option<&mut MeasRecord>,
    ) -> Vec<f32> {
        let CircuitConfig { depth: depth_conf, gates: mut gate_conf } = config;
        let mut gates: Vec<Gate> = Vec::new();
        let mut outcomes: MeasLayer = vec![None; self.n];
        let mut s: f32 = self.entropy();
        let mut entropy: Vec<f32> = vec![s];
        let mut d: usize = 0;
        let mut sbar: f32 = 0.0;
        let mut check: f32;
        loop {
            gates.clear();
            match &mut gate_conf {
                GateConfig::Simple => {
                    self.sample_simple(d % 2 == 1, &mut gates);
                    self.state.apply_circuit(&gates);
                },
                GateConfig::Clifford => {
                    let cliff = Clifford::gen(self.n, &mut self.rng);
                    gates.append(&mut cliff.unpack().0);
                    self.state.apply_circuit(&gates);
                },
                GateConfig::GateSet(g1, g2) => {
                    self.sample_gateset(g1, g2, d % 2 == 1, &mut gates);
                    self.state.apply_circuit(&gates);
                },
                GateConfig::Circuit(circ) => {
                    self.state.apply_circuit(&*circ);
                },
                GateConfig::Feedback(f) => {
                    match f(d, s, &outcomes) {
                        Feedback::Halt => { break entropy; },
                        Feedback::Simple => {
                            self.sample_simple(d % 2 == 1, &mut gates);
                            self.state.apply_circuit(&gates);
                        },
                        Feedback::Clifford => {
                            let cliff = Clifford::gen(self.n, &mut self.rng);
                            gates.append(&mut cliff.unpack().0);
                            self.state.apply_circuit(&gates);
                        },
                        Feedback::GateSet(g1, g2) => {
                            self.sample_gateset(
                                &g1, &g2, d % 2 == 1, &mut gates);
                            self.state.apply_circuit(&gates);
                        },
                        Feedback::Circuit(circ) => {
                            self.state.apply_circuit(&circ);
                        },
                    }
                },
            }

            self.measure(&mut outcomes);
            if let Some(rcd) = meas.as_mut() { rcd.push(outcomes.clone()); }

            s = self.entropy();
            entropy.push(s);
            match depth_conf {
                DepthConfig::Converge(tol) => {
                    check = (
                        2.0 * (sbar - s) / ((2 * d + 3) as f32 * sbar + s)
                    ).abs();
                    if check < tol.unwrap_or(1e-4) { break entropy; }
                    sbar = (sbar + (d + 1) as f32 + s) / (d + 2) as f32;
                },
                DepthConfig::Const(d0) => {
                    if d >= d0 { break entropy; }
                },
            }
            d += 1;
        }
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

/// Method for handling bipartitions on a 1 chain for the purpose of calculting
/// the entanglement entropy.
#[derive(Copy, Clone, Debug)]
pub enum EntropyCut {
    /// Average over all possible bipartitions.
    Average,
    /// A single bipartition. `Single(k)` places `k` qubits in one partition and
    /// the remainder in the other.
    Single(usize),
}

/// Post-measurement determination of a mid-circuit action.
#[derive(Clone, Debug)]
pub enum Feedback {
    /// Immediately halt the circuit.
    Halt,
    /// Draw the next layer of gates from the "simple" set (all single-qubit
    /// gates and tiling CXs).
    Simple,
    /// Draw the next layer of gates uniformly from the set of *N*-qubit
    /// Clifford gates.
    Clifford,
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
    /// Run until the entanglement entropy converges on a steady-state value to
    /// within some tolerance (defaults to 10<sup>-4</sup>).
    ///
    /// The criterion for convergence is
    /// ```math
    /// 2 \left| \frac{\mu_{k+1} - \mu_k}{\mu_{k+1} + \mu_k} \right| < \text{tol}
    /// ```
    /// where `$\mu_k$` is the running average of the first `$k$` entropy
    /// measurement, including the first before the circuit has begun; i.e. the
    /// entropy has reached a steady state when the absolute difference between
    /// consecutive values of the running average divided by their mean is less
    /// than `tol`.
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
    /// Uniformly sampled *N*-qubit Clifford gates.
    Clifford,
    /// A particular gate set.
    GateSet(G1Set, G2Set),
    /// A particular sequence of gates.
    Circuit(Vec<Gate>),
    /// Gates based on a feedback function on measurement outcomes.
    Feedback(FeedbackFn<'a>),
}

/// Top-level config for a circuit.
pub struct CircuitConfig<'a> {
    /// Set the depth of the circuit.
    pub depth: DepthConfig,
    /// Set available gates to draw from.
    pub gates: GateConfig<'a>,
}

