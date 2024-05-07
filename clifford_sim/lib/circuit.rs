//! Abstractions for driving randomized Clifford circuits and measuring
//! measurement-induced phase transitions (MIPTs).

use rand::{ rngs::StdRng, Rng, SeedableRng };
use crate::{
    gate::{ Clifford, Gate },
    stab::{ Outcome, Stab },
};

/// Main driver for running circuits of alternating unitary evolution and
/// measurement.
#[derive(Clone, Debug)]
pub struct StabCircuit<const N: usize> {
    pub state: Stab<N>,
    pub outcomes: Vec<[Option<Outcome>; N]>,
    pub p_meas: f32,
    pub cut: Option<EntropyCut>,
    pub rng: StdRng,
}

impl<const N: usize> StabCircuit<N> {
    /// Create a new `StabCircuit` for a 1D chain of `N` qubits, with state
    /// initialized to ∣0...0⟩ and no recorded outcomes.
    ///
    /// `p_meas` is the probability with which qubits are measured in the
    /// Z-basis at each layer in the circuit. `cut` is the location in the chain
    /// at which to place the bipartition for calculation of the entanglement
    /// entropy, defaulting to a single cut at `floor(N / 2)`. Optionally also
    /// seed the internal random number generator.
    ///
    /// *Panics if `meas_rate` is not a valid probability.*
    pub fn new(p_meas: f32, cut: Option<EntropyCut>, seed: Option<u64>)
        -> Self
    {
        if !(0.0..=1.0).contains(&p_meas) {
            panic!("StabCircuit: measurement rate must be a valid probability");
        }
        let rng
            = seed.map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_entropy);
        Self { state: Stab::default(), outcomes: Vec::new(), p_meas, cut, rng }
    }

    fn gates_simple(&mut self, cnot_offs: bool, buf: &mut Vec<Gate>) {
        (0..N).for_each(|k| {
            buf.push(Gate::sample_single(k, &mut self.rng));
        });
        CNots::new(cnot_offs, N).for_each(|cx| { buf.push(cx); });
    }

    fn measure(&mut self, buf: &mut [Option<Outcome>; N]) {
        buf.iter_mut()
            .enumerate()
            .for_each(|(k, ok)| {
                if self.rng.gen::<f32>() < self.p_meas {
                    *ok = Some(self.state.measure(k, &mut self.rng));
                } else {
                    *ok = None;
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
    pub fn run_simple(&mut self, depth: usize) -> Vec<f32> {
        let mut gates: Vec<Gate> = Vec::new();
        let mut outcomes: [Option<Outcome>; N] = [None; N];
        let mut entropy: Vec<f32> = Vec::with_capacity(depth + 1);
        entropy.push(self.entropy());
        for d in 0..depth {
            self.gates_simple(d % 2 == 1, &mut gates);
            self.state.apply_circuit(&gates);
            gates.clear();

            self.measure(&mut outcomes);
            self.outcomes.push(outcomes);

            entropy.push(self.entropy());
        }
        entropy
    }

    /// Run a simple MIPT procedure with feedback on measurement outcomes.
    ///
    /// At each layer in the circuit, the previous layer's measurement outcomes
    /// are passed to the supplied closure along with the current circuit depth
    /// to determine when to halt or what gates to apply. For the first layer,
    /// no measurements have been performed and hence an array of `None`s is
    /// passed to the closure. After gates are applied, measurements are
    /// performed on each qubit with probability `self.p_meas`, whose outcomes
    /// are then passed back to the closure.
    ///
    /// Returns the entanglement entropy measured once before the first layer,
    /// and then after each round of measurements.
    pub fn run_feedback<F>(&mut self, mut feedback: F) -> Vec<f32>
    where F: FnMut(usize, &[Option<Outcome>; N]) -> Feedback
    {
        let mut outcomes: [Option<Outcome>; N] = [None; N];
        let mut entropy: Vec<f32> = vec![self.entropy()];
        let mut gates: Vec<Gate> = Vec::new();
        let mut d: usize = 0;
        loop {
            match feedback(0, &outcomes) {
                Feedback::Halt => { break entropy; },
                Feedback::Simple => {
                    self.gates_simple(d % 2 == 1, &mut gates);
                    self.state.apply_circuit(&gates);
                    gates.clear();
                },
                Feedback::Gates(fgates) => {
                    self.state.apply_circuit(&fgates);
                },
            }

            self.measure(&mut outcomes);
            self.outcomes.push(outcomes);

            entropy.push(self.entropy());
            d += 1;
        }
    }

    /// Run the MIPT procedure, replacing the unitary evolution between
    /// measurements in the "simple" procedure with a random `N`-qubit Clifford
    /// gate.
    ///
    /// Returns the entanglement entropy measured once before the first layer,
    /// and then after each round of measurements.
    pub fn run_clifford(&mut self, depth: usize) -> Vec<f32> {
        let mut gates: Clifford<N>;
        let mut outcomes: [Option<Outcome>; N] = [None; N];
        let mut entropy: Vec<f32> = Vec::with_capacity(depth + 1);
        entropy.push(self.entropy());
        for _ in 0..depth {
            gates = Clifford::gen(&mut self.rng);
            self.state.apply_circuit(&gates);

            self.measure(&mut outcomes);
            self.outcomes.push(outcomes);

            entropy.push(self.entropy());
        }
        entropy
    }
}

struct CNots {
    iter: std::ops::Range<usize>
}

impl CNots {
    fn new(offs: bool, stop: usize) -> Self {
        Self { iter: if offs { 1 } else { 0 } .. stop }
    }
}

impl Iterator for CNots {
    type Item = Gate;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
            .zip(self.iter.next())
            .map(|(a, b)| Gate::CX(a, b))
    }
}

/// Method for handling bipartitions on a 1D chain for the purpose of calculting
/// the entanglement entropy.
#[derive(Copy, Clone, Debug)]
pub enum EntropyCut {
    /// Average over all possible bipartitions.
    Average,
    /// A single bipartition. `Single(k)` places `k` qubits in one partition and
    /// the remainder in the other.
    Single(usize),
}

/// Post-measurement determination 
#[derive(Clone, Debug)]
pub enum Feedback {
    Halt,
    Simple,
    Gates(Vec<Gate>),
}

