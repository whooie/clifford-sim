//! Like [`stab`][crate::stab], but allowing for mixed stabilizer states.
//!
//! Here, a stabilizer state of *n* qubits is allowed to have *m* ≤ *n*
//! (de)stabilizers; any state with *m* < *n* contains some degree of mixing.

#![allow(unused_imports)]

use std::fmt;
use itertools::{ Itertools, MultiProduct };
use nalgebra as na;
use num_complex::Complex64 as C64;
use rand::Rng;
use once_cell::sync::Lazy;
use crate::{
    gate::{ Gate, Pauli, Phase },
    indexmap::IndexMap,
    stab::*,
};

/// A (possibly mixed) state of a finite register of qubits, identified by its
/// stabilizer group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StabM {
    pub(crate) m: usize, // number of stabilizers
    pub(crate) stab: Stab,
}

impl From<Stab> for StabM {
    fn from(stab: Stab) -> Self {
        let m = stab.num_qubits();
        Self { m, stab }
    }
}

impl TryFrom<StabM> for Stab {
    type Error = Box<StabM>;

    fn try_from(stabm: StabM) -> Result<Self, Self::Error> {
        if stabm.m == stabm.stab.n {
            Ok(stabm.stab)
        } else {
            Err(stabm.into())
        }
    }
}

impl StabM {
    /// Create a new (pure) stabilizer state of size `n` initialized to ∣0...0⟩.
    pub fn new(n: usize) -> Self {
        let stab = Stab::new(n);
        Self { m: n, stab }
    }

    /// Return the number of qubits.
    pub fn num_qubits(&self) -> usize { self.stab.num_qubits() }

    /// Return `true` if `self` is a pure state.
    pub fn is_pure(&self) -> bool { self.m == self.stab.n }

    /// Return a reference to `self` as a `Stab` if there is no mixing.
    pub fn as_pure(&self) -> Option<&Stab> {
        self.is_pure().then_some(&self.stab)
    }

    /// Return a mutable reference to `self` as a `Stab` if there is no mixing.
    pub fn as_pure_mut(&mut self) -> Option<&mut Stab> {
        self.is_pure().then_some(&mut self.stab)
    }

    /// Apply a Hadamard gate to the `k`-th qubit.
    pub fn apply_h(&mut self, k: usize) -> &mut Self {
        self.stab.apply_h(k);
        self
    }

    /// Apply an S gate (= Z(π/2)) to the `k`-th qubit.
    pub fn apply_s(&mut self, k: usize) -> &mut Self {
        self.stab.apply_s(k);
        self
    }

    /// Apply an S<sup>†</sup> gate (= Z(-π/2)) to the `k`-th qubit.
    pub fn apply_sinv(&mut self, k: usize) -> &mut Self {
        self.stab.apply_sinv(k);
        self
    }

    /// Apply X gate to the `k`-th qubit.
    pub fn apply_x(&mut self, k: usize) -> &mut Self {
        self.stab.apply_x(k);
        self
    }

    /// Apply an Y gate to the `k`-th qubit.
    pub fn apply_y(&mut self, k: usize) -> &mut Self {
        self.stab.apply_y(k);
        self
    }

    /// Apply an Z gate to the `k`-th qubit.
    pub fn apply_z(&mut self, k: usize) -> &mut Self {
        self.stab.apply_z(k);
        self
    }

    /// Apply a CNOT gate to the `b`-th qubit, with the `a`-th qubit as control.
    pub fn apply_cnot(&mut self, a: usize, b: usize) -> &mut Self {
        self.stab.apply_cnot(a, b);
        self
    }

    /// Apply a CZ gate to the `a`-th and `b`-th qubits.
    pub fn apply_cz(&mut self, a: usize, b: usize) -> &mut Self {
        self.stab.apply_cz(a, b);
        self
    }

    /// Apply a SWAP gate to the `a`-th and `b`-th qubits.
    pub fn apply_swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.stab.apply_swap(a, b);
        self
    }

    /// Perform the action of a gate.
    ///
    /// Does nothing if any qubit indices are out of bounds.
    pub fn apply_gate(&mut self, gate: Gate) -> &mut Self {
        self.stab.apply_gate(gate);
        self
    }

    /// Perform a series of gates.
    pub fn apply_circuit<'a, I>(&mut self, gates: I) -> &mut Self
    where I: IntoIterator<Item = &'a Gate>
    {
        self.stab.apply_circuit(gates);
        self
    }

    // convert a row index in the destabilizer to the corresponding position in
    // the stabilizer, and vice-versa
    fn row_idx_conj(&self, idx: usize) -> usize {
        if idx < self.stab.n {
            idx + self.stab.n
        } else {
            idx - self.stab.n
        }
    }

    /// Perform a projective measurement on a qubit `k` in the Z-basis,
    /// returning the outcome of the measurement.
    ///
    /// **Note**: this measurement is either deterministic (when the target
    /// qubit is ∣±z⟩) or random (otherwise). For post-selected measurements,
    /// see [`Self::measure_postsel`].
    pub fn measure<R>(&mut self, k: usize, rng: &mut R) -> Outcome
    where R: Rng + ?Sized
    {
        let mut rnd: bool = false;
        let k5: usize = k >> 5;
        let pw: u32 = PW[k & 31];
        let mut p: usize = 0;
        let mut m: usize = 0;

        for (q, x_q_k5) in
            self.stab.x.column(k5).iter()
                .enumerate()
                .take(2 * self.stab.n)
                .skip(self.m)
        {
            rnd = *x_q_k5 & pw != 0;
            if rnd { p = q; break; }
        }

        if rnd && (self.stab.n .. self.stab.n + self.m).contains(&p) {
            self.stab.row_copy(p, p - self.stab.n);
            self.stab.row_set(k + self.stab.n, p);
            self.stab.r[p] = 2 * u8::from(rng.gen::<bool>());
            for i in 0 .. 2 * self.stab.n {
                if i != p - self.stab.n && self.stab.x[(i, k5)] & pw != 0 {
                    self.stab.row_mul(p - self.stab.n, i);
                }
            }
            if self.stab.r[p] != 0 {
                Outcome::Rand1
            } else {
                Outcome::Rand0
            }
        } else if rnd {
            let pconj = self.row_idx_conj(p);
            self.stab.row_copy(p, pconj);
            self.stab.row_set(k + self.stab.n, p);
            self.stab.r[p] = 2 * u8::from(rng.gen::<bool>());
            for i in 0 .. 2 * self.stab.n {
                if i != pconj && self.stab.x[(i, k5)] & pw != 0 {
                    self.stab.row_mul(pconj, i);
                }
            }
            let outcome =
                if self.stab.r[p] != 0 {
                    Outcome::Rand1
                } else {
                    Outcome::Rand0
                };
            self.stab.row_swap(p, self.stab.n + self.m);
            if p != self.m { self.stab.row_swap(pconj, self.m); }
            self.m += 1;
            outcome
        } else {
            for (q, x_q_k5) in
                self.stab.x.column(k5).iter()
                    .take(self.stab.n)
                    .enumerate()
            {
                if x_q_k5 & pw != 0 { m = q; break; }
            }
            self.stab.row_copy(m + self.stab.n, 2 * self.stab.n);
            for i in m + 1 .. self.stab.n {
                if self.stab.x[(i, k5)] & pw != 0 {
                    self.stab.row_mul(i + self.stab.n, 2 * self.stab.n);
                }
            }
            if self.stab.r[2 * self.stab.n] != 0 {
                Outcome::Det1
            } else {
                Outcome::Det0
            }
        }
    }

    /// Return the single-qubit Z-basis measurement probabilities for the `k`-th
    /// qubit, but do not actually perform a measurement.
    ///
    /// Note that these probabilities are always either exactly even or
    /// concentrated on one of the two possible outcomes.
    pub fn probs(&self, k: usize) -> Probs {
        let mut rnd: bool = false;
        let k5: usize = k >> 5;
        let pw: u32 = PW[k & 31];
        let mut m: usize = 0;

        for x_q_k5 in
            self.stab.x.column(k5).iter()
                .take(2 * self.stab.n)
                .skip(self.m)
        {
            rnd = *x_q_k5 & pw != 0;
            if rnd { break; }
        }

        if rnd {
            Probs::Rand
        } else {
            for (q, x_q_k5) in
                self.stab.x.column(k5).iter()
                    .take(self.stab.n)
                    .enumerate()
            {
                if x_q_k5 & pw != 0 { m = q; break; }
            }
            let mut scratch = self.stab.extract_row(m + self.stab.n);
            for i in m + 1 .. self.stab.n {
                if self.stab.x[(i, k5)] & pw != 0 {
                    self.stab.row_mul_s(i + self.stab.n, &mut scratch);
                }
            }
            if scratch.r != 0 { Probs::Det1 } else { Probs::Det0 }
        }
    }

    /// Like [`Self::measure`], but deterministically reset the qubit state to
    /// ∣0⟩ after the measurement.
    ///
    /// The outcome of the original measurement is returned.
    pub fn measure_reset<R>(&mut self, k: usize, rng: &mut R) -> Outcome
    where R: Rng + ?Sized
    {
        let outcome = self.measure(k, rng);
        if outcome.is_1() { self.apply_x(k); }
        outcome
    }

    /// Like [`Self::measure`], but deterministically post-selects on a desired
    /// measurement outcome.
    ///
    /// The tradeoff is that if the desired outcome is incompatible with `self`,
    /// the `self` must be invalidated. This method therefore consumes `self`,
    /// returning it as a [`Ok`] if the post-selection was valid and [`Err`]
    /// otherwise.
    pub fn measure_postsel(mut self, k: usize, postsel: Postsel)
        -> Result<Self, Box<Self>>
    {
        let mut rnd: bool = false;
        let k5: usize = k >> 5;
        let pw: u32 = PW[k & 31];
        let mut p: usize = 0;
        let mut m: usize = 0;

        for (q, x_q_k5) in
            self.stab.x.column(k5).iter()
                .enumerate()
                .take(2 * self.stab.n)
                .skip(self.m)
        {
            rnd = *x_q_k5 & pw != 0;
            if rnd { p = q; break; }
        }

        if rnd && (self.stab.n .. self.stab.n + self.m).contains(&p) {
            self.stab.row_copy(p, p - self.stab.n);
            self.stab.row_set(k + self.stab.n, p);
            self.stab.r[p] = 2 * postsel as u8;
            for i in 0 .. 2 * self.stab.n {
                if i != p - self.stab.n && self.stab.x[(i, k5)] & pw != 0 {
                    self.stab.row_mul(p - self.stab.n, i);
                }
            }
            Ok(self)
        } else if rnd {
            let pconj = self.row_idx_conj(p);
            self.stab.row_copy(p, pconj);
            self.stab.row_set(k + self.stab.n, p);
            self.stab.r[p] = 2 * postsel as u8;
            for i in 0 .. 2 * self.stab.n {
                if i != pconj && self.stab.x[(i, k5)] & pw != 0 {
                    self.stab.row_mul(pconj, i);
                }
            }
            self.stab.row_swap(p, self.stab.n + self.m);
            if p != self.m { self.stab.row_swap(pconj, self.m); }
            self.m += 1;
            Ok(self)
        } else {
            for (q, x_q_k5) in
                self.stab.x.column(k5).iter()
                    .take(self.stab.n)
                    .enumerate()
            {
                if x_q_k5 & pw != 0 { m = q; break; }
            }
            self.stab.row_copy(m + self.stab.n, 2 * self.stab.n);
            for i in m + 1 .. self.stab.n {
                if self.stab.x[(i, k5)] & pw != 0 {
                    self.stab.row_mul(i + self.stab.n, 2 * self.stab.n);
                }
            }
            match (self.stab.r[2 * self.stab.n] != 0, postsel) {
                (true,  Postsel::One ) => Ok(self),
                (false, Postsel::Zero) => Ok(self),
                _ => Err(self.into()),
            }
        }
    }

    /// Convert `self` to a more human-readable stabilizer/destabilizer group
    /// representation.
    pub fn as_group(&self) -> StabMGroup {
        let n0 = self.stab.n;
        let m = self.m;
        let mut j5: usize;
        let mut pw: u32;
        let mut npauli: NPauli =
            NPauli { phase: Phase::Pi0, ops: vec![Pauli::I; n0] };
        let mut ops: Vec<NPauli> = Vec::with_capacity(2 * n0);
        let iter =
            self.stab.x.row_iter()
            .zip(self.stab.z.row_iter())
            .zip(self.stab.r.iter())
            .take(2 * n0);
        for ((xi, zi), ri) in iter {
            npauli.phase = Phase::from_uint(*ri);
            for (j, op) in npauli.ops.iter_mut().enumerate() {
                j5 = j >> 5;
                pw = PW[j & 31];
                match (xi[j5] & pw, zi[j5] & pw) {
                    (0, 0) => { *op = Pauli::I; },
                    (_, 0) => { *op = Pauli::X; },
                    (0, _) => { *op = Pauli::Z; },
                    (_, _) => { *op = Pauli::Y; },
                }
            }
            ops.push(npauli.clone());
        }
        let mut logical_extra = ops.split_off(n0 + m);
        let stab = ops.split_off(n0);
        let mut logical = ops.split_off(m);
        let destab = ops;
        logical.append(&mut logical_extra);
        StabMGroup { destab, stab, logical }
    }

    // /// Calculate the entanglement entropy of the state in a given subsystem.
    // ///
    // /// The partition defaults to the leftmost `floor(n / 2)` qubits.
    // pub fn entanglement_entropy(&self, part: Option<Partition>) -> f32 {
    //     let part = part.unwrap_or(Partition::Left(self.stab.n / 2 - 1));
    //     self.stab.x.row_iter()
    //         .zip(self.stab.z.row_iter())
    //         .skip(self.stab.n)
    //         .take(self.m)
    //         .filter(|(xi, zi)| {
    //             let maybe_l =
    //                 xi.iter()
    //                 .zip(zi)
    //                 .enumerate()
    //                 .find_map(|(j5, (xij5, zij5))| {
    //                     PW.iter()
    //                         .enumerate()
    //                         .find_map(|(jj, pwjj)| {
    //                             (xij5 & pwjj != 0 || zij5 & pwjj != 0)
    //                                 .then_some(jj)
    //                         })
    //                         .map(|jj| (j5 << 5) | jj)
    //                 });
    //             let maybe_r =
    //                 xi.iter()
    //                 .zip(zi)
    //                 .enumerate()
    //                 .rev()
    //                 .find_map(|(j5, (xij5, zij5))| {
    //                     PW.iter()
    //                         .enumerate()
    //                         .rev()
    //                         .find_map(|(jj, pwjj)| {
    //                             (xij5 & pwjj != 0 || zij5 & pwjj != 0)
    //                                 .then_some(jj)
    //                         })
    //                         .map(|jj| (j5 << 5) | jj)
    //                 });
    //             maybe_l.zip(maybe_r)
    //                 .map(|(l, r)| {
    //                     (part.contains(l) && !part.contains(r))
    //                         || (!part.contains(l) && part.contains(r))
    //                 })
    //                 .unwrap_or(false)
    //         })
    //         .count() as f32 * 0.5
    // }
    //
    // /// Calculate the mutual information between two subsystems of a given size,
    // /// maximally spaced from each other around a periodic 1D chain.
    // ///
    // /// The subsystem size defaults to `floor(3 * n / 8)`.
    // ///
    // /// *Panics* if the subsystem size is greater than `floor(n / 2)`.
    // pub fn mutual_information(&self, size: Option<usize>) -> f32 {
    //     let size = size.unwrap_or(3 * self.stab.n / 8);
    //     if size > self.stab.n / 2 {
    //         panic!(
    //             "Stab::mutual_information: subsystem size must be less than \
    //             floor(n/2)"
    //         );
    //     }
    //     let part_a = Partition::Left(size);
    //     let part_b = Partition::Range(self.stab.n / 2, self.stab.n / 2 + size);
    //
    //     let mut cond_a: bool;
    //     let mut n_a: usize = 0;
    //     let mut cond_b: bool;
    //     let mut n_b: usize = 0;
    //     let mut cond_ab: bool;
    //     let mut n_ab: usize = 0;
    //     let mut j5 = 0;
    //     let mut pw = 0;
    //     for (xi, zi) in
    //         self.stab.x.row_iter()
    //             .zip(self.stab.z.row_iter())
    //             .skip(self.stab.n)
    //             .take(self.m)
    //     {
    //         cond_a = (0..self.stab.n)
    //             .filter(|&j| !part_a.contains(j))
    //             .all(|j| {
    //                 j5 = j >> 5;
    //                 pw = PW[j & 31];
    //                 xi[j5] & pw == 0 && zi[j5] & pw == 0
    //             });
    //         if cond_a { n_a += 1; }
    //
    //         cond_b = (0..self.stab.n)
    //             .filter(|&j| !part_b.contains(j))
    //             .all(|j| {
    //                 j5 = j >> 5;
    //                 pw = PW[j & 31];
    //                 xi[j5] & pw == 0 && zi[j5] & pw == 0
    //             });
    //         if cond_b { n_b += 1; }
    //
    //         cond_ab = (0..self.stab.n)
    //             .filter(|&j| !part_a.contains(j) && !part_b.contains(j))
    //             .all(|j| {
    //                 j5 = j >> 5;
    //                 pw = PW[j & 31];
    //                 xi[j5] & pw == 0 && zi[j5] & pw == 0
    //             });
    //         if cond_ab { n_ab += 1; }
    //     }
    //     n_ab as f32 - n_a as f32 - n_b as f32
    // }

    /// Construct a density matrix representation of `self`.
    pub fn as_matrix(&self) -> na::DMatrix<C64> {
        let size = 2_usize.pow(self.stab.n as u32);
        let id: na::DMatrix<C64> = na::DMatrix::identity(size, size);
        let mut acc: na::DMatrix<C64> = id.clone();
        let mut phase: C64;
        let mut row_acc: na::DMatrix<C64>;
        let mut j5: usize;
        let mut pw: u32;
        let iter =
            self.stab.x.row_iter()
            .zip(self.stab.z.row_iter())
            .zip(self.stab.r.iter())
            .skip(self.stab.n)
            .take(self.m);
        for ((xi, zi), ri) in iter {
            phase =
                match ri % 4 {
                    0 => C64::from(1.0),
                    1 => C64::i(),
                    2 => C64::from(-1.0),
                    3 => -C64::i(),
                    _ => unreachable!(),
                };
            row_acc = na::DMatrix::from_diagonal_element(1, 1, phase);
            for j in 0..self.stab.n {
                j5 = j >> 5;
                pw = PW[j & 31];
                match(xi[j5] & pw, zi[j5] & pw) {
                    (0, 0) =>
                        { row_acc = row_acc.kronecker(Lazy::force(&PAULI_I)); },
                    (_, 0) =>
                        { row_acc = row_acc.kronecker(Lazy::force(&PAULI_X)); },
                    (0, _) =>
                        { row_acc = row_acc.kronecker(Lazy::force(&PAULI_Z)); },
                    (_, _) =>
                        { row_acc = row_acc.kronecker(Lazy::force(&PAULI_Y)); },
                }
            }
            acc *= &id + &row_acc;
        }
        acc /= acc.trace();
        acc
    }

    // count the number of unique, non-identity (de)stabilizer Paulis in a
    // single column
    fn col_nub(&self, k: usize, skip: usize) -> ColNub {
        let k5: usize = k >> 5;
        let pw: u32 = PW[k & 31];
        let mut nub = ColNub::I;
        let mut many: bool;
        for (i, (x_i_k5, z_i_k5)) in
            self.stab.x.column(k5).iter()
                .zip(self.stab.z.column(k5).iter())
                .enumerate()
                .skip(skip)
                .take(self.m)
        {
            match (*x_i_k5 & pw, *z_i_k5 & pw) {
                (0, 0) => { continue; }
                (_, 0) => { many = nub.push_x(i); },
                (0, _) => { many = nub.push_z(i); },
                (_, _) => { many = nub.push_y(i); },
            }
            if many { return nub; }
        }
        nub
    }

    // find the first (de)stabilizer and its row index in a particular column
    // that matches a nub
    fn nub_match(&self, k: usize, skip: usize, nub: ColNub) -> Option<ColNub> {
        let k5: usize = k >> 5;
        let pw: u32 = PW[k & 31];
        self.stab.x.column(k5).iter()
            .zip(self.stab.z.column(k5).iter())
            .enumerate()
            .skip(skip)
            .take(self.m)
            .map(|(i, (x_i_k5, z_i_k5))| (i, (*x_i_k5 & pw, *z_i_k5 & pw)))
            .find(|(_i, (x, z))| nub.contains_xz(*x, *z))
            .map(|(i, (x, z))| ColNub::from_xz(x, z, i))
    }

    // find the first (de)stabilizer and its row index in a particular column
    // that does not match a nub
    fn nub_amatch(&self, k: usize, skip: usize, nub: ColNub) -> Option<ColNub> {
        let k5: usize = k >> 5;
        let pw: u32 = PW[k & 31];
        self.stab.x.column(k5).iter()
            .zip(self.stab.z.column(k5).iter())
            .enumerate()
            .skip(skip)
            .take(self.m)
            .map(|(i, (x_i_k5, z_i_k5))| (i, (*x_i_k5 & pw, *z_i_k5 & pw)))
            .find(|(_i, (x, z))| !nub.contains_xz(*x, *z))
            .map(|(i, (x, z))| ColNub::from_xz(x, z, i))
    }

    // perform row operations to bring the stabilizer array into a roughly
    // triangular shape
    //
    // this is like a RREF, with the following exceptions:
    // * row leaders will, in general, not be the only non-identity elements in
    //   their columns
    // * column indexing is reversed, so the end result will be an
    //   upper-triangular stabilizer array where the lower-*right* half is
    //   identities -- this is done to make column removal in the partial trace
    //   more efficient
    pub fn rref_rev(&mut self) {
        let n = self.stab.n;
        let mut col_min: usize = 1;
        let mut row_min: usize = n;
        let mut col_nub: ColNub;
        while col_min <= n && row_min < n + self.m {
            col_nub = self.col_nub(n - col_min, row_min);
            match col_nub {
                ColNub::I => {
                    col_min += 1;
                    continue;
                },
                ColNub::X(k) | ColNub::Y(k) | ColNub::Z(k) => {
                    if k != row_min {
                        self.stab.row_swap(k, row_min);
                        self.stab.row_swap(k - n, row_min - n);
                    }
                    let mut k_scan = row_min + 1;
                    while let Some(k_elim) =
                        self.nub_match(n - col_min, k_scan, col_nub)
                        .and_then(|nub| nub.idx())
                    {
                        self.stab.row_mul(row_min, k_elim);
                        self.stab.row_mul(row_min - n, k_elim - n);
                        k_scan += 1;
                    }
                    col_min += 1;
                    row_min += 1;
                },
                ColNub::XY(k0, k1) | ColNub::YX(k0, k1)
                | ColNub::XZ(k0, k1) | ColNub::ZX(k0, k1)
                | ColNub::YZ(k0, k1) | ColNub::ZY(k0, k1)
                => {
                    if k0 != row_min {
                        self.stab.row_swap(k0, row_min);
                        self.stab.row_swap(k0 - n, row_min - n);
                    }
                    if k1 != row_min + 1 {
                        self.stab.row_swap(k1, row_min + 1);
                        self.stab.row_swap(k1 - n, row_min + 1 - n);
                    }
                    let mut k_scan = row_min + 2;
                    while let Some(nub_elim) =
                        self.nub_amatch(n - col_min, k_scan, ColNub::I)
                    {
                        let k_elim = nub_elim.idx().unwrap();
                        match col_nub.matches(&nub_elim) {
                            Which::Left => {
                                self.stab.row_mul(row_min, k_elim);
                                self.stab.row_mul(row_min - n, k_elim - n);
                            },
                            Which::Right => {
                                self.stab.row_mul(row_min + 1, k_elim);
                                self.stab.row_mul(row_min + 1 - n, k_elim - n);
                            },
                            Which::None => {
                                self.stab.row_mul(row_min, k_elim);
                                self.stab.row_mul(row_min + 1, k_elim);
                                self.stab.row_mul(row_min - n, k_elim - n);
                                self.stab.row_mul(row_min + 1 - n, k_elim - n);
                            },
                            Which::Both => unreachable!(),
                        }
                        k_scan += 1;
                    }
                    col_min += 1;
                    row_min += 2;
                },
            }
        }
    }

    // compute the appropriate logical operators to fill in a tableau with fewer
    // generators than qubits
    //
    // logical operates are named X'[i] and Z'[i] (for i ∊ 0 .. n - m) and fit
    // into the remaining n - r rows of the destabilizer and stabilizer blocks,
    // respectively
    //
    // they are computed to satisfy the (anti)commutation relations with
    // stabilizers M[k] and destabilizers M'[k]:
    // * [ M[k], X'[j] ]
    //     = [ M[k], Z'[j] ]
    //     = [ M'[k], X'[j] ]
    //     = [ M'[k], Z'[j] ] = 0  ∀ j, k
    // * [ X'[i], X'[j] ]
    //     = [ Z'[i], Z'[j] ]
    //     = [ X'[i], Z'[j] ] = 0  ∀ i ≠ j
    // * { X'[i], Z'[i] } = 0  ∀ i
    //
    // note that for two Pauli operators
    //   P0 ≡ (-i)^{z0·x0} Z^{z0} X^{x0}
    //   P1 ≡ (-i)^{z1·x1} Z^{z1} X^{x1}
    // where z0, x0, z1, x1 are taken to be binary vectors (and exponentiation
    // by such vectors is taken to be a tensor product), we have
    // * [ P0, P1 ] = 0 iff z0·x1 + x0·z1 ≡ 0 mod 2
    // * { P0, P1 } = 0 iff z0·x1 + x0·z1 ≡ 1 mod 2
    //
    // therefore the logical operators X'[i] and Z'[i] can be computed by
    // solving a system of linear binary equations with lhs generated from the
    // bits already present in `self` and rhs determined by whether we want
    // commutation or anticommutation with a particular operator
    //
    // here we start with a system containing only constraints generated from
    // (de)stabilizer generators, and iteratively add in extra rows with
    // computed logical operators -- each new operator is taken as the first
    // non-identity solution to the system
    fn compute_logical(&mut self) {
        let n0 = self.stab.n;
        let mut sys = XZSystem::new(n0);
        // load all (de)stabilizers in the system with rhs == 0
        let mut lhs_x: Vec<u32>;
        let mut lhs_z: Vec<u32>;
        let destab_iter =
            self.stab.x.row_iter().zip(self.stab.z.row_iter())
            .take(self.m);
        for (xrow, zrow) in destab_iter {
            lhs_x = xrow.iter().copied().collect();
            lhs_z = zrow.iter().copied().collect();
            sys.push_row(lhs_x, lhs_z, 0);
        }
        let stab_iter =
            self.stab.x.row_iter().zip(self.stab.z.row_iter())
            .skip(n0)
            .take(self.m);
        for (xrow, zrow) in stab_iter {
            lhs_x = xrow.iter().copied().collect();
            lhs_z = zrow.iter().copied().collect();
            sys.push_row(lhs_x, lhs_z, 0);
        }
        // compute logical operators in X/Z pairs until all rows of the tableau
        // are filled
        //
        // assume that it is always possible to find a non-identity operator
        // that satisfies all (anti)commutation constraints
        let mut solution: XZSolution;
        for i in self.m..n0 {
            // TODO: make sure computed logical operators are unique (actually,
            // is this always possible?)

            // compute Z'[i - self.m]
            solution = sys.solve().find(|sol| !sol.is_identity()).unwrap();
            self.stab.x.row_mut(n0 + i).copy_from_slice(&solution.x);
            self.stab.z.row_mut(n0 + i).copy_from_slice(&solution.z);
            sys.push_row(solution.x, solution.z, 1);

            // compute X'[i - self.m]
            solution = sys.solve().find(|sol| !sol.is_identity()).unwrap();
            self.stab.x.row_mut(i).copy_from_slice(&solution.x);
            self.stab.z.row_mut(i).copy_from_slice(&solution.z);
            sys.push_row(solution.x, solution.z, 0);
            sys.set_rhs(2 * i, 0);
        }
    }

    // copy data in self except for a number of leading rows and trailing
    // columns into a new `StabM`
    fn remove_upper_right(&self, skiprows: usize, skipcols: usize) -> Self {
        let n = self.stab.n.saturating_sub(skipcols);
        let m = self.m.saturating_sub(skiprows);
        let over32: usize = (n >> 5) + 1;
        let mut x: na::DMatrix<u32> = na::DMatrix::zeros(2 * n + 1, over32);
        let mut z: na::DMatrix<u32> = na::DMatrix::zeros(2 * n + 1, over32);
        let mut r: na::DVector<u8> = na::DVector::zeros(2 * n + 1);

        // copy x/z data
        // most efficient to copy over the entire relevant portion of the
        // original arrays and then zero out extraneous bit positions
        let n0 = self.stab.n;
        x.view_mut((0, 0), (m, over32))
            .copy_from(&self.stab.x.view((skiprows, 0), (m, over32)));
        x.view_mut((n, 0), (m, over32))
            .copy_from(&self.stab.x.view((n0 + skiprows, 0), (m, over32)));
        z.view_mut((0, 0), (m, over32))
            .copy_from(&self.stab.z.view((skiprows, 0), (m, over32)));
        z.view_mut((n, 0), (m, over32))
            .copy_from(&self.stab.z.view((n0 + skiprows, 0), (m, over32)));
        let extcols = (skipcols % 32) as u32;
        x.column_mut(over32 - 1).iter_mut()
            .for_each(|col| { *col = col.wrapping_shl(extcols) >> extcols; });
        z.column_mut(over32 - 1).iter_mut()
            .for_each(|col| { *col = col.wrapping_shl(extcols) >> extcols; });
        // copy r data
        r.view_mut((0, 0), (m, 1))
            .copy_from(&self.stab.r.view((skiprows, 0), (m, 1)));
        r.view_mut((n, 0), (m, 1))
            .copy_from(&self.stab.r.view((n0 + skiprows, 0), (m, 1)));

        Self { stab: Stab { n, x, z, r }, m }
    }

    // for use when filling out the logical operators of the completely mixed
    // state (self.m == 0)
    //
    // sets the logical operators X'[i] and Z'[i] to X or Z on the i-th qubit
    // and identities everywhere else
    fn compute_logical_trivial(&mut self) {
        assert_eq!(self.m, 0);
        let n = self.stab.n;
        let mut j: usize;
        let iter =
            self.stab.x.row_iter_mut().take(n)
            .chain(self.stab.z.row_iter_mut().skip(n).take(n))
            .enumerate();
        for (i, mut row) in iter {
            if i < self.stab.n {
                row[i >> 5] = PW[i & 31];
            } else {
                j = i - self.stab.n;
                row[j >> 5] = PW[j & 31];
            }
        }
    }

    /// Trace out a single qubit, yielding a new `StabM` with the qubit removed.
    pub fn partial_trace(mut self, k: usize) -> Self {
        // move the qubit to be traced out into the rightmost column
        let tr_col = self.stab.n - 1;
        (k..tr_col).for_each(|j| { self.apply_swap(j, j + 1); });
        // bring to reversed rref -- the rows with column leaders in the
        // rightmost column will be removed
        self.rref_rev();
        let tr5 = tr_col >> 5;
        let pw = PW[tr_col & 31];
        let skiprows =
            self.stab.x.column(tr5).iter()
            .zip(self.stab.z.column(tr5).iter())
            .skip(self.stab.n).take(self.m)
            .take_while(|(xi, zi)| *xi & pw != 0 || *zi & pw != 0)
            .count();
        let mut new = self.remove_upper_right(skiprows, 1);
        if new.m == 0 {
            new.compute_logical_trivial();
        } else {
            new.compute_logical();
        }
        new
    }

    /// Like [`partial_trace`][Self::partial_trace], but for multiple qubits.
    pub fn partial_trace_multi<I>(mut self, trace: I) -> Self
    where I: IntoIterator<Item = usize>
    {
        // move the qubits to trace out into the rightmost positions, ignoring
        // duplicats in the input
        let mut to_trace: Vec<usize> = Vec::new();
        let mut tr_col: usize = self.stab.n - 1;
        for k in trace {
            if (0..self.stab.n).contains(&k) && !to_trace.contains(&k) {
                to_trace.push(k);
                (k..tr_col).for_each(|j| { self.apply_swap(j, j + 1); });
                tr_col -= 1;
            }
        }
        let skipcols: usize = to_trace.len();
        if skipcols == 0 { return self; }
        // bring to reversed rref -- rows with column leaders in the columns to
        // be traced out will be removed
        self.rref_rev();
        let mut skiprows: usize = 0;
        let mut j5: usize;
        let mut pw: u32;
        let mut max_leader: usize;
        for j in (self.stab.n - skipcols ..= self.stab.n - 1).rev() {
            j5 = j >> 5;
            pw = PW[j & 31];
            max_leader =
                self.stab.x.column(j5).iter()
                .zip(self.stab.z.column(j5).iter())
                .skip(self.stab.n).take(self.m)
                .enumerate()
                .rev()
                .find(|(_, (xi, zi))| *xi & pw != 0 || *zi & pw != 0)
                .map(|(i, _)| i + 1)
                .unwrap_or(0);
            skiprows = skiprows.max(max_leader);
        }
        let mut new = self.remove_upper_right(skiprows, skipcols);
        if new.m == 0 {
            new.compute_logical_trivial();
        } else {
            new.compute_logical();
        }
        new
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Which { None, Left, Right, Both }

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ColNub {
    I,
    X(usize),
    Y(usize),
    Z(usize),
    XY(usize, usize),
    YX(usize, usize),
    XZ(usize, usize),
    ZX(usize, usize),
    YZ(usize, usize),
    ZY(usize, usize),
}

impl ColNub {
    fn from_xz(x: u32, z: u32, k: usize) -> Self {
        match (x, z) {
            (0, 0) => Self::I,
            (_, 0) => Self::X(k),
            (0, _) => Self::Z(k),
            (_, _) => Self::Y(k),
        }
    }

    fn contains_xz(&self, x: u32, z: u32) -> bool {
        match self {
            Self::I => x == 0 && z == 0,
            Self::X(_) => x != 0 && z == 0,
            Self::Y(_) => x != 0 && z != 0,
            Self::Z(_) => x == 0 && z != 0,
            Self::XY(..) | Self::YX(..) => x != 0,
            Self::XZ(..) | Self::ZX(..) => (x != 0) ^ (z != 0),
            Self::YZ(..) | Self::ZY(..) => z != 0,
        }
    }

    fn matches(&self, other: &Self) -> Which {
        match (self, other) {
            (Self::I, Self::I) => Which::Left,
            (Self::X(_), Self::X(_)) => Which::Left,
            (Self::Y(_), Self::Y(_)) => Which::Left,
            (Self::Z(_), Self::Z(_)) => Which::Left,
            (Self::XY(..), Self::X(_)) | (Self::X(_), Self::XY(..)) => Which::Left,
            (Self::XY(..), Self::Y(_)) | (Self::Y(_), Self::XY(..)) => Which::Right,
            (Self::YX(..), Self::X(_)) | (Self::X(_), Self::YX(..)) => Which::Right,
            (Self::YX(..), Self::Y(_)) | (Self::Y(_), Self::YX(..)) => Which::Left,
            (Self::XY(..), Self::XY(..)) | (Self::YX(..), Self::YX(..)) => Which::Both,
            (Self::XZ(..), Self::X(_)) | (Self::X(_), Self::XZ(..)) => Which::Left,
            (Self::XZ(..), Self::Z(_)) | (Self::Z(_), Self::XZ(..)) => Which::Right,
            (Self::ZX(..), Self::X(_)) | (Self::X(_), Self::ZX(..)) => Which::Right,
            (Self::ZX(..), Self::Z(_)) | (Self::Z(_), Self::ZX(..)) => Which::Left,
            (Self::XZ(..), Self::XZ(..)) | (Self::ZX(..), Self::ZX(..)) => Which::Both,
            (Self::YZ(..), Self::Y(_)) | (Self::Y(_), Self::YZ(..)) => Which::Left,
            (Self::YZ(..), Self::Z(_)) | (Self::Z(_), Self::YZ(..)) => Which::Right,
            (Self::ZY(..), Self::Y(_)) | (Self::Y(_), Self::ZY(..)) => Which::Right,
            (Self::ZY(..), Self::Z(_)) | (Self::Z(_), Self::ZY(..)) => Which::Left,
            (Self::YZ(..), Self::YZ(..)) | (Self::ZY(..), Self::ZY(..)) => Which::Both,
            _ => Which::None,
        }
    }

    // return true if `self` has two or more elements after the push
    fn push_x(&mut self, k: usize) -> bool {
        match self {
            Self::I => { *self = Self::X(k); false },
            Self::X(_) => { false },
            Self::Y(k0) => { *self = Self::YX(*k0, k); true },
            Self::Z(k0) => { *self = Self::ZX(*k0, k); true },
            _ => { true },
        }
    }

    fn push_y(&mut self, k: usize) -> bool {
        match self {
            Self::I => { *self = Self::Y(k); false },
            Self::X(k0) => { *self = Self::XY(*k0, k); true },
            Self::Y(_) => { false },
            Self::Z(k0) => { *self = Self::ZY(*k0, k); true },
            _ => { true },
        }
    }

    fn push_z(&mut self, k: usize) -> bool {
        match self {
            Self::I => { *self = Self::Z(k); false },
            Self::X(k0) => { *self = Self::XZ(*k0, k); true },
            Self::Y(k0) => { *self = Self::YZ(*k0, k); true },
            Self::Z(_) => { false },
            _ => { true },
        }
    }

    fn idx(&self) -> Option<usize> {
        match self {
            Self::X(k) | Self::Y(k) | Self::Z(k) => Some(*k),
            _ => None,
        }
    }

    fn idx2(&self) -> Option<(usize, usize)> {
        match self {
            Self::XY(k0, k1) | Self::YX(k0, k1)
            | Self::XZ(k0, k1) | Self::ZX(k0, k1)
            | Self::YZ(k0, k1) | Self::ZY(k0, k1) =>
                Some((*k0, *k1)),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum BColNub {
    Empty,
    One(usize),
    Many,
}

impl BColNub {
    fn inc(&mut self, idx: usize) -> bool {
        match self {
            Self::Empty => { *self = Self::One(idx); false },
            Self::One(_) => { *self = Self::Many; true },
            Self::Many => { true },
        }
    }
}

/// A single-qubit identity matrix.
pub static PAULI_I: Lazy<na::DMatrix<C64>> =
    Lazy::new(|| na::DMatrix::identity(2, 2));

/// A single-qubit Pauli *X* matrix.
pub static PAULI_X: Lazy<na::DMatrix<C64>> =
    Lazy::new(|| {
        let mut x = na::DMatrix::zeros(2, 2);
        x[(0, 1)] = C64::from(1.0);
        x[(1, 0)] = C64::from(1.0);
        x
    });

/// A single-qubit Pauli *Y* matrix.
pub static PAULI_Y: Lazy<na::DMatrix<C64>> =
    Lazy::new(|| {
        let mut y = na::DMatrix::zeros(2, 2);
        y[(0, 1)] = -C64::i();
        y[(1, 0)] =  C64::i();
        y
    });

/// A single-qubit Pauli *Z* matrix.
pub static PAULI_Z: Lazy<na::DMatrix<C64>> =
    Lazy::new(|| {
        let mut z = na::DMatrix::zeros(2, 2);
        z[(0, 0)] = C64::from( 1.0);
        z[(1, 1)] = C64::from(-1.0);
        z
    });

#[derive(Clone, Debug, PartialEq, Eq)]
struct XZSystem {
    nxz: usize, // number of x/z binary variables
    nxz_over32: usize, // floor(nxz / 32) + 1
    lhs_x: Vec<Vec<u32>>, // bit-packed bools;
    lhs_z: Vec<Vec<u32>>, // vectors to make appending easier
    rhs: Vec<u8>, // use u8's as bools for space efficiency
}
    
impl XZSystem {
    fn new(nxz: usize) -> Self {
        let nxz_over32 = (nxz >> 5) + 1;
        let lhs_x: Vec<Vec<u32>> = Vec::new();
        let lhs_z: Vec<Vec<u32>> = Vec::new();
        let rhs: Vec<u8> = Vec::new();
        Self { nxz, nxz_over32, lhs_x, lhs_z, rhs }
    }

    fn num_rows(&self) -> usize { self.lhs_x.len() }

    fn set_rhs(&mut self, row_idx: usize, rhs: u8) {
        if let Some(rhs_val) = self.rhs.get_mut(row_idx) {
            *rhs_val = rhs % 2;
        }
    }

    fn push_row(&mut self, lhs_x: Vec<u32>, lhs_z: Vec<u32>, rhs: u8) {
        assert_eq!(lhs_x.len(), self.nxz_over32);
        assert_eq!(lhs_z.len(), self.nxz_over32);
        self.lhs_x.push(lhs_x);
        self.lhs_z.push(lhs_z);
        self.rhs.push(rhs & 1);
    }

    // return an iterator over the unpacked bits (as bools) of a row
    fn row_iter(&self, row_idx: usize)
        -> Option<impl Iterator<Item = bool> + '_>
    {
        self.lhs_x.get(row_idx).zip(self.lhs_z.get(row_idx))
            .map(|(row_x, row_z)| {
                let iter_x =
                    (0..self.nxz)
                    .map(move |j| {
                        let j5 = j >> 5;
                        let pw = PW[j & 31];
                        row_x[j5] & pw != 0
                    });
                let iter_z =
                    (0..self.nxz)
                    .map(move |j| {
                        let j5 = j >> 5;
                        let pw = PW[j & 31];
                        row_z[j5] & pw != 0
                    });
                iter_x.chain(iter_z)
            })
    }

    // return the column index of the leader in a row
    fn row_leader(&self, row_idx: usize) -> Option<usize> {
        self.row_iter(row_idx)
            .and_then(|row_iter| {
                row_iter.enumerate().find_map(|(j, b)| b.then_some(j))
            })
    }

    // return the row index of the first non-zero entry in a column
    fn find_first_col(&self, mut col_idx: usize, skiprows: usize)
        -> Option<usize>
    {
        if col_idx < self.nxz {
            let c5: usize = col_idx >> 5;
            let pw: u32 = PW[col_idx & 31];
            self.lhs_x.iter().enumerate()
                .skip(skiprows)
                .find_map(|(i, row)| (row[c5] & pw != 0).then_some(i))
        } else if col_idx < 2 * self.nxz {
            col_idx -= self.nxz;
            let c5: usize = col_idx >> 5;
            let pw: u32 = PW[col_idx & 31];
            self.lhs_z.iter().enumerate()
                .skip(skiprows)
                .find_map(|(i, row)| (row[c5] & pw != 0).then_some(i))
        } else {
            None
        }
    }

    // return the number of nonzero entries in a column
    fn count_col(&self, mut col_idx: usize) -> Option<BColNub> {
        if col_idx < self.nxz {
            let c5: usize = col_idx >> 5;
            let pw: u32 = PW[col_idx & 31];
            let mut nub = BColNub::Empty;
            for (i, row) in self.lhs_x.iter().enumerate() {
                if row[c5] & pw != 0 && nub.inc(i) { break; }
            }
            Some(nub)
        } else if col_idx < 2 * self.nxz {
            col_idx -= self.nxz;
            let c5: usize = col_idx >> 5;
            let pw: u32 = PW[col_idx & 31];
            let mut nub = BColNub::Empty;
            for (i, row) in self.lhs_z.iter().enumerate() {
                if row[c5] & pw != 0 && nub.inc(i) { break; }
            }
            Some(nub)
        } else {
            None
        }
    }

    // XOR row `a` with row `b` and store the result in row `b`
    fn row_xor(&mut self, a: usize, b: usize) {
        assert!(a < self.lhs_x.len() && a < self.lhs_z.len());
        assert!(b < self.lhs_x.len() && b < self.lhs_z.len());
        assert_ne!(a, b);
        unsafe {
            let lhs_x_a: &Vec<u32> =
                self.lhs_x.as_ptr().add(a).as_ref().unwrap();
            let lhs_x_b: &mut Vec<u32> =
                self.lhs_x.as_mut_ptr().add(b).as_mut().unwrap();
            lhs_x_a.iter().zip(lhs_x_b.iter_mut())
                .for_each(|(a_j, b_j)| { *b_j ^= *a_j; });
            let lhs_z_a: &Vec<u32> =
                self.lhs_z.as_ptr().add(a).as_ref().unwrap();
            let lhs_z_b: &mut Vec<u32> =
                self.lhs_z.as_mut_ptr().add(b).as_mut().unwrap();
            lhs_z_a.iter().zip(lhs_z_b.iter_mut())
                .for_each(|(a_j, b_j)| { *b_j ^= *a_j; });
        }
        self.rhs[b] ^= self.rhs[a];
    }

    // swap row `a` with row `b`
    fn row_swap(&mut self, a: usize, b: usize) {
        if a == b { return; }
        assert!(a < self.lhs_x.len() && a < self.lhs_z.len());
        assert!(b < self.lhs_x.len() && b < self.lhs_z.len());
        self.lhs_x.swap(a, b);
        self.lhs_z.swap(a, b);
        self.rhs.swap(a, b);
    }

    fn rref(&mut self) {
        assert_eq!(self.lhs_x.len(), self.rhs.len());
        assert_eq!(self.lhs_z.len(), self.rhs.len());
        let n = 2 * self.nxz;
        let m = self.lhs_x.len();
        let mut col_min: usize = 0;
        let mut row_min: usize = 0;
        let mut c5: usize;
        let mut pw: u32;
        while col_min < n && row_min < m {
            if let Some(k) = self.find_first_col(col_min, row_min) {
                if k != row_min { self.row_swap(k, row_min); }
                if col_min < self.nxz {
                    c5 = col_min >> 5;
                    pw = PW[col_min & 31];
                    for i in 0..m {
                        if i == row_min { continue; }
                        if self.lhs_x[i][c5] & pw != 0 {
                            self.row_xor(row_min, i);
                        }
                    }
                } else {
                    c5 = (col_min - self.nxz) >> 5;
                    pw = PW[(col_min - self.nxz) & 31];
                    for i in 0..m {
                        if i == row_min { continue; }
                        if self.lhs_z[i][c5] & pw != 0 {
                            self.row_xor(row_min, i);
                        }
                    }
                }
                col_min += 1;
                row_min += 1;
            } else {
                col_min += 1;
            }
        }
    }

    fn solve(&mut self) -> XZSolutionIter {
        self.rref();
        let mut free_idx: IndexMap<usize> = IndexMap::new();
        let mut free_counter: usize = 0;
        let mut exprs: Vec<BExpr> = Vec::with_capacity(2 * self.nxz);
        for var_idx in 0 .. 2 * self.nxz {
            match self.count_col(var_idx).unwrap() {
                BColNub::Empty => {
                    exprs.push(BExpr { offs: 0, deps: Vec::with_capacity(0) });
                },
                BColNub::One(row_idx) => {
                    let rlead = self.row_leader(row_idx).unwrap();
                    let offs: u32 = self.rhs[row_idx].into();
                    let mut deps: Vec<usize> = Vec::new();
                    let iter =
                        self.row_iter(row_idx).unwrap()
                        .enumerate()
                        .skip(rlead + 1);
                    for (j, b) in iter {
                        if b {
                            if let Some(free) = free_idx.get(j) {
                                deps.push(*free);
                            } else {
                                deps.push(free_counter);
                                free_idx.insert(j, free_counter);
                                free_counter += 1;
                            }
                        }
                    }
                    exprs.push(BExpr { offs, deps });
                },
                BColNub::Many => {
                    if let Some(free) = free_idx.get(var_idx) {
                        exprs.push(BExpr { offs: 0, deps: vec![*free] });
                    } else {
                        exprs.push(BExpr { offs: 0, deps: vec![free_counter] });
                        free_idx.insert(var_idx, free_counter);
                        free_counter += 1;
                    }
                },
            }
        }
        XZSolutionIter::new(exprs, free_counter)
    }

    fn pretty_print(&self) {
        let iter =
            self.lhs_x.iter().zip(self.lhs_z.iter()).zip(self.rhs.iter());
        let mut j5: usize;
        let mut pw: u32;
        for ((lhs_x_i, lhs_z_i), rhs_i) in iter {
            print!("[ ");
            for j in 0..self.nxz {
                j5 = j >> 5;
                pw = PW[j & 31];
                if lhs_x_i[j5] & pw != 0 {
                    print!("1 ");
                } else {
                    print!("0 ");
                }
            }
            print!(": ");
            for j in 0..self.nxz {
                j5 = j >> 5;
                pw = PW[j & 31];
                if lhs_z_i[j5] & pw != 0 {
                    print!("1 ");
                } else {
                    print!("0 ");
                }
            }
            if *rhs_i != 0 { print!("| 1 ]"); } else { print!("| 0 ]"); }
            println!();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct XZSolution {
    x: Vec<u32>,
    z: Vec<u32>,
}

impl XZSolution {
    fn is_identity(&self) -> bool {
        self.x.iter().all(|b| *b == 0)
            && self.z.iter().all(|b| *b == 0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct XZSolutionIter {
    exprs: Vec<BExpr>,
    nxz: usize,
    nxz_over32: usize,
    free: Vec<u32>,
    len: usize,
}

impl XZSolutionIter {
    fn new(exprs: Vec<BExpr>, num_free: usize) -> Self {
        assert!(exprs.len() % 2 == 0);
        let nxz = exprs.len() / 2;
        let nxz_over32 = (nxz >> 5) + 1;
        let free: Vec<u32> = vec![0; (num_free >> 5) + 1];
        let len = 1_usize << num_free as u32;
        Self { exprs, nxz, nxz_over32, free, len }
    }

    fn advance(&mut self) {
        let mut carry: bool = true;
        for b in self.free.iter_mut() {
            if !carry { break; }
            let (b_new, c) = b.overflowing_add(1);
            *b = b_new;
            carry = c;
        }
        self.len -= 1;
    }
}

impl Iterator for XZSolutionIter {
    type Item = XZSolution;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len > 0 {
            let mut z: Vec<u32> = vec![0; self.nxz_over32];
            let mut x: Vec<u32> = vec![0; self.nxz_over32];
            let mut k5: usize;
            let mut k_: usize;
            // note that columns in lhs_x give z solution variables and
            // vice-versa
            for (k, expr) in self.exprs.iter().take(self.nxz).enumerate() {
                k5 = k >> 5;
                k_ = k & 31;
                z[k5] |= expr.eval(&self.free).wrapping_shl(k_ as u32);
            }
            for (k, expr) in self.exprs.iter().skip(self.nxz).enumerate() {
                k5 = k >> 5;
                k_ = k & 31;
                x[k5] |= expr.eval(&self.free).wrapping_shl(k_ as u32);
            }
            self.advance();
            Some(XZSolution { x, z })
        } else {
            None
        }
    }
}

impl ExactSizeIterator for XZSolutionIter {
    fn len(&self) -> usize { self.len }
}

impl std::iter::FusedIterator for XZSolutionIter { }

#[derive(Clone, Debug, PartialEq, Eq)]
struct BExpr {
    offs: u32,
    deps: Vec<usize>,
}

impl BExpr {
    fn eval(&self, free_vals: &[u32]) -> u32 {
        self.deps.iter().copied()
            .map(|k| {
                let k5 = k >> 5;
                let k_ = k & 31;
                let pw = PW[k_];
                free_vals.get(k5).copied()
                    .map(|b| (b & pw).wrapping_shr(k_ as u32))
                    .unwrap_or(0)
            })
            .fold(self.offs, |acc, b| acc ^ b)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn xzsystem_rref_0() {
        let mut sys = XZSystem::new(3);
        sys.push_row(vec![0b011_u32], vec![0b000_u32], 0);
        sys.push_row(vec![0b110_u32], vec![0b000_u32], 0);
        sys.push_row(vec![0b000_u32], vec![0b011_u32], 0);
        sys.push_row(vec![0b000_u32], vec![0b110_u32], 0);

        let mut sol = XZSystem::new(3);
        sol.push_row(vec![0b101_u32], vec![0b000_u32], 0);
        sol.push_row(vec![0b110_u32], vec![0b000_u32], 0);
        sol.push_row(vec![0b000_u32], vec![0b101_u32], 0);
        sol.push_row(vec![0b000_u32], vec![0b110_u32], 0);

        println!("init");
        sys.pretty_print();
        println!("rref");
        sys.rref();
        sys.pretty_print();
        println!("expected");
        sol.pretty_print();
        assert_eq!(sys, sol);
    }

    #[test]
    fn xzsystem_rref_1() {
        let mut sys = XZSystem::new(3);
        sys.push_row(vec![0b011_u32], vec![0b000_u32], 0);
        sys.push_row(vec![0b110_u32], vec![0b000_u32], 0);
        sys.push_row(vec![0b111_u32], vec![0b000_u32], 1);
        sys.push_row(vec![0b000_u32], vec![0b011_u32], 0);
        sys.push_row(vec![0b000_u32], vec![0b110_u32], 0);

        let mut sol = XZSystem::new(3);
        sol.push_row(vec![0b001_u32], vec![0b000_u32], 1);
        sol.push_row(vec![0b010_u32], vec![0b000_u32], 1);
        sol.push_row(vec![0b100_u32], vec![0b000_u32], 1);
        sol.push_row(vec![0b000_u32], vec![0b101_u32], 0);
        sol.push_row(vec![0b000_u32], vec![0b110_u32], 0);

        println!("init");
        sys.pretty_print();
        println!("rref");
        sys.rref();
        sys.pretty_print();
        println!("expected");
        sol.pretty_print();
        assert_eq!(sys, sol);
    }

    #[test]
    fn xzsystem_rref_2() {
        let mut sys = XZSystem::new(3);
        sys.push_row(vec![0b011_u32], vec![0b000_u32], 0);
        sys.push_row(vec![0b110_u32], vec![0b000_u32], 0);
        sys.push_row(vec![0b000_u32], vec![0b111_u32], 1);
        sys.push_row(vec![0b000_u32], vec![0b011_u32], 0);
        sys.push_row(vec![0b000_u32], vec![0b110_u32], 0);

        let mut sol = XZSystem::new(3);
        sol.push_row(vec![0b101_u32], vec![0b000_u32], 0);
        sol.push_row(vec![0b110_u32], vec![0b000_u32], 0);
        sol.push_row(vec![0b000_u32], vec![0b001_u32], 1);
        sol.push_row(vec![0b000_u32], vec![0b010_u32], 1);
        sol.push_row(vec![0b000_u32], vec![0b100_u32], 1);

        println!("init");
        sys.pretty_print();
        println!("rref");
        sys.rref();
        sys.pretty_print();
        println!("expected");
        sol.pretty_print();
        assert_eq!(sys, sol);
    }

    #[test]
    fn xzsystem_rref_3() {
        let mut sys = XZSystem::new(3);
        sys.push_row(vec![0b011_u32], vec![0b000_u32], 0);
        sys.push_row(vec![0b110_u32], vec![0b000_u32], 0);
        sys.push_row(vec![0b111_u32], vec![0b111_u32], 1);
        sys.push_row(vec![0b000_u32], vec![0b011_u32], 0);
        sys.push_row(vec![0b000_u32], vec![0b110_u32], 0);

        let mut sol = XZSystem::new(3);
        sol.push_row(vec![0b001_u32], vec![0b100_u32], 1);
        sol.push_row(vec![0b010_u32], vec![0b100_u32], 1);
        sol.push_row(vec![0b100_u32], vec![0b100_u32], 1);
        sol.push_row(vec![0b000_u32], vec![0b101_u32], 0);
        sol.push_row(vec![0b000_u32], vec![0b110_u32], 0);

        println!("init");
        sys.pretty_print();
        println!("rref");
        sys.rref();
        sys.pretty_print();
        println!("expected");
        sol.pretty_print();
        assert_eq!(sys, sol);
    }

    #[test]
    fn xzsolutioniter() {
        let exprs: Vec<BExpr> =
            vec![
                BExpr {
                    offs: 0,
                    deps: vec![0],
                },
                BExpr {
                    offs: 0,
                    deps: vec![1],
                },
                BExpr {
                    offs: 0,
                    deps: vec![2],
                },
                BExpr {
                    offs: 0,
                    deps: vec![3],
                },
            ];
        let mut iter = XZSolutionIter::new(exprs, 4);
        assert_eq!(iter.next(), Some(XZSolution { x: vec![0], z: vec![0] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![0], z: vec![1] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![0], z: vec![2] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![0], z: vec![3] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![1], z: vec![0] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![1], z: vec![1] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![1], z: vec![2] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![1], z: vec![3] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![2], z: vec![0] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![2], z: vec![1] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![2], z: vec![2] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![2], z: vec![3] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![3], z: vec![0] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![3], z: vec![1] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![3], z: vec![2] }));
        assert_eq!(iter.next(), Some(XZSolution { x: vec![3], z: vec![3] }));
        assert_eq!(iter.next(), None                                       );
    }
}

/// Like [`StabGroup`], but including the "logical operators" that fill out
/// extra rows in a mixed state tableau.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StabMGroup {
    pub destab: Vec<NPauli>,
    pub stab: Vec<NPauli>,
    pub logical: Vec<NPauli>,
}

impl fmt::Display for StabMGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.destab.is_empty() {
            writeln!(f, "Destab")?;
            for destab in self.destab.iter() { destab.fmt(f)?; writeln!(f)?; }
        }
        if !self.stab.is_empty() {
            writeln!(f, "Stab")?;
            for stab in self.stab.iter() { stab.fmt(f)?; writeln!(f)?; }
        }
        if !self.logical.is_empty() {
            writeln!(f, "Logical")?;
            for op in self.logical.iter() { op.fmt(f)?; writeln!(f)?; }
        }
        Ok(())
    }
}

