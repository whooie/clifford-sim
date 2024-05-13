//! Like [`stab`][crate::stab], but not statically sized.

use std::fmt;
use ndarray::{ self as nd, s };
use rand::Rng;
use crate::{
    gate::{ Gate, Pauli, Phase },
    graph::Graph,
    graphd::GraphD,
    stab::{ Outcome, Postsel, Qubit, Stab },
};

const PW: [u32; 32] = [ // PW[i] = 2^i
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216,
    33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648
];

/// A stabilizer state of a finite register of qubits, identified by its
/// stabilizer group.
///
/// Like [`Stab`], but for non-static system sizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StabD {
    pub(crate) n: usize,
    // `x` and `z` are bit arrays of size (2n + 1) × n; for space efficiency,
    // the columns are packed into u32s
    pub(crate) x: nd::Array2<u32>, // Pauli-X bits; size (2n + 1) × (floor(N / 8) + 1)
    pub(crate) z: nd::Array2<u32>, // Pauli-Z bits; size (2n + 1) × (floor(N / 8) + 1)
    pub(crate) r: nd::Array1<u8>, // Phases (0 for +1, 1 for i, 2 for -1, 3 for -i); size 2n + 1
    pub(crate) over32: usize, // = floor(N / 8) + 1
}

impl<const N: usize> From<Stab<N>> for StabD {
    fn from(stab: Stab<N>) -> Self {
        Self { n: N, x: stab.x, z: stab.z, r: stab.r, over32: stab.over32 }
    }
}

impl StabD {
    /// Create a new stabilizer state of size `n` initialized to ∣0...0⟩.
    pub fn new(n: usize) -> Self {
        let over32: usize = (n >> 5) + 1;
        let x: nd::Array2<u32> = nd::Array2::zeros((2 * n + 1, over32));
        let z: nd::Array2<u32> = nd::Array2::zeros((2 * n + 1, over32));
        let r: nd::Array1<u8> = nd::Array1::zeros(2 * n + 1);
        let mut q = Self { n, x, z, r, over32 };
        let mut j: usize;
        for (i, ((mut xi, mut zi), _)) in q.row_iter_mut().enumerate() {
            if i < n {
                xi[i >> 5] = PW[i & 31];
            } else if i < 2 * n {
                j = i - n;
                zi[j >> 5] = PW[j & 31];
            }
        }
        q
    }

    pub(crate) fn row_iter(&self)
        -> impl Iterator<
            Item = ((nd::ArrayView1<u32>, nd::ArrayView1<u32>), nd::ArrayView0<u8>)
        >
    {
        self.x.axis_iter(nd::Axis(0))
            .zip(self.z.axis_iter(nd::Axis(0)))
            .zip(self.r.axis_iter(nd::Axis(0)))
            .take(2 * self.n)
    }

    pub(crate) fn row_iter_mut(&mut self)
        -> impl Iterator<
            Item = ((nd::ArrayViewMut1<u32>, nd::ArrayViewMut1<u32>), nd::ArrayViewMut0<u8>)
        >
    {
        self.x.axis_iter_mut(nd::Axis(0))
            .zip(self.z.axis_iter_mut(nd::Axis(0)))
            .zip(self.r.axis_iter_mut(nd::Axis(0)))
            .take(2 * self.n)
    }

    pub(crate) fn col_iter(&self)
        -> impl Iterator<Item = (nd::ArrayView1<u32>, nd::ArrayView1<u32>)>
    {
        self.x.axis_iter(nd::Axis(1))
            .zip(self.z.axis_iter(nd::Axis(1)))
    }

    pub(crate) fn col_iter_mut(&mut self)
        -> impl Iterator<Item = (nd::ArrayViewMut1<u32>, nd::ArrayViewMut1<u32>)>
    {
        self.x.axis_iter_mut(nd::Axis(1))
            .zip(self.z.axis_iter_mut(nd::Axis(1)))
    }

    pub fn apply_h(&mut self, k: usize) -> &mut Self {
        let k5: usize = k >> 5;
        let pw: u32 = PW[k & 31];
        let mut tmp: u32;
        for ((x_i_k5, z_i_k5), r_i) in
            self.x.slice_mut(s![.., k5]).iter_mut()
                .zip(self.z.slice_mut(s![.., k5]).iter_mut())
                .zip(self.r.iter_mut())
                .take(2 * self.n)
        {
            tmp = *x_i_k5;
            *x_i_k5 ^= (*x_i_k5 ^ *z_i_k5) & pw;
            *z_i_k5 ^= (*z_i_k5 ^ tmp) & pw;
            if *x_i_k5 & pw != 0 && *z_i_k5 & pw != 0 { *r_i = (*r_i + 2) % 4; }
        }
        // for i in 0..2 * self.n {
        //     tmp = self.x[[i, k5]];
        //     self.x[[i, k5]] ^= (self.x[[i, k5]] ^ self.z[[i, k5]]) & pw;
        //     self.z[[i, k5]] ^= (self.z[[i, k5]] ^ tmp) & pw;
        //     if self.x[[i, k5]] & pw != 0 && self.z[[i, k5]] & pw != 0 {
        //         self.r[i] = (self.r[i] + 2) % 4;
        //     }
        // }
        self
    }

    pub fn apply_s(&mut self, k: usize) -> &mut Self {
        let k5: usize = k >> 5;
        let pw: u32 = PW[k & 31];
        for ((x_i_k5, z_i_k5), r_i) in
            self.x.slice_mut(s![.., k5]).iter_mut()
                .zip(self.z.slice_mut(s![.., k5]).iter_mut())
                .zip(self.r.iter_mut())
                .take(2 * self.n)
        {
            if *x_i_k5 & pw != 0 && *z_i_k5 & pw != 0 { *r_i = (*r_i + 2) % 4; }
            *z_i_k5 ^= *x_i_k5 & pw;
        }
        // for i in 0..2 * self.n {
        //     if self.x[[i, k5]] & pw != 0 && self.z[[i, k5]] & pw != 0 {
        //         self.r[i] = (self.r[i] + 2) % 4;
        //     }
        //     self.z[[i, k5]] ^= self.x[[i, k5]] & pw;
        // }
        self
    }

    pub fn apply_x(&mut self, k: usize) -> &mut Self {
        self.apply_h(k).apply_z(k).apply_h(k)
    }

    pub fn apply_y(&mut self, k: usize) -> &mut Self {
        self.apply_s(k).apply_x(k).apply_z(k).apply_s(k)
    }

    pub fn apply_z(&mut self, k: usize) -> &mut Self {
        self.apply_s(k).apply_s(k)
    }

    pub fn apply_cnot(&mut self, a: usize, b: usize) -> &mut Self {
        let a5: usize = a >> 5;
        let b5: usize = b >> 5;
        let pwa: u32 = PW[a & 31];
        let pwb: u32 = PW[b & 31];
        for ((mut x_i, mut z_i), r_i) in
            self.x.axis_iter_mut(nd::Axis(0))
                .zip(self.z.axis_iter_mut(nd::Axis(0)))
                .zip(self.r.iter_mut())
                .take(2 * self.n)
        {
            if x_i[a5] & pwa != 0 { x_i[b5] ^= pwb; }
            if z_i[b5] & pwb != 0 { z_i[a5] ^= pwa; }
            if x_i[a5] & pwa != 0 && z_i[b5] & pwb != 0
                && x_i[b5] & pwb != 0 && z_i[a5] & pwa != 0
            { *r_i = (*r_i + 2) % 4; }
            if x_i[a5] & pwa != 0 && z_i[b5] & pwb != 0
                && x_i[b5] & pwb == 0 && z_i[a5] & pwa == 0
            { *r_i = (*r_i + 2) % 4; }
        }
        // for i in 0..2 * self.n {
        //     if self.x[[i, a5]] & pwa != 0 { self.x[[i, b5]] ^= pwb; }
        //     if self.z[[i, b5]] & pwb != 0 { self.z[[i, a5]] ^= pwa; }
        //     if self.x[[i, a5]] & pwa != 0 && self.z[[i, b5]] & pwb != 0
        //         && self.x[[i, b5]] & pwb != 0 && self.z[[i, a5]] & pwa != 0
        //     { self.r[i] = (self.r[i] + 2) % 4; }
        //     if self.x[[i, a5]] & pwa != 0 && self.z[[i, b5]] & pwb != 0
        //         && self.x[[i, b5]] & pwb == 0 && self.z[[i, a5]] & pwa == 0
        //     { self.r[i] = (self.r[i] + 2) % 4; }
        // }
        self
    }

    pub fn apply_swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.apply_cnot(a, b).apply_cnot(b, a).apply_cnot(a, b)
    }

    /// Perform the action of a gate.
    ///
    /// Does nothing if any qubit indices are out of bounds.
    pub fn apply_gate(&mut self, gate: Gate) -> &mut Self {
        match gate {
            Gate::H(k) if k < self.n => self.apply_h(k),
            Gate::X(k) if k < self.n => self.apply_x(k),
            Gate::Y(k) if k < self.n => self.apply_y(k),
            Gate::Z(k) if k < self.n => self.apply_z(k),
            Gate::S(k) if k < self.n => self.apply_s(k),
            Gate::CX(a, b) if a < self.n && b < self.n => self.apply_cnot(a, b),
            Gate::Swap(a, b) if a < self.n && b < self.n => self.apply_swap(a, b),
            _ => self,
        }
    }

    /// Perform a series of gates.
    pub fn apply_circuit<'a, I>(&mut self, gates: I) -> &mut Self
    where I: IntoIterator<Item = &'a Gate>
    {
        gates.into_iter().copied().for_each(|g| { self.apply_gate(g); });
        self
    }

    pub(crate) fn row_copy(&mut self, a: usize, b: usize) -> &mut Self {
        // set row b equal to row a
        for (mut x__j, mut z__j) in
            self.x.axis_iter_mut(nd::Axis(1))
                .zip(self.z.axis_iter_mut(nd::Axis(1)))
        {
            x__j[b] = x__j[a];
            z__j[b] = z__j[a];
        }
        // for j in 0..self.over32 {
        //     self.x[[b, j]] = self.x[[a, j]];
        //     self.z[[b, j]] = self.z[[a, j]];
        // }
        self.r[b] = self.r[a];
        self
    }

    pub(crate) fn row_swap(&mut self, a: usize, b: usize) -> &mut Self {
        let n = self.n;
        // swap rows a and b
        self.row_copy(b, 2 * n)
            .row_copy(a, b)
            .row_copy(2 * n, a)
    }

    // set row k equal to the o-th observable (X_1, ... X_n, Z_1, ..., Z_n)
    pub(crate) fn row_set(&mut self, o: usize, k: usize) -> &mut Self {
        let o5: usize;
        let o31: usize;
        self.x.slice_mut(nd::s![k, ..]).fill(0);
        self.z.slice_mut(nd::s![k, ..]).fill(0);
        self.r[k] = 0;
        if o < self.n {
            o5 = o >> 5;
            o31 = o & 31;
            self.x[[k, o5]] = PW[o31];
        } else {
            o5 = (o - self.n) >> 5;
            o31 = (o - self.n) & 31;
            self.z[[k, o5]] = PW[o31];
        }
        self
    }

    // return the phase (0, ..., 3) when row b is left-multiplied by row a
    pub(crate) fn row_mul_phase(&self, a: usize, b: usize) -> u8 {
        let mut e: i32 = 0;
        let xa = self.x.slice(nd::s![a, ..]);
        let xb = self.x.slice(nd::s![b, ..]);
        let za = self.z.slice(nd::s![a, ..]);
        let zb = self.z.slice(nd::s![b, ..]);
        for ((&xaj, &xbj), (&zaj, &zbj)) in
            xa.iter().zip(xb).zip(za.iter().zip(zb))
        {
            for &pw in PW.iter() {
                if xaj & pw != 0 && zaj & pw == 0 {
                    if xbj & pw != 0 && zbj & pw != 0 { e += 1; }
                    if xbj & pw == 0 && zbj & pw != 0 { e -= 1; }
                }
                if xaj & pw != 0 && zaj & pw != 0 {
                    if xbj & pw == 0 && zbj & pw != 0 { e += 1; }
                    if xbj & pw != 0 && zbj * pw == 0 { e -= 1; }
                }
                if xaj & pw == 0 && zaj & pw != 0 {
                    if xbj & pw != 0 && zbj & pw == 0 { e += 1; }
                    if xbj & pw != 0 && zbj & pw != 0 { e -= 1; }
                }
            }
        }
        // for j in 0..self.over32 {
        //     for &pw in PW.iter() {
        //         if self.x[[a, j]] & pw != 0 && self.z[[a, j]] & pw == 0 {
        //             if self.x[[b, j]] & pw != 0 && self.z[[b, j]] & pw != 0 {
        //                 e += 1;
        //             }
        //             if self.x[[b, j]] & pw == 0 && self.z[[b, j]] & pw != 0 {
        //                 e -= 1;
        //             }
        //         }
        //         if self.x[[a, j]] & pw != 0 && self.z[[a, j]] & pw != 0 {
        //             if self.x[[b, j]] & pw == 0 && self.z[[b, j]] & pw != 0 {
        //                 e += 1;
        //             }
        //             if self.x[[b, j]] & pw != 0 && self.z[[b, j]] & pw == 0 {
        //                 e -= 1;
        //             }
        //         }
        //         if self.x[[a, j]] & pw == 0 && self.z[[a, j]] & pw != 0 {
        //             if self.x[[b, j]] & pw != 0 && self.z[[b, j]] & pw == 0 {
        //                 e += 1;
        //             }
        //             if self.x[[b, j]] & pw != 0 && self.z[[b, j]] & pw != 0 {
        //                 e -= 1;
        //             }
        //         }
        //     }
        // }
        e = (e + i32::from(self.r[b]) + i32::from(self.r[a])).rem_euclid(4);
        e as u8
    }

    // left-multiply row b by row a
    pub(crate) fn row_mul(&mut self, a: usize, b: usize) -> &mut Self {
        self.r[b] = self.row_mul_phase(a, b);
        for (mut x__j, mut z__j) in
            self.x.axis_iter_mut(nd::Axis(1))
                .zip(self.z.axis_iter_mut(nd::Axis(1)))
        {
            x__j[b] ^= x__j[a];
            z__j[b] ^= z__j[a];
        }
        // for j in 0..self.over32 {
        //     self.x[[b, j]] ^= self.x[[a, j]];
        //     self.z[[b, j]] ^= self.z[[a, j]];
        // }
        self
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

        for (q, x_qpN_k5) in
            self.x.slice(nd::s![.., k5]).iter()
                .take(2 * self.n)
                .skip(self.n)
                .enumerate()
        {
            rnd = *x_qpN_k5 & pw != 0;
            if rnd { p = q; break; }
        }

        if rnd {
            self.row_copy(p + self.n, p);
            self.row_set(k + self.n, p + self.n);
            self.r[p + self.n] = 2 * u8::from(rng.gen::<bool>());
            for i in 0..2 * self.n {
                if i != p && self.x[[i, k5]] & pw != 0 { self.row_mul(p, i); }
            }
            if self.r[p + self.n] != 0 {
                Outcome::Rand1
            } else {
                Outcome::Rand0
            }
        } else {
            for (q, x_q_k5) in
                self.x.slice(nd::s![.., k5]).iter()
                    .take(self.n)
                    .enumerate()
            {
                if x_q_k5 & pw != 0 { m = q; break; }
            }
            self.row_copy(m + self.n, 2 * self.n);
            for i in m + 1..self.n {
                if self.x[[i, k5]] & pw != 0 { self.row_mul(i + self.n, 2 * self.n); }
            }
            if self.r[2 * self.n] != 0 {
                Outcome::Det1
            } else {
                Outcome::Det0
            }
        }
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

        for (q, x_qpN_k5) in
            self.x.slice(nd::s![.., k5]).iter()
                .take(2 * self.n)
                .skip(self.n)
                .enumerate()
        {
            rnd = *x_qpN_k5 & pw != 0;
            if rnd { p = q; break; }
        }

        if rnd {
            self.row_copy(p + self.n, p);
            self.row_set(k + self.n, p + self.n);
            self.r[p + self.n] = 2 * postsel as u8;
            for i in 0..2 * self.n {
                if i != p && self.x[[i, k5]] & pw != 0 { self.row_mul(p, i); }
            }
            Ok(self)
        } else {
            for (q, x_q_k5) in
                self.x.slice(nd::s![.., k5]).iter()
                    .take(self.n)
                    .enumerate()
            {
                if x_q_k5 & pw != 0 { m = q; break; }
            }
            self.row_copy(m + self.n, 2 * self.n);
            for i in m + 1..self.n {
                if self.x[[i, k5]] & pw != 0 { self.row_mul(i + self.n, 2 * self.n); }
            }
            match (self.r[2 * self.n] != 0, postsel) {
                (true,  Postsel::One ) => Ok(self),
                (false, Postsel::Zero) => Ok(self),
                _ => Err(self.into()),
            }
        }
    }

    /// Convert `self` to a more human-readable stabilizer/destabilizer group
    /// representation.
    pub fn as_group(&self) -> StabGroupD {
        let mut j5: usize;
        let mut pw: u32;
        let mut npauli: NPauliD
            = NPauliD { phase: Phase::Pi0, ops: vec![Pauli::I; self.n] };
        let mut stab: Vec<NPauliD> = vec![npauli.clone(); self.n];
        let mut destab: Vec<NPauliD> = vec![npauli.clone(); self.n];
        let iter
            = self.row_iter()
            .zip(stab.iter_mut().chain(destab.iter_mut()));
        for (((xi, zi), ri), g) in iter {
            npauli.phase
                = match ri[()] {
                    0 => Phase::Pi0,
                    1 => Phase::Pi1h,
                    2 => Phase::Pi3h,
                    _ => unreachable!(),
                };
            for (j, op) in npauli.ops.iter_mut().enumerate() {
                j5 = j >> 5;
                pw = PW[j & 31];
                if xi[j5] & pw == 0 && zi[j5] & pw == 0 { *op = Pauli::I; }
                if xi[j5] & pw != 0 && zi[j5] & pw == 0 { *op = Pauli::X; }
                if xi[j5] & pw != 0 && zi[j5] & pw != 0 { *op = Pauli::Y; }
                if xi[j5] & pw == 0 && zi[j5] & pw != 0 { *op = Pauli::Z; }
            }
            *g = npauli.clone();
        }
        StabGroupD { stab, destab }
    }

    /// Convert `self` to a bare tableau representation of only the stabilizer
    /// group.
    ///
    /// Here, the "tableau" representation is an `N × 2N` binary matrix with an
    /// extra vector of length `N` specifying the phase of each stabilizer as
    /// a power on *i* (that is, the phase of the `i`-th stabilizer is given by
    /// *i* raised to the value of the `i`-th element of the vector). For `0 ≤
    /// i, j < N`, the `i`-th row is the `i`-th element of the stabilizer group
    /// while the `j`-th and `N + j`-th columns identify the Pauli operator
    /// acting on the `j`-th qubit. For tableau `T`,
    ///
    /// | <code>T[_, j]</code> | <code>T[_, N + j]</code> | Pauli |
    /// | :------------------: | :----------------------: | :---: |
    /// | 0                    | 0                        | *I*   |
    /// | 1                    | 0                        | *X*   |
    /// | 1                    | 1                        | *Y*   |
    /// | 0                    | 1                        | *Z*   |
    pub fn as_tableau(&self) -> (nd::Array2<u8>, nd::Array1<u8>) {
        let mut j5: usize;
        let mut pw: u32;
        let mut tab: nd::Array2<u8> = nd::Array2::zeros((self.n, 2 * self.n));
        let ph: nd::Array1<u8> = self.r.slice(nd::s![..self.n]).to_owned();
        for ((xi, zi), mut ti) in
            self.x.axis_iter(nd::Axis(0))
            .zip(self.z.axis_iter(nd::Axis(0)))
            .zip(tab.axis_iter_mut(nd::Axis(0)))
        {
            for j in 0..self.n {
                j5 = j >> 5;
                pw = PW[j & 31];
                ti[j] = (xi[j5] & pw).try_into().unwrap();
                ti[self.n + j] = (zi[j5] & pw).try_into().unwrap();
            }
        }
        (tab, ph)
    }

    // like `as_tableau`, but outputting f32s for entanglement entropy
    // calculations via matrix rank via SVD, and without the phases
    fn as_tableau_f32(&self) -> nd::Array2<f32> {
        let mut j5: usize;
        let mut pw: u32;
        let mut tab: nd::Array2<f32> = nd::Array2::zeros((self.n, 2 * self.n));
        for ((xi, zi), mut ti) in
            self.x.axis_iter(nd::Axis(0))
            .zip(self.z.axis_iter(nd::Axis(0)))
            .zip(tab.axis_iter_mut(nd::Axis(0)))
        {
            for j in 0..self.n {
                j5 = j >> 5;
                pw = PW[j & 31];
                ti[j] = (xi[j5] & pw) as f32;
                ti[self.n + j] = (zi[j5] & pw) as f32;
            }
        }
        tab
    }

    /// Calculate the entanglement entropy of the state as the average over all
    /// bipartitions.
    ///
    /// See equation A19 of [arXiv:1901.08092][arxiv1], footnote 11 of
    /// [arXiv:1608.09650][arxiv2] and [this Stack Exchange thread][stackex].
    ///
    /// [arxiv1]: https://arxiv.org/abs/1901.08092
    /// [arxiv2]: https://arxiv.org/abs/1608.06950
    /// [stackex]: https://quantumcomputing.stackexchange.com/questions/16718/measuring-entanglement-entropy-using-a-stabilizer-circuit-simulator
    pub fn entanglement_entropy(&self) -> f32 {
        (1..self.n - 1)
            .map(|cut| {
                self.entanglement_entropy_single(Some(cut)) / (self.n - 1) as f32
            })
            .sum()
    }

    /// Calculate the entanglement entropy of the state across only a single
    /// bipartition placed at `cut`, defaulting to `floor(n / 2)`.
    ///
    /// See also [`Self::entanglement_entropy`].
    pub fn entanglement_entropy_single(&self, cut: Option<usize>) -> f32 {
        let mut j5 = 0;
        let mut pw = 0;
        let cut = cut.unwrap_or(self.n / 2).min(self.n - 1);
        if cut <= self.n / 2 {
            let n
                = self.x.axis_iter(nd::Axis(0))
                .zip(self.z.axis_iter(nd::Axis(0)))
                .take(self.n)
                .filter(|(xi, zi)| {
                    (0..cut).all(|j| {
                        j5 = j >> 5;
                        pw = PW[j & 31];
                        xi[j5] & pw == 0 && zi[j5] & pw == 0
                    })
                })
                .count() as f32;
            (self.n - cut) as f32 - n
        } else {
            let n
                = self.x.axis_iter(nd::Axis(0))
                .zip(self.z.axis_iter(nd::Axis(0)))
                .take(self.n)
                .filter(|(xi, zi)| {
                    (cut..self.n).all(|j| {
                        j5 = j >> 5;
                        pw = PW[j & 31];
                        xi[j5] & pw == 0 && zi[j5] & pw == 0
                    })
                })
                .count() as f32;
            cut as f32 - n
        }
    }

    // do Gaussian elimination to put the stabilizer generators in the following
    // form:
    // - at the top, a minimal set of generators containins X's and Y's, in
    //   "quasi-upper-triangular" form
    // - at the bottom, generators containins Z's only in
    //   quasi-upper-triangular form
    //
    // returns the number of such generators, equal to the log_2 of the number
    // of nonzero basis states
    fn gaussian_elim(&mut self) -> usize {
        let mut j5: usize;
        let mut pw: u32;
        let mut maybe_k: Option<usize>;
        let mut i: usize = self.n;
        for j in 0..self.n {
            j5 = j >> 5;
            pw = PW[j & 31];
            maybe_k = (i..2 * self.n).find(|q| self.x[[*q, j5]] & pw != 0);
            if let Some(k) = maybe_k {
                if k < 2 * self.n {
                    self.row_swap(k, i);
                    self.row_swap(k - self.n, i - self.n);
                    for k2 in i + 1..2 * self.n {
                        if self.x[[k2, j5]] & pw != 0 {
                            self.row_mul(i, k2);
                            self.row_mul(k2 - self.n, i - self.n);
                        }
                    }
                    i += 1;
                }
            }
        }
        let g: usize = i - self.n;
        for j in 0..self.n {
            j5 = j >> 5;
            pw = PW[j & 31];
            maybe_k = (i..2 * self.n).find(|q| self.z[[*q, j5]] & pw != 0);
            if let Some(k) = maybe_k {
                if k < 2 * self.n {
                    self.row_swap(k, i);
                    self.row_swap(k - self.n, i - self.n);
                    for k2 in i + 1..2 * self.n {
                        if self.z[[k2, j5]] & pw != 0 {
                            self.row_mul(i, k2);
                            self.row_mul(k2 - self.n, i - self.n);
                        }
                    }
                    i += 1;
                }
            }
        }
        g
    }

    // finds a Pauli operator P such that the basis state P |0...0> occurs with
    // nonzero amplitude, and writes P to the scratch row.
    //
    // self.gaussian_elim should be called before this method (and its output
    // should be provided as argument)
    fn seed_scratch(&mut self, g: usize) {
        self.r[2 * self.n] = 0;
        self.x.slice_mut(nd::s![2 * self.n, ..]).fill(0);
        self.z.slice_mut(nd::s![2 * self.n, ..]).fill(0);

        let mut f: u8;
        let mut j5: usize;
        let mut pw: u32;
        let mut min: usize = 0;
        for i in (self.n + g..2 * self.n).rev() {
            f = self.r[i];
            for j in (0..self.n).rev() {
                j5 = j >> 5;
                pw = PW[j & 31];
                if self.z[[i, j5]] & pw != 0 {
                    min = j;
                    if self.x[[2 * self.n, j5]] & pw != 0 { f = (f + 2) % 4; }
                }
            }
            if f % 4 == 2 {
                j5 = min >> 5;
                pw = PW[min & 31];
                self.x[[2 * self.n, j5]] ^= pw;
            }
        }
    }

    // returns the result of applying the Pauli operator in the scratch row to
    // |0...0> as a basis state
    fn as_basis_state(&self) -> BasisStateD {
        let mut j5: usize;
        let mut pw: u32;
        let mut e: u8 = self.r[2 * self.n];

        for j in 0..self.n {
            j5 = j >> 5;
            pw = PW[j & 31];
            if self.x[[2 * self.n, j5]] & pw != 0 && self.z[[2 * self.n, j5]] & pw != 0 {
                e = (e + 1) % 4;
            }
        }
        let phase
            = match e % 4 {
                0 => Phase::Pi0,
                1 => Phase::Pi1h,
                2 => Phase::Pi,
                3 => Phase::Pi3h,
                _ => unreachable!(),
            };
        let mut state: Vec<Qubit> = vec![Qubit::Zero; self.n];
        for (j, b) in state.iter_mut().enumerate() {
            j5 = j >> 5;
            pw = PW[j & 31];
            if self.x[[2 * self.n, j5]] & pw != 0 { *b = Qubit::One; }
        }
        BasisStateD { phase, state }
    }

    /// Perform Gaussian elimination and construct a basis state-based
    /// representation of `self`.
    ///
    /// Fails if the number of non-zero terms in the representation is greater
    /// than 2^31.
    pub fn as_kets(&mut self) -> Option<StateD> {
        let g = self.gaussian_elim();
        if g > 31 { return None; }
        let mut acc: Vec<BasisStateD>
            = Vec::with_capacity(2_usize.pow(g as u32));
        self.seed_scratch(g);
        acc.push(self.as_basis_state());
        let mut t2: u32;
        for t in 0..PW[g] - 1 {
            t2 = t ^ (t + 1);
            for (i, pw) in PW.iter().enumerate().take(g) {
                if t2 & pw != 0 { self.row_mul(self.n + i, 2 * self.n); }
            }
            acc.push(self.as_basis_state());
        }
        Some(StateD(acc))
    }
}

impl<const N: usize> From<Graph<N>> for StabD {
    fn from(graph: Graph<N>) -> Self { graph.to_stabd() }
}

impl From<GraphD> for StabD {
    fn from(graphd: GraphD) -> Self { graphd.to_stabd() }
}

/// A single `n`-qubit Pauli operator with a phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NPauliD {
    pub phase: Phase,
    pub ops: Vec<Pauli>,
}

impl fmt::Display for NPauliD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.phase.fmt(f)?;
        write!(f, " ")?;
        self.ops.iter()
            .try_for_each(|p| p.fmt(f))
    }
}

/// The complete `n`-qubit stabilizer/destabilizer groups for a given state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StabGroupD {
    pub stab: Vec<NPauliD>,
    pub destab: Vec<NPauliD>,
}

impl fmt::Display for StabGroupD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.stab.len();
        for (k, (stab, destab)) in
            self.stab.iter().zip(&self.destab).enumerate()
        {
            stab.fmt(f)?;
            write!(f, " | ")?;
            destab.fmt(f)?;
            if k < n - 1 { writeln!(f)?; }
        }
        Ok(())
    }
}

/// A single basis state in the product space of `n` qubits, with a phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasisStateD {
    pub phase: Phase,
    pub state: Vec<Qubit>,
}

impl fmt::Display for BasisStateD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.phase)?;
        write!(f, "∣")?;
        for q in self.state.iter() { write!(f, "{}", q)?; }
        write!(f, "⟩")?;
        Ok(())
    }
}

/// A superposition of basis states.
///
/// All amplitudes are equal in magnitude.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateD(pub Vec<BasisStateD>);

impl fmt::Display for StateD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0.len();
        for (k, bs) in self.0.iter().enumerate() {
            write!(f, "{}", bs)?;
            if k < n - 1 { write!(f, " ")?; }
        }
        Ok(())
    }
}


