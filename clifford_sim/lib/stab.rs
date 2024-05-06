//! *N*-qubit stabilizer states in the Gottesman-Knill tableau representation.
//!
//! In the tableau representation, states are identified not by complex
//! amplitudes but by the set of *N*-qubit Pauli operators (that is, the tensor
//! product of *N* Pauli operators including the identity) that stabilize them,
//! of which there are *N*. Since there are only four single-qubit Pauli
//! operators, the operator corresponding to each qubit requires a minimum of
//! only two bits to encode. Along with another four bits to identify an overall
//! complex phase limited to integer multiples of π/2, this means a complete
//! stabilizer state can be encoded by `*N* × 4 (*N* + 1)  bits, which is
//! significantly better than the *O*(2<sup>*N*</sup>) scaling of a naive state
//! vector representation.
//!
//! Properly, a tableau is a binary matrix of size *N* × 2*N*, with an extra
//! vector of values 0, ..., 3, which encode phases by the exponent on the
//! complex unit **i**. The (*i*, *j*)-th and (*i*, *N* + *j*)-th entries of the
//! binary matrix encode the *j*-th Pauli of the *i*-th stabilizer, with phase
//! given by the *i*-th entry of the phase vector. In later work by Gottesman
//! and Aaronson, this encoding is extended to include *N* more "destabilizers"
//! (with phases), which are the complementary set of *N*-qubit Paulis that,
//! together with the stabilizers, generate the full group of all possible
//! *N*-qubit Paulis[^1].
//!
//! Additionally, the actions of the generating operators for the *N*-qubit
//! Clifford group (Hadamard, π/2 phase, and CNOT) on the stabilizer state can
//! then be written as bitwise operations on the columns of the binary matrix,
//! which are *O*(*N*) in runtime. Thus, the tableau representation enables
//! efficient classical simulation of Clifford circuits.
//!
//! The code in this module is pretty much a direct translation of Scott
//! Aaronson's code (see [arXiv:quant-ph/0406196][tableau] for details on his
//! extension and [here][chp] for his implementation) with some extra parts to
//! handle the calculation of entanglement entropy, using methods from
//! [arXiv:1901.08092][entropy1] and [arXiv:1608.09650][entropy2].
//!
//! # Example
//! ```
//! use clifford_sim::{ stab::Stab, gate::Gate };
//!
//! const N: usize = 5; // number of qubits
//!
//! fn main() {
//!     // initialize a new state to ∣00000⟩
//!     let mut stab: Stab<N> = Stab::new();
//!
//!     // generate a Bell state on qubits 0, 1
//!     stab.apply_gate(Gate::H(0));
//!     stab.apply_gate(Gate::CX(0, 1));
//!
//!     // print out the stabilizers and destabilizers
//!     println!("{:#}", stab.as_group()); // `#` formatter suppresses identities
//!     // +1 Z.... | +1 XX...
//!     // +1 .X... | +1 ZZ...
//!     // +1 ..X.. | +1 ..Z..
//!     // +1 ...X. | +1 ...Z.
//!     // +1 ....X | +1 ....Z
//!
//!     // convert to ket notation; fails if there are > 2^31 basis states
//!     println!("{}", stab.as_kets().unwrap());
//!     // +1∣00000⟩ +1∣11000⟩
//! }
//! ```
//!
//! [^1]: In their implementation, Aaronson and Gottesman also includes a single
//! extra row and phase as a useful scratch space for many of his operations.
//!
//! [tableau]: https://arxiv.org/abs/quant-ph/0406196
//! [chp]: https://www.scottaaronson.com/chp/
//! [entropy1]: https://arxiv.org/abs/1901.08092
//! [entropy2]: https://arxiv.org/abs/1608.09650

use std::fmt;
use ndarray::{ self as nd, s };
use ndarray_linalg::SVD;
use rand::Rng;
use crate::gate::{ Gate, Pauli, Phase };

const PW: [u32; 32] = [ // PW[i] = 2^i
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216,
    33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648
];

/// An `N`-qubit stabilizer state, identified by its stabilizer group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stab<const N: usize> {
    // `x` and `z` are bit arrays of size (2N + 1) x N; for space efficiency,
    // the columns are packed into u32s
    x: nd::Array2<u32>, // Pauli-X bits; size (2N + 1) x (floor(N / 8) + 1)
    z: nd::Array2<u32>, // Pauli-Z bits; size (2N + 1) x (floor(N / 8) + 1)
    r: nd::Array1<u8>, // Phases (0 for +1, 1 for i, 2 for -1 3 for -i); size 2N + 1
    over32: usize, // = floor(N / 8) + 1
}

impl<const N: usize> Default for Stab<N> {
    fn default() -> Self { Self::new() }
}

impl<const N: usize> Stab<N> {
    /// Create a new stabilizer state initialized to ∣0...0⟩.
    pub fn new() -> Self {
        let over32: usize = (N >> 5) + 1;
        let x: nd::Array2<u32> = nd::Array2::zeros((2 * N + 1, over32));
        let z: nd::Array2<u32> = nd::Array2::zeros((2 * N + 1, over32));
        let r: nd::Array1<u8> = nd::Array1::zeros(2 * N + 1);
        let mut q = Self { x, z, r, over32 };
        let mut j: usize;
        for (i, ((mut xi, mut zi), _)) in q.row_iter_mut().enumerate() {
            if i < N {
                xi[i >> 5] = PW[i & 31];
            } else if i < 2 * N {
                j = i - N;
                zi[j >> 5] = PW[j & 31];
            }
        }
        q
    }

    fn row_iter(&self)
        -> impl Iterator<
            Item = ((nd::ArrayView1<u32>, nd::ArrayView1<u32>), nd::ArrayView0<u8>)
        >
    {
        self.x.axis_iter(nd::Axis(0))
            .zip(self.z.axis_iter(nd::Axis(0)))
            .zip(self.r.axis_iter(nd::Axis(0)))
            .take(2 * N)
    }

    fn row_iter_mut(&mut self)
        -> impl Iterator<
            Item = ((nd::ArrayViewMut1<u32>, nd::ArrayViewMut1<u32>), nd::ArrayViewMut0<u8>)
        >
    {
        self.x.axis_iter_mut(nd::Axis(0))
            .zip(self.z.axis_iter_mut(nd::Axis(0)))
            .zip(self.r.axis_iter_mut(nd::Axis(0)))
            .take(2 * N)
    }

    fn col_iter(&self)
        -> impl Iterator<Item = (nd::ArrayView1<u32>, nd::ArrayView1<u32>)>
    {
        self.x.axis_iter(nd::Axis(1))
            .zip(self.z.axis_iter(nd::Axis(1)))
    }

    fn col_iter_mut(&mut self)
        -> impl Iterator<Item = (nd::ArrayViewMut1<u32>, nd::ArrayViewMut1<u32>)>
    {
        self.x.axis_iter_mut(nd::Axis(1))
            .zip(self.z.axis_iter_mut(nd::Axis(1)))
    }

    fn apply_h(&mut self, k: usize) -> &mut Self {
        let k5: usize = k >> 5;
        let pw: u32 = PW[k & 31];
        let mut tmp: u32;
        for ((x_i_k5, z_i_k5), r_i) in
            self.x.slice_mut(s![.., k5]).iter_mut()
                .zip(self.z.slice_mut(s![.., k5]).iter_mut())
                .zip(self.r.iter_mut())
                .take(2 * N)
        {
            tmp = *x_i_k5;
            *x_i_k5 ^= (*x_i_k5 ^ *z_i_k5) & pw;
            *z_i_k5 ^= (*z_i_k5 ^ tmp) & pw;
            if *x_i_k5 & pw != 0 && *z_i_k5 & pw != 0 { *r_i = (*r_i + 2) % 4; }
        }
        // for i in 0..2 * N {
        //     tmp = self.x[[i, k5]];
        //     self.x[[i, k5]] ^= (self.x[[i, k5]] ^ self.z[[i, k5]]) & pw;
        //     self.z[[i, k5]] ^= (self.z[[i, k5]] ^ tmp) & pw;
        //     if self.x[[i, k5]] & pw != 0 && self.z[[i, k5]] & pw != 0 {
        //         self.r[i] = (self.r[i] + 2) % 4;
        //     }
        // }
        self
    }

    fn apply_s(&mut self, k: usize) -> &mut Self {
        let k5: usize = k >> 5;
        let pw: u32 = PW[k & 31];
        for ((x_i_k5, z_i_k5), r_i) in
            self.x.slice_mut(s![.., k5]).iter_mut()
                .zip(self.z.slice_mut(s![.., k5]).iter_mut())
                .zip(self.r.iter_mut())
                .take(2 * N)
        {
            if *x_i_k5 & pw != 0 && *z_i_k5 & pw != 0 { *r_i = (*r_i + 2) % 4; }
            *z_i_k5 ^= *x_i_k5 & pw;
        }
        // for i in 0..2 * N {
        //     if self.x[[i, k5]] & pw != 0 && self.z[[i, k5]] & pw != 0 {
        //         self.r[i] = (self.r[i] + 2) % 4;
        //     }
        //     self.z[[i, k5]] ^= self.x[[i, k5]] & pw;
        // }
        self
    }

    fn apply_x(&mut self, k: usize) -> &mut Self {
        self.apply_h(k).apply_z(k).apply_h(k)
    }

    fn apply_y(&mut self, k: usize) -> &mut Self {
        self.apply_s(k).apply_x(k).apply_z(k).apply_s(k)
    }

    fn apply_z(&mut self, k: usize) -> &mut Self {
        self.apply_s(k).apply_s(k)
    }

    fn apply_cnot(&mut self, a: usize, b: usize) -> &mut Self {
        let a5: usize = a >> 5;
        let b5: usize = b >> 5;
        let pwa: u32 = PW[a & 31];
        let pwb: u32 = PW[b & 31];
        for ((mut x_i, mut z_i), r_i) in
            self.x.axis_iter_mut(nd::Axis(0))
                .zip(self.z.axis_iter_mut(nd::Axis(0)))
                .zip(self.r.iter_mut())
                .take(2 * N)
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
        // for i in 0..2 * N {
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

    fn apply_swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.apply_cnot(a, b).apply_cnot(b, a).apply_cnot(a, b)
    }

    /// Perform the action of a gate.
    ///
    /// Does nothing if any qubit indices are out of bounds.
    pub fn apply_gate(&mut self, gate: Gate) -> &mut Self {
        match gate {
            Gate::H(k) if k < N => self.apply_h(k),
            Gate::X(k) if k < N => self.apply_x(k),
            Gate::Y(k) if k < N => self.apply_y(k),
            Gate::Z(k) if k < N => self.apply_z(k),
            Gate::S(k) if k < N => self.apply_s(k),
            Gate::CX(a, b) if a < N && b < N => self.apply_cnot(a, b),
            Gate::Swap(a, b) if a < N && b < N => self.apply_swap(a, b),
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

    fn row_copy(&mut self, a: usize, b: usize) -> &mut Self {
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

    fn row_swap(&mut self, a: usize, b: usize) -> &mut Self {
        // swap rows a and b
        self.row_copy(b, 2 * N)
            .row_copy(a, b)
            .row_copy(2 * N, a)
    }

    // set row k equal to the o-th observable (X_1, ... X_n, Z_1, ..., Z_n)
    fn row_set(&mut self, o: usize, k: usize) -> &mut Self {
        let o5: usize;
        let o31: usize;
        self.x.slice_mut(nd::s![k, ..]).fill(0);
        self.z.slice_mut(nd::s![k, ..]).fill(0);
        self.r[k] = 0;
        if o < N {
            o5 = o >> 5;
            o31 = o & 31;
            self.x[[k, o5]] = PW[o31];
        } else {
            o5 = (o - N) >> 5;
            o31 = (o - N) & 31;
            self.z[[k, o5]] = PW[o31];
        }
        self
    }

    // return the phase (0, ..., 3) when row b is left-multiplied by row a
    fn row_mul_phase(&self, a: usize, b: usize) -> u8 {
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
    fn row_mul(&mut self, a: usize, b: usize) -> &mut Self {
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
                .take(2 * N)
                .skip(N)
                .enumerate()
        {
            rnd = *x_qpN_k5 & pw != 0;
            if rnd { p = q; break; }
        }

        if rnd {
            self.row_copy(p + N, p);
            self.row_set(k + N, p + N);
            self.r[p + N] = 2 * u8::from(rng.gen::<bool>());
            for i in 0..2 * N {
                if i != p && self.x[[i, k5]] & pw != 0 { self.row_mul(p, i); }
            }
            if self.r[p + N] != 0 {
                Outcome::Rand1
            } else {
                Outcome::Rand0
            }
        } else {
            for (q, x_q_k5) in
                self.x.slice(nd::s![.., k5]).iter()
                    .take(N)
                    .enumerate()
            {
                if x_q_k5 & pw != 0 { m = q; break; }
            }
            self.row_copy(m + N, 2 * N);
            for i in m + 1..N {
                if self.x[[i, k5]] & pw != 0 { self.row_mul(i + N, 2 * N); }
            }
            if self.r[2 * N] != 0 {
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
    /// returning it as a [`Some`] if the post-selection was valid and [`None`]
    /// otherwise.
    pub fn measure_postsel(mut self, k: usize, postsel: Postsel) -> Option<Self>
    {
        let mut rnd: bool = false;
        let k5: usize = k >> 5;
        let pw: u32 = PW[k & 31];
        let mut p: usize = 0;
        let mut m: usize = 0;

        for (q, x_qpN_k5) in
            self.x.slice(nd::s![.., k5]).iter()
                .take(2 * N)
                .skip(N)
                .enumerate()
        {
            rnd = *x_qpN_k5 & pw != 0;
            if rnd { p = q; break; }
        }

        if rnd {
            self.row_copy(p + N, p);
            self.row_set(k + N, p + N);
            self.r[p + N] = 2 * postsel as u8;
            for i in 0..2 * N {
                if i != p && self.x[[i, k5]] & pw != 0 { self.row_mul(p, i); }
            }
            Some(self)
        } else {
            for (q, x_q_k5) in
                self.x.slice(nd::s![.., k5]).iter()
                    .take(N)
                    .enumerate()
            {
                if x_q_k5 & pw != 0 { m = q; break; }
            }
            self.row_copy(m + N, 2 * N);
            for i in m + 1..N {
                if self.x[[i, k5]] & pw != 0 { self.row_mul(i + N, 2 * N); }
            }
            match (self.r[2 * N] != 0, postsel) {
                (true,  Postsel::One ) => Some(self),
                (false, Postsel::Zero) => Some(self),
                _ => None,
            }
        }
    }

    /// Convert `self` to a more human-readable stabilizer/destabilizer group
    /// representation.
    pub fn as_group(&self) -> StabGroup<N> {
        let mut j5: usize;
        let mut pw: u32;
        let mut npauli: NPauli<N>
            = NPauli { phase: Phase::Pi0, ops: [Pauli::I; N] };
        let mut stab: [NPauli<N>; N] = [npauli; N];
        let mut destab: [NPauli<N>; N] = [npauli; N];
        let iter
            = self.row_iter()
            .zip(stab.iter_mut().chain(destab.iter_mut()));
        for (((xi, zi), ri), g) in iter {
            npauli.phase
                = match ri[()] {
                    0 => Phase::Pi0,
                    1 => Phase::Pi1h,
                    2 => Phase::Pi,
                    3 => Phase::Pi3h,
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
            *g = npauli;
        }
        StabGroup { stab, destab }
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
        let mut tab: nd::Array2<u8> = nd::Array2::zeros((N, 2 * N));
        let ph: nd::Array1<u8> = self.r.slice(nd::s![..N]).to_owned();
        for ((xi, zi), mut ti) in
            self.x.axis_iter(nd::Axis(0))
            .zip(self.z.axis_iter(nd::Axis(0)))
            .zip(tab.axis_iter_mut(nd::Axis(0)))
        {
            for j in 0..N {
                j5 = j >> 5;
                pw = PW[j & 31];
                ti[j] = (xi[j5] & pw).try_into().unwrap();
                ti[N + j] = (zi[j5] & pw).try_into().unwrap();
            }
        }
        (tab, ph)
    }

    // like `as_tableau`, but outputting f32s for entanglement entropy
    // calculations via matrix rank via SVD, and without the phases
    fn as_tableau_f32(&self) -> nd::Array2<f32> {
        let mut j5: usize;
        let mut pw: u32;
        let mut tab: nd::Array2<f32> = nd::Array2::zeros((N, 2 * N));
        for ((xi, zi), mut ti) in
            self.x.axis_iter(nd::Axis(0))
            .zip(self.z.axis_iter(nd::Axis(0)))
            .zip(tab.axis_iter_mut(nd::Axis(0)))
        {
            for j in 0..N {
                j5 = j >> 5;
                pw = PW[j & 31];
                ti[j] = (xi[j5] & pw) as f32;
                ti[N + j] = (zi[j5] & pw) as f32;
            }
        }
        tab
    }

    fn do_entanglement_entropy(
        tab: &nd::Array2<f32>, // shape N x 2N
        subtab: &mut nd::Array2<f32>, // shape N x N
        cut: usize,
    ) -> f32
    {
        let n = cut.min(N - cut);
        if cut <= N / 2 {
            subtab.slice_mut(nd::s![.., ..n])
                .assign(&tab.slice(nd::s![.., ..cut]));
            subtab.slice_mut(nd::s![.., n..2 * n])
                .assign(&tab.slice(nd::s![.., N..N + cut]));
        } else {
            subtab.slice_mut(nd::s![.., ..n])
                .assign(&tab.slice(nd::s![.., cut..N]));
            subtab.slice_mut(nd::s![.., n..2 * n])
                .assign(&tab.slice(nd::s![.., N + cut..]));
        }
        let (_, s, _): (_, nd::Array1<f32>, _)
            = subtab.slice(nd::s![.., ..2 * n]).svd(false, false).unwrap();
        let smax: f32
            = s.iter()
            .max_by(|l, r| {
                l.partial_cmp(r).unwrap_or(std::cmp::Ordering::Greater)
            })
            .copied()
            .unwrap();
        let eps: f32 = smax * (N as f32) * f32::EPSILON;
        s.into_iter().filter(|sk| *sk > eps).count() as f32 - n as f32
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
        let tab: nd::Array2<f32> = self.as_tableau_f32();
        let mut subtab: nd::Array2<f32> = nd::Array2::zeros((N, N));
        (1..N - 1)
            .map(|cut| {
                Self::do_entanglement_entropy(&tab, &mut subtab, cut)
                    / (N - 1) as f32
            })
            .sum()
    }

    /// Calculate the entanglement entropy of the state across only a single
    /// bipartition placed at `cut`, defaulting to `floor(N / 2)`.
    ///
    /// See also [`Self::entanglement_entropy`].
    pub fn entanglement_entropy_single(&self, cut: Option<usize>) -> f32 {
        let cut = cut.unwrap_or(N / 2);
        let n = cut.min(N - cut);
        let tab: nd::Array2<f32> = self.as_tableau_f32();
        let mut subtab: nd::Array2<f32> = nd::Array2::zeros((N, 2 * n));
        Self::do_entanglement_entropy(&tab, &mut subtab, cut)
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
        let mut i: usize = N;
        for j in 0..N {
            j5 = j >> 5;
            pw = PW[j & 31];
            maybe_k = (i..2 * N).find(|q| self.x[[*q, j5]] & pw != 0);
            if let Some(k) = maybe_k {
                if k < 2 * N {
                    self.row_swap(k, i);
                    self.row_swap(k - N, i - N);
                    for k2 in i + 1..2 * N {
                        if self.x[[k2, j5]] & pw != 0 {
                            self.row_mul(i, k2);
                            self.row_mul(k2 - N, i - N);
                        }
                    }
                    i += 1;
                }
            }
        }
        let g: usize = i - N;
        for j in 0..N {
            j5 = j >> 5;
            pw = PW[j & 31];
            maybe_k = (i..2 * N).find(|q| self.z[[*q, j5]] & pw != 0);
            if let Some(k) = maybe_k {
                if k < 2 * N {
                    self.row_swap(k, i);
                    self.row_swap(k - N, i - N);
                    for k2 in i + 1..2 * N {
                        if self.z[[k2, j5]] & pw != 0 {
                            self.row_mul(i, k2);
                            self.row_mul(k2 - N, i - N);
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
        self.r[2 * N] = 0;
        self.x.slice_mut(nd::s![2 * N, ..]).fill(0);
        self.z.slice_mut(nd::s![2 * N, ..]).fill(0);

        let mut f: u8;
        let mut j5: usize;
        let mut pw: u32;
        let mut min: usize = 0;
        for i in (N + g..2 * N).rev() {
            f = self.r[i];
            for j in (0..N).rev() {
                j5 = j >> 5;
                pw = PW[j & 31];
                if self.z[[i, j5]] & pw != 0 {
                    min = j;
                    if self.x[[2 * N, j5]] & pw != 0 { f = (f + 2) % 4; }
                }
            }
            if f % 4 == 2 {
                j5 = min >> 5;
                pw = PW[min & 31];
                self.x[[2 * N, j5]] ^= pw;
            }
        }
    }

    // returns the result of applying the Pauli operator in the scratch row to
    // |0...0> as a basis state
    fn as_basis_state(&self) -> BasisState<N> {
        let mut j5: usize;
        let mut pw: u32;
        let mut e: u8 = self.r[2 * N];

        for j in 0..N {
            j5 = j >> 5;
            pw = PW[j & 31];
            if self.x[[2 * N, j5]] & pw != 0 && self.z[[2 * N, j5]] & pw != 0 {
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
        let mut state: [Qubit; N] = [Qubit::Zero; N];
        for (j, b) in state.iter_mut().enumerate() {
            j5 = j >> 5;
            pw = PW[j & 31];
            if self.x[[2 * N, j5]] & pw != 0 { *b = Qubit::One; }
        }
        BasisState { phase, state }
    }

    /// Perform Gaussian elimination and construct a basis state-based
    /// representation of `self`.
    ///
    /// Fails if the number of non-zero terms in the representation is greater
    /// than 2^31.
    pub fn as_kets(&mut self) -> Option<State<N>> {
        let g = self.gaussian_elim();
        if g > 31 { return None; }
        let mut acc: Vec<BasisState<N>>
            = Vec::with_capacity(2_usize.pow(g as u32));
        self.seed_scratch(g);
        acc.push(self.as_basis_state());
        let mut t2: u32;
        for t in 0..PW[g] - 1 {
            t2 = t ^ (t + 1);
            for (i, pw) in PW.iter().enumerate().take(g) {
                if t2 & pw != 0 { self.row_mul(N + i, 2 * N); }
            }
            acc.push(self.as_basis_state());
        }
        Some(State(acc))
    }

}

/// A single `N`-qubit Pauli operator with a phase.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NPauli<const N: usize> {
    pub phase: Phase,
    pub ops: [Pauli; N],
}

impl<const N: usize> fmt::Display for NPauli<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.phase.fmt(f)?;
        write!(f, " ")?;
        self.ops.iter()
            .try_for_each(|p| p.fmt(f))
    }
}

/// The complete `N`-qubit stabilizer/destabilizer groups for a given state.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct StabGroup<const N: usize> {
    pub stab: [NPauli<N>; N],
    pub destab: [NPauli<N>; N],
}

impl<const N: usize> fmt::Display for StabGroup<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (k, (stab, destab)) in
            self.stab.iter().zip(self.destab).enumerate()
        {
            stab.fmt(f)?;
            write!(f, " | ")?;
            destab.fmt(f)?;
            if k < N - 1 { writeln!(f)?; }
        }
        Ok(())
    }
}

/// The result of a measurement, generated by [`Stab::measure`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// A deterministic outcome resulting in ∣0⟩
    Det0,
    /// A deterministic outcome resulting in ∣1⟩
    Det1,
    /// A random outcome resulting in ∣0⟩
    Rand0,
    /// A random outcome resulting in ∣1⟩
    Rand1,
}

/// A post-selected measurement result, required by [`Stab::measure_postsel`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Postsel {
    /// Post-selection for ∣0⟩
    Zero,
    /// Pose-selection for ∣1⟩
    One,
}

/// A qubit state in the standard (Z) basis.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Qubit {
    /// ∣0⟩
    Zero,
    /// ∣1⟩
    One,
}

impl fmt::Display for Qubit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Zero => write!(f, "0"),
            Self::One => write!(f, "1"),
        }
    }
}

/// A single basis state in the product space of `N` qubits, with a phase.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BasisState<const N: usize> {
    pub phase: Phase,
    pub state: [Qubit; N],
}

impl<const N: usize> fmt::Display for BasisState<N> {
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
pub struct State<const N: usize>(pub Vec<BasisState<N>>);

impl<const N: usize> fmt::Display for State<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0.len();
        for (k, bs) in self.0.iter().enumerate() {
            write!(f, "{}", bs)?;
            if k < n - 1 { write!(f, " ")?; }
        }
        Ok(())
    }
}

