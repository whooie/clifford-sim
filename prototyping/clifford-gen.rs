---cargo
[package]
edition = "2021"

[dependencies]
itertools = "*"
ndarray = "*"
num-complex = "*"
rand = "*"
---

#![allow(unused_imports, dead_code)]

use itertools::Itertools;
use ndarray::{ self as nd, linalg::kron };
use num_complex::Complex64 as C64;
use rand::{ Rng, thread_rng };

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Pauli { I, X, Y, Z }

impl Pauli {
    fn gen<R>(rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        match rng.gen_range(0..4_usize) {
            0 => Self::I,
            1 => Self::X,
            2 => Self::Y,
            3 => Self::Z,
            _ => unreachable!(),
        }
    }

    fn commutes_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I, _) | (_, Self::I) => true,
            (l, r) => l == r,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PauliString(bool, Vec<Pauli>);

impl PauliString {
    fn gen<R>(n: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        Self(rng.gen(), (0..n).map(|_| Pauli::gen(rng)).collect())
    }

    fn commutes_with(&self, other: &Self, skip: Option<usize>) -> bool {
        if self.1.len() != other.1.len() { panic!(); }
        self.1.iter().zip(&other.1)
            .skip(skip.unwrap_or(0))
            .filter(|(l, r)| !l.commutes_with(r))
            .count() % 2 == 0
    }

    fn sample_anticomm<R>(&self, skip: Option<usize>, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let n = self.1.len();
        let mut other: Self;
        for _ in 0..100 {
            other = Self::gen(n, rng);
            if !self.commutes_with(&other, skip) {
                return other;
            }
        }
        panic!()
        // panic!("{:?}", self);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Col { a: bool, b: bool }

#[derive(Clone, Debug, PartialEq, Eq)]
struct Tableau {
    n: usize,
    x: Vec<Col>,
    z: Vec<Col>,
    s: Col,
    gates: Vec<Gate>,
}

impl Tableau {
    fn new(n: usize) -> Self {
        Self {
            n,
            x: vec![Col { a: false, b: false }; n],
            z: vec![Col { a: false, b: false }; n],
            s: Col { a: false, b: false },
            gates: Vec::new(),
        }
    }

    fn set(&mut self, a: &PauliString, b: &PauliString) {
        assert_eq!(a.1.len(), b.1.len());
        assert_eq!(self.n, a.1.len());
        self.x.iter_mut().zip(self.z.iter_mut())
            .zip(a.1.iter().zip(b.1.iter()))
            .for_each(|((xj, zj), (aj, bj))| {
                match aj {
                    Pauli::I => { xj.a = false; zj.a = false; },
                    Pauli::X => { xj.a = true;  zj.a = false; },
                    Pauli::Y => { xj.a = true;  zj.a = true;  },
                    Pauli::Z => { xj.a = false; zj.a = true;  },
                }
                match bj {
                    Pauli::I => { xj.b = false; zj.b = false; },
                    Pauli::X => { xj.b = true;  zj.b = false; },
                    Pauli::Y => { xj.b = true;  zj.b = true;  },
                    Pauli::Z => { xj.b = false; zj.b = true;  },
                }
            });
        self.s.a = a.0;
        self.s.b = b.0;
    }

    fn h(&mut self, k: usize) {
        let xk = &mut self.x[k];
        let zk = &mut self.z[k];
        std::mem::swap(xk, zk);
        self.s.a ^= xk.a && zk.a;
        self.s.b ^= xk.b && zk.b;
        self.gates.push(Gate::H(k));
    }

    fn s(&mut self, k: usize) {
        let xk = &mut self.x[k];
        let zk = &mut self.z[k];
        self.s.a ^= xk.a && zk.a;
        self.s.b ^= xk.b && zk.b;
        zk.a ^= xk.a;
        zk.b ^= xk.b;
        self.gates.push(Gate::S(k));
    }

    fn cnot(&mut self, c: usize, j: usize) {
        self.x[j].a ^= self.x[c].a;
        self.x[j].b ^= self.x[c].b;
        self.z[c].a ^= self.z[j].a;
        self.z[c].b ^= self.z[j].b;
        self.s.a ^= self.x[c].a && self.z[j].a &&  self.x[j].a &&  self.z[c].a;
        self.s.a ^= self.x[c].a && self.z[j].a && !self.x[j].a && !self.z[c].a;
        self.s.b ^= self.x[c].b && self.z[j].b &&  self.x[j].b &&  self.z[c].b;
        self.s.b ^= self.x[c].b && self.z[j].b && !self.x[j].b && !self.z[c].b;
        self.gates.push(Gate::CX(c, j));
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.x.swap(a, b);
        self.z.swap(a, b);
        self.gates.push(Gate::Swap(a, b));
    }

    fn iter_xz(&self) -> impl Iterator<Item = (&Col, &Col)> + '_ {
        self.x.iter().zip(self.z.iter())
    }

    fn iter_xz_mut(&mut self) -> impl Iterator<Item = (&mut Col, &mut Col)> + '_ {
        self.x.iter_mut().zip(self.z.iter_mut())
    }

    fn sweep(&mut self, llim: usize) {
        let mut idx_scratch: Vec<usize> = Vec::with_capacity(self.n - llim);
        macro_rules! step_12 {
            ( $tab:ident, $llim:ident, $idx_scratch:ident, $row:ident )
            => {
                // (1)
                // clear top row of z: H
                $tab.iter_xz().enumerate().skip($llim)
                    .filter(|(_, (txj, tzj))| tzj.$row && !txj.$row)
                    .for_each(|(j, _)| { $idx_scratch.push(j); });
                $idx_scratch.drain(..)
                    .for_each(|j| { $tab.h(j); });
                // clear top row of z: S
                $tab.iter_xz().enumerate().skip($llim)
                    .filter(|(_, (txj, tzj))| tzj.$row && txj.$row)
                    .for_each(|(j, _)| { $idx_scratch.push(j); });
                $idx_scratch.drain(..)
                    .for_each(|j| { $tab.s(j); });

                // (2)
                // clear top row of x, all but one: CNOTs
                $tab.iter_xz().enumerate().skip($llim)
                    .filter(|(_, (txj, _))| txj.$row) // guaranteed at least 1 such
                    .for_each(|(j, _)| { $idx_scratch.push(j); });
                while $idx_scratch.len() > 1 {
                    $idx_scratch
                        = $idx_scratch.into_iter()
                        .chunks(2).into_iter()
                        .map(|mut chunk| {
                            let Some(a)
                                = chunk.next() else { unreachable!() };
                            if let Some(b) = chunk.next() {
                                $tab.cnot(a, b);
                            }
                            a
                        })
                        .collect();
                }
            }
        }
        step_12!(self, llim, idx_scratch, a);

        // (3)
        // move the remaining x in the top row to the leftmost column
        if let Some(j) = idx_scratch.first() {
            if *j != llim { self.swap(*j, llim); }
            idx_scratch.pop();
        }

        // (4)
        // apply a hadamard if p1 != Z1.I.I...
        if !self.z[llim].b
            || self.x[llim].b
            || self.iter_xz().skip(llim + 1).any(|(txj, tzj)| txj.b || tzj.b)
        {
            self.h(llim);
            // repeat (1) and (2) above for the bottom row
            step_12!(self, llim, idx_scratch, b);
            self.h(llim);
        }

        // (5)
        // clear signs
        match self.s {
            Col { a: false, b: false } => { },
            Col { a: false, b: true  } => { self.gates.push(Gate::X(llim)); },
            Col { a: true,  b: true  } => { self.gates.push(Gate::Y(llim)); },
            Col { a: true,  b: false } => { self.gates.push(Gate::Z(llim)); },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Clifford {
    n: usize,
    gates: Vec<Gate>,
}

impl Clifford {
    fn iter(&self) -> std::slice::Iter<'_, Gate> { self.gates.iter() }

    fn gen<R>(n: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let mut stab_a: PauliString;
        let mut stab_b: PauliString;
        let mut tab = Tableau::new(n);
        for llim in 0..n {
            stab_a = PauliString::gen(n, rng);
            while stab_a.1.iter().skip(llim).all(|p| *p == Pauli::I) {
                stab_a = PauliString::gen(n, rng);
            }
            stab_b = stab_a.sample_anticomm(Some(llim), rng);
            tab.set(&stab_a, &stab_b);
            tab.sweep(llim);
        }
        Self { n, gates: tab.gates }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Gate {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    SInv(usize),
    CX(usize, usize),
    CZ(usize, usize),
    Swap(usize, usize),
}

impl Gate {
    fn to_matrix(&self) -> nd::Array2<C64> {
        use std::f64::consts::FRAC_1_SQRT_2;
        const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
        const ONE: C64 = C64 { re: 1.0, im: 0.0 };
        const I: C64 = C64 { re: 0.0, im: 1.0 };
        const ORT2: C64 = C64 { re: FRAC_1_SQRT_2, im: 0.0 };
        const IORT2: C64 = C64 { re: 0.0, im: FRAC_1_SQRT_2 };
        let eye: nd::Array2<C64> = nd::array![[ONE, ZERO], [ZERO, ONE]];
        match self {
            Gate::H(k) => {
                let mat: nd::Array2<C64> =
                    nd::array![[ORT2, ORT2], [ORT2, -ORT2]];
                if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
            },
            Gate::X(k) => {
                let mat: nd::Array2<C64> =
                    nd::array![[ZERO, ONE], [ONE, ZERO]];
                if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
            },
            Gate::Y(k) => {
                let mat: nd::Array2<C64> =
                    nd::array![[ZERO, -I], [I, ZERO]];
                if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
            },
            Gate::Z(k) => {
                let mat: nd::Array2<C64> =
                    nd::array![[ONE, ZERO], [ZERO, -ONE]];
                if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
            },
            Gate::S(k) => {
                let mat: nd::Array2<C64> =
                    nd::array![[ONE, ZERO], [ZERO, I]];
                if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
            },
            Gate::SInv(k) => {
                let mat: nd::Array2<C64> =
                    nd::array![[ONE, ZERO], [ZERO, -I]];
                if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
            },
            Gate::CX(a, _) => {
                if *a == 0 {
                    nd::array![
                        [ONE, ZERO, ZERO, ZERO],
                        [ZERO, ONE, ZERO, ZERO],
                        [ZERO, ZERO, ZERO, ONE],
                        [ZERO, ZERO, ONE, ZERO],
                    ]
                } else {
                    nd::array![
                        [ONE, ZERO, ZERO, ZERO],
                        [ZERO, ZERO, ZERO, ONE],
                        [ZERO, ZERO, ONE, ZERO],
                        [ZERO, ONE, ZERO, ZERO],
                    ]
                }
            },
            Gate::CZ(..) => {
                nd::array![
                    [ONE,  ZERO, ZERO,  ZERO],
                    [ZERO, ONE,  ZERO,  ZERO],
                    [ZERO, ZERO, ONE,   ZERO],
                    [ZERO, ZERO, ZERO, -ONE ],
                ]
            },
            Gate::Swap(..) => {
                nd::array![
                    [ONE, ZERO, ZERO, ZERO],
                    [ZERO, ZERO, ONE, ZERO],
                    [ZERO, ONE, ZERO, ZERO],
                    [ZERO, ZERO, ZERO, ONE],
                ]
            },
        }
    }
}

fn cliff_matrix(gates: &Clifford) -> nd::Array2<C64> {
    gates.iter()
        .rev()
        .map(|g| g.to_matrix())
        .fold(nd::Array2::eye(4), |acc, gate| acc.dot(&gate))
}

fn matrix_approx_eq(a: &nd::Array2<C64>, b: &nd::Array2<C64>) -> bool {
    a.iter().zip(b).all(|(ak, bk)| (ak - bk).norm() < 1e-6)
}

fn card_cn(n: usize) -> u128 {
    (1..=n).map(|k| 2_u128 * 4_u128.pow(k as u32) * (4_u128.pow(k as u32) - 1))
        .product()
}


fn main() {
    const N: usize = 2;
    const MC: usize = 500_000;
    println!("|C{}| = {}", N, card_cn(N));

    let mut rng = thread_rng();
    let mut elems: Vec<Clifford> = Vec::new();
    let mut mats: Vec<nd::Array2<C64>> = Vec::new();
    print!("\r 0 / {} ", MC);
    for k in 0..MC {
        let cliff = Clifford::gen(N, &mut rng);
        // println!("{:?}", cliff);
        let mat = cliff_matrix(&cliff);
        if !mats.iter().any(|g| matrix_approx_eq(g, &mat)) {
            mats.push(mat);
            elems.push(cliff);
        }
        print!("\r {} / {} ", k + 1, MC);
    }
    println!();
    println!("{} unique elements found from {} draws", mats.len(), MC);

    // let mut tab = Tableau::new(4);
    // let a = PauliString(false, vec![Pauli::X, Pauli::Y, Pauli::Y, Pauli::X]);
    // let b = PauliString(false, vec![Pauli::Y, Pauli::Y, Pauli::Y, Pauli::X]);
    // tab.set(&a, &b);
    // tab.sweep(0);
    // println!("{:?}", tab.gates);
    //
    // let mut tab = Tableau::new(3);
    // let a = PauliString(false, vec![Pauli::I, Pauli::Z, Pauli::I]);
    // let b = PauliString(false, vec![Pauli::Y, Pauli::Y, Pauli::I]);
    // tab.set(&a, &b);
    // tab.sweep(0);
    // println!("{:?}", tab.gates);
    //
    // let mut tab = Tableau::new(2);
    // let a = PauliString(false, vec![Pauli::I, Pauli::X]);
    // let b = PauliString(false, vec![Pauli::I, Pauli::Z]);
    // tab.set(&a, &b);
    // tab.sweep(0);
    // println!("{:?}", tab.gates);
    //
    // let mut tab = Tableau::new(1);
    // let a = PauliString(false, vec![Pauli::Z]);
    // let b = PauliString(false, vec![Pauli::X]);
    // tab.set(&a, &b);
    // tab.sweep(0);
    // println!("{:?}", tab.gates);
}
