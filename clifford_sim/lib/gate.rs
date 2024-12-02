//! Gates whose operations belong to the *n*-qubit Clifford group.
//!
//! See also: <https://en.wikipedia.org/wiki/Clifford_gates>

use std::{
    fmt,
    ops::{
        Neg,
        Add,
        AddAssign,
        Sub,
        SubAssign,
        Mul,
        MulAssign,
        Div,
        DivAssign,
    },
};
use itertools::Itertools;
use num_complex::Complex64 as C64;
use rand::Rng;
use crate::tree::Qubit;

/// The argument of a complex phase factor, limited to integer multiples of π/4.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    /// 0
    Pi0,
    /// π/2
    Pi1h,
    /// π
    Pi,
    /// 3π/2
    Pi3h,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Pi0 => write!(f, "+1"),
            Self::Pi1h => write!(f, "+i"),
            Self::Pi => write!(f, "-1"),
            Self::Pi3h => write!(f, "-i"),
        }
    }
}

impl Phase {
    /// Convert to the bare multiple of π/4.
    pub fn to_int(&self) -> i8 {
        match self {
            Self::Pi0  => 0,
            Self::Pi1h => 1,
            Self::Pi   => 2,
            Self::Pi3h => 3,
        }
    }

    /// Convert from a bare multiple of π/4 (modulo 8).
    pub fn from_int(i: i8) -> Self {
        match i.rem_euclid(4) {
            0 => Self::Pi0,
            1 => Self::Pi1h,
            2 => Self::Pi,
            3 => Self::Pi3h,
            _ => unreachable!(),
        }
    }

    pub fn as_complex(self) -> C64 {
        match self {
            Self::Pi0  => 1.0_f64.into(),
            Self::Pi1h => C64::i(),
            Self::Pi   => (-1.0_f64).into(),
            Self::Pi3h => -C64::i(),
        }
    }
}

impl Neg for Phase {
    type Output = Self;

    fn neg(self) -> Self::Output { Self::from_int(-self.to_int()) }
}

macro_rules! impl_phase_math {
    (
        $trait:ident,
        $trait_fn:ident,
        $trait_assign:ident,
        $trait_assign_fn:ident,
        $op:tt
    ) => {
        impl $trait for Phase {
            type Output = Self;

            fn $trait_fn(self, rhs: Self) -> Self::Output {
                Self::from_int(self.to_int() $op rhs.to_int())
            }
        }

        impl $trait_assign for Phase {
            fn $trait_assign_fn(&mut self, rhs: Self) {
                *self = *self $op rhs;
            }
        }
    }
}
impl_phase_math!(Add, add, AddAssign, add_assign, +);
impl_phase_math!(Sub, sub, SubAssign, sub_assign, -);
impl_phase_math!(Mul, mul, MulAssign, mul_assign, *);
impl_phase_math!(Div, div, DivAssign, div_assign, /);

impl Mul<i8> for Phase {
    type Output = Self;

    fn mul(self, i: i8) -> Self::Output {
        Self::from_int(self.to_int() * i)
    }
}

impl Mul<Phase> for i8 {
    type Output = Phase;

    fn mul(self, ph: Phase) -> Self::Output {
        Phase::from_int(self * ph.to_int())
    }
}

/// Identifier for a single one-qubit gate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum G1 {
    /// Hadamard
    H,
    /// π rotation about X
    X,
    /// π rotation about Y
    Y,
    /// π rotation about Z
    Z,
    /// π/2 rotation about Z
    S,
    /// -π/2 rotation about Z
    SInv,
}

impl G1 {
    /// Returns `true` if `self` is `H`.
    pub fn is_h(&self) -> bool { matches!(self, Self::H) }

    /// Returns `true` if `self` is `X`.
    pub fn is_x(&self) -> bool { matches!(self, Self::X) }

    /// Returns `true` if `self` is `Y`.
    pub fn is_y(&self) -> bool { matches!(self, Self::Y) }

    /// Returns `true` if `self` is `Z`.
    pub fn is_z(&self) -> bool { matches!(self, Self::Z) }

    /// Returns `true` if `self` is `S`.
    pub fn is_s(&self) -> bool { matches!(self, Self::S) }

    /// Returns `true` if `self` is `SInv`.
    pub fn is_sinv(&self) -> bool { matches!(self, Self::SInv) }
}

/// Identifier for a single two-qubit gate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum G2 {
    /// Z-controlled π rotation about X.
    CX,
    /// Z-controlled π rotation about Z.
    CZ,
    /// Swap
    Swap,
}

impl G2 {
    /// Returns `true` if `self` is `CX`.
    pub fn is_cx(&self) -> bool { matches!(self, Self::CX) }

    /// Returns `true` if `self` is `CZ`.
    pub fn is_cz(&self) -> bool { matches!(self, Self::CZ) }

    /// Returns `true` if `self` is `Swap`.
    pub fn is_swap(&self) -> bool { matches!(self, Self::Swap) }
}

/// Identifier for a single gate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum G {
    /// A one-qubit gate.
    Q1(G1),
    /// A two-qubit gate.
    Q2(G2),
}

impl From<G1> for G {
    fn from(kind1: G1) -> Self { Self::Q1(kind1) }
}

impl From<G2> for G {
    fn from(kind2: G2) -> Self { Self::Q2(kind2) }
}

impl G {
    /// Returns `true` if `self` is `Q1`.
    pub fn is_q1(&self) -> bool { matches!(self, Self::Q1(..)) }

    /// Returns `true` if `self` is `H`.
    pub fn is_h(&self) -> bool { matches!(self, Self::Q1(g) if g.is_h()) }

    /// Returns `true` if `self` is `X`.
    pub fn is_x(&self) -> bool { matches!(self, Self::Q1(g) if g.is_x()) }

    /// Returns `true` if `self` is `Y`.
    pub fn is_y(&self) -> bool { matches!(self, Self::Q1(g) if g.is_y()) }

    /// Returns `true` if `self` is `Z`.
    pub fn is_z(&self) -> bool { matches!(self, Self::Q1(g) if g.is_z()) }

    /// Returns `true` if `self` is `S`.
    pub fn is_s(&self) -> bool { matches!(self, Self::Q1(g) if g.is_s()) }

    /// Returns `true` if `self` is `SInv`.
    pub fn is_sinv(&self) -> bool { matches!(self, Self::Q1(g) if g.is_sinv()) }

    /// Returns `true` if `self` is `Q2`.
    pub fn is_q2(&self) -> bool { matches!(self, Self::Q2(..)) }

    /// Returns `true` if `self` is `CX`.
    pub fn is_cx(&self) -> bool { matches!(self, Self::Q2(g) if g.is_cx()) }

    /// Returns `true` if `self` is `CZ`.
    pub fn is_cz(&self) -> bool { matches!(self, Self::Q2(g) if g.is_cz()) }

    /// Returns `true` if `self` is `Swap`.
    pub fn is_swap(&self) -> bool { matches!(self, Self::Q2(g) if g.is_swap()) }
}

/// Describes the general behavior for a gate identifier token, e.g. [`G1`] and
/// [`G2`].
pub trait GateToken {
    /// Data required to construct a full [`Gate`].
    type Arg;

    /// Construct a [`Gate`].
    fn make(&self, arg: Self::Arg) -> Gate;
}

impl GateToken for G1 {
    type Arg = usize;

    fn make(&self, arg: usize) -> Gate {
        match *self {
            Self::H => Gate::H(arg),
            Self::X => Gate::X(arg),
            Self::Y => Gate::Y(arg),
            Self::Z => Gate::Z(arg),
            Self::S => Gate::S(arg),
            Self::SInv => Gate::SInv(arg),
        }
    }
}

impl GateToken for G2 {
    type Arg = (usize, usize);

    fn make(&self, arg: (usize, usize)) -> Gate {
        match *self {
            Self::CX => Gate::CX(arg.0, arg.1),
            Self::CZ => Gate::CZ(arg.0, arg.1),
            Self::Swap => Gate::Swap(arg.0, arg.1),
        }
    }
}

/// Description of a single gate for a register of qubits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Gate {
    /// Hadamard
    H(usize),
    /// π rotation about X
    X(usize),
    /// π rotation about Y
    Y(usize),
    /// π rotation about Z
    Z(usize),
    /// π/2 rotation about Z
    S(usize),
    /// -π/2 rotation about Z
    SInv(usize),
    /// Z-controlled π rotation about X.
    ///
    /// The first qubit index is the control.
    CX(usize, usize),
    /// Z-controlled π rotation about Z.
    ///
    /// The first qubit index is the control.
    CZ(usize, usize),
    /// Swap
    Swap(usize, usize),
}

impl Gate {
    /// Return `true` if `self` is `H`.
    pub fn is_h(&self) -> bool { matches!(self, Self::H(..)) }

    /// Return `true` if `self` is `X`.
    pub fn is_x(&self) -> bool { matches!(self, Self::X(..)) }

    /// Return `true` if `self` is `Y`.
    pub fn is_y(&self) -> bool { matches!(self, Self::Y(..)) }

    /// Return `true` if `self` is `Z`.
    pub fn is_z(&self) -> bool { matches!(self, Self::Z(..)) }

    /// Return `true` if `self` is `S`.
    pub fn is_s(&self) -> bool { matches!(self, Self::S(..)) }

    /// Return `true` if `self` is `SInv`.
    pub fn is_sinv(&self) -> bool { matches!(self, Self::SInv(..)) }

    /// Return `true` if `self` is `CX`.
    pub fn is_cx(&self) -> bool { matches!(self, Self::CX(..)) }

    /// Return `true` if `self` is `CZ`.
    pub fn is_cz(&self) -> bool { matches!(self, Self::CZ(..)) }

    /// Return `true` if `self` is `Swap`.
    pub fn is_swap(&self) -> bool { matches!(self, Self::Swap(..)) }

    /// Return the [kind][G] of `self`.
    pub fn kind(&self) -> G {
        use G::*;
        use G1::*;
        use G2::*;
        match *self {
            Self::H(..) => Q1(H),
            Self::X(..) => Q1(X),
            Self::Y(..) => Q1(Y),
            Self::Z(..) => Q1(Z),
            Self::S(..) => Q1(S),
            Self::SInv(..) => Q1(SInv),
            Self::CX(..) => Q2(CX),
            Self::CZ(..) => Q2(CZ),
            Self::Swap(..) => Q2(Swap),
        }
    }

    /// Sample a random single-qubit gate (`H`, `X`, `Y`, `Z`, or `S`) for a
    /// given qubit index.
    pub fn sample_single<R>(idx: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        match rng.gen_range(0..6_usize) {
            0 => Self::H(idx),
            1 => Self::X(idx),
            2 => Self::Y(idx),
            3 => Self::Z(idx),
            4 => Self::S(idx),
            5 => Self::SInv(idx),
            _ => unreachable!(),
        }
    }

    /// Sample a random gate uniformly from a set.
    pub fn sample<'a, I, G, R>(kinds: I, arg: G::Arg, rng: &mut R) -> Self
    where
        I: IntoIterator<Item = &'a G>,
        G: GateToken + 'a,
        R: Rng + ?Sized
    {
        let kinds: Vec<&G> = kinds.into_iter().collect();
        let n = kinds.len();
        kinds[rng.gen_range(0..n)].make(arg)
    }

    /// Return `true` if `self` and `other` are inverses.
    pub fn is_inv(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::H(a), Self::H(b)) if a == b => true,
            (Self::X(a), Self::X(b)) if a == b => true,
            (Self::Y(a), Self::Y(b)) if a == b => true,
            (Self::Z(a), Self::Z(b)) if a == b => true,
            (Self::S(a), Self::SInv(b)) if a == b => true,
            (Self::CX(ca, ta), Self::CX(cb, tb)) if ca == cb && ta == tb => true,
            (Self::CZ(ca, ta), Self::CZ(cb, tb))
                if (ca == cb && ta == tb) || (ca == tb && cb == ta) => true,
            (Self::Swap(aa, ba), Self::Swap(ab, bb))
                if (aa == ba && ab == bb) || (aa == bb && ab == ba) => true,
            _ => false,
        }
    }

    /// Shift qubit indices by `d`.
    pub fn shift(&mut self, d: usize) {
        match self {
            Self::H(a) => { *a += d; },
            Self::X(a) => { *a += d; },
            Self::Y(a) => { *a += d; },
            Self::Z(a) => { *a += d; },
            Self::S(a) => { *a += d; },
            Self::SInv(a) => { *a += d; },
            Self::CX(c, t) => { *c += d; *t += d; },
            Self::CZ(c, t) => { *c += d; *t += d; },
            Self::Swap(a, b) => { *a += d; *b += d; },
        }
    }

    /// Shift qubit indices by `d`, modulo `m`.
    pub fn shift_mod(&mut self, d: usize, m: usize) {
        match self {
            Self::H(a) => { *a += d; *a %= m; },
            Self::X(a) => { *a += d; *a %= m; },
            Self::Y(a) => { *a += d; *a %= m; },
            Self::Z(a) => { *a += d; *a %= m; },
            Self::S(a) => { *a += d; *a %= m; },
            Self::SInv(a) => { *a += d; *a %= m; },
            Self::CX(c, t) => { *c += d; *c %= m; *t += d; *t %= m; },
            Self::CZ(c, t) => { *c += d; *c %= m; *t += d; *t %= m; },
            Self::Swap(a, b) => { *a += d; *a %= m; *b += d; *b %= m; },
        }
    }

}

/// A single-qubit Pauli operator.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Pauli {
    /// Identity
    I,
    /// σ<sub>*x*</sub>
    X,
    /// σ<sub>*y*</sub>
    Y,
    /// σ<sub>*z*</sub>
    Z,
}

impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::I => write!(f, "{}", if f.alternate() { "." } else { "I" }),
            _ => write!(f, "{:?}", self),
        }
    }
}

impl Pauli {
    /// Draw a single Pauli operator uniformly at random.
    pub fn gen<R>(rng: &mut R) -> Self
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

    /// Return `true` if `self` commutes with `other`.
    pub fn commutes_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I, _) | (_, Self::I) => true,
            (l, r) => l == r,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NPauli(bool, Vec<Pauli>);

impl NPauli {
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
        loop {
            other = Self::gen(n, rng);
            if !self.commutes_with(&other, skip) {
                return other;
            }
        }
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

    fn set(&mut self, a: &NPauli, b: &NPauli) {
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

/// A series of [`Gate`]s implementing an element of the *N*-qubit Clifford
/// group.
///
/// All gates sourced from this type are guaranteed to apply to qubit indices
/// less than the output of [`Clifford::n`], and all two-qubit gate indices are
/// guaranteed to be non-equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clifford {
    n: usize,
    gates: Vec<Gate>,
}

impl IntoIterator for Clifford {
    type Item = Gate;
    type IntoIter = <Vec<Gate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.gates.into_iter() }
}

impl<'a> IntoIterator for &'a Clifford {
    type Item = &'a Gate;
    type IntoIter = <&'a Vec<Gate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.gates.iter() }
}

impl Clifford {
    /// Convert a series of gates to a new `n`-qubit Clifford element, verifying
    /// that all qubit indices are less than `n` and that all two-qubit gate
    /// indices are non-equal.
    ///
    /// If the above conditions do not hold, all gates are returned in a new
    /// vector as `Err`.
    pub fn new<I>(n: usize, gates: I) -> Result<Self, Vec<Gate>>
    where I: IntoIterator<Item = Gate>
    {
        let gates: Vec<Gate> = gates.into_iter().collect();
        if gates.iter()
            .all(|gate| {
                match gate {
                    Gate::H(k)
                    | Gate::X(k)
                    | Gate::Y(k)
                    | Gate::Z(k)
                    | Gate::S(k)
                    | Gate::SInv(k)
                    => *k < n,
                    Gate::CX(a, b)
                    | Gate::CZ(a, b)
                    | Gate::Swap(a, b)
                    => *a < n && *b < n && a != b,
                }
            })
        {
            Ok(Self { n, gates })
        } else {
            Err(gates)
        }
    }

    /// Return the number of qubits.
    pub fn n(&self) -> usize { self.n }

    /// Return the number of gates.
    pub fn len(&self) -> usize { self.gates.len() }

    /// Return `true` if the number of gates is zero.
    pub fn is_empty(&self) -> bool { self.gates.is_empty() }

    /// Return an iterator over the gates implementing the Clifford group
    /// element.
    pub fn iter(&self) -> std::slice::Iter<'_, Gate> { self.gates.iter() }

    pub fn unpack(self) -> (Vec<Gate>, usize) { (self.gates, self.n) }

    pub fn gen<R>(n: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let mut stab_a: NPauli;
        let mut stab_b: NPauli;
        let mut tab = Tableau::new(n);
        for llim in 0..n {
            stab_a = NPauli::gen(n, rng);
            while stab_a.1.iter().skip(llim).all(|p| *p == Pauli::I) {
                stab_a = NPauli::gen(n, rng);
            }
            stab_b = stab_a.sample_anticomm(Some(llim), rng);
            tab.set(&stab_a, &stab_b);
            tab.sweep(llim);
        }
        Self { n, gates: reduce(tab.gates) }
    }
}

fn reduce(gates: Vec<Gate>) -> Vec<Gate> {
    let mut reduced: Vec<Gate> = Vec::with_capacity(gates.len());
    for gate in gates.into_iter() {
        if let Some(g) = reduced.last() {
            if g.is_inv(&gate) {
                reduced.pop();
            } else {
                reduced.push(gate);
            }
        } else {
            reduced.push(gate);
        }
    }
    reduced
}

/// Specify the basis in which to perform a projective measurement.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Basis {
    /// X-basis
    X,
    /// Y-basis
    Y,
    /// Z-basis
    Z,
}

impl Basis {
    /// Return the possible outcomes for a given measurement basis, with the
    /// plus state first.
    pub(crate) fn outcomes(self) -> (Qubit, Qubit) {
        match self {
            Self::X => (Qubit::Xp, Qubit::Xm),
            Self::Y => (Qubit::Yp, Qubit::Ym),
            Self::Z => (Qubit::Zp, Qubit::Zm),
        }
    }
}

