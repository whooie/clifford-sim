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
use nalgebra as na;
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

/// Description of a single gate for a register of `N` qubits.
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

    // shift qubit indices by `d`
    pub(crate) fn shift(&mut self, d: usize) {
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

    // shift qubit indices by `d`, modulo `m`
    pub(crate) fn shift_mod(&mut self, d: usize, m: usize) {
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
    fn commutes_with(self, other: Self) -> bool {
        match (self, other) {
            (_, Self::I) => true,
            (Self::I, _) => true,
            (a, b) if a == b => true,
            _ => false,
        }
    }

    fn from_int(u: usize) -> Self {
        match u % 4 {
            0 => Self::I,
            1 => Self::X,
            2 => Self::Y,
            3 => Self::Z,
            _ => unreachable!(),
        }
    }

    fn gen_nqubit<R>(n: usize, rng: &mut R) -> Vec<Self>
    where R: Rng + ?Sized
    {
        (0..n).map(|_| Self::from_int(rng.gen_range(0..4))).collect()
    }

    fn gen_anticomm<R>(&mut self, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let r: bool = rng.gen();
        match self {
            Self::I => {
                *self = Self::from_int(rng.gen_range(1..=3));
                self.gen_anticomm(rng)
            },
            Self::X => if r { Self::Y } else { Self::Z },
            Self::Y => if r { Self::X } else { Self::Z },
            Self::Z => if r { Self::X } else { Self::Y },
        }
    }
}

/// A series of [`Gate`]s implementing a element of the `N`-qubit Clifford
/// group.
///
/// All gates sourced from this type are guaranteed to apply to qubit indices
/// less than or equal to `N` and all two-qubit gate indices are guaranteed to
/// be non-equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clifford(Vec<Gate>, usize);

impl IntoIterator for Clifford {
    type Item = Gate;
    type IntoIter = <Vec<Gate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

impl<'a> IntoIterator for &'a Clifford {
    type Item = &'a Gate;
    type IntoIter = <&'a Vec<Gate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.iter() }
}

impl Clifford {
    /// Convert a series of gates to a new `n`-qubit Clifford circuit, verifying
    /// that all qubit indices are less than or equal to `n` and that all
    /// two-qubit gate indices are non-equal.
    ///
    /// If the above conditions do not hold, all gates are returned in a new
    /// vector.
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
                    => *k <= n,
                    Gate::CX(a, b)
                    | Gate::CZ(a, b)
                    | Gate::Swap(a, b)
                    => *a <= n && *b <= n && a != b,
                }
            })
        {
            Ok(Self(gates, n))
        } else {
            Err(gates)
        }
    }

    /// Unpack `self` into a bare sequence of [`Gate`]s and the number of
    /// qubits.
    #[inline]
    pub fn unpack(self) -> (Vec<Gate>, usize) { (self.0, self.1) }

    /// Return the number of qubits.
    pub fn n(&self) -> usize { self.1 }

    /// Return the number of gates.
    pub fn len(&self) -> usize { self.0.len() }

    /// Return `true` if the number of gates is zero.
    pub fn is_empty(&self) -> bool { self.0.is_empty() }

    /// Generates a random element of the `n`-qubit Clifford group as a
    /// particular sequence of [`Gate`]s.
    pub fn gen<R>(n: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        // See arXiv:2008.06011v4 for details on the algorithm and
        // arXiv:0406196v5 for background.
        //
        // The `N`-qubit Clifford group is defined as the set of all operations
        // that transform any tensor product of single-qubit Pauli matrices to
        // another product of Pauli matrices. Broadly, it therefore suffices to
        // uniformly sample two `N`-qubit Pauli matrices and find the set of
        // gates required to transform both to an equivalent canonical form. The
        // following procedure is proven to guarantee uniform sampling over the
        // entire Clifford group.

        let mut p0: Vec<Pauli>;
        let mut p1: Vec<Pauli>;
        let mut p: Vec<Pauli>;
        let mut tab: Tableau = Tableau::new(n);
        let mut idx_scratch: Vec<usize> = Vec::new();
        for llim in 0..n {
            // init
            p0 = Pauli::gen_nqubit(n, rng);
            while p0.iter().all(|p| *p == Pauli::I) {
                p0 = Pauli::gen_nqubit(n, rng);
            }
            p1 = {
                p = Pauli::gen_nqubit(n, rng);
                let n_anti_comm
                    = p0.iter().zip(&p).skip(llim)
                    .filter(|(p0k, pk)| !p0k.commutes_with(**pk))
                    .count();
                if n_anti_comm % 2 == 0 {
                    let k = rng.gen_range(llim..n);
                    p[k] = p0[k].gen_anticomm(rng);
                }
                p
            };
            tab.init_with(&p0, &p1, &[rng.gen(), rng.gen()]);

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
            step_12!(tab, llim, idx_scratch, a);

            // (3)
            // move the remaining x in the top row to the leftmost column
            if let Some(j) = idx_scratch.first() {
                if *j != llim { tab.swap(*j, llim); }
                idx_scratch.pop();
            }

            // (4)
            // apply a hadamard if p1 != Z1.I.I...
            if !tab.tabz[llim].b
                || tab.tabx[llim].b
                || tab.iter_xz().skip(llim + 1).any(|(txj, tzj)| txj.b || tzj.b)
            {
                tab.h(llim);
                // repeat (1) and (2) above for the bottom row
                step_12!(tab, llim, idx_scratch, b);
                tab.h(llim);
            }

            // (5)
            // clear signs
            match tab.sign {
                TabCol { a: false, b: false } => { },
                TabCol { a: false, b: true  }
                    => { tab.circuit.push(Gate::X(llim)); },
                TabCol { a: true,  b: true  }
                    => { tab.circuit.push(Gate::Y(llim)); },
                TabCol { a: true,  b: false }
                    => { tab.circuit.push(Gate::Z(llim)); },
            }
        }
        Self(reduce(tab.unpack()), n)
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

#[derive(Copy, Clone, Debug)]
struct TabCol {
    a: bool,
    b: bool,
}

#[derive(Clone, Debug)]
struct Tableau {
    n: usize,
    tabx: Vec<TabCol>,
    tabz: Vec<TabCol>,
    sign: TabCol,
    // tabx: Vec<[bool; 2]>,
    // tabz: Vec<[bool; 2]>,
    // sign: [bool; 2],
    circuit: Vec<Gate>,
}

impl Tableau {
    fn new(n: usize) -> Self {
        Self {
            n,
            tabx: vec![TabCol { a: false, b: false }; n],
            tabz: vec![TabCol { a: false, b: false }; n],
            sign: TabCol { a: false, b: false },
            circuit: Vec::new(),
        }
    }

    fn is_normal(&self, llim: usize) -> bool {
        self.tabx[0].a && !self.tabx[0].b
            && !self.tabz[0].a && self.tabz[0].b
            && self.tabx.iter().skip(llim + 1).all(|txj| !txj.a && !txj.b)
            && self.tabz.iter().skip(llim + 1).all(|tzj| !tzj.a && !tzj.b)
    }

    fn init_with(&mut self, p0: &[Pauli], p1: &[Pauli], sign: &[bool; 2])
    {
        let iter
            = p0.iter().zip(p1)
            .zip(self.tabx.iter_mut().zip(self.tabz.iter_mut()));
        for ((p0j, p1j), (txj, tzj)) in iter {
            match p0j {
                Pauli::I => { },
                Pauli::X => { txj.a = true; },
                Pauli::Y => { txj.a = true; tzj.a = true; },
                Pauli::Z => { tzj.a = true; },
            }
            match p1j {
                Pauli::I => { },
                Pauli::X => { txj.b = true; },
                Pauli::Y => { txj.b = true; tzj.b = true; },
                Pauli::Z => { tzj.b = true; },
            }
        }
        self.sign.a = sign[0];
        self.sign.b = sign[1];
    }

    #[inline]
    fn iter_xz(&self) -> impl Iterator<Item = (&TabCol, &TabCol)> + '_
    {
        self.tabx.iter().zip(self.tabz.iter())
    }

    #[inline]
    fn h(&mut self, j: usize) {
        std::mem::swap(&mut self.tabx[j], &mut self.tabz[j]);
        self.circuit.push(Gate::H(j));
    }

    #[inline]
    fn s(&mut self, j: usize) {
        self.tabz[j].a ^= self.tabx[j].a;
        self.tabz[j].b ^= self.tabx[j].b;
        self.circuit.push(Gate::S(j));
    }

    #[inline]
    fn cnot(&mut self, c: usize, j: usize) {
        self.tabx[j].a ^= self.tabx[c].a;
        self.tabx[j].b ^= self.tabx[c].b;
        self.tabz[c].a ^= self.tabz[j].a;
        self.tabz[c].b ^= self.tabz[j].b;
        self.circuit.push(Gate::CX(c, j));
    }

    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        self.tabx.swap(a, b);
        self.tabz.swap(a, b);
        self.circuit.push(Gate::Swap(a, b));
    }

    fn unpack(self) -> Vec<Gate> { self.circuit }
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

/// A member of the `N`-qubit Pauli group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NPauli(pub Phase, pub Vec<Pauli>);

impl NPauli {
    /// Create a new `N`-qubit Pauli operator.
    pub fn new<I>(phase: Phase, paulis: I) -> Self
    where I: IntoIterator<Item = Pauli>
    {
        Self(phase, paulis.into_iter().collect())
    }

    /// Return the number of qubits that `self` acts on.
    pub fn n(&self) -> usize { self.1.len() }

    /// Sample a random `n`-qubit Pauli operator from the uniform distribution.
    pub fn gen<R>(n: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let phase = Phase::from_int(rng.gen_range(0..4));
        let paulis: Vec<Pauli>
            = (0..n)
            .map(|_| Pauli::from_int(rng.gen_range(0..4)))
            .collect();
        Self(phase, paulis)
    }
}

/// A series of [`Gate`]s implementing a element of the `N`-qubit Clifford
/// group.
///
/// All gates sourced from this type are guaranteed to apply to qubit indices
/// less than or equal to `N` and all two-qubit gate indices are guaranteed to
/// be non-equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clifford2(Vec<Gate>, usize);

impl IntoIterator for Clifford2 {
    type Item = Gate;
    type IntoIter = <Vec<Gate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

impl<'a> IntoIterator for &'a Clifford2 {
    type Item = &'a Gate;
    type IntoIter = <&'a Vec<Gate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.iter() }
}

impl Clifford2 {
    /// Convert a series of gates to a new `n`-qubit Clifford2 circuit, verifying
    /// that all qubit indices are less than or equal to `n` and that all
    /// two-qubit gate indices are non-equal.
    ///
    /// If the above conditions do not hold, all gates are returned in a new
    /// vector.
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
                    => *k <= n,
                    Gate::CX(a, b)
                    | Gate::CZ(a, b)
                    | Gate::Swap(a, b)
                    => *a <= n && *b <= n && a != b,
                }
            })
        {
            Ok(Self(gates, n))
        } else {
            Err(gates)
        }
    }

    /// Unpack `self` into a bare sequence of [`Gate`]s and the number of
    /// qubits.
    #[inline]
    pub fn unpack(self) -> (Vec<Gate>, usize) { (self.0, self.1) }

    /// Return the number of qubits.
    pub fn n(&self) -> usize { self.1 }

    /// Return the number of gates.
    pub fn len(&self) -> usize { self.0.len() }

    /// Return `true` if the number of gates is zero.
    pub fn is_empty(&self) -> bool { self.0.is_empty() }

    /// Generates a random element of the `n`-qubit Clifford2 group as a
    /// particular sequence of [`Gate`]s.
    pub fn gen<R>(n: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        // init
        let mut delta: na::DMatrix<bool>
            = na::DMatrix::from_element(n, n, false);
        delta.fill_diagonal(true);
        let mut deltap = delta.clone();
        let mut gamma: na::DMatrix<bool>
            = na::DMatrix::from_element(n, n, false);
        let mut gammap = gamma.clone();

        // sample hadamards and permutation from the quantum mallows distribution
        let (h, mut s) = sample_qmallows(n, rng);

        // fill out gamma and delta matrices
        let mut b: bool;
        for i in 0..n {
            gammap[(i, i)] = rng.gen();
            if h[i] { gamma[(i, i)] = rng.gen(); }
        }
        for j in 0..n {
            for i in j + 1..n {
                b = rng.gen();
                gammap[(i, j)] = b;
                gammap[(j, i)] = b;
                deltap[(i, j)] = rng.gen();

                if h[i] && h[j] {
                    b = rng.gen();
                    gamma[(i, j)] = b;
                    gamma[(j, i)] = b;
                }
                if h[i] && !h[j] && s[i] < s[j] {
                    b = rng.gen();
                    gamma[(i, j)] = b;
                    gamma[(j, i)] = b;
                }
                if !h[i] && h[j] && s[i] > s[j] {
                    b = rng.gen();
                    gamma[(i, j)] = b;
                    gamma[(j, i)] = b;
                }
                if !h[i] && h[j] {
                    delta[(i, j)] = rng.gen();
                }
                if h[i] && h[j] && s[i] > s[j] {
                    delta[(i, j)] = rng.gen();
                }
                if !h[i] && !h[j] && s[i] < s[j] {
                    delta[(i, j)] = rng.gen();
                }
            }
        }

        // sample a random n-qubit pauli
        let pauli = NPauli::gen(n, rng);

        // assemble circuit
        let mut gates: Vec<Gate> = Vec::new();
        borel(n, Some(&pauli), &gammap, &deltap, &mut gates);
        permutation(n, &mut s, &mut gates);
        hadamard_layer(&h, &mut gates);
        borel(n, None, &gamma, &delta, &mut gates);
        Self(gates, n)
    }
}

fn sample_qmallows<R>(n: usize, rng: &mut R) -> (Vec<bool>, Vec<usize>)
where R: Rng + ?Sized
{
    let mut h: Vec<bool> = vec![false; n];
    let mut s: Vec<usize> = vec![0; n];
    let mut a: Vec<usize> = (0..n).collect();
    let mut m: usize;
    let mut r: f32;
    let mut idx: usize;
    let mut k: usize;
    for i in 0..n {
        m = n - i; // number of elements in `a`
        r = rng.gen();
        idx = (1.0 + (1.0 - r) * 4.0_f32.powi(-(m as i32)))
            .log2()
            .ceil() as usize;
        h[i] = idx < m;
        k = if idx < m { idx } else { 2 * m - idx - 1 };
        s[i] = a[k];
        a.remove(k);
    }
    (h, s)
}

fn borel(
    n: usize,
    pauli: Option<&NPauli>,
    gamma: &na::DMatrix<bool>,
    delta: &na::DMatrix<bool>,
    buf: &mut Vec<Gate>,
) {
    for (c, d_c) in delta.column_iter().enumerate().take(n - 1).rev() {
        for (t, dtc) in d_c.iter().enumerate().skip(c + 1).rev() {
            if *dtc { buf.push(Gate::CX(c, t)); }
        }
    }
    for (c, gc_) in gamma.row_iter().enumerate() {
        for (t, gct) in gc_.iter().enumerate().skip(c + 1) {
            if *gct { buf.push(Gate::CZ(c, t)); }
        }
    }
    for (i, gii) in gamma.diagonal().iter().enumerate() {
        if *gii { buf.push(Gate::S(i)); }
    }
    if let Some(NPauli(phase, paulis)) = pauli {
        for (k, p) in paulis.iter().enumerate() {
            match *p {
                Pauli::I => { continue; },
                Pauli::X => { buf.push(Gate::X(k)); }
                Pauli::Y => { buf.push(Gate::Y(k)); }
                Pauli::Z => { buf.push(Gate::Z(k)); }
            }
        }
        match *phase {
            Phase::Pi0 => { },
            Phase::Pi1h => {
                buf.push(Gate::S(0));
                buf.push(Gate::X(0));
                buf.push(Gate::S(0));
                buf.push(Gate::X(0));
            },
            Phase::Pi => {
                buf.push(Gate::Z(0));
                buf.push(Gate::X(0));
                buf.push(Gate::Z(0));
                buf.push(Gate::X(0));
            },
            Phase::Pi3h => {
                buf.push(Gate::SInv(0));
                buf.push(Gate::X(0));
                buf.push(Gate::SInv(0));
                buf.push(Gate::X(0));
            },
        }
    }
}

fn permutation(n: usize, s: &mut [usize], buf: &mut Vec<Gate>) {
    // implement a permutation operator with only swaps by finding the sequence
    // (of swaps) that sorts s and reversing
    let mut acc: Vec<Gate> = Vec::with_capacity(n - 1);
    let mut k: usize;
    for t in 0..n - 1 {
        if s[t] != t {
            k = s.iter()
                .enumerate()
                .skip(t + 1)
                .find_map(|(j, p)| (*p == t).then_some(j))
                .unwrap();
            s.swap(t, k);
            acc.push(Gate::Swap(t, k));
        }
    }
    acc.reverse();
    buf.append(&mut acc);
}

fn hadamard_layer(h: &[bool], buf: &mut Vec<Gate>) {
    for (k, hk) in h.iter().enumerate() {
        if *hk { buf.push(Gate::H(k)); }
    }
}

