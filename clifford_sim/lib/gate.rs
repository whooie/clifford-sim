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
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    /// 0
    Pi0,
    /// π/4
    Pi1q,
    /// π/2
    Pi1h,
    /// 3π/4
    Pi3q,
    /// π
    Pi,
    /// 5π/4
    Pi5q,
    /// 3π/2
    Pi3h,
    /// 7π/4
    Pi7q,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Pi0 => write!(f, "+1"),
            Self::Pi1q => write!(f, "+e^iπ/4"),
            Self::Pi1h => write!(f, "+i"),
            Self::Pi3q => write!(f, "+e^i3π/4"),
            Self::Pi => write!(f, "-1"),
            Self::Pi5q => write!(f, "+e^i5π/4"),
            Self::Pi3h => write!(f, "-i"),
            Self::Pi7q => write!(f, "+e^i7π/4"),
        }
    }
}

impl Phase {
    /// Convert to the bare multiple of π/4.
    pub fn to_int(&self) -> i8 {
        match self {
            Self::Pi0  => 0,
            Self::Pi1q => 1,
            Self::Pi1h => 2,
            Self::Pi3q => 3,
            Self::Pi   => 4,
            Self::Pi5q => 5,
            Self::Pi3h => 6,
            Self::Pi7q => 7,
        }
    }

    /// Convert from a bare multiple of π/4 (modulo 8).
    pub fn from_int(i: i8) -> Self {
        match i.rem_euclid(8) {
            0 => Self::Pi0,
            1 => Self::Pi1q,
            2 => Self::Pi1h,
            3 => Self::Pi3q,
            4 => Self::Pi,
            5 => Self::Pi5q,
            6 => Self::Pi3h,
            7 => Self::Pi7q,
            _ => unreachable!(),
        }
    }

    pub fn as_complex(self) -> C64 {
        use std::f64::consts::FRAC_PI_4 as PI4;
        match self {
            Self::Pi0  => 1.0_f64.into(),
            Self::Pi1q => C64::cis(PI4),
            Self::Pi1h => C64::i(),
            Self::Pi3q => C64::cis(3.0 * PI4),
            Self::Pi   => (-1.0_f64).into(),
            Self::Pi5q => C64::cis(5.0 * PI4),
            Self::Pi3h => -C64::i(),
            Self::Pi7q => C64::cis(7.0 * PI4),
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

/// Description of a single gate for a register of `N` qubits.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

    /// Return `true` if `self` is `CX`.
    pub fn is_cx(&self) -> bool { matches!(self, Self::CX(..)) }

    /// Return `true` if `self` is `Swap`.
    pub fn is_swap(&self) -> bool { matches!(self, Self::Swap(..)) }

    /// Sample a random single-qubit gate (`H`, `X`, `Y`, `Z`, or `S`) for a
    /// given qubit index.
    pub fn sample_single<R>(idx: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        match rng.gen_range(0..5_usize) {
            0 => Self::H(idx),
            1 => Self::X(idx),
            2 => Self::Y(idx),
            3 => Self::Z(idx),
            4 => Self::S(idx),
            _ => unreachable!(),
        }
    }
}

/// A single-qubit Pauli operator.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

    fn gen_nqubit<const N: usize, R>(rng: &mut R) -> [Self; N]
    where R: Rng + ?Sized
    {
        let mut p = [Self::I; N];
        p.iter_mut()
            .for_each(|pk| { *pk = Self::from_int(rng.gen_range(0..4)); });
        p
    }

    fn gen_nqubitd<R>(n: usize, rng: &mut R) -> Vec<Self>
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Clifford<const N: usize>(Vec<Gate>);

impl<const N: usize> IntoIterator for Clifford<N> {
    type Item = Gate;
    type IntoIter = <Vec<Gate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

impl<'a, const N: usize> IntoIterator for &'a Clifford<N> {
    type Item = &'a Gate;
    type IntoIter = <&'a Vec<Gate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.iter() }
}

impl<const N: usize> Clifford<N> {
    /// Convert a series of gates to a new `N`-qubit Clifford circuit, verifying
    /// that all qubit indices are less than or equal to `N` and that all
    /// two-qubit gate indices are non-equal.
    ///
    /// If the above conditions do not hold, all gates are returned in a new
    /// vector.
    pub fn new<I>(gates: I) -> Result<Self, Vec<Gate>>
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
                    => *k <= N,
                    Gate::CX(a, b)
                    | Gate::CZ(a, b)
                    | Gate::Swap(a, b)
                    => *a <= N && *b <= N && a != b,
                }
            })
        {
            Ok(Self(gates))
        } else {
            Err(gates)
        }
    }

    pub fn len(&self) -> usize { self.0.len() }

    pub fn is_empty(&self) -> bool { self.0.is_empty() }

    /// Generates a random element of the `N`-qubit Clifford group as a
    /// particular sequence of [`Gate`]s.
    pub fn gen<R>(rng: &mut R) -> Self
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

        let mut p0: [Pauli; N];
        let mut p1: [Pauli; N];
        let mut p: [Pauli; N];
        let mut tab: Tableau<N> = Tableau::new();
        let mut idx_scratch: Vec<usize> = Vec::new();
        for llim in 0..N {
            // init
            p0 = Pauli::gen_nqubit(rng);
            p1 = {
                p = Pauli::gen_nqubit(rng);
                let n_anti_comm
                    = p0.iter().zip(&p).skip(llim)
                    .filter(|(p0k, pk)| !p0k.commutes_with(**pk))
                    .count();
                if n_anti_comm % 2 == 0 {
                    *p.last_mut().unwrap()
                        = p0.last_mut().unwrap().gen_anticomm(rng);
                }
                p
            };
            // p1 = loop { // make sure p0 and p1 anti-commute
            //     p = Pauli::gen_nqubit(rng);
            //     let n_anti_comm
            //         = p0.iter().zip(&p).skip(llim)
            //         .filter(|(p0k, pk)| !p0k.commutes_with(**pk))
            //         .count();
            //     if n_anti_comm % 2 == 1 {
            //         break p;
            //     } else { continue; }
            // };
            tab.init_with(&p0, &p1, &[rng.gen(), rng.gen()]);

            macro_rules! step_12 {
                ( $tab:ident, $llim:ident, $idx_scratch:ident, $row:literal )
                => {
                    // (1)
                    // clear top row of z: H
                    $tab.iter_xz().enumerate().skip($llim)
                        .filter(|(_, (txj, tzj))| tzj[$row] && !txj[$row])
                        .for_each(|(j, _)| { $idx_scratch.push(j); });
                    $idx_scratch.drain(..)
                        .for_each(|j| { $tab.h(j); });
                    // clear top row of z: S
                    $tab.iter_xz().enumerate().skip($llim)
                        .filter(|(_, (txj, tzj))| tzj[$row] && txj[$row])
                        .for_each(|(j, _)| { $idx_scratch.push(j); });
                    $idx_scratch.drain(..)
                        .for_each(|j| { $tab.s(j); });

                    // (2)
                    // clear top row of x, all but one: CNOTs
                    $tab.iter_xz().enumerate().skip($llim)
                        .filter(|(_, (txj, _))| txj[$row]) // guaranteed at least 1 such
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
            step_12!(tab, llim, idx_scratch, 0);

            // (3)
            // move the remaining x in the top row to the leftmost column
            if let Some(j) = idx_scratch.first() {
                if *j != llim { tab.swap(*j, llim); }
                idx_scratch.pop();
            }

            // (4)
            // apply a hadamard if p1 != Z1.I.I...
            if tab.tabx[llim][1]
                || !tab.tabz[llim][1]
                || tab.iter_xz().any(|(txj, tzj)| txj[1] || tzj[1])
            {
                tab.h(llim);
                // repeat (1) and (2) above for the bottom row
                step_12!(tab, llim, idx_scratch, 1);
            }
            tab.h(llim);

            // (5)
            // clear signs
            match tab.sign {
                [false, false] => { },
                [false, true ] => { tab.circuit.push(Gate::X(llim)); },
                [true,  false] => { tab.circuit.push(Gate::Z(llim)); },
                [true,  true ] => { tab.circuit.push(Gate::Y(llim)); },
            }
        }
        Self(tab.unpack())
    }
}

#[derive(Clone, Debug)]
struct Tableau<const N: usize> {
    tabx: [[bool; 2]; N],
    tabz: [[bool; 2]; N],
    sign: [bool; 2],
    circuit: Vec<Gate>,
}

impl<const N: usize> Default for Tableau<N> {
    fn default() -> Self { Self::new() }
}

impl<const N: usize> Tableau<N> {
    fn new() -> Self {
        Self {
            tabx: [[false; 2]; N],
            tabz: [[false; 2]; N],
            sign: [false; 2],
            circuit: Vec::new(),
        }
    }

    fn is_normal(&self, llim: usize) -> bool {
        self.tabx[0][0] && !self.tabx[0][1]
            && !self.tabz[0][0] && self.tabz[0][1]
            && self.tabx.iter().skip(llim + 1).all(|txj| !txj[0] && !txj[1])
            && self.tabz.iter().skip(llim + 1).all(|tzj| !tzj[0] && !tzj[1])
    }

    fn init_with(&mut self, p0: &[Pauli; N], p1: &[Pauli; N], sign: &[bool; 2])
    {
        let iter
            = p0.iter().zip(p1)
            .zip(self.tabx.iter_mut().zip(self.tabz.iter_mut()));
        for ((p0j, p1j), (txj, tzj)) in iter {
            match p0j {
                Pauli::I => { },
                Pauli::X => { txj[0] = true; },
                Pauli::Y => { txj[0] = true; tzj[0] = true; },
                Pauli::Z => { tzj[0] = true; },
            }
            match p1j {
                Pauli::I => { },
                Pauli::X => { txj[1] = true; },
                Pauli::Y => { txj[1] = true; tzj[1] = true; },
                Pauli::Z => { tzj[1] = true; },
            }
        }
        self.sign[0] = sign[0];
        self.sign[1] = sign[1];
    }

    fn iter_xz(&self) -> impl Iterator<Item = (&[bool; 2], &[bool; 2])> + '_ {
        self.tabx.iter().zip(self.tabz.iter())
    }

    fn h(&mut self, j: usize) {
        std::mem::swap(&mut self.tabx[j], &mut self.tabz[j]);
        self.circuit.push(Gate::H(j));
    }

    fn s(&mut self, j: usize) {
        self.tabz[j][0] ^= self.tabx[j][0];
        self.tabz[j][1] ^= self.tabx[j][1];
        self.circuit.push(Gate::S(j));
    }

    fn cnot(&mut self, c: usize, j: usize) {
        self.tabx[j][0] ^= self.tabx[c][0];
        self.tabx[j][1] ^= self.tabx[c][1];
        self.tabz[c][0] ^= self.tabz[j][0];
        self.tabz[c][1] ^= self.tabz[j][1];
        self.circuit.push(Gate::CX(c, j));
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.tabx.swap(a, b);
        self.tabz.swap(a, b);
        self.circuit.push(Gate::Swap(a, b));
    }

    fn unpack(self) -> Vec<Gate> { self.circuit }
}

/// Like [`Clifford`], but for non-statically sized systems.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CliffordD(Vec<Gate>, usize);

impl IntoIterator for CliffordD {
    type Item = Gate;
    type IntoIter = <Vec<Gate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

impl<'a> IntoIterator for &'a CliffordD {
    type Item = &'a Gate;
    type IntoIter = <&'a Vec<Gate> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.iter() }
}

impl CliffordD {
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

    pub fn n(&self) -> usize { self.1 }

    pub fn len(&self) -> usize { self.0.len() }

    pub fn is_empty(&self) -> bool { self.0.is_empty() }

    /// Generates a random element of the `n`-qubit Clifford group as a
    /// particular sequence of [`Gate`]s.
    pub fn gen<R>(n: usize, rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        let mut p0: Vec<Pauli>;
        let mut p1: Vec<Pauli>;
        let mut p: Vec<Pauli>;
        let mut tab: TableauD = TableauD::new(n);
        let mut idx_scratch: Vec<usize> = Vec::new();
        for llim in 0..n {
            // init
            p0 = Pauli::gen_nqubitd(n, rng);
            p1 = {
                p = Pauli::gen_nqubitd(n, rng);
                let n_anti_comm
                    = p0.iter().zip(&p).skip(llim)
                    .filter(|(p0k, pk)| !p0k.commutes_with(**pk))
                    .count();
                if n_anti_comm % 2 == 0 {
                    *p.last_mut().unwrap()
                        = p0.last_mut().unwrap().gen_anticomm(rng);
                }
                p
            };
            // p1 = loop { // make sure p0 and p1 anti-commute
            //     p = Pauli::gen_nqubit(rng);
            //     let n_anti_comm
            //         = p0.iter().zip(&p).skip(llim)
            //         .filter(|(p0k, pk)| !p0k.commutes_with(**pk))
            //         .count();
            //     if n_anti_comm % 2 == 1 {
            //         break p;
            //     } else { continue; }
            // };
            tab.init_with(&p0, &p1, &[rng.gen(), rng.gen()]);

            macro_rules! step_12 {
                ( $tab:ident, $llim:ident, $idx_scratch:ident, $row:literal )
                => {
                    // (1)
                    // clear top row of z: H
                    $tab.iter_xz().enumerate().skip($llim)
                        .filter(|(_, (txj, tzj))| tzj[$row] && !txj[$row])
                        .for_each(|(j, _)| { $idx_scratch.push(j); });
                    $idx_scratch.drain(..)
                        .for_each(|j| { $tab.h(j); });
                    // clear top row of z: S
                    $tab.iter_xz().enumerate().skip($llim)
                        .filter(|(_, (txj, tzj))| tzj[$row] && txj[$row])
                        .for_each(|(j, _)| { $idx_scratch.push(j); });
                    $idx_scratch.drain(..)
                        .for_each(|j| { $tab.s(j); });

                    // (2)
                    // clear top row of x, all but one: CNOTs
                    $tab.iter_xz().enumerate().skip($llim)
                        .filter(|(_, (txj, _))| txj[$row]) // guaranteed at least 1 such
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
            step_12!(tab, llim, idx_scratch, 0);

            // (3)
            // move the remaining x in the top row to the leftmost column
            if let Some(j) = idx_scratch.first() {
                if *j != llim { tab.swap(*j, llim); }
                idx_scratch.pop();
            }

            // (4)
            // apply a hadamard if p1 != Z1.I.I...
            if tab.tabx[llim][1]
                || !tab.tabz[llim][1]
                || tab.iter_xz().any(|(txj, tzj)| txj[1] || tzj[1])
            {
                tab.h(llim);
                // repeat (1) and (2) above for the bottom row
                step_12!(tab, llim, idx_scratch, 1);
            }
            tab.h(llim);

            // (5)
            // clear signs
            match tab.sign {
                [false, false] => { },
                [false, true ] => { tab.circuit.push(Gate::X(llim)); },
                [true,  false] => { tab.circuit.push(Gate::Z(llim)); },
                [true,  true ] => { tab.circuit.push(Gate::Y(llim)); },
            }
        }
        Self(tab.unpack(), n)
    }
}

#[derive(Clone, Debug)]
struct TableauD {
    n: usize,
    tabx: Vec<[bool; 2]>,
    tabz: Vec<[bool; 2]>,
    sign: [bool; 2],
    circuit: Vec<Gate>,
}

impl TableauD {
    fn new(n: usize) -> Self {
        Self {
            n,
            tabx: vec![[false; 2]; n],
            tabz: vec![[false; 2]; n],
            sign: [false; 2],
            circuit: Vec::new(),
        }
    }

    fn is_normal(&self, llim: usize) -> bool {
        self.tabx[0][0] && !self.tabx[0][1]
            && !self.tabz[0][0] && self.tabz[0][1]
            && self.tabx.iter().skip(llim + 1).all(|txj| !txj[0] && !txj[1])
            && self.tabz.iter().skip(llim + 1).all(|tzj| !tzj[0] && !tzj[1])
    }

    fn init_with(&mut self, p0: &[Pauli], p1: &[Pauli], sign: &[bool; 2])
    {
        let iter
            = p0.iter().zip(p1)
            .zip(self.tabx.iter_mut().zip(self.tabz.iter_mut()));
        for ((p0j, p1j), (txj, tzj)) in iter {
            match p0j {
                Pauli::I => { },
                Pauli::X => { txj[0] = true; },
                Pauli::Y => { txj[0] = true; tzj[0] = true; },
                Pauli::Z => { tzj[0] = true; },
            }
            match p1j {
                Pauli::I => { },
                Pauli::X => { txj[1] = true; },
                Pauli::Y => { txj[1] = true; tzj[1] = true; },
                Pauli::Z => { tzj[1] = true; },
            }
        }
        self.sign[0] = sign[0];
        self.sign[1] = sign[1];
    }

    fn iter_xz(&self) -> impl Iterator<Item = (&[bool; 2], &[bool; 2])> + '_ {
        self.tabx.iter().zip(self.tabz.iter())
    }

    fn h(&mut self, j: usize) {
        std::mem::swap(&mut self.tabx[j], &mut self.tabz[j]);
        self.circuit.push(Gate::H(j));
    }

    fn s(&mut self, j: usize) {
        self.tabz[j][0] ^= self.tabx[j][0];
        self.tabz[j][1] ^= self.tabx[j][1];
        self.circuit.push(Gate::S(j));
    }

    fn cnot(&mut self, c: usize, j: usize) {
        self.tabx[j][0] ^= self.tabx[c][0];
        self.tabx[j][1] ^= self.tabx[c][1];
        self.tabz[c][0] ^= self.tabz[j][0];
        self.tabz[c][1] ^= self.tabz[j][1];
        self.circuit.push(Gate::CX(c, j));
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.tabx.swap(a, b);
        self.tabz.swap(a, b);
        self.circuit.push(Gate::Swap(a, b));
    }

    fn unpack(self) -> Vec<Gate> { self.circuit }
}

/// Specify the basis in which to perform a projective measurement.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

