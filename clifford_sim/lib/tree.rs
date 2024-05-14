//! Tree-based representation of variable-basis *N*-qubit register states.
//!
//! Here, *N*-qubit "register" states are a series of *N* positions on the Bloch
//! sphere, limited to the usual six cardinal directions: ∣±x⟩, ∣±y⟩, and ∣±z⟩.
//! General states are represented by a quasi-binary tree of these register
//! states which take advantage of the fact that, given a single Clifford
//! generator gate (Hadamard, π/2 phase, or CNOT) acting on a single register
//! (basis) state, the action of the gate will be to either apply a simple
//! rotation to a single qubit, or to create a Bell state. In the latter case, a
//! basis can always be chosen such that the Bell state can be written as the
//! sum of only two basis states. This is mirrored in how non-destructive
//! measurements act on pure states in the creation of mixed states.
//!
//! Thus a given [`Pure`] state is either a single *N*-qubit register state or a
//! superposition of two register states with a relative phase, and its depth in
//! a larger tree encodes the magnitude of both their amplitudes. Likewise, a
//! [`State`] is either `Pure` or an even (classical) mixture of two `Pure`s.
//!
//! Although states are not principally represented by a naive complex-valued
//! vector or matrix and gate operations are straightforwardly not performed via
//! matrix multiplication, the action of gates and measurements still grow the
//! encoding trees on average, which results in spatial and temporal runtime
//! requirements that are still super-polynomial, making this an *invalid
//! approach for efficient simulation*.

use std::{
    boxed::Box,
    hash::{ Hash, Hasher },
    ops::DerefMut,
};
use itertools::Itertools;
use nalgebra as na;
use num_complex::Complex64 as C64;
use rustc_hash::FxHasher;
use crate::gate::{ Gate, Basis, Phase };

/* Qubit **********************************************************************/

/// Base qubit object defined as the 6 cardinal directions on the Bloch sphere.
///
/// The ∣+z⟩ state corresponds to ∣0⟩.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Qubit {
    /// ∣+x⟩
    Xp,
    /// ∣–x⟩
    Xm,
    /// ∣+y⟩
    Yp,
    /// ∣–y⟩
    Ym,
    /// ∣+z⟩
    Zp,
    /// ∣–z⟩
    Zm,
}

/// Creates a [`Self::Zp`].
impl Default for Qubit {
    fn default() -> Self { Self::Zp }
}

impl std::fmt::Display for Qubit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Xp => write!(f, "+x"),
            Self::Xm => write!(f, "-x"),
            Self::Yp => write!(f, "+y"),
            Self::Ym => write!(f, "-y"),
            Self::Zp => write!(f, "+z"),
            Self::Zm => write!(f, "-z"),
        }
    }
}

impl Qubit {
    /// Return *z*-basis amplitudes, with +z ordered first.
    pub fn z_amps(self) -> [C64; 2] {
        use std::f64::consts::FRAC_1_SQRT_2;
        const ZERO:  C64 = C64 { re: 1.0, im: 0.0 };
        const ONE:   C64 = C64 { re: 0.0, im: 0.0 };
        const ORT2:  C64 = C64 { re: FRAC_1_SQRT_2, im: 0.0 };
        const iORT2: C64 = C64 { re: 0.0, im: FRAC_1_SQRT_2 };
        match self {
            Self::Xp => [ORT2,   ORT2 ],
            Self::Xm => [ORT2,  -ORT2 ],
            Self::Yp => [ORT2,   iORT2],
            Self::Ym => [ORT2,  -iORT2],
            Self::Zp => [ONE,    ZERO ],
            Self::Zm => [ZERO,   ONE  ],
        }
    }
}

/* Register *******************************************************************/

/// Newtype based on a fixed-sized array of [`Qubit`]s.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Register<const N: usize>([Qubit; N]);

/// Creates an `N`-qubit register of all [`Qubit::Zp`]s.
impl<const N: usize> Default for Register<N> {
    fn default() -> Self { Self([Qubit::default(); N]) }
}

impl<const N: usize> std::fmt::Display for Register<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (k, q) in self.0.iter().enumerate() {
            write!(f, "{}", q)?;
            if k < N - 1 { write!(f, ", ")?; }
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<const N: usize> AsRef<[Qubit; N]> for Register<N> {
    fn as_ref(&self) -> &[Qubit; N] { &self.0 }
}

impl<const N: usize> std::ops::Deref for Register<N> {
    type Target = [Qubit; N];

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<const N: usize, T> From<T> for Register<N>
where [Qubit; N]: From<T>
{
    fn from(t: T) -> Self { Self(t.into()) }
}

impl<const N: usize> IntoIterator for Register<N> {
    type Item = Qubit;
    type IntoIter = <[Qubit; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

impl<'a, const N: usize> IntoIterator for &'a Register<N> {
    type Item = &'a Qubit;
    type IntoIter = <&'a [Qubit; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter { self.0.iter() }
}

impl<const N: usize, I> std::ops::Index<I> for Register<N>
where [Qubit; N]: std::ops::Index<I>
{
    type Output = <[Qubit; N] as std::ops::Index<I>>::Output;

    fn index(&self, idx: I) -> &Self::Output {
        std::ops::Index::index(&self.0, idx)
    }
}

impl<const N: usize, I> std::ops::IndexMut<I> for Register<N>
where [Qubit; N]: std::ops::IndexMut<I>
{
    fn index_mut(&mut self, idx: I) -> &mut Self::Output {
        std::ops::IndexMut::index_mut(&mut self.0, idx)
    }
}

impl<const N: usize> Register<N> {
    /// Represent `self` as a complex-valued 1D array in the *z*-basis.
    ///
    /// The length of this array is guaranteed to be `2^N`.
    pub fn as_vector(&self) -> na::DVector<C64> {
        na::DVector::from_iterator(
            2_usize.pow(N as u32),
            self.0.iter()
                .map(|q| q.z_amps())
                .multi_cartesian_product()
                .map(|amps| amps.into_iter().product()),
        )
    }

    /// Represent `self` as a complex-valued 2D array in the *z*-basis.
    ///
    /// This array is guaranteed to be Hermitian and idempotent with shape `2^N
    /// × 2^N`.
    pub fn as_matrix(&self) -> na::DMatrix<C64> {
        let psi = self.as_vector();
        outer_prod(&psi, &psi)
    }

    fn ident(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/* Pure ***********************************************************************/

/// A pure state of an `N`-qubit register.
///
/// Constraint to only Clifford gates implies that for any such one- or
/// two-qubit rotation, an appropriate basis can be chosen such that the output
/// state can be written as a superposition of at most two terms. Hence, a pure
/// state is defined recursively here as a quasi-binary tree.
#[derive(Clone, Debug)]
pub enum Pure<const N: usize> {
    /// A physically impossible null state.
    Null,
    /// A pure state comprising only a single register state.
    Single(Register<N>),
    /// A pure state comprising the (even) superposition of two pure states
    /// with a relative phase defined as that of the second with respect to the
    /// first.
    Superpos(Box<Pure<N>>, Box<Pure<N>>, Phase),
}

/// Initialize a register as all [`Qubit::Zp`].
impl<const N: usize> Default for Pure<N> {
    fn default() -> Self { Self::new() }
}

impl<const N: usize> From<Register<N>> for Pure<N> {
    fn from(reg: Register<N>) -> Self { Self::Single(reg) }
}

impl<const N: usize> Pure<N> {
    /// Create a new pure state as a single register state initialized to all
    /// [`Qubit::Zp`].
    pub fn new() -> Self { Self::Single(Register::default()) }

    /// Return `true` if `self` is `Null`.
    pub fn is_null(&self) -> bool { matches!(self, Self::Null) }

    /// Return `true` if `self` is `Single`.
    pub fn is_single(&self) -> bool { matches!(self, Self::Single(_)) }

    /// Return `true` if `self` is `Superpos`.
    pub fn is_superpos(&self) -> bool { matches!(self, Self::Superpos(..)) }

    /// Return the number of terms in the superposition.
    ///
    /// Note that this sum may count some register states twice.
    pub fn num_terms(&self) -> usize {
        match self {
            Self::Null => 0,
            Self::Single(_) => 1,
            Self::Superpos(l, r, _) => l.num_terms() + r.num_terms(),
        }
    }

    /// Return a vector of all register states with associated amplitudes.
    pub fn terms(&self) -> Vec<(&Register<N>, C64)> {
        fn do_iter<'b, const M: usize>(
            state: &'b Pure<M>,
            depth: i32,
            phase: Phase,
            acc: &mut Vec<(&'b Register<M>, C64)>,
        ) {
            use std::f64::consts::FRAC_1_SQRT_2 as ORT2;
            match state {
                Pure::Null => { },
                Pure::Single(reg) => {
                    let amp = ORT2.powi(depth) * phase.as_complex();
                    acc.push((reg, amp));
                },
                Pure::Superpos(l, r, ph) => {
                    do_iter(l.as_ref(), depth + 1, phase, acc);
                    do_iter(r.as_ref(), depth + 1, phase + *ph, acc);
                },
            }
        }

        let mut acc: Vec<(&Register<N>, C64)> = Vec::new();
        do_iter(self, 0, Phase::Pi0, &mut acc);
        acc
    }

    // like `terms`, but calculate amplitudes by tree depth and phase instead of
    // a single C64, and identify registers by a hash.
    fn terms_dph(&self) -> Vec<(u64, usize, Phase)> {
        fn do_iter<const M: usize> (
            state: &Pure<M>,
            depth: usize,
            phase: Phase,
            acc: &mut Vec<(u64, usize, Phase)>,
        ) {
            match state {
                Pure::Null => { },
                Pure::Single(reg) => {
                    acc.push((reg.ident(), depth, phase));
                },
                Pure::Superpos(l, r, ph) => {
                    do_iter(l.as_ref(), depth + 1, phase, acc);
                    do_iter(r.as_ref(), depth + 1, phase + *ph, acc);
                },
            }
        }

        let mut acc: Vec<(u64, usize, Phase)> = Vec::new();
        do_iter(self, 0, Phase::Pi0, &mut acc);
        acc
    }

    // set a single term with specified `(hash, depth, phase)` to `Null`.
    fn remove_term(
        &mut self,
        target: (u64, usize, Phase),
        depth: usize,
        phase: Phase,
    ) -> bool
    {
        match self {
            Self::Null => { false },
            Self::Single(reg) => {
                if reg.ident() == target.0
                    && depth == target.1
                    && phase == target.2
                {
                    *self = Self::Null;
                    true
                } else {
                    false
                }
            },
            Self::Superpos(l, r, ph) => {
                l.remove_term(target, depth + 1, phase)
                    || r.remove_term(target, depth + 1, phase + *ph)
            },
        }
    }

    /// Search the tree for canceling terms and eliminate them, converting to
    /// `Null` where applicable.
    pub fn reduce_terms(&mut self) {
        let mut terms = self.terms_dph();
        let mut remove_pair: Option<(usize, usize)>;
        while !terms.is_empty() {
            remove_pair
                = terms.iter().enumerate()
                .cartesian_product(terms.iter().enumerate())
                .find_map(|((i, term_i), (j, term_j))| {
                    (
                        term_i.0 == term_j.0
                        && term_i.1 == term_j.1
                        && term_i.2 - term_j.2 == Phase::Pi
                    )
                    .then_some((i, j))
                });
            if let Some((i0, j0)) = remove_pair {
                self.remove_term(terms.swap_remove(i0.min(j0)), 0, Phase::Pi0);
                self.remove_term(terms.swap_remove(i0.max(j0) - 1), 0, Phase::Pi0);
            } else {
                break;
            }
        }
    }

    /// Perform the action of a gate.
    pub fn apply_gate(&mut self, gate: Gate) -> Option<Phase> {
        use Qubit::*;
        use Phase::*;
        macro_rules! zsup {
            (
                $self:ident,
                $reg:ident,
                $c:ident,
                $t:ident,
                $targetl:expr,
                $targetr:expr,
                $rel_ph:expr,
                $glob_ph:expr
            ) => {
                {
                    let mut l = *$reg;
                    l[$c] = Zp;
                    l[$t] = $targetl;
                    let mut r = *$reg;
                    r[$c] = Zm;
                    r[$t] = $targetr;
                    *$self = Self::Superpos(
                        Box::new(l.into()), Box::new(r.into()), $rel_ph);
                    Some($glob_ph)
                }
            }
        }
        match self {
            Self::Null => { None },
            Self::Single(reg) => match gate {
                Gate::H(k) if k < N => match reg[k] {
                    Xp => { reg[k] = Zp; Some(Pi0) },
                    Xm => { reg[k] = Zm; Some(Pi) },
                    Yp => { reg[k] = Ym; Some(Pi1q) },
                    Ym => { reg[k] = Yp; Some(Pi7q) },
                    Zp => { reg[k] = Xp; Some(Pi0) },
                    Zm => { reg[k] = Xm; Some(Pi0) },
                },
                Gate::X(k) if k < N => match reg[k] {
                    Xp => { Some(Pi0) },
                    Xm => { Some(Pi) },
                    Yp => { reg[k] = Ym; Some(Pi1h) },
                    Ym => { reg[k] = Yp; Some(Pi3h) },
                    Zp => { reg[k] = Zm; Some(Pi0) },
                    Zm => { reg[k] = Zp; Some(Pi0) },
                },
                Gate::Y(k) if k < N => match reg[k] {
                    Xp => { reg[k] = Xm; Some(Pi3h) },
                    Xm => { reg[k] = Xp; Some(Pi1h) },
                    Yp => { Some(Pi0) },
                    Ym => { Some(Pi) },
                    Zp => { reg[k] = Zm; Some(Pi1h) },
                    Zm => { reg[k] = Zp; Some(Pi3h) },
                },
                Gate::Z(k) if k < N => match reg[k] {
                    Xp => { reg[k] = Xm; Some(Pi0) },
                    Xm => { reg[k] = Xp; Some(Pi0) },
                    Yp => { reg[k] = Ym; Some(Pi0) },
                    Ym => { reg[k] = Yp; Some(Pi0) },
                    Zp => { Some(Pi0) },
                    Zm => { Some(Pi) },
                },
                Gate::S(k) if k < N => match reg[k] {
                    Xp => { reg[k] = Yp; Some(Pi0) },
                    Xm => { reg[k] = Ym; Some(Pi0) },
                    Yp => { reg[k] = Xm; Some(Pi0) },
                    Ym => { reg[k] = Xp; Some(Pi0) },
                    Zp => { Some(Pi0) },
                    Zm => { Some(Pi1h) },
                },
                Gate::CX(a, b) if a < N && b < N && a != b
                => match (reg[a], reg[b]) {
                    (Xp, Xp) => { Some(Pi0) },
                    (Xp, Xm) => { reg[a] = Xm; Some(Pi0) },
                    (Xm, Xp) => { Some(Pi0) },
                    (Xm, Xm) => { reg[a] = Xp; Some(Pi0) },
                    //
                    (Xp, Yp) => { zsup!(self, reg, a, b, Yp, Ym, Pi1h, Pi0) },
                    (Xp, Ym) => { zsup!(self, reg, a, b, Ym, Yp, Pi3h, Pi0) },
                    (Xm, Yp) => { zsup!(self, reg, a, b, Yp, Ym, Pi3h, Pi0) },
                    (Xm, Ym) => { zsup!(self, reg, a, b, Ym, Yp, Pi1h, Pi0) },
                    //
                    (Xp, Zp) => { zsup!(self, reg, a, b, Zp, Zm, Pi0,  Pi0) },
                    (Xp, Zm) => { zsup!(self, reg, a, b, Zm, Zp, Pi0,  Pi0) },
                    (Xm, Zp) => { zsup!(self, reg, a, b, Zp, Zm, Pi,   Pi0) },
                    (Xm, Zm) => { zsup!(self, reg, a, b, Zm, Zp, Pi,   Pi0) },
                    //
                    (Yp, Xp) => { Some(Pi0) },
                    (Yp, Xm) => { reg[a] = Ym; Some(Pi0) },
                    (Ym, Xp) => { Some(Pi0) },
                    (Ym, Xm) => { reg[a] = Yp; Some(Pi0) },
                    //
                    (Yp, Yp) => { zsup!(self, reg, a, b, Yp, Ym, Pi,   Pi0) },
                    (Yp, Ym) => { zsup!(self, reg, a, b, Ym, Yp, Pi0,  Pi0) },
                    (Ym, Yp) => { zsup!(self, reg, a, b, Yp, Ym, Pi0,  Pi0) },
                    (Ym, Ym) => { zsup!(self, reg, a, b, Ym, Yp, Pi,   Pi0) },
                    //
                    (Yp, Zp) => { zsup!(self, reg, a, b, Zp, Zm, Pi1h, Pi0) },
                    (Yp, Zm) => { zsup!(self, reg, a, b, Zm, Zp, Pi1h, Pi0) },
                    (Ym, Zp) => { zsup!(self, reg, a, b, Zp, Zm, Pi3h, Pi0) },
                    (Ym, Zm) => { zsup!(self, reg, a, b, Zm, Zp, Pi3h, Pi0) },
                    //
                    (Zp, Xp) => { Some(Pi0) },
                    (Zp, Xm) => { Some(Pi0) },
                    (Zm, Xp) => { Some(Pi0) },
                    (Zm, Xm) => { Some(Pi) },
                    //
                    (Zp, Yp) => { Some(Pi0) },
                    (Zp, Ym) => { Some(Pi0) },
                    (Zm, Yp) => { reg[b] = Ym; Some(Pi1h) },
                    (Zm, Ym) => { reg[b] = Yp; Some(Pi3h) },
                    //
                    (Zp, Zp) => { Some(Pi0) },
                    (Zp, Zm) => { Some(Pi0) },
                    (Zm, Zp) => { reg[b] = Zm; Some(Pi0) },
                    (Zm, Zm) => { reg[b] = Zp; Some(Pi0) },
                },
                Gate::CZ(a, b) if a < N && b < N && a != b
                => match (reg[a], reg[b]) {
                    (Xp, Xp) => { zsup!(self, reg, a, b, Xp, Xm, Pi0,  Pi0) },
                    (Xp, Xm) => { zsup!(self, reg, a, b, Xm, Xp, Pi0,  Pi0) },
                    (Xm, Xp) => { zsup!(self, reg, a, b, Xp, Xm, Pi,   Pi0) },
                    (Xm, Xm) => { zsup!(self, reg, a, b, Xm, Xp, Pi,   Pi0) },
                    //
                    (Xp, Yp) => { zsup!(self, reg, a, b, Yp, Ym, Pi0,  Pi0) },
                    (Xp, Ym) => { zsup!(self, reg, a, b, Ym, Yp, Pi0,  Pi0) },
                    (Xm, Yp) => { zsup!(self, reg, a, b, Yp, Ym, Pi,   Pi0) },
                    (Xm, Ym) => { zsup!(self, reg, a, b, Ym, Yp, Pi,   Pi0) },
                    //
                    (Xp, Zp) => { Some(Pi0) },
                    (Xp, Zm) => { reg[a] = Xm; Some(Pi0) },
                    (Xm, Zp) => { Some(Pi0) },
                    (Xm, Zm) => { reg[a] = Xp; Some(Pi0) },
                    //
                    (Yp, Xp) => { zsup!(self, reg, a, b, Xp, Xm, Pi1h, Pi0) },
                    (Yp, Xm) => { zsup!(self, reg, a, b, Xm, Xp, Pi1h, Pi0) },
                    (Ym, Xp) => { zsup!(self, reg, a, b, Xp, Xm, Pi3h, Pi0) },
                    (Ym, Xm) => { zsup!(self, reg, a, b, Xm, Xp, Pi3h, Pi0) },
                    //
                    (Yp, Yp) => { zsup!(self, reg, a, b, Yp, Ym, Pi1h, Pi0) },
                    (Yp, Ym) => { zsup!(self, reg, a, b, Ym, Yp, Pi1h, Pi0) },
                    (Ym, Yp) => { zsup!(self, reg, a, b, Yp, Ym, Pi3h, Pi0) },
                    (Ym, Ym) => { zsup!(self, reg, a, b, Ym, Yp, Pi3h, Pi0) },
                    //
                    (Yp, Zp) => { Some(Pi0) },
                    (Yp, Zm) => { reg[a] = Ym; Some(Pi0) },
                    (Ym, Zp) => { Some(Pi0) },
                    (Ym, Zm) => { reg[a] = Yp; Some(Pi0) },
                    //
                    (Zp, Xp) => { Some(Pi0) },
                    (Zp, Xm) => { Some(Pi0) },
                    (Zm, Xp) => { reg[b] = Xm; Some(Pi0) },
                    (Zm, Xm) => { reg[b] = Xp; Some(Pi0) },
                    //
                    (Zp, Yp) => { Some(Pi0) },
                    (Zp, Ym) => { Some(Pi0) },
                    (Zm, Yp) => { reg[b] = Ym; Some(Pi0) },
                    (Zm, Ym) => { reg[b] = Yp; Some(Pi0) },
                    //
                    (Zp, Zp) => { Some(Pi0) },
                    (Zp, Zm) => { Some(Pi0) },
                    (Zm, Zp) => { Some(Pi0) },
                    (Zm, Zm) => { Some(Pi) },
                },
                Gate::Swap(a, b) if a < N && b < N => {
                    reg.0.swap(a, b);
                    Some(Pi0)
                },
                _ => { Some(Pi0) },
            },
            Self::Superpos(l, r, ph) => {
                let maybe_ph_l = l.apply_gate(gate);
                let maybe_ph_r = r.apply_gate(gate);
                match (maybe_ph_l, maybe_ph_r) {
                    (Some(ph_l), Some(ph_r)) => {
                        *ph += ph_r - ph_l;
                        Some(ph_l)
                    },
                    (Some(ph_l), None) => {
                        *ph -= ph_l;
                        Some(ph_l)
                    },
                    (None, Some(ph_r)) => {
                        *ph += ph_r;
                        Some(Pi0)
                    },
                    (None, None) => {
                        *self = Self::Null;
                        None
                    },
                }
            },
        }
    }

    /// Perform a series of gates.
    pub fn apply_circuit<'b, I>(&mut self, gates: I)
    where I: IntoIterator<Item = &'b Gate>
    {
        gates.into_iter()
            .copied()
            .for_each(|g| {
                self.apply_gate(g);
                if g.is_cx() { self.reduce_terms(); }
            });
    }

    /// Perform a projective measurement on a single qubit, post-selected to a
    /// particular outcome.
    pub fn measure(&mut self, index: usize, outcome: Qubit) -> Option<Phase> {
        use Qubit::*;
        use Phase::*;
        match self {
            Self::Null => None,
            Self::Single(reg) => {
                reg.0.get_mut(index)
                    .and_then(|q| {
                        (
                            !matches!(
                                (*q, outcome),
                                (Xp, Xm) | (Xm, Xp)
                                | (Yp, Ym) | (Ym, Yp)
                                | (Zp, Zm) | (Zm, Zp)
                            )
                        )
                        .then(|| { *q = outcome; Pi0 })
                    })
            },
            Self::Superpos(l, r, ph) => {
                let res_l = l.measure(index, outcome);
                let res_r = r.measure(index, outcome);
                match (res_l, res_r) {
                    (None, None) => {
                        *self = Self::Null;
                        None
                    },
                    (None, Some(ph_r)) => {
                        let ph = *ph;
                        *self = std::mem::replace(r.deref_mut(), Self::Null);
                        Some(ph + ph_r)
                    },
                    (Some(ph_l), None) => {
                        *self = std::mem::replace(l.deref_mut(), Self::Null);
                        Some(ph_l)
                    },
                    (Some(ph_l), Some(ph_r)) => {
                        *ph += ph_r - ph_l;
                        Some(ph_l)
                    },
                }
            },
        }
    }

    /// Represent `self` as a complex-valued 1D array in the *z*-basis.
    ///
    /// The length of this array is guaranteed to be `2^N`.
    pub fn as_vector(&self) -> na::DVector<C64> {
        self.terms()
            .into_iter()
            .fold(
                na::DVector::<C64>::zeros(2_usize.pow(N as u32)),
                |acc, (reg, amp)| {
                    let mut v = reg.as_vector();
                    v.iter_mut().for_each(|a| { *a *= amp; });
                    acc + v
                }
            )
    }

    /// Represent `self` as a complex-valued 2D array in the *z*-basis.
    ///
    /// This array is guaranteed to be Hermitian and idempotent with shape `2^N
    /// × 2^N`.
    pub fn as_matrix(&self) -> na::DMatrix<C64> {
        let psi = self.as_vector();
        outer_prod(&psi, &psi)
    }
}

/* State **********************************************************************/

/// A classical mixture of `N`-qubit register states.
#[derive(Clone, Debug)]
pub enum State<const N: usize> {
    /// A physically impossible null state.
    Null,
    /// A single [`Pure`] state.
    Pure(Pure<N>),
    /// A mixed state comprising the (even) classical mix of two states.
    Mixed(Box<State<N>>, Box<State<N>>),
}

/// Initialize a register as all [`Qubit::Zp`].
impl<const N: usize> Default for State<N> {
    fn default() -> Self { Self::new() }
}

impl<const N: usize> From<Register<N>> for State<N> {
    fn from(reg: Register<N>) -> Self { Self::Pure(reg.into()) }
}

impl<const N: usize> From<Pure<N>> for State<N> {
    fn from(pure: Pure<N>) -> Self { Self::Pure(pure) }
}

impl<const N: usize> State<N> {
    /// Create a new state as a pure register initialized to all [`Qubit::Zp`].
    pub fn new() -> Self { Self::Pure(Pure::new()) }

    /// Return `true` if `self` is `Null`.
    pub fn is_null(&self) -> bool { matches!(self, Self::Null) }

    /// Return `true` if `self` is `Pure`.
    pub fn is_pure(&self) -> bool { matches!(self, Self::Pure(_)) }

    /// Return `true` if `self` is `Mixed`.
    pub fn is_mixed(&self) -> bool { matches!(self, Self::Mixed(..)) }

    /// Return the number of pure states in the classical distribution.
    ///
    /// Note that this sum may count some pure states twice.
    pub fn num_terms(&self) -> usize {
        match self {
            Self::Null => 0,
            Self::Pure(_) => 1,
            Self::Mixed(l, r) => l.num_terms() + r.num_terms(),
        }
    }

    /// Return a vector of all pure states with associated probabilities.
    pub fn terms(&self) -> Vec<(&Pure<N>, f64)> {
        fn do_iter<'b, const M: usize>(
            state: &'b State<M>,
            depth: i32,
            acc: &mut Vec<(&'b Pure<M>, f64)>,
        ) {
            match state {
                State::Null => { },
                State::Pure(pure) => {
                    let prob = 0.5_f64.powi(depth);
                    acc.push((pure, prob));
                },
                State::Mixed(l, r) => {
                    do_iter(l.as_ref(), depth + 1, acc);
                    do_iter(r.as_ref(), depth + 1, acc);
                },
            }
        }

        let mut acc: Vec<(&Pure<N>, f64)> = Vec::new();
        do_iter(self, 0, &mut acc);
        acc
    }

    /// Perform the action of a gate.
    pub fn apply_gate(&mut self, gate: Gate) {
        match self {
            Self::Null => { },
            Self::Pure(pure) => { pure.apply_gate(gate); }
            Self::Mixed(l, r) => {
                l.apply_gate(gate);
                r.apply_gate(gate);
            },
        }
    }

    /// Perform a sequence of gates.
    pub fn apply_circuit<'b, I>(&mut self, gates: I)
    where I: IntoIterator<Item = &'b Gate>
    {
        gates.into_iter()
            .copied()
            .for_each(|g| { self.apply_gate(g); });
    }

    /// Perform a projective measurement on a single qubit.
    pub fn measure(&mut self, index: usize, basis: Basis) {
        match self {
            Self::Null => { },
            Self::Pure(pure) => {
                let (op, om) = basis.outcomes();
                let mut pure_alt = pure.clone();
                pure.measure(index, op);
                pure_alt.measure(index, om);
                match (pure.is_null(), pure_alt.is_null()) {
                    (true,  true ) => { *self = Self::Null; },
                    (true,  false) => { *pure = pure_alt; },
                    (false, true ) => { },
                    (false, false) => {
                        let pure = std::mem::replace(pure, Pure::Null);
                        *self = Self::Mixed(
                            Box::new(pure.into()), Box::new(pure_alt.into()));
                    },
                }
            }
            Self::Mixed(l, r) => {
                l.measure(index, basis);
                r.measure(index, basis);
                match (l.is_null(), r.is_null()) {
                    (true,  true ) => { *self = Self::Null; },
                    (true,  false) => {
                        *self = std::mem::replace(r.deref_mut(), State::Null);
                    },
                    (false, true ) => {
                        *self = std::mem::replace(l.deref_mut(), State::Null);
                    },
                    (false, false) => { },
                }
            },
        }
    }

    /// Represent `self` as a complex-valued 2D array in the *z*-basis.
    ///
    /// This array is guaranteed to be Hermitian with shape `2^N × 2^N`.
    pub fn as_matrix(&self) -> na::DMatrix<C64> {
        let n = 2_usize.pow(N as u32);
        self.terms()
            .into_iter()
            .fold(
                na::DMatrix::<C64>::zeros(n, n),
                |acc, (pure, prob)| {
                    let mut r = pure.as_matrix();
                    r.iter_mut().for_each(|a| { *a *= prob; });
                    acc + r
                }
            )
    }
}

/* Auxiliary structures *******************************************************/

fn outer_prod(a: &na::DVector<C64>, b: &na::DVector<C64>) -> na::DMatrix<C64> {
    let na = a.len();
    let nb = b.len();
    na::DMatrix::from_iterator(
        na, nb,
        a.iter().cartesian_product(b)
            .map(|(ai, bj)| *ai * bj.conj()),
    )
}

