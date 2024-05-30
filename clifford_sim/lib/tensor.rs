//! Tensor network-based circuit simulator.

#![allow(unused_imports)]

use std::cmp::Ordering;
use ndarray as nd;
use num_complex::Complex32 as C32;
use tensor_net::{
    tensor::{ Idx, Tensor },
    network::Network,
};

/// An index representing a qubit degree of freedom.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Q(
    /// Qubit index.
    pub usize,
    /// Bra- (`true`) or ket- (`false`) side of a density matrix.
    pub bool,
    /// Flag to keep track of gate inputs/outputs. All outgoing indices in
    /// [`QNet`] will have this field as `false`.
    pub(crate) bool,
);

impl PartialOrd for Q {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Q {
    fn cmp(&self, other: &Self) -> Ordering {
        self.2.cmp(&other.2)
            .reverse()
            .then(self.1.cmp(&other.1))
            .then(self.0.cmp(&other.0))
    }
}

impl From<(usize, bool)> for Q {
    fn from(x: (usize, bool)) -> Self { Self(x.0, x.1, false) }
}

impl Idx for Q {
    fn dim(&self) -> usize { 2 }

    fn label(&self) -> String {
        format!(
            "q:{}{}",
            self.0,
            if self.1 { "'" } else { "" },
        )
    }
}

impl Q {
    fn reset_io(&mut self) { self.2 = false; }
}

/// A node in a qubit tensor network.
pub type QTensor = Tensor<Q, C32>;

/// Definitions of common one- and two-qubit gates.
pub mod gates {
    use super::*;

    /// Make a single-qubit unitary from its Euler angles.
    ///
    /// This gate is equivalent to `Z(γ) × X(β) × Z(α)`.
    pub fn u(q: usize, adj: bool, alpha: f32, beta: f32, gamma: f32)
        -> QTensor
    {
        let q_in = Q(q, adj, false);
        let q_out = Q(q, adj, true);
        let b2 = beta / 2.0;
        let ag = alpha + gamma;
        let prefactor = if adj { C32::cis(-b2) } else { C32::cis(b2) };
        let ondiag0 = b2.cos();
        let ondiag1 = C32::cis(if adj { -ag } else { ag }) * ondiag0;
        let offdiag = b2.sin();
        let offdiag0 = if adj {
            C32::i() * C32::cis(-alpha) * offdiag
        } else {
            -C32::i() * C32::cis(gamma) * offdiag
        };
        let offdiag1 = if adj {
            C32::i() * C32::cis(-gamma) * offdiag
        } else {
            -C32::i() * C32::cis(alpha) * offdiag
        };
        unsafe {
            QTensor::from_array_unchecked(
                [q_in, q_out],
                nd::array![
                    [prefactor * ondiag0,  prefactor * offdiag0],
                    [prefactor * offdiag1, prefactor * ondiag1 ],
                ]
            )
        }
    }

    /// Make a Hadamard gate.
    pub fn h(q: usize, adj: bool) -> QTensor {
        use std::f32::consts::FRAC_1_SQRT_2;
        let q_in = Q(q, adj, false);
        let q_out = Q(q, adj, true);
        unsafe {
            QTensor::from_array_unchecked(
                [q_in, q_out],
                nd::array![
                    [FRAC_1_SQRT_2.into(),   FRAC_1_SQRT_2.into() ],
                    [FRAC_1_SQRT_2.into(), (-FRAC_1_SQRT_2).into()],
                ]
            )
        }
    }

    /// Make an X gate.
    pub fn x(q: usize, adj: bool) -> QTensor {
        let q_in = Q(q, adj, false);
        let q_out = Q(q, adj, true);
        unsafe {
            QTensor::from_array_unchecked(
                [q_in, q_out],
                nd::array![
                    [0.0.into(), 1.0.into()],
                    [1.0.into(), 0.0.into()],
                ]
            )
        }
    }

    /// Make a Y gate.
    pub fn y(q: usize, adj: bool) -> QTensor {
        let q_in = Q(q, adj, false);
        let q_out = Q(q, adj, true);
        unsafe {
            QTensor::from_array_unchecked(
                [q_in, q_out],
                nd::array![
                    [ 0.0.into(), C32::i()  ],
                    [-C32::i(),   0.0.into()],
                ]
            )
        }
    }

    /// Make a Z gate.
    pub fn z(q: usize, adj: bool) -> QTensor {
        let q_in = Q(q, adj, false);
        let q_out = Q(q, adj, true);
        unsafe {
            QTensor::from_array_unchecked(
                [q_in, q_out],
                nd::array![
                    [1.0.into(),   0.0.into() ],
                    [0.0.into(), (-1.0).into()],
                ]
            )
        }
    }

    /// Make an X-rotation gate.
    pub fn xrot(q: usize, adj: bool, angle: f32) -> QTensor {
        let q_in = Q(q, adj, false);
        let q_out = Q(q, adj, true);
        let ang2 = angle / 2.0;
        let prefactor = if adj { C32::cis(-ang2) } else { C32::cis(ang2) };
        let ondiag = ang2.cos();
        let offdiag = if adj { C32::i() } else { -C32::i() } * ang2.sin();
        unsafe {
            QTensor::from_array_unchecked(
                [q_in, q_out],
                nd::array![
                    [prefactor * ondiag,  prefactor * offdiag],
                    [prefactor * offdiag, prefactor * ondiag ],
                ]
            )
        }
    }

    /// Make a Y-rotation gate.
    pub fn yrot(q: usize, adj: bool, angle: f32) -> QTensor {
        let q_in = Q(q, adj, false);
        let q_out = Q(q, adj, true);
        let ang2 = angle / 2.0;
        let prefactor = if adj { C32::cis(-ang2) } else { C32::cis(ang2) };
        let ondiag = ang2.cos();
        let offdiag = ang2.sin();
        unsafe {
            QTensor::from_array_unchecked(
                [q_in, q_out],
                nd::array![
                    [ prefactor * ondiag,  prefactor * offdiag],
                    [-prefactor * offdiag, prefactor * ondiag ],
                ]
            )
        }
    }

    /// Make a Z-rotation gate.
    pub fn zrot(q: usize, adj: bool, angle: f32) -> QTensor {
        let q_in = Q(q, adj, false);
        let q_out = Q(q, adj, true);
        let ph = if adj { C32::cis(-angle) } else { C32::cis(angle) };
        unsafe {
            QTensor::from_array_unchecked(
                [q_in, q_out],
                nd::array![
                    [1.0.into(), 0.0.into()],
                    [0.0.into(), ph        ],
                ]
            )
        }
    }

    /// Make a CX gate.
    pub fn cx(c: usize, t: usize, adj: bool) -> QTensor {
        let q_in_c = Q(c, adj, false);
        let q_in_t = Q(t, adj, false);
        let q_out_c = Q(c, adj, true);
        let q_out_t = Q(t, adj, true);
        let z0 = C32::from(0.0);
        let z1 = C32::from(1.0);
        unsafe {
            QTensor::from_array_unchecked(
                [q_in_c, q_in_t, q_out_c, q_out_t],
                nd::array![
                    z1, z0, z0, z0,
                    z0, z1, z0, z0,
                    z0, z0, z0, z1,
                    z0, z0, z1, z0,
                ]
                .into_shape((2, 2, 2, 2))
                .unwrap()
            )
        }
    }

    /// Make a CY gate.
    pub fn cy(c: usize, t: usize, adj: bool) -> QTensor {
        let q_in_c = Q(c, adj, false);
        let q_in_t = Q(t, adj, false);
        let q_out_c = Q(c, adj, true);
        let q_out_t = Q(t, adj, true);
        let z0 = C32::from(0.0);
        let z1 = C32::from(1.0);
        let i = C32::i();
        unsafe {
            QTensor::from_array_unchecked(
                [q_in_c, q_in_t, q_out_c, q_out_t],
                nd::array![
                    z1, z0, z0, z0,
                    z0, z1, z0, z0,
                    z0, z0, z0,  i,
                    z0, z0, -i, z0,
                ]
                .into_shape((2, 2, 2, 2))
                .unwrap()
            )
        }
    }

    /// Make a CZ gate.
    pub fn cz(c: usize, t: usize, adj: bool) -> QTensor {
        let q_in_c = Q(c, adj, false);
        let q_in_t = Q(t, adj, false);
        let q_out_c = Q(c, adj, true);
        let q_out_t = Q(t, adj, true);
        let z0 = C32::from(0.0);
        let z1 = C32::from(1.0);
        unsafe {
            QTensor::from_array_unchecked(
                [q_in_c, q_in_t, q_out_c, q_out_t],
                nd::array![
                    z1,  z0,  z0,  z0,
                    z0,  z1,  z0,  z0,
                    z0,  z0,  z1,  z0,
                    z0,  z0,  z0, -z1,
                ]
                .into_shape((2, 2, 2, 2))
                .unwrap()
            )
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Gate {
    U(usize, f32, f32, f32),
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    XRot(usize, f32),
    YRot(usize, f32),
    ZRot(usize, f32),
    CX(usize, usize),
    CY(usize, usize),
    CZ(usize, usize),
}

impl Gate {
    fn as_tensor(&self, adj: bool) -> QTensor {
        match *self {
            Self::U(q, alpha, beta, gamma)
                => gates::u(q, adj, alpha, beta, gamma),
            Self::H(q) => gates::h(q, adj),
            Self::X(q) => gates::x(q, adj),
            Self::Y(q) => gates::y(q, adj),
            Self::Z(q) => gates::z(q, adj),
            Self::XRot(q, ang) => gates::xrot(q, adj, ang),
            Self::YRot(q, ang) => gates::yrot(q, adj, ang),
            Self::ZRot(q, ang) => gates::zrot(q, adj, ang),
            Self::CX(c, t) => gates::cx(c, t, adj),
            Self::CY(c, t) => gates::cy(c, t, adj),
            Self::CZ(c, t) => gates::cz(c, t, adj),
        }
    }
}

pub type QNetwork = Network<Q, C32>;

/// A general N-qubit state represented as a tensor network.
#[derive(Clone, Debug)]
pub struct TNet {
    pub(crate) n: usize,
    pub(crate) net: QNetwork,
}

#[allow(unused_variables, unused_mut)]
impl TNet {
    /// Create a new `n`-qubit state initialized to ∣0...0⟩.
    pub fn new(n: usize) -> Self {
        let mut net = QNetwork::new();
        let indices: Vec<Q>
            = (0..n).map(|q| Q(q, false, false))
            .chain((0..n).map(|q| Q(q, true, false)))
            .collect();
        let mut shape = vec![2_usize; 2 * n];
        let mut arr: nd::ArrayD<C32> = nd::ArrayD::zeros(shape.as_slice());
        shape.fill(0);
        arr[shape.as_slice()] = C32::from(1.0);
        let init = unsafe { QTensor::from_array_unchecked(indices, arr) };
        net.push(init).unwrap();
        Self { n, net }
    }

    /// Perform the action of a gate.
    pub fn apply_gate(&mut self, gate: &Gate) -> &mut Self {
        let g_ket = gate.as_tensor(false);
        let g_bra = gate.as_tensor(true);
        self.net.push(g_ket)
            .expect("unexpected duplicate ket index in network");
        self.net.push(g_bra)
            .expect("unexpected duplicate bra index in network");
        self.net.contract_network()
            .expect("unexpected error in network contraction");
        self.update_io_idx();
        self
    }

    fn update_io_idx(&mut self) {
        unsafe { self.net.update_indices(|idx| { idx.reset_io(); }); }
    }

    /// Perform a series of gates.
    pub fn apply_circuit<'a, I>(&mut self, gates: I) -> &mut Self
    where I: IntoIterator<Item = &'a Gate>
    {
        gates.into_iter().for_each(|g| { self.apply_gate(g); });
        self
    }

    /// Apply a projective measurement to the `k`-th qubit.
    pub fn measure(&mut self, k: usize) {
        todo!()
    }

    /// Calculate the Von Neumann entanglement entropy of the state
    pub fn entanglement_entropy(&self) -> f32 {
        todo!()
    }

    pub fn mutual_information(self /* partitions */) -> f32 {
        todo!()
    }

    /// Convert to a full density matrix.
    pub fn into_matrix(self) -> nd::Array2<C32> {
        todo!()
    }
}

// /// Main driver for running circuit of alternating unitary evolution and
// /// measurement.
// #[derive(Clone, Debug)]
// pub struct NetworkCircuit {
//     pub net: QNetwork,
//     pub part: Option<Partition>,
//     n: usize,
// }

