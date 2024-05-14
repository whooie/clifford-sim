//! *N*-qubit graph states.
//!
//! A graph state can be represented as a graph in which each node is a qubit in
//! the ∣+⟩ = (∣0⟩ + ∣1⟩) / √2 state, and there is an edge between every
//! interacting pair of qubits. Graph states are a common language for reasoning
//! about measurement-based quantum computing models and can easily represent
//! some many-body entangled states.
//!
//! Every stabilizer state is called "locally equivalent" to a graph state.
//! Specifically: for every stabilizer state, there exists a series of local
//! Clifford gates that transform the state into one with the required
//! structure. In the reverse direction, every graph state is a stabilizer
//! state, whose stabilizers can be generated by the following simple rules. For
//! each node *i* in the graph, the generating stabilizer *S*<sub>*i*</sub> is
//! equal to *X* applied to *i* and *Z* applied to each of *i*'s neighbors.
//!
//! This relationship allows for conversion from [`Stab`] states to [`Graph`]
//! states in *O*(*N*<sup>3</sup>) time as well as the reverse in
//! *O*(*N*<sup>2</sup>) time.
//!
//! The code in this module follows information from a nice introductory [blog
//! post by Peter Rohde][rohde] and the stabilizer-to-graph conversion algorithm
//! by [Vijayan *et al*][vijayan].
//!
//! # Example
//! ```
//! use clifford_sim::graph::Graph;
//!
//! const N: usize = 5; // number of qubits
//!
//! fn main() {
//!     // initialize a totally disconnected graph
//!     let mut graph: Graph = Graph::new(N);
//!
//!     // apply CZs to form the 5-qubit "star" state; this graph is locally
//!     // equivalent to the 5-qubit GHZ state
//!     graph.apply_cz(0, 1);
//!     graph.apply_cz(0, 2);
//!     graph.apply_cz(0, 3);
//!     graph.apply_cz(0, 4);
//!
//!     // print out its stabilizers
//!     println!("{:#}", graph.stabilizers()); // `#` formatter suppresses identities
//!     // +1 XZZZZ
//!     // +1 ZX...
//!     // +1 Z.X..
//!     // +1 Z..X.
//!     // +1 Z...X
//!
//!     // convert to an ordinary stabilizer state
//!     let mut stab = graph.to_stab();
//!
//!     // apply Hadamards to convert the CZs above to CNOTs for the usual GHZ
//!     // preparation circuit
//!     stab.apply_h(1);
//!     stab.apply_h(2);
//!     stab.apply_h(3);
//!     stab.apply_h(4);
//!
//!     // print out the stabilizers and destabilizers
//!     println!("{:#}", stab.as_group());
//!     // +1 XXXXX | +1 Z....
//!     // +1 ZZ... | +1 .X...
//!     // +1 Z.Z.. | +1 ..X..
//!     // +1 Z..Z. | +1 ...X.
//!     // +1 Z...Z | +1 ....X
//!
//!     // convert to ket notation
//!     println!("{}", stab.as_kets().unwrap());
//!     // +1∣00000⟩ +1∣11111⟩
//! }
//! ```
//!
//! [rohde]: https://peterrohde.org/an-introduction-to-graph-states/
//! [vijayan]: https://arxiv.org/abs/2209.07345

use std::{
    fmt,
    fs,
    io::{ self, Write },
    path::Path,
};
use itertools::Itertools;
use ndarray as nd;
use crate:: {
    gate::{ Basis, Pauli, Phase },
    stab::{ Stab, NPauli },
};

/// A graph state of a collection of qubits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    n: usize,
    adj: nd::Array2<bool>,
}

impl Graph {
    /// Create a new, totally disconnected graph state of `n` qubits.
    pub fn new(n: usize) -> Self {
        Self { n, adj: nd::Array2::from_elem((n, n), false) } }

    /// Return the number of qubits.
    pub fn n(&self) -> usize { self.n }

    /// Return an iterator over the indices of all nodes that share an edge with
    /// a given source node.
    pub fn neighbors_of(&self, node: usize) -> Neighbors<'_> {
        if node < self.n {
            let iter
                = self.adj.slice(nd::s![node, ..])
                .into_iter()
                .enumerate();
            NeighborsData::Iter(iter).into()
        } else {
            NeighborsData::Empty.into()
        }
    }

    fn neighbors_of_unchecked(&self, node: usize) -> Neighbors<'_> {
        let iter = self.adj.slice(nd::s![node, ..]).into_iter().enumerate();
        NeighborsData::Iter(iter).into()
    }

    fn toggle_edge_unchecked(&mut self, a: usize, b: usize) -> &mut Self {
        self.adj[[a, b]] ^= true;
        self.adj[[b, a]] ^= true;
        self
    }

    /// Add an edge between nodes `a` and `b` if it doesn't already exist, or
    /// remove it otherwise.
    ///
    /// This is equivalent to applying a CZ gate to `a` and `b`.
    pub fn toggle_edge(&mut self, a: usize, b: usize) -> &mut Self {
        if a >= self.n || b >= self.n || a == b { return self; }
        self.toggle_edge_unchecked(a, b)
    }

    fn add_edge_unchecked(&mut self, a: usize, b: usize) -> &mut Self {
        self.adj[[a, b]] = true;
        self.adj[[b, a]] = true;
        self
    }

    /// Add an edge between `a` and `b`.
    ///
    /// Does nothing if an edge already exists.
    pub fn add_edge(&mut self, a: usize, b: usize) -> &mut Self {
        if a >= self.n || b >= self.n || a == b { return self; }
        self.add_edge_unchecked(a, b)
    }

    fn remove_edge_unchecked(&mut self, a: usize, b: usize) -> &mut Self {
        self.adj[[a, b]] = false;
        self.adj[[b, a]] = false;
        self
    }

    /// Remove an edge between `a` and `b`.
    ///
    /// Does nothing if `a` and `b` are not connected.
    pub fn remove_edge(&mut self, a: usize, b: usize) -> &mut Self {
        if a >= self.n || b >= self.n { return self; }
        self.remove_edge_unchecked(a, b)
    }

    /// Apply a CZ gate to `a` and `b`.
    ///
    /// Synonym for [`toggle_edge`][Self::toggle_edge].
    pub fn apply_cz(&mut self, a: usize, b: usize) -> &mut Self {
        self.toggle_edge(a, b)
    }

    fn disconnect_node_unchecked(&mut self, node: usize) -> &mut Self {
        (0..self.n).for_each(|k| { self.remove_edge_unchecked(node, k); });
        self
    }

    /// Disconnect `node` from all of its neighbors.
    pub fn disconnect_node(&mut self, node: usize) -> &mut Self {
        if node >= self.n { return self; }
        self.disconnect_node_unchecked(node)
    }

    fn local_complement_unchecked(&mut self, node: usize) -> &mut Self {
        let neighbors: Vec<usize> = self.neighbors_of_unchecked(node).collect();
        neighbors.iter()
            .cartesian_product(neighbors.iter())
            .filter(|(a, b)| a != b && a < b)
            .for_each(|(a, b)| { self.toggle_edge_unchecked(*a, *b); });
        self
    }

    /// Perform a local complementation on `node`.
    ///
    /// That is, toggle all edges in the subgraph induced by the neighborhood of
    /// `node`.
    pub fn local_complement(&mut self, node: usize) -> &mut Self {
        if node >= self.n { return self; }
        self.local_complement_unchecked(node)
    }

    /// Return a generating set for the stabilizer group of `self`.
    pub fn stabilizers(&self) -> Stabilizers {
        let npauli: NPauli
            = NPauli { phase: Phase::Pi0, ops: vec![Pauli::I; self.n] };
        let mut stabs: Vec<NPauli> = vec![npauli.clone(); self.n];
        let iter1
            = self.adj.axis_iter(nd::Axis(0))
            .zip(stabs.iter_mut())
            .enumerate();
        for (i, (row, stab)) in iter1 {
            let iter2
                = row.iter()
                .zip(stab.ops.iter_mut())
                .enumerate();
            for (j, (col, pauli)) in iter2 {
                if i == j {
                    *pauli = Pauli::X;
                } else if *col {
                    *pauli = Pauli::Z;
                } else {
                    *pauli = Pauli::I;
                }
            }
        }
        Stabilizers(stabs)
    }

    /// Perform a projective measurement on a single qubit in a given basis.
    ///
    /// Measured qubits are not removed.
    pub fn measure(&mut self, node: usize, basis: Basis) -> &mut Self {
        if node >= self.n { return self; }
        match basis {
            Basis::Z => { self.disconnect_node_unchecked(node); },
            Basis::Y => {
                self.local_complement_unchecked(node);
                self.disconnect_node_unchecked(node);
            },
            Basis::X => {
                let some_neighbor: Option<usize>
                    = self.adj.slice(nd::s![node, ..]).iter()
                    .enumerate()
                    .find_map(|(k, a)| a.then_some(k));
                if let Some(k) = some_neighbor {
                    self.local_complement_unchecked(k);
                    self.local_complement_unchecked(node);
                    self.disconnect_node_unchecked(node);
                    self.local_complement_unchecked(k);
                }
            },
        }
        self
    }

    /// Convert to a [`Stab`].
    pub fn to_stab(&self) -> Stab {
        // any graph state can be formed by taking an initial ∣0...0⟩ state,
        // applying a Hadamard to every qubit, and then applying CZ gates for
        // every edge in the graph; this implies that the resulting stabilizer
        // tableau should have the following structure:
        // 1. the top-left block (rows 0..N of X) should be all zeros
        // 2. the top-right and bottom-left blocks (rows N..2 * N of X and 0..N
        //    of Z) should be the identity
        // 3. the bottom-right block (rows N..2 * N of Z) should be the
        //    adjacency matrix of the graph
        let over32: usize = (self.n >> 5) + 1;
        let mut x: nd::Array2<u32>
            = nd::Array2::zeros((2 * self.n + 1, over32));
        let mut z: nd::Array2<u32>
            = nd::Array2::zeros((2 * self.n + 1, over32));
        let r: nd::Array1<u8> = nd::Array1::zeros(2 * self.n + 1);
        for (i, mut xi) in
            x.axis_iter_mut(nd::Axis(0)).skip(self.n).take(self.n).enumerate()
        {
            xi[i >> 5] = 1 << (i & 31);
        }
        for (i, mut zi) in
            z.axis_iter_mut(nd::Axis(0)).take(self.n).enumerate()
        {
            zi[i >> 5] = 1 << (i & 31);
        }
        let mut j5: usize;
        for ((i, j), adjij) in self.adj.indexed_iter() {
            if *adjij {
                j5 = j >> 5;
                z[[i, j5]] ^= 1 << (j & 31);
            }
        }
        Stab { n: self.n, x, z, r, over32 }
    }

    /// Convert from a [`Stab`].
    pub(crate) fn from_stab(mut stab: Stab) -> Self {
        // see Vijayan et al (arXiv:2209.07345) for details on this algorithm
        let mut j5: usize;
        let mut pw: u32;
        // make the X block full rank (O(N^3))
        let mut all_zeros: bool;
        for j in 0..stab.n {
            j5 = j >> 5;
            pw = 1 << (j & 31);
            all_zeros
                = stab.x.slice(nd::s![.., j5]).iter()
                .skip(stab.n)
                .all(|xij| xij & pw == 0);
            if all_zeros { stab.apply_h(j); }
            for i in stab.n + j + 1..2 * stab.n {
                if stab.x[[i, j5]] & pw != 0 {
                    stab.row_swap(i, j);
                    break;
                }
            }
            for i in stab.n + j + 1..2 * stab.n {
                if stab.x[[i, j5]] & pw != 0 { stab.row_mul(i, j); }
            }
        }
        // diagonalize the X block (O(N^3))
        for i in (stab.n..2 * stab.n - 1).rev() {
            for j in (i + 1 - stab.n..stab.n).rev() {
                j5 = j >> 5;
                pw = 1 << (j & 31);
                if stab.x[[i, j5]] & pw == 0 { stab.row_mul(j, i); }
            }
        }
        // make Z block diagonal zero and correct phases (O(N))
        for i in stab.n..2 * stab.n {
            j5 = i >> 5;
            pw = 1 << (i & 31);
            if stab.z[[i, j5]] & pw != 0 {
                stab.apply_s(i).apply_s(i).apply_s(i); // phase gate inverse
            }
            if stab.r[i] % 4 == 2 {
                stab.apply_z(i);
            }
        }
        // convert to boolean adjacency matrix
        let mut adj: nd::Array2<bool>
            = nd::Array2::from_elem((stab.n, stab.n), false);
        let iter
            = stab.z.axis_iter(nd::Axis(0)).skip(stab.n)
            .zip(adj.axis_iter_mut(nd::Axis(0)));
        for (zi, mut adji) in iter {
            for (j, adjij) in adji.iter_mut().enumerate() {
                j5 = j >> 5;
                pw = 1 << (j & 31);
                *adjij = zi[j5] & pw != 0;
            }
        }
        Self { n: stab.n, adj }
    }

    /// Return an object containing an encoding of `self` in the [dot
    /// language][dot-lang].
    ///
    /// Rendering this object using the default formatter will result in a full
    /// dot string representation of the diagram.
    ///
    /// [dot-lang]: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
    pub fn to_graphviz(&self, name: &str) -> tabbycat::Graph {
        use tabbycat::*;
        use tabbycat::attributes::*;

        const FONT: &str = "DejaVu Sans";
        const FONTSIZE: f64 = 10.0; // pt
        const NODE_MARGIN: f64 = 0.025; // in
        const NODE_HEIGHT: f64 = 0.200; // in
        const NODE_COLOR: Color = Color::Rgb(115, 150, 250);

        // initial declarations
        let mut statements
            = StmtList::new()
            .add_attr(
                AttrType::Graph,
                AttrList::new().add_pair(rankdir(RankDir::LR)),
            )
            .add_attr(
                AttrType::Node,
                AttrList::new()
                    .add_pair(fontname(FONT))
                    .add_pair(fontsize(FONTSIZE))
                    .add_pair(margin(NODE_MARGIN))
                    ,
            );
        // add nodes
        for k in 0..self.n {
            let attrs
                = AttrList::new()
                .add_pair(label(k.to_string()))
                .add_pair(shape(Shape::Circle))
                .add_pair(height(NODE_HEIGHT))
                .add_pair(style(Style::Filled))
                .add_pair(fillcolor(NODE_COLOR));
            statements = statements.add_node(k.into(), None, Some(attrs));
        }
        // add edges
        for ((i, j), a) in self.adj.indexed_iter() {
            if *a && j < i {
                statements
                    = statements.add_edge(
                        Edge::head_node(i.into(), None)
                            .line_to_node(j.into(), None)
                    );
            }
        }
        GraphBuilder::default()
            .graph_type(GraphType::Graph)
            .strict(false)
            .id(Identity::quoted(name))
            .stmts(statements)
            .build()
            .expect("error building graphviz")
    }

    /// Like [`to_graphviz`][Self::to_graphviz], but render directly to a string
    /// and write it to `path`.
    pub fn save_graphviz<P>(&self, name: &str, path: P)
        -> Result<&Self, io::Error>
    where P: AsRef<Path>
    {
        let graphviz = self.to_graphviz(name);
        fs::OpenOptions::new()
            .write(true)
            .append(false)
            .create(true)
            .truncate(true)
            .open(path)?
            .write_all(format!("{}", graphviz).as_bytes())?;
        Ok(self)
    }
}

impl From<Stab> for Graph {
    fn from(stab: Stab) -> Self { Self::from_stab(stab) }
}

#[derive(Clone)]
pub struct Neighbors<'a> {
    data: NeighborsData<'a>,
}

#[derive(Clone)]
enum NeighborsData<'a> {
    Empty,
    Iter(std::iter::Enumerate<<nd::ArrayView1<'a, bool> as IntoIterator>::IntoIter>),
}

impl<'a> From<NeighborsData<'a>> for Neighbors<'a> {
    fn from(data: NeighborsData<'a>) -> Self { Self { data } }
}

impl<'a> Iterator for Neighbors<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.data {
            NeighborsData::Empty => None,
            NeighborsData::Iter(ref mut iter)
                => iter.find_map(|(k, bk)| bk.then_some(k)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stabilizers(pub Vec<NPauli>);

impl fmt::Display for Stabilizers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0.len();
        for (k, stab) in self.0.iter().enumerate() {
            stab.fmt(f)?;
            if k < n - 1 { writeln!(f)?; }
        }
        Ok(())
    }
}

