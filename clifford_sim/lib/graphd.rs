//! Like [`graph`][crate::graph], but not statically sized.

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
    graph::Graph,
    stab::Stab,
    stabd::{ StabD, NPauliD },
};

/// A graph state of a finite register of qubits.
/// 
/// Like [`Graph`], but for non-static system sizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphD {
    n: usize,
    adj: nd::Array2<bool>,
}

impl<const N: usize> From<Graph<N>> for GraphD {
    fn from(graph: Graph<N>) -> Self {
        let Graph { adj } = graph;
        let adj: nd::Array2<bool>
            = nd::Array2::from_shape_vec(
                (N, N), adj.into_iter().flatten().collect())
            .expect("error reshaping adjacency data");
        Self { n: N, adj }
    }
}

impl GraphD {
    /// Create a new, totally disconnected graph state of `n` qubits.
    pub fn new(n: usize) -> Self {
        Self { n, adj: nd::Array2::from_elem((n, n), false) } }

    /// Return an iterator over the indices of all nodes who share an edge with
    /// a given source node.
    ///
    /// Returns `Err` if the node doesn't exist.
    pub fn neighbors_of(&self, node: usize) -> NeighborsD<'_> {
        if node < self.n {
            let iter
                = self.adj.slice(nd::s![node, ..])
                .into_iter()
                .enumerate();
            NeighborsDataD::Iter(iter).into()
        } else {
            NeighborsDataD::Empty.into()
        }
    }

    fn neighbors_of_unchecked(&self, node: usize) -> NeighborsD<'_> {
        let iter = self.adj.slice(nd::s![node, ..]).into_iter().enumerate();
        NeighborsDataD::Iter(iter).into()
    }

    fn toggle_edge_unchecked(&mut self, a: usize, b: usize) -> &mut Self {
        self.adj[[a, b]] ^= true;
        self.adj[[b, a]] ^= true;
        self
    }

    /// Add an edge between nodes `a` and `b` if it doesn't already exist, or
    /// remove it if it does.
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
    pub fn stabilizers(&self) -> StabilizersD {
        let npauli: NPauliD
            = NPauliD { phase: Phase::Pi0, ops: vec![Pauli::I; self.n] };
        let mut stabs: Vec<NPauliD> = vec![npauli.clone(); self.n];
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
        StabilizersD(stabs)
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
                } else {
                    self.local_complement_unchecked(node);
                    self.disconnect_node_unchecked(node);
                }
            },
        }
        self
    }

    /// Convert to a [`StabD`].
    pub fn to_stabd(&self) -> StabD {
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
        StabD { n: self.n, x, z, r, over32 }
    }

    /// Convert from a [`Stab`].
    pub(crate) fn from_stab<const N: usize>(stab: Stab<N>) -> Self {
        Self::from_stabd(stab.into())
    }

    /// Convert from a [`StabD`].
    pub(crate) fn from_stabd(mut stabd: StabD) -> Self {
        // see Vijayan et al (arXiv:2209.07345) for details on this algorithm
        let mut j5: usize;
        let mut pw: u32;
        // make the X block full rank (O(N^3))
        let mut all_zeros: bool;
        for j in 0..stabd.n {
            j5 = j >> 5;
            pw = 1 << (j & 31);
            all_zeros
                = stabd.x.slice(nd::s![.., j5]).iter()
                .skip(stabd.n)
                .all(|xij| xij & pw == 0);
            if all_zeros { stabd.apply_h(j); }
            for i in stabd.n + j + 1..2 * stabd.n {
                if stabd.x[[i, j5]] & pw != 0 {
                    stabd.row_swap(i, j);
                    break;
                }
            }
            for i in stabd.n + j + 1..2 * stabd.n {
                if stabd.x[[i, j5]] & pw != 0 { stabd.row_mul(i, j); }
            }
        }
        // diagonalize the X block (O(N^3))
        for i in (stabd.n..2 * stabd.n - 1).rev() {
            for j in (i + 1 - stabd.n..stabd.n).rev() {
                j5 = j >> 5;
                pw = 1 << (j & 31);
                if stabd.x[[i, j5]] & pw == 0 { stabd.row_mul(j, i); }
            }
        }
        // make Z block diagonal zero and correct phases (O(N))
        for i in stabd.n..2 * stabd.n {
            j5 = i >> 5;
            pw = 1 << (i & 31);
            if stabd.z[[i, j5]] & pw != 0 {
                stabd.apply_s(i).apply_s(i).apply_s(i); // phase gate inverse
            }
            if stabd.r[i] % 4 == 2 {
                stabd.apply_z(i);
            }
        }
        // convert to boolean adjacency matrix
        let mut adj: nd::Array2<bool>
            = nd::Array2::from_elem((stabd.n, stabd.n), false);
        let iter
            = stabd.z.axis_iter(nd::Axis(0)).skip(stabd.n)
            .zip(adj.axis_iter_mut(nd::Axis(0)));
        for (zi, mut adji) in iter {
            for (j, adjij) in adji.iter_mut().enumerate() {
                j5 = j << 5;
                pw = 1 << (j & 31);
                *adjij = zi[j5] & pw != 0;
            }
        }
        Self { n: stabd.n, adj }
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

impl<const N: usize> From<Stab<N>> for GraphD {
    fn from(stab: Stab<N>) -> Self { Self::from_stab(stab) }
}

impl From<StabD> for GraphD {
    fn from(stabd: StabD) -> Self { Self::from_stabd(stabd) }
}

#[derive(Clone)]
pub struct NeighborsD<'a> {
    data: NeighborsDataD<'a>,
}

#[derive(Clone)]
enum NeighborsDataD<'a> {
    Empty,
    Iter(std::iter::Enumerate<<nd::ArrayView1<'a, bool> as IntoIterator>::IntoIter>),
}

impl<'a> From<NeighborsDataD<'a>> for NeighborsD<'a> {
    fn from(data: NeighborsDataD<'a>) -> Self { Self { data } }
}

impl<'a> Iterator for NeighborsD<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.data {
            NeighborsDataD::Empty => None,
            NeighborsDataD::Iter(ref mut iter)
                => iter.find_map(|(k, bk)| bk.then_some(k)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StabilizersD(pub Vec<NPauliD>);

impl fmt::Display for StabilizersD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0.len();
        for (k, stab) in self.0.iter().enumerate() {
            stab.fmt(f)?;
            if k < n - 1 { writeln!(f)?; }
        }
        Ok(())
    }
}

