#![allow(dead_code, non_snake_case, non_upper_case_globals)]

//! Tools for simulating measurement-induced phase transitions in registers of
//! qubits.
//!
//! Assumes all operations will be limited to Clifford-group transformations
//! (i.e. Hadamard, Pauli, singly controlled Pauli, or phase rotations that are
//! integer multiples of π/2).

pub mod tree;
pub mod stab;
pub mod gate;
pub mod circuit;
pub mod graph;
