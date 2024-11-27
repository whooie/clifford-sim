#![allow(unused_imports, dead_code)]

use std::{ fs, io::Write };
use clifford_sim::gate::{ Clifford, Gate };
use ndarray::{ self as nd, linalg::kron };
use num_complex::Complex32 as C32;
use rand::thread_rng;
use whooie::print_flush;

fn gate_matrix(gate: &Gate) -> nd::Array2<C32> {
    use std::f32::consts::FRAC_1_SQRT_2;
    const ZERO: C32 = C32 { re: 0.0, im: 0.0 };
    const ONE: C32 = C32 { re: 1.0, im: 0.0 };
    const I: C32 = C32 { re: 0.0, im: 1.0 };
    const ORT2: C32 = C32 { re: FRAC_1_SQRT_2, im: 0.0 };
    const IORT2: C32 = C32 { re: 0.0, im: FRAC_1_SQRT_2 };
    let eye: nd::Array2<C32> = nd::array![[ONE, ZERO], [ZERO, ONE]];
    match gate {
        Gate::H(k) => {
            let mat: nd::Array2<C32> = nd::array![[ORT2, ORT2], [ORT2, -ORT2]];
            if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
        },
        Gate::X(k) => {
            let mat: nd::Array2<C32> = nd::array![[ZERO, ONE], [ONE, ZERO]];
            if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
        },
        Gate::Y(k) => {
            let mat: nd::Array2<C32> = nd::array![[ZERO, -I], [I, ZERO]];
            if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
        },
        Gate::Z(k) => {
            let mat: nd::Array2<C32> = nd::array![[ONE, ZERO], [ZERO, -ONE]];
            if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
        },
        Gate::S(k) => {
            let mat: nd::Array2<C32> = nd::array![[ONE, ZERO], [ZERO, I]];
            if *k == 0 { kron(&mat, &eye) } else { kron(&eye, &mat) }
        },
        Gate::SInv(k) => {
            let mat: nd::Array2<C32> = nd::array![[ONE, ZERO], [ZERO, -I]];
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

fn cliff_matrix(gates: &Clifford) -> nd::Array2<C32> {
    gates.into_iter()
        .rev()
        .map(gate_matrix)
        .fold(nd::Array2::eye(4), |acc, gate| acc.dot(&gate))
}

fn matrix_approx_eq(a: &nd::Array2<C32>, b: &nd::Array2<C32>) -> bool {
    a.iter().zip(b).all(|(ak, bk)| (ak - bk).norm() < 1e-6)
}

fn card_cn(n: usize) -> u128 {
    (1..=n).map(|k| 2_u128 * 4_u128.pow(k as u32) * (4_u128.pow(k as u32) - 1))
        .product()
}

fn main() {
    const N: usize = 2;
    const MC: usize = 1_000_000;
    println!("|C{}| = {}", N, card_cn(N));
    let mut rng = thread_rng();
    
    // let mut elems: Vec<Clifford> = Vec::new();
    // for _ in 0..MC {
    //     let cliff = Clifford::gen(N, &mut rng);
    //     if !elems.contains(&cliff) {
    //         elems.push(cliff);
    //     }
    // }
    // println!("{} unique elements found from {} draws", elems.len(), MC);
    // // for elem in elems.into_iter() {
    // //     println!("{:?}", elem);
    // // }
    // {
    //     let mut out
    //         = fs::OpenOptions::new()
    //         .create(true)
    //         .write(true)
    //         .truncate(true)
    //         .append(false)
    //         .open("cliffords.txt")
    //         .unwrap();
    //     for elem in elems.into_iter() {
    //         writeln!(out, "{:?}", elem.unpack().0).unwrap();
    //     }
    // }

    let mut elems: Vec<Clifford> = Vec::new();
    let mut mats: Vec<nd::Array2<C32>> = Vec::new();
    eprint!("\r 0 / {} ", MC);
    for k in 0..MC {
        eprint!("\r {} / {} ", k + 1, MC);
        let cliff = Clifford::gen(N, &mut rng);
        // println!("{:?}", cliff);
        let mat = cliff_matrix(&cliff);
        if !mats.iter().any(|g| matrix_approx_eq(g, &mat)) {
            mats.push(mat);
            elems.push(cliff);
        }
    }
    eprintln!();
    eprintln!("{} unique elements found from {} draws", mats.len(), MC);
    // for elem in elems.into_iter() {
    //     println!("{:?}", elem);
    // }
    {
        let mut out
            = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .append(false)
            .open("cliffords.txt")
            .unwrap();
        for elem in elems.into_iter() {
            writeln!(out, "{:?}", elem.unpack().0).unwrap();
        }
    }
}

