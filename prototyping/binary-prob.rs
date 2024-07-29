```cargo
[package]
edition = "2021"

[dependencies]
rand = "*"
```

#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

use rand::Rng;

#[derive(Copy, Clone, Debug)]
struct BinRatio {
    num: u16,
    pow: u16,
}

impl BinRatio {
    fn as_float(self) -> f32 {
        f32::from(self.num) / 2.0_f32.powi(self.pow.into())
    }
}

fn to_float(num: u16, pow: u16) -> f32 {
    f32::from(num) / 2.0_f32.powi(pow.into())
}

fn approximate_prob(p: f32, eps: f32) -> BinRatio {
    assert!((0.0..=1.0).contains(&p));
    let mut l: u16 = 0;
    let mut r: u16 = 1;
    let mut dist_l: f32;
    let mut dist_r: f32;
    let mut prec: f32;
    for pow in 0..16_u16 {
        dist_l = (to_float(l, pow) - p).abs();
        dist_r = (to_float(r, pow) - p).abs();

        if dist_l < eps || dist_r < eps {
            return BinRatio { num: if dist_l < dist_r { l } else { r }, pow };
        } else if dist_l < dist_r {
            l *= 2;
            r *= 2;
            r -= 1;
        } else {
            l *= 2;
            l += 1;
            r *= 2;
        }
    }
    // println!("failed to meet precision threshold");
    dist_l = (to_float(l, 15) - p).abs();
    dist_r = (to_float(r, 15) - p).abs();
    BinRatio { num: if dist_l < dist_r { l } else { r }, pow: 15 }
}

fn get_bits_pos(k: usize, total: usize) -> usize {
    assert!(total.is_power_of_two() && total != 1);
    assert!(k <= total);
    if total == 2 {
        k & 1
    } else if k & 1 == 1 {
        total / 2 + get_bits_pos(k >> 1, total / 2)
    } else {
        get_bits_pos(k >> 1, total / 2)
    }
}

fn make_pattern(ratio: BinRatio) -> Vec<bool> {
    let n = 2_usize.pow(ratio.pow.into());
    let mut pattern: Vec<bool> = vec![false; n];
    for k in 0..ratio.num as usize {
        pattern[get_bits_pos(k, n)] = true;
    }
    pattern
}

fn main() {
    let mut rng = rand::thread_rng();
    let mc: usize = 100000;

    let mut p;
    let mut approx: BinRatio;
    let mut approx_f: f32;
    let mut err: f32 = 0.0;
    let mut pow: f32 = 0.0;
    for _ in 0..mc {
        p = rng.gen();
        approx = approximate_prob(p, 1e-3);
        approx_f = approx.as_float();
        err += (approx_f - p).abs() / mc as f32;
        pow += (approx.pow as f32) / mc as f32;
    }

    println!("avg. error = {:.9}", err);
    println!("avg. pow   = {:.9}", pow);

    println!("{:?}", make_pattern(BinRatio { num: 0, pow: 3 }));
    println!("{:?}", make_pattern(BinRatio { num: 1, pow: 3 }));
    println!("{:?}", make_pattern(BinRatio { num: 2, pow: 3 }));
    println!("{:?}", make_pattern(BinRatio { num: 3, pow: 3 }));
    println!("{:?}", make_pattern(BinRatio { num: 4, pow: 3 }));
    println!("{:?}", make_pattern(BinRatio { num: 5, pow: 3 }));
    println!("{:?}", make_pattern(BinRatio { num: 6, pow: 3 }));
    println!("{:?}", make_pattern(BinRatio { num: 7, pow: 3 }));
    println!("{:?}", make_pattern(BinRatio { num: 8, pow: 3 }));
}
