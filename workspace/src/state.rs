use clifford_sim::stab::*;

fn main() {
    let mut stab = Stab::new(4);
    println!("t=0");
    println!("{}", stab.as_kets().unwrap());

    (0..4).for_each(|k| { stab.apply_h(k); });
    stab.apply_cz(0, 1).apply_cz(2, 3);
    println!("t=1");
    println!("{}", stab.as_kets().unwrap());

    (0..4).for_each(|k| { stab.apply_h(k); });
    stab.apply_cz(1, 2);
    println!("t=2");
    println!("{}", stab.as_kets().unwrap());

    (0..4).for_each(|k| { stab.apply_h(k); });
    stab.apply_cz(0, 1).apply_cz(2, 3);
    println!("t=3");
    println!("{}", stab.as_kets().unwrap());

    (0..4).for_each(|k| { stab.apply_h(k); });
    stab.apply_cz(1, 2);
    println!("t=4");
    println!("{}", stab.as_kets().unwrap());

    (0..4).for_each(|k| { stab.apply_h(k); });
    stab.apply_cz(0, 1).apply_cz(2, 3);
    println!("t=5");
    println!("{}", stab.as_kets().unwrap());

    (0..4).for_each(|k| { stab.apply_h(k); });
    stab.apply_cz(1, 2);
    println!("t=6");
    println!("{}", stab.as_kets().unwrap());

    (0..4).for_each(|k| { stab.apply_h(k); });
    stab.apply_cz(0, 1).apply_cz(2, 3);
    println!("t=7");
    println!("{}", stab.as_kets().unwrap());

    (0..4).for_each(|k| { stab.apply_h(k); });
    stab.apply_cz(1, 2);
    println!("t=8");
    println!("{}", stab.as_kets().unwrap());
}

