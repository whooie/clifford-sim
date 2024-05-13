from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

infile = Path("output/convergence_test.npz")
data = np.load(str(infile))

depth = data["depth"]
entropy = data["entropy"]
size = data["size"][0]
p_meas = data["p_meas"][0]
tol = data["tol"][0]

d0 = np.mean(depth)
dpm = np.std(depth)

s0 = np.mean(entropy)
spm = np.std(entropy)

(
    pd.Plotter()
    .hist(depth, bins=50)
    .axvline(d0, color="k")
    .axvline(d0 + dpm, color="k", linestyle=":")
    .axvline(d0 - dpm, color="k", linestyle=":")
    .ggrid().grid(False, which="both")
    .set_xlabel("Circuit depth")
    .set_title(f"N = {size}, p_meas = {p_meas:.3f}, tol = {tol:.1e}")
)

(
    pd.Plotter()
    .hist(entropy, bins=50)
    .axvline(s0, color="k")
    .axvline(s0 + spm, color="k", linestyle=":")
    .axvline(s0 - spm, color="k", linestyle=":")
    .ggrid().grid(False, which="both")
    .set_title(f"N = {size}, p_meas = {p_meas:.3f}, tol = {tol:.1e}")
    .set_xlabel("Entropy")
    .set_title(f"N = {size}, p_meas = {p_meas:.3f}, tol = {tol:.1e}")
)

pd.show()

