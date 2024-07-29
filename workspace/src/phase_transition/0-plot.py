from pathlib import Path
import sys
import numpy as np
import whooie.pyplotdefs as pd

fname = sys.argv[1] if len(sys.argv) > 1 else "output/phase_transition.npz"
print(f"reading from '{fname}'")

outdir = Path("output")
infile = Path(fname)
# infile = outdir.joinpath("phase_transition_h-s-cx-open_every-prob.npz")
# infile = outdir.joinpath("phase_transition_h-s-cx-open_every-prob_small.npz")

data = np.load(str(infile))
p_meas = data["p_meas"]
size = data["size"]
s_mean = data["s_mean"]
s_std_p = data["s_std_p"]
s_std_m = data["s_std_m"]

P = pd.Plotter()
it = enumerate(zip(p_meas, s_mean, s_std_p, s_std_m))
for (k, (p_k, mean_k, std_p_k, std_m_k)) in it:
    P.errorbar(
        size, mean_k, np.array([std_m_k, std_p_k]),
        marker="o", linestyle="-", color=f"C{k % 10}",
        label=f"$p = {p_k:.3f}$",
    )
if size.max() > 30:
    P.set_xscale("log").set_yscale("log")
(
    P
    .ggrid()
    .legend(
        fontsize=4.0,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    )
    .set_xlabel("System size")
    .set_ylabel("Entanglement entropy")
    .savefig(outdir.joinpath("phase_transition.png"))
    # .show()
    .close()
)

