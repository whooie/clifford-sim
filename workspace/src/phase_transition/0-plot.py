from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output")
infile = outdir.joinpath("phase_transition.npz")
# infile = outdir.joinpath("phase_transition_h-s-cx-open_every-prob.npz")
# infile = outdir.joinpath("phase_transition_h-s-cx-open_every-prob_small.npz")

data = np.load(str(infile))
p_meas = data["p_meas"]
size = data["size"]
entropy = data["entropy"]

P = pd.Plotter()
for (k, (pk, sk)) in enumerate(zip(p_meas, entropy)):
    P.plot(
        size, sk,
        marker="o", linestyle="-", color=f"C{k % 10}",
        label=f"$p = {pk:.3f}$",
    )
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
    .show()
    .close()
)

