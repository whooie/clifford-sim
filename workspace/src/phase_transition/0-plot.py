from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output")
infile = outdir.joinpath("phase_transition.npz")

data = np.load(str(infile))
p_meas = data["p_meas"]
size = data["size"]
entropy = data["entropy"]

P = pd.Plotter()
for (k, (pk, sk)) in enumerate(zip(p_meas, entropy)):
    P.loglog(
        size, sk,
        marker="o", linestyle="-", color=f"C{k % 10}",
        label=f"$p = {pk:.3f}$",
    )
(
    P
    .ggrid()
    .legend(fontsize="xx-small")
    .set_xlabel("System size")
    .set_ylabel("Entanglement entropy")
    .savefig(outdir.joinpath("phase_transition.png"))
    .show()
    .close()
)

