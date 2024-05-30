from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output")
infile = outdir.joinpath("mutual_information.npz")

data = np.load(str(infile))
size = data["size"]
p_meas = data["p_meas"]
mutinf = data["mutinf"]

P = pd.Plotter()
for (k, (lk, ik)) in enumerate(zip(size, mutinf)):
    (
        P
        .plot(
            p_meas, ik,
            marker="o", linestyle="-", color=f"C{k % 10}",
            label=f"$L = {lk:.0f}$",
        )
        .axvline(
            p_meas[ik.argmax()],
            linestyle="--", color=f"C{k % 10}",
        )
    )
(
    P
    .ggrid()
    .legend(fontsize="xx-small")
    .set_xlabel("Measurement probability")
    .set_ylabel("Mutual information")
    .savefig(outdir.joinpath("mutual_information.png"))
    .show()
    .close()
)

