from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output")
infile = outdir.joinpath("entropy_test.npz")
data = np.load(str(infile))
s = data["entropy"]

t = np.arange(s.shape[0])

(
    pd.Plotter()
    .plot(t, s)
    .ggrid()
    .set_xlabel("Time")
    .set_ylabel("Entanglement entropy")
    .show()
)

