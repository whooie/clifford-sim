import numpy as np
import whooie.pyplotdefs as pd
fs = pd.pp.rcParams["figure.figsize"]
pd.pp.rcParams["legend.handlelength"] = 4.0

period = np.array([1, 2, 4, 8])

pc_hcz_prob = np.array([[0.0590, 0.0004], [0.059, 0.001], [0.0740, 0.0006], [0.0886, 0.0010]]).T
pc_hcz_cyc = np.array([[0.0980, 0.0009], [0.068, 0.002], [0.130, 0.002], [0.279, 0.003]]).T
pc_hscz_prob = np.array([[0.0752, 0.0003], [0.1409, 0.0002], [0.2739, 0.0006], [0.4967, 0.0005]]).T
pc_hscz_cyc = np.array([[0.056, 0.001], [0.10015, 0.00008], [0.2272, 0.0004], [0.4167, 0.0005]]).T

nu_hcz_prob = np.array([[0.92, 0.01], [0.74, 0.02], [0.736, 0.006], [0.635, 0.008]]).T
nu_hcz_cyc = np.array([[0.36, 0.04], [0.73, 0.03], [0.53, 0.04], [0.26, 0.01]]).T
nu_hscz_prob = np.array([[1.32, 0.01], [1.197, 0.009], [1.29, 0.01], [1.207, 0.009]]).T
nu_hscz_cyc = np.array([[3.3, 0.1], [1.21, 0.03], [0.61, 0.01], [0.58, 0.01]]).T

lmin_hcz_prob = np.array([32, 64, 128, 128])
lmin_hcz_cyc = np.array([16, 64, 32, 64])
lmin_hscz_prob = np.array([8, 32, 64, 32])
lmin_hscz_cyc = np.array([16, 32, 16, 16])

(
    pd.Plotter.new(
        nrows=3,
        sharex=True,
        figsize=[fs[0], 1.5 * fs[1]],
        as_plotarray=True,
    )
    [0]
    .errorbar(
        period, *pc_hcz_prob,
        marker="o", linestyle="-", color="C0",
        label="H/CZ; probabilistic",
    )
    .errorbar(
        period, *pc_hcz_cyc,
        marker="o", linestyle="-", color="C1",
        label="H/CZ; cycling",
    )
    .errorbar(
        period, *pc_hscz_prob,
        marker="D", linestyle="--", color="C0",
        label="H/S/CZ; probabilistic",
    )
    .errorbar(
        period, *pc_hscz_cyc,
        marker="D", linestyle="--", color="C1",
        label="H/S/CZ; cycling",
    )
    .ggrid()
    .legend(
        fontsize=4.0,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    )
    .set_ylabel("$p_c$")
    [1]
    .errorbar(
        period, *nu_hcz_prob,
        marker="o", linestyle="-", color="C4",
        label="H/CZ; probabilistic",
    )
    .errorbar(
        period, *nu_hcz_cyc,
        marker="o", linestyle="-", color="C5",
        label="H/CZ; cycling",
    )
    .errorbar(
        period, *nu_hscz_prob,
        marker="D", linestyle="--", color="C4",
        label="H/S/CZ; probabilistic",
    )
    .errorbar(
        period, *nu_hscz_cyc,
        marker="D", linestyle="--", color="C5",
        label="H/S/CZ; cycling",
    )
    .ggrid()
    .legend(
        fontsize=4.0,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    )
    .set_ylabel("$\\nu$")
    [2]
    .plot(
        period, lmin_hcz_prob,
        marker="o", linestyle="-", color="C8",
        label="H/CZ; probabilistic",
    )
    .plot(
        period, lmin_hcz_cyc,
        marker="o", linestyle="-", color="C3",
        label="H/CZ; cycling",
    )
    .plot(
        period, lmin_hscz_prob,
        marker="D", linestyle="--", color="C8",
        label="H/S/CZ; probabilistic",
    )
    .plot(
        period, lmin_hscz_cyc,
        marker="D", linestyle="--", color="C3",
        label="H/S/CZ; cycling",
    )
    .ggrid()
    .legend(
        fontsize=4.0,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=1.0,
    )
    .set_ylabel("$L_\\mathregular{min}$")
    .set_xlabel("Period")
    .tight_layout()
    .savefig("output/period.png")
    .close()
)

