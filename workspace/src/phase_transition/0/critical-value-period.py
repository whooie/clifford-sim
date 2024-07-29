from typing import Callable
import lmfit
import numpy as np
from whooie.analysis import ExpVal
import whooie.pyplotdefs as pd

P = pd.Plotter()
P2 = pd.Plotter()

# H/CZ
data = np.array([
    # [ period, pc, pc error ]
    [1,  0.0590, 0.0004],
    [2,  0.059,  0.001 ],
    [4,  0.0740, 0.0006],
    [8,  0.0886, 0.0010],
    [9,  0.9916, 0.0002],
    [10, 0.9984, 0.0003],
    [12, 0.9997, 0.0004],
    [14, 0.990,  0.001 ],
])

period, pc, pc_err = data.T
P.errorbar(
    period, pc, pc_err,
    marker="o", linestyle=":", color="C0",
    label="H/CZ; probabilistic",
)
P2.errorbar(
    period, pc / period, pc_err / period,
    marker="o", linestyle=":", color="C0",
    label="H/CZ; probabilistic",
)

# H/S/CZ
data = np.array([
    # [ period, pc, pc error ]
    [1,  0.0752, 0.0003],
    [2,  0.1409, 0.0002],
    [4,  0.2739, 0.0006],
    [8,  0.4967, 0.0005],
    [10, 0.5852, 0.0005],
    [12, 0.667,  0.001 ],
    [14, 0.748,  0.001 ],
    [16, 0.7718, 0.0008],
    [18, 0.8298, 0.0007],
    [20, 0.8545, 0.0003],
    [22, 0.8725, 0.0009],
    [24, 0.8933, 0.0002],
    [26, 0.9177, 0.0007],
    [28, 0.945,  0.001 ],
    [29, 0.9436, 0.0005],
    [30, 0.9462, 0.0002],
    [35, 0.9730, 0.0004],
])

period, pc, pc_err = data.T
P.errorbar(
    period, pc, pc_err,
    marker="o", linestyle="", color="C1",
    label="H/S/CZ; probabilistic",
)
P2.errorbar(
    period, pc / period, pc_err / period,
    marker="o", linestyle="", color="C1",
    label="H/S/CZ; probabilistic",
)

def m1(
    params: lmfit.Parameters,
    x: np.ndarray[float, 1],
) -> np.ndarray[float, 1]:
    a = params["a"].value
    b = params["b"].value
    return 1 - a / x ** b

def m2(
    params: lmfit.Parameters,
    x: np.ndarray[float, 1],
) -> np.ndarray[float, 1]:
    a = params["a"].value
    b = params["b"].value
    return 1 - a / 2 ** (b * x)

def residuals(
    params: lmfit.Parameters,
    model: Callable[[lmfit.Parameters, np.ndarray[float, 1]], np.ndarray[float, 1]],
    x: np.ndarray[float, 1],
    y: np.ndarray[float, 1],
    err: np.ndarray[float, 1],
) -> np.ndarray[float, 1]:
    m = model(params, x)
    return ((m - y) / err)**2

params = lmfit.Parameters()
params.add("a", value=1.0, min=0.0)
params.add("b", value=1.0, min=0.0)
fit_m1 = lmfit.minimize(residuals, params, args=(m1, period, pc, pc_err))
fit_m2 = lmfit.minimize(residuals, params, args=(m2, period, pc, pc_err))
if not (fit_m1.success and fit_m2.success):
    raise Exception("failure in fit")
xplot = np.linspace(period.min(), period.max(), 1000)
yplot_m1 = m1(fit_m1.params, xplot)
yplot_m2 = m2(fit_m2.params, xplot)

a_m1 = ExpVal(fit_m1.params["a"].value, fit_m1.params["a"].stderr)
b_m1 = ExpVal(fit_m1.params["b"].value, fit_m1.params["b"].stderr)

a_m2 = ExpVal(fit_m2.params["a"].value, fit_m2.params["a"].stderr)
b_m2 = ExpVal(fit_m2.params["b"].value, fit_m2.params["b"].stderr)

print(
f"""
-- m1 --
a = {a_m1.value_str()}
b = {b_m1.value_str()}
-- m2 --
a = {a_m2.value_str()}
b = {b_m2.value_str()}
"""[1:-1]
)

(
    P
    .plot(
        xplot, yplot_m2,
        marker="", linestyle="-", color="C1",
        label="$1 - a / 2^{b T}$",
    )
    .ggrid()
    .legend(fontsize="xx-small")
    .set_xlabel("Period")
    .set_ylabel("$p_c$")
    .savefig("output/critical-value-period.png")
    .close()
)
(
    P2
    .plot(
        xplot, yplot_m2 / xplot,
        marker="", linestyle="-", color="C1",
        label="$(1 - a / 2^{b T}) / T$",
    )
    .ggrid()
    .legend(fontsize="xx-small")
    .set_xlabel("Period")
    .set_ylabel("$p_c / T$")
    .savefig("output/critical-value-period-per.png")
    .close()
)

