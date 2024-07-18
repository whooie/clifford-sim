import numpy as np
from scipy.linalg import logm, svd

def multikron(*mats) -> np.ndarray[..., 2]:
    if len(mats) == 0:
        return np.array([[1]])
    elif len(mats) == 1:
        return mats[0]
    else:
        return np.kron(mats[0], multikron(*(mats[1:])))

psi0 = np.array([1] + 15 * [0], dtype=complex)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
I = np.array([[1, 0], [0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

U = (
      multikron(SWAP, I, I)
    @ multikron(I, SWAP, I)
    @ multikron(I, I, I, H)
    @ multikron(I, I, CNOT)
    @ multikron(I, I, H, I)
    @ multikron(I, SWAP, I)
    @ multikron(SWAP, I, I)
)

psi = U @ psi0
# print(psi)

def trace(rho: np.ndarray) -> float:
    return sum(rk for rk in np.diag(rho) if abs(rk) > 1e-12)

def trace_qubit(n: int, k: int, rho: np.ndarray) -> np.ndarray:
    T0 = np.kron(
        np.kron(np.eye(2 ** k), np.array([[1, 0]]).T),
        np.eye(2 ** (n - k - 1)),
    )
    T1 = np.kron(
        np.kron(np.eye(2 ** k), np.array([[0, 1]]).T),
        np.eye(2 ** (n - k - 1)),
    )
    return (T0.T @ rho @ T0) + (T1.T @ rho @ T1)

def entropy_log(rho: np.ndarray) -> float:
    return -trace(rho @ logm(rho))

def entropy_svd(rho: np.ndarray) -> float:
    s = svd(rho)[1]
    return -sum(sk ** 2 * np.log(sk**2) for sk in s if abs(sk**2) > 1e-12)

rho = np.outer(psi, psi.conj())
# print(rho)

print(entropy_log(rho))
print(entropy_log(trace_qubit(3, 2, trace_qubit(4, 3, rho))))
print(entropy_log(trace_qubit(3, 0, trace_qubit(4, 0, rho))))


