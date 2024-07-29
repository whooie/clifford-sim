import numpy as np
from numpy.random import random
import whooie.pyplotdefs as pd

def floyd_steinberg():
    n = 32
    p = random()
    arr = p * np.ones((n, n), dtype=float)
    for y in range(n):
        for x in range(n):
            old = arr[y, x].copy()
            new = int(round(old))
            arr[y, x] = new
            err = old - new
            if x < n - 1:
                arr[y, x + 1] += err * 7 / 16
            if y < n - 1:
                if x > 0:
                    arr[y + 1, x - 1] += err * 3 / 16
                arr[y + 1, x] += err * 5 / 16
                if x < n - 1:
                    arr[y + 1, x + 1] += err / 16

    img = arr.astype(int)
    pd.Plotter().imshow(1 - img, cmap="gray").set_title(f"{p = :.6}; unrolled")
    for k in range(n):
        img[k, :] = np.roll(img[k, :], k)
    pd.Plotter().imshow(1 - img, cmap="gray").set_title(f"{p = :.6}; rolled")
    pd.show()

def bayer():
    def bayer_matrix(n: int) -> np.ndarray:
        if n < 2:
            raise Exception(f"bayer_matrix: {n=} too small")
        elif abs(np.log2(n) % 1) > 1e-9:
            raise Exception(f"bayer_matrix: bad value {n=}: must be a power of 2")
        elif n == 2:
            return np.array([[0, 2], [3, 1]], dtype=float) / 4
        else:
            n2 = n // 2
            pref = n ** 2
            sub = pref * bayer_matrix(n2)
            m = np.zeros((n, n), dtype=float)
            m[:n2, :n2] = sub
            m[:n2, n2:] = sub + 2
            m[n2:, :n2] = sub + 3
            m[n2:, n2:] = sub + 1
            return m / pref
    n = 32
    m = 8
    p = random()
    M = bayer_matrix(m)
    arr = p * np.ones((n, n), dtype=float)
    img = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            img[i, j] = 1 if p >= M[i % m, j % m] else 0
    pd.Plotter().imshow(1 - img, cmap="gray").set_title(f"{p = :.6}; unrolled")
    for k in range(n):
        img[k, :] = np.roll(img[k, :], k)
    pd.Plotter().imshow(1 - img, cmap="gray").set_title(f"{p = :.6}; rolled")
    pd.show()

if __name__ == "__main__":
    floyd_steinberg()
    # bayer()

