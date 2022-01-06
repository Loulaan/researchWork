from collections import Counter
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from utils.hmatrStatistics import Hmatr

rpy2.robjects.numpy2ri.activate()

rssa = importr('Rssa')

N = 700
w1 = 1 / 10
w2 = 1 / 5
Q = 301  # 301 номер, значит разладка в ряде будет на 302й точке, если ряд задан с 0.
B = 100
T_ = 100
L = 50
r = 2
method = "svd"

def generate_series(omega, C=1, N=700, Q=301):
    w1, w2 = omega
    series = lambda n: C*np.sin(2*np.pi*w1*n) if n < Q-1 else C*np.sin(2*np.pi*w2*n)
    return [series(i) for i in range(N)]

def new_computation():
    f = generate_series((w1, w2), N=N, Q=Q)
    hm = Hmatr(f, B, T_, L, neig=r, svdMethod=method)
    hm.compute_single_row(0)


def analytical_computation():
    f = generate_series((w1, w2), N=N, Q=Q)
    omegas = [[w1, w1]] * Q + [[w1, w2]] * (N - Q)
    hm = Hmatr(f, B, T_, L, neig=r, svdMethod=method)
    hm.compute_row_analytical(omegas)
    print(hm.compute_single_val_analytical(0.1, 0.1))
    print(hm.compute_single_val_analytical(0.1, 0.5))



if __name__ == "__main__":
    new_computation()
