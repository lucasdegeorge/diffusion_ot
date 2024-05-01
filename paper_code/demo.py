import numpy as np


from utils import solve


if __name__ == '__main__':
    np.random.seed(42)

    d = 3               # Dimension
    a = -8.             # Spatial grid lower bound
    b = +8.             # Spatial grid upper bound
    t = 5.0             # Final time
    n = 100             # Number of spatial points
    m = 100             # Number of time points
    s = 1000            # Number of samples for ODE
    e = 1.E-6           # Accuracy
    nswp = 10           # Number of cross sweeps
    e_vld = 1.E-4       # Validation accuracy for TT-cross
    is_random = False   # If true, then use random initial density

    solve(d, a, b, t, n, m, s, e, nswp, e_vld, is_random=is_random)
