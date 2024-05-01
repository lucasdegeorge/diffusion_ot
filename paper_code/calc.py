import numpy as np
import pickle
import sys


from utils import solve


def calc(d, a, b, t, n, m, s, e, nswp, e_vld, reps, is_random):
    result = {'ot_true': [], 'ot_ours': [], 'seed': []}
    rep = 0
    seed = 100
    while rep < reps:
        seed += 1
        np.random.seed(seed)

        try:
            ot_true, ot_ours = solve(d, a, b, t, n, m, s, e, nswp, e_vld,
                rep+1, is_random)
        except Exception as ex:
            print('\n\n!!! Bad density. Skip.\n\n')
            continue

        result['seed'].append(seed)
        result['ot_true'].append(ot_true)
        result['ot_ours'].append(ot_ours)

        with open(f'result/result_{d}d.pkl', 'wb') as f:
            pickle.dump(result, f)

        rep += 1


if __name__ == '__main__':
    d = int(sys.argv[1])  # Dimension

    a = -8.              # Spatial grid lower bound
    b = +8.               # Spatial grid upper bound
    t = 5.0               # Final time
    e = 1.E-6             # Accuracy
    nswp = 10             # Number of cross sweeps
    reps = 100            # Number of runs
    s = 1000              # Number of samples for ODE
    e_vld = 1.E-4         # Validation accuracy for TT-cross
    is_random = False     # If true, then use random initial density

    if d == 2:
        n = 250           # Number of spatial points
        m = 250           # Number of time points

    elif d == 3:
        n = 100           # Number of spatial points
        m = 100           # Number of time points

    elif d == 7:          # Use random density in this case
        n = 50            # Number of spatial points
        m = 50            # Number of time points
        e = 1.E-8         # Accuracy
        nswp = 2          # Number of cross sweeps
        is_random = True

    else:
        raise ValueError('Dimension is not supported')

    calc(d, a, b, t, n, m, s, e, nswp, e_vld, reps, is_random)
