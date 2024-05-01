import numpy as np
import pickle
import sys


def check(d):
    with open(f'result/result_{d}d.pkl', 'rb') as f:
        result = pickle.load(f)

    err = []
    for ot_true, ot_ours in  zip(result['ot_true'], result['ot_ours']):
        err.append(abs((ot_true - ot_ours) / ot_ours))
    err = np.array(err)

    print(f'Iterations        : {err.size:-7.0f}')
    print(f'Error max         : {np.max(err):-7.1e}')
    print(f'Error avg         : {np.mean(err):-7.1e}')
    print(f'Errors which are bigger than "1.E-14"  :')

    for i in range(len(err)):
        if err[i] > 1.E-14:
            text = ''
            text += f'>>> error = {err[i]:-7.1e} | '
            text += f'ot_true = {result["ot_true"][i]:12.8f} | '
            text += f'ot_ours = {result["ot_ours"][i]:12.8f} | '
            text += f'seed = {result["seed"][i]} | '
            print(text)


if __name__ == '__main__':
    np.random.seed(42)

    d = int(sys.argv[1])

    check(d)
