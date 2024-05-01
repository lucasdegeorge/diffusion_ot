from time import perf_counter as tpc


from density import density
from density_random import density_random
from fpdimod import fpdimod


def solve(d, a, b, t, n, m, s, e, nswp, e_vld, iter=None, is_random=False):
    _t = tpc()

    text = '\n\n'
    text += f'='*46 + '\n'
    text += f'--- fpdimod start... '
    text += f'[dim = {d}] '
    text += f'(iter = {iter:-6d}) ' if iter else ''
    text += f'\n'
    print(text)

    if is_random:
        p0_func = None
        P0 = density_random(d, a, b, n, e)
    else:
        p0_func = density(d)
        P0 = None

    X0, Xt, ot_true, ot_ours = fpdimod(p0_func, d, a, b, t, n, m, s,
        e, nswp, e_vld, P0)

    _t = tpc() - _t

    print(f'\n\n---------------               Done')
    print(f'Dimension     : {d:-18.0f}')
    print(f'Spatial grid  : {n:-18.0f}')
    print(f'Time    grid  : {m:-18.0f}')
    print(f'Samples (ODE) : {s:-18.0f}')
    print(f'---------------')
    print(f'Solve time    : {_t:-18.5f}')
    print(f'OT true       : {ot_true:-18.14f}')
    print(f'OT ours       : {ot_ours:-18.14f}')
    print(f'---------------\n\n')

    return ot_true, ot_ours
