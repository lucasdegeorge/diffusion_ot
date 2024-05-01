import numpy as np
import teneva


def tensor_rand(n, r):
    def rand_func(size):
        return np.random.uniform(0., 1., size=size)
    return teneva.tensor_rand(n, r, rand_func)


def density_random(d, a, b, n, e, r=2, m_tst=1.E+4):
    Y = tensor_rand([n]*d, r)

    def gauss(X, coef=1.):
        alpha = 2. * coef
        r = np.exp(-np.sum(X*X, axis=1) / alpha)
        r /= (np.pi * alpha)**(d/2)
        return r

    func = teneva.Func(d, f_comp=gauss, name='gauss')
    func.set_lim(a, b)
    func.set_grid(n, kind='cheb')
    func.build_vld_ind(m_tst)
    func.rand(r)
    func.cross(nswp=20, e_vld=e, dr_min=1, dr_max=1, log=False)

    Y = teneva.mul(Y, func.Y)
    Y = teneva.truncate(Y, e)

    A = teneva.cheb_int(Y)
    s = teneva.cheb_sum(A, a, b)
    Y = teneva.mul(1. / s, Y)

    return Y
