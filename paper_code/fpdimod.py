from fpcross import Equation
from fpcross import FPCross
import numpy as np
import ot
from scipy.linalg import solve_lyapunov
import teneva
from time import perf_counter as tpc
from tqdm import tqdm


def fpdimod(p0_func, d, a, b, t, n, m, s, e, nswp, e_vld, P0_init=None):
    if P0_init is None:
        P0 = _density_build(p0_func, d, a, b, n, e, nswp, e_vld)
        P_all, P_coef_all = _fpe_solve(p0_func, d, a, b, t, n, m, e, nswp)
    else:
        P0 = P0_init
        P_all, P_coef_all = _fpe_solve(None, d, a, b, t, n, m, e, nswp, P0=P0)

    P_der_coef_all = _pdf_der_build(P_all, d, a, b, n)

    X0 = _samples_build(P0, a, b, n, s)

    Xt = _ode_solve(X0, P_coef_all, P_der_coef_all, a, b, t, m)

    ot_true, ot_ours = _ot_build(X0, Xt)

    return X0, Xt, ot_true, ot_ours


def _density_build(p0_func, d, a, b, n, eps, nswp, e_vld, m_tst=1.E+4, r=10):
    info = {}
    func = teneva.Func(d, f_comp=p0_func, name='P0')
    func.set_lim(a, b)
    func.set_grid(n, kind='cheb')
    func.build_vld_ind(m_tst/2)
    func.build_tst_poi(m_tst/2)
    func.rand(r)
    func.cross(eps=eps, nswp=nswp, e_vld=e_vld, dr_min=1, dr_max=1,
        info=info, log=False)
    if info['e_vld'] > e_vld:
        raise ValueError('The TT-cross did not converged')
    return func.Y


def _fpe_solve(p0_func, d, a, b, t, n, m, e, nswp, dr_min=1, dr_max=1, P0=None):
    """Solve Fokker-Planck equation and return solutions from all time steps.

    Function returns two lists of the lenth "m": "Y_list" and "A_list". The
    first corresponds to the solutions at each time step (which are the tensors
    on a spatial grid in the TT-format), and the second corresponds to the
    Chebyshev coefficient tensor in the TT-format (which have the same
    dimensions as the solution tensors).

    """
    class EquationFP(Equation):
        def build_r0(self):
            if P0 is None:
                return super().build_r0()

            self.Y0 = teneva.copy(P0)
            return self.Y0

        def r0(self, X):
            return p0_func(X)

        def f(self, X, t):
            return -X

        def f1(self, X, t):
            return -np.ones_like(X)

        def init(self):
            self.with_rs = True
            self.with_rt = False

            self.W = solve_lyapunov(np.eye(self.d), 2.*self.coef*np.eye(self.d))
            self.Wi = np.linalg.inv(self.W)
            self.Wd = np.linalg.det(self.W)

        def rs(self, X):
            r = np.exp(-0.5 * np.diag(X @ self.Wi @ X.T))
            r /= np.sqrt(2**self.d * np.pi**self.d * self.Wd)
            return r

    eq = EquationFP(d, e, name='FPE')
    eq.set_grid(n, a, b)
    eq.set_grid_time(m, t)
    eq.set_coef(1.)
    eq.set_cross_opts(nswp=nswp, dr_min=dr_min, dr_max=dr_max, r=1)
    eq.init()

    fpc = FPCross(eq, with_hist=False, with_y_list=True, with_a_list=True)
    fpc.solve()
    # fpc.plot(f'./result/result_fpcross_{d}d')

    return fpc.Y_list, fpc.A_list


def _ode_solve(X0, P_coef_all, P_der_coef_all, a, b, t, m):
    """Solve ODE dx/dt = -(x + nabla log p(x, t)).

    Note:
        When solving the Fokker-Planck equation, we used a time grid with "m"
        points and with a step size "h = t/m". The Runge-Kutta method uses
        "step" and "half-step" values, and for the ODE we use a 2-x coarser time
        grid (i.e., h -> 2h, m -> m/2), hence we do not need to interpolate our
        PDE values ("P_coef_all" and "P_der_coef_all") in time (TODO: check!).

    """
    def solve(f, y0, t, m, h):
        """Runge-Kutta solver of the 4th order."""
        y = y0.copy()
        for _ in range(1, m):
            k1 = h * f(y, t)
            k2 = h * f(y + 0.5 * k1, t + 0.5 * h)
            k3 = h * f(y + 0.5 * k2, t + 0.5 * h)
            k4 = h * f(y + k3, t + h)
            y += (k1 + 2 * (k2 + k3) + k4) / 6.
            t += h

            tqdm_.set_postfix_str('', refresh=True)
            tqdm_.update(1)

        return y

    def rhs(x_cur, t_cur):
        t_cur_ind = int(t_cur / h)
        P_coef = P_coef_all[t_cur_ind]
        P_der_coef = P_der_coef_all[t_cur_ind]

        p = teneva.cheb_get(x_cur, P_coef, a, b)
        p_der = [teneva.cheb_get(x_cur, P, a, b) for P in P_der_coef]
        p_res = [p_der_ / (p + 1.E-15) for p_der_ in p_der]
        p_res = np.array(p_res).T

        return -1. * (x_cur + p_res)

    h = t / m
    tqdm_ = tqdm(desc='Sampl', unit='iter', total=int(m/2)-2, ncols=90)

    return solve(rhs, X0, 0., int(m/2), h*2)


def _ot_build(X0, Xt):
    s = Xt.shape[0]
    a = np.ones((s,)) / s
    b = np.ones((s,)) / s
    M = ot.dist(X0, Xt)

    G0, info = ot.emd(a, b, M, log=True, numItermax=500000)
    ot_true = info['cost']

    ot_ours = np.diag(M).sum() / s

    return ot_true, ot_ours


def _pdf_der_build(P_all, d, a, b, n):
    """Builds interpolation for gradient for all TT-tensors from the given list.

    Function returns the list "P_der_coef_all", which has the same length as
    "P_all". Its items "P_der_coef" are the lists of the length "d" (where "d"
    is a number of dimensions), and the k-th item of "P_der_coef" is a tensor
    of Chebyshev coefficients in the TT-format for the k-th partial derivative.

    """
    D = teneva.cheb_diff_matrix(a, b, n)
    P_der_coef_all = []
    for P in P_all:
        P_der_coef_all_cur = []
        for k in range(d):
            P_der = teneva.copy(P)
            P_der[k] = np.einsum('ij,kjm->kim', D, P_der[k])
            P_der_coef = teneva.cheb_int(P_der)
            P_der_coef_all_cur.append(P_der_coef)
        P_der_coef_all.append(P_der_coef_all_cur)
    return P_der_coef_all


def _samples_build(P, a, b, n, s, unique=True):
    """Build spatial samples from the given distribution in the TT-format.

    Function returns numpy array "X" of the shape "[s, d]", where "d" is a
    number of dimensions and "s" is a number of samples. The samples are
    generated from the given d-dimensional TT-tensor "P".

    """
    I = teneva.sample_ind_rand(P, s, unique, m_fact=5, max_rep=100000)
    X = teneva.ind_to_poi(I, a, b, n, 'cheb')
    return X
