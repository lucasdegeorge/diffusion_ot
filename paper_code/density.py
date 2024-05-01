import numpy as np


def density(d):
    return _mixture_factory(d, np.random.randint(1, 5))


def _mixture_factory(d, num=5):
    ps = [_p_random(d) for _ in range(num)]

    wt = np.ones(num) / num
    wt = wt.reshape(1, num)

    def mix(X):
        vals = np.stack([p(X) for p in ps], axis=1)
        return (vals * wt).sum(axis=1)

    return mix


def _p_factory(Q1, Q2, c1, c2):
    def t1(x):
        x1 = (x - c1)
        t1 = ((x1 @ Q1) * x1).sum(axis=1)
        return t1

    def t2(x):
        x2 = (x - c2)**2
        t2 = ((x2 @ Q2) * x2).sum(axis=1)
        return t2

    def log_p(x):
        return 0.05 * (-t1(x)-t2(x))

    def p(x):
        return np.exp(log_p(x))

    return p


def _p_random(d):
    Q1_ = np.random.randn(d, d)
    Q1 = Q1_.T @ Q1_

    Q2_ = np.random.randn(d, d)
    Q2 = Q2_.T @ Q2_

    c1 = np.random.rand(d) * 4 - 2
    c2 = np.random.rand(d) * 4 - 2

    return _p_factory(Q1, Q2, c1, c2)
