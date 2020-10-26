"""
Generic bisection algorithm for simplex optimization
For strongly convex g, solve:

 maximize  <p, x> - sum_j g(p_j)
 s.t.      p >= 0
           <1, p> = 1

by reduction to 1-d root finding.
"""


import numpy as np
from scipy import optimize

try:
    import pytest
except:
    pass

_xtol = 2e-12
_rtol = 4 * np.finfo(float).eps
_METHODS = {
    'brentq': optimize.brentq,
    'brenth': optimize.brenth,
    'ridder': optimize.ridder,
    'bisect': optimize.bisect
}

class TsallisSeparable(object):
    """Tsallis up to a constant.
    g(t) = t^α / α(α-1)
    g'(t) = t^(α-1) / (α-1)
    (g')^-1 (x) = (t/β)^β where β = 1 / (α-1)
    Satisfies A.2 and A.3 but not A.1
    """

    def __init__(self, alpha=1.5):
        self.alpha = alpha

    def g(self, t):
        val = t ** (self.alpha)
        val /= self.alpha * (self.alpha - 1)
        return val

    def gp(self, t):
        return (t ** (self.alpha - 1)) / (self.alpha - 1)

    def gp_inv(self, x):
        beta = 1 / (self.alpha - 1)
        return (x / beta) ** beta


def _make_funcs_bounds(x, penalty):
    d = x.shape[0]
    x_max = np.max(x)

    gp_zero = penalty.gp(0)

    def p(tau):
        return penalty.gp_inv(np.maximum(x - tau, gp_zero))

    def nu(tau):
        res = gp_zero - x + tau
        res[x - tau >= gp_zero] = 0
        return res

    def f(tau):
        p0 = p(tau)
        return np.sum(p0) - 1

    tau_min = x_max - penalty.gp(1)
    tau_max = x_max - penalty.gp(1 / d)

    return f, tau_min, tau_max, p, nu



def optimize_bisect(x, penalty, method="brentq", maxiter=100, xtol=_xtol, rtol=_rtol):

    f, tau_min, tau_max, p, nu = _make_funcs_bounds(x, penalty)

    # scipy solvers check f(tau_min) and f(tau_max) to have opposite signs.
    # This might fail if either of them is 0 and numerical instability comes
    # into play, so check manually. Unfortunately adds 2 redundant func_calls

    if f(tau_min) < 0:
        tau0 = tau_min
        results = optimize.zeros.RootResults(tau0, 0, 1, 0)
    elif f(tau_max) > 0:
        tau0 = tau_max
        results = optimize.zeros.RootResults(tau0, 0, 2, 0)
    else:
        solver = _METHODS[method]
        tau0, results = solver(f, tau_min, tau_max, xtol=xtol, rtol=rtol,
                maxiter=maxiter, full_output=True, disp=False)
        results.function_calls += 2

    return p(tau0), nu(tau0), tau0, results


def optimize_secant(x, penalty, maxiter=100):
    f, tau_min, tau_max, p, nu = _make_funcs_bounds(x, penalty)

    lo, hi = tau_min, tau_max

    f_lo = f(lo)
    # print(lo, f_lo)
    if f_lo == 0:
        return lo, p(lo)

    f_hi = f(hi)
    # print(hi, f_hi)
    if f_hi == 0:
        return hi, p(hi)

    for it in range(maxiter):

        if np.abs(hi - lo) < _rtol * np.abs(hi + lo):
            break

        # tau = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        tau = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)

        lo, hi = hi, tau
        f_lo, f_hi = f_hi, f(tau)

        # print(hi, f_hi)

    return hi, p(hi)


def optimize_falsi(x, penalty, maxiter=100):
    f, lo, hi, p, nu = _make_funcs_bounds(x, penalty)

    f_lo = f(lo)
    if f_lo == 0:
        return lo, p(lo)

    f_hi = f(hi)
    if f_hi == 0:
        return hi, p(hi)

    f_lo_sign = np.sign(f_lo)
    f_hi_sign = np.sign(f_hi)

    side = 0
    tau = lo

    for it in range(maxiter):
        if f_hi == f_lo:
            break

        tau = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
        if np.abs(hi - lo) < _rtol * np.abs(hi + lo) / 2:
            break

        f_tau = f(tau)
        f_tau_sign = np.sign(f_tau)

        if f_tau_sign == f_lo_sign:
            lo, f_lo = tau, f_tau

            if side == -1:
                f_hi /= 2

            side = -1

        else:
            hi, f_hi = tau, f_tau

            if side == +1:
                f_lo /= 2

            side = +1

    # if (it >= maxiter - 1):
    #     print('max iter reached')

    return tau, p(tau)


def optimize_bisect_custom(x, penalty, maxiter=100, tol=1e-10, verbose=False):

    f, tau_min, tau_max, p, nu = _make_funcs_bounds(x, penalty)

    # roll our own bisection
    lower, upper = tau_min, tau_max
    current = np.inf

    for it in range(maxiter):
        if current ** 2 < tol:
            break
        tau = (upper + lower) / 2
        current = f(tau)
        if current <= 0:
            upper = tau
        else:
            lower = tau

        if verbose:
            p0 = p(tau)
            nu0 = nu(tau)
            grad = x + nu0 - penalty.gp(p0) - tau
            norm_grad = np.dot(grad, grad)
            print(f"""\
Iteration {it}
    Primal infeasability: {current ** 2},
    Dual infeasability: {np.sum(nu0[nu0 < 0] ** 2)},
    Compl slackness: {np.dot(p0, nu0) ** 2}
    Non-stationarity: {norm_grad}""")

    return tau, p(tau)


# pytest generate vectors
@pytest.fixture
def _vectors():
    rng = np.random.RandomState(42)
    # a few special case vectors
    all_pos = 1 * np.ones(10)
    all_neg = -1 * np.ones(10)

    all_neg_rand = -np.abs(rng.randn(10))

    all_large = 10 * np.ones(10)
    all_small = -10 * np.ones(10)

    one_large = -1 * np.ones(10)
    one_large[0] = 100

    one_small = 1 * np.ones(10)
    one_small[0] = -100

    # make worst case vector
    worst_case = np.zeros(1)
    for rho in range(1, 10):
        worst_case -= worst_case.mean()
        s = np.sum(worst_case ** 2)
        worst_case = np.append(worst_case, - np.sqrt((1 - s) / rho))
    worst_case *= 2

    vectors = [all_pos, all_neg, all_neg_rand, all_large, all_small, one_large,
        one_small, worst_case]

    for _ in range(100000):
        dim = rng.randint(1, 20)
        std = np.exp(rng.uniform(-4, 4))
        x = rng.randn(dim) * std
        vectors.append(x)

    return vectors


@pytest.mark.parametrize('alpha', (1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 2.5, 3, 4))
def test_custom_bisect(alpha, _vectors):
    tol = 1e-8
    penalty = TsallisSeparable(alpha)
    for x in _vectors:

        f, tau_min, tau_max, _, nu = _make_funcs_bounds(x, penalty)
        tau, p0 = optimize_bisect_custom(x, penalty, tol=tol)

        assert (np.sum(p0) - 1) ** 2 < tol  # primal feas

        nu0 = nu(tau)
        assert np.all(nu0 >= 0)  # dual feas
        assert np.dot(p0, nu(tau)) == 0  # compl slackness

        grad = x + nu0 - penalty.gp(p0) - tau   # stationarity
        assert(np.allclose(grad, np.zeros_like(x)))


@pytest.mark.parametrize('alpha', (1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 2.5, 3, 4))
def test_falsi(alpha, _vectors):

    penalty = TsallisSeparable(alpha)
    for x in _vectors:
        tau, p = optimize_falsi(x, penalty)
        if not np.allclose(np.sum(p) - 1, 0):
            print(x)
        assert np.allclose(np.sum(p) - 1, 0, atol=1e-6)


@pytest.mark.parametrize('alpha', (1.1, 1.3, 1.5, 1.7, 1.9, 2.0))
def test_secant(alpha, _vectors):

    penalty = TsallisSeparable(alpha)
    for x in _vectors:
        tau, p = optimize_secant(x, penalty)
        assert np.allclose(np.sum(p) - 1, 0)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n_samples = 1000
    dim = 200
    xs = np.random.RandomState(0).randn(n_samples, dim)

    for alpha in (1.1, 1.5, 1.8, 2.2, 3):
        plt.figure()
        plt.title(f"Tsallis alpha={alpha}")
        penalty = TsallisSeparable(alpha)

        func_calls = {key: [] for key in _METHODS}

        for x in xs:

            for method in _METHODS.keys():
                p0, nu0, tau0, results = optimize_bisect(x, penalty,
                                                         xtol=1e-3,
                                                         method=method,
                                                         maxiter=10000)
                assert results.converged
                func_calls[method].append(results.function_calls)

        plt.boxplot([func_calls[m] for m in sorted(_METHODS)],
                    labels=sorted(_METHODS))
    alt.show()

# todo falsi alpha=4 is not very accurate for this point
# [ 0.54893122  1.03712757 -0.00524181 -0.17058755 -0.17790246 -0.36595707
# -0.14739792  0.70380788 -0.05965505  0.15079314  0.10152055  0.05851401
#  -0.07907953  0.19227826 -0.56079428 -0.64078167 -0.38494253  0.26910706
#    0.31196961]

