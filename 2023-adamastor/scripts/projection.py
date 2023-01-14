import numpy as np
from sklearn.isotonic import isotonic_regression
from scipy.optimize import check_grad
from lightning.impl.penalty import prox_tv1d

# isotonic regression is a fundamental building block

def prox_owl(v, w):
    """Proximal operator of the OWL norm dot(w, reversed(sort(v)))
    Follows description and notation from:
    X. Zeng, M. Figueiredo,
    The ordered weighted L1 norm: Atomic formulation, dual norm,
    and projections.
    eprint http://arxiv.org/abs/1409.4271
    """

    # wlog operate on absolute values
    v_abs = np.abs(v)
    ix = np.argsort(v_abs)[::-1]
    v_abs = v_abs[ix]
    # project to K+ (monotone non-negative decreasing cone)
    v_abs = isotonic_regression(v_abs - w, y_min=0, increasing=False)

    # undo the sorting
    inv_ix = np.zeros_like(ix)
    inv_ix[ix] = np.arange(len(v))
    v_abs = v_abs[inv_ix]

    return np.sign(v) * v_abs


def _oscar_weights(alpha, beta, size):
    w = np.arange(size - 1, -1, -1, dtype=np.double)
    w *= beta
    w += alpha
    return w


def prox_oscar(x, alpha=1.0):
    w = _oscar_weights(alpha=0, beta=alpha, size=len(x))
    return prox_owl(x, w)


def project_simplex(v, z=1):
    v = np.array(v)
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


class Squared(object):
    title = 'sparsemax'
    def __init__(self, theta):
        self.theta = theta

    def val(self, y):
        return np.dot(y, self.theta) - 0.5 * np.sum(y ** 2)

    def project(self, *args, **kwargs):
        return project_simplex(self.theta)


class Gini(object):
    title = 'sparsemax (gini)'
    def __init__(self, theta):
        self.theta = theta

    def val(self, y):
        return np.dot(y, self.theta) + 0.5 * np.sum(y * (1 - y))

    def grad(self, y):
        return self.theta - y + 0.5

    def project(self, *args, **kwargs):
        return project_simplex(self.theta)


class Linear(object):
    title = 'linear'

    def __init__(self, theta):
        self.theta = theta

    def val(self, y):
        return np.dot(y, self.theta)

    def grad(self, y):
        return self.theta

    def project(self, *args, **kwargs):
        out = np.zeros_like(self.theta)
        out[np.argmax(self.theta)] = 1
        return out


class Entropy(object):
    title = 'softmax'
    def __init__(self, theta):
        self.theta = theta

    def val(self, y):
        return np.dot(y, self.theta) - np.sum(y * np.log(y))

    def grad(self, y):
        return self.theta - 1 - np.log(y)

    def project(self, *args, **kwargs):
        return np.exp(self.theta) / np.sum(np.exp(self.theta))


class Oscarmax(object):
    title = 'oscarmax'
    def __init__(self, theta, alpha=0.1):
        self.theta = theta
        self.alpha = alpha

    def val(self, x):
        norm = np.sum(x ** 2)
        pen = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                pen += max(x[i], x[j])
        return np.dot(x, self.theta) - 0.5 * norm - self.alpha * pen

    def project(self, *args, **kwargs):
        tmp = prox_oscar(self.theta, self.alpha)
        return project_simplex(tmp)


class Fusedmax(object):
    title = 'fusedmax'
    def __init__(self, theta, alpha=0.1):
        self.theta = theta
        self.alpha = alpha

    def val(self, x):
        norm = np.sum(x ** 2)
        pen = 0
        for i in range(1, len(x)):
            pen += np.abs(x[i] - x[i - 1])
        return np.dot(x, self.theta) - 0.5 * norm - self.alpha * pen

    def project(self, *args, **kwargs):
        tmp = self.theta.copy()
        prox_tv1d(tmp, self.alpha)
        return project_simplex(tmp)



class TsallisMax(object):
    def __init__(self, theta, p=1.5):
        self.theta = theta
        self.p = p
        self.n_features = theta.shape[0]

    def val(self, y):
        p = self.p
        return np.dot(y, self.theta) - (np.sum(y - y ** p)) / (p * (1 - p))

    def grad(self, y):
        p = self.p
        return self.theta - (1 - p * (y ** (p - 1))) / (p * (1 - p))

    @property
    def title(self):
        return 'tsallismax-{}'.format(self.p)


class Tsallis15(TsallisMax):
    def __init__(self, theta):
        self.theta = theta
        self.p = 1.5

    def project(self):
        x = self.theta / 2
        sigma = np.argsort(x)[::-1]

        best_nnz = None
        best_delta = -np.inf

        # first: check delta for all possible nnz
        for nnz in range(1, len(x) + 1):
            nzvals = x[sigma[:nnz]]
            delta = np.sum(nzvals) ** 2 - nnz * np.sum(nzvals ** 2) + nnz

            if delta > best_delta:
                best_nnz = nnz
                best_delta = delta

        nzvals = x[sigma[:best_nnz]]
        theta = (np.sum(nzvals) - np.sqrt(best_delta)) / best_nnz

        out_nzvals = (nzvals - theta) ** 2
        y = np.zeros_like(x)
        y[sigma[:best_nnz]] = out_nzvals

        return y


if __name__ == '__main__':
    # print("test tsallis f grad")
    # m = Tsallis15(np.zeros(5))
    # m.check_grad()

    print(Tsallis15(np.array([0, 1, -1])).project())

