import numpy as np
from auxiliaryFunctions import maxVertex
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

"""# LP Oracles"""

# Birkhoff Polytope feasible region.
class PSDUnitTrace:
    def __init__(self, dim):
        self.dim = dim
        self.matdim = int(np.sqrt(dim))
        self.warmStart = None

    def LPOracle(self, X):
        objective = X.reshape((self.matdim, self.matdim))
        if self.warmStart is not None:
            w, v = eigsh(-objective, 1, which="LA", v0=self.warmStart, maxiter=100000)
        else:
            w, v = eigsh(-objective, 1, which="LA", maxiter=100000)
        self.warmStart = v
        return (np.outer(v, v)).reshape(self.dim)

    def initialPoint(self):
        return (np.identity(self.matdim) / self.matdim).flatten()

    # Input is the vector over which we calculate the inner product.
    def AwayOracle(self, grad, activeVertex):
        return maxVertex(grad, activeVertex)


# Birkhoff Polytope feasible region.
class BirkhoffPolytope:
    def __init__(self, dim):
        self.dim = dim
        self.matdim = int(np.sqrt(dim))

    def LPOracle(self, x):
        from scipy.optimize import linear_sum_assignment

        objective = x.reshape((self.matdim, self.matdim))
        matching = linear_sum_assignment(objective)
        solution = np.zeros((self.matdim, self.matdim))
        solution[matching] = 1
        return solution.reshape(self.dim)

    def initialPoint(self):
        return np.identity(self.matdim).flatten()

    # Input is the vector over which we calculate the inner product.
    def AwayOracle(self, grad, activeVertex):
        return maxVertex(grad, activeVertex)


# Birkhoff Polytope feasible region.
class L1UnitBallPolytope:
    def __init__(self, dim, lambdaVal):
        self.dim = dim
        self.lambdaVal = lambdaVal

    def LPOracle(self, x):
        v = np.zeros(len(x), dtype=float)
        maxInd = np.argmax(np.abs(x))
        v[maxInd] = -1.0 * np.sign(x[maxInd])
        return self.lambdaVal * v

    # Input is the vector over which we calculate the inner product.
    def AwayOracle(self, grad, activeVertex):
        return maxVertex(grad, activeVertex)

    def initialPoint(self):
        v = np.zeros(self.dim)
        v[0] = 1.0
        return self.lambdaVal * v
