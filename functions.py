import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize_scalar
from scipy import sparse

"""# Objective Functions

"""

# Custom Quadratic
# sizeVectorY denotes the dimension of the y subspace.
# sizeVectorX denotes the dimension of the x subspace.
# \Sum ||y_i - Ax_i||^2 where y_i and x_i are vectors.
# The dimension of the A vector is the dimension of the x vector times the y vector.
class funcQuadraticCustom:
    def __init__(self, n, sizeVectorY, sizeVectorX, AMat):
        self.AMat = AMat
        self.sizeY = sizeVectorY
        self.sizeX = sizeVectorX
        assert AMat.shape == (self.sizeY, self.sizeX), "Incorrect Input."
        self.numSamples = n
        # Generate the samples.
        self.Y = np.zeros((self.sizeY, self.numSamples))
        self.X = np.random.normal(size=(self.sizeX, self.numSamples))
        for i in range(0, self.numSamples):
            self.Y[:, i] = np.dot(self.AMat, self.X[:, i])

        # Find the largest eigenvalue.
        # The Hessian as a block diagonal matrix where each diagonal is this matrix.
        self.Hessian = 2.0 * np.outer(self.X[:, 0], self.X[:, 0])
        for i in range(1, self.numSamples):
            self.Hessian += 2.0 * np.outer(self.X[:, i], self.X[:, i])

        from scipy.linalg import eigvalsh

        dim = len(self.Hessian)
        self.L = eigvalsh(self.Hessian, eigvals=(dim - 1, dim - 1))[0]
        self.Mu = eigvalsh(self.Hessian, eigvals=(0, 0))[0]
        return

    # Evaluate function.
    def fEval(self, A):
        assert A.shape == (int(self.sizeY * self.sizeX),), "Incorrect Input."
        # Transform into a matrix of the correct size.
        Ainner = A.reshape(self.sizeY, self.sizeX)
        aux = np.matmul(Ainner, self.X)
        val = 0.0
        for i in range(self.numSamples):
            val += np.dot(self.Y[:, i] - aux[:, i], self.Y[:, i] - aux[:, i])
        return val

    # Evaluate gradient.
    def fEvalGrad(self, A):
        assert A.shape == (int(self.sizeY * self.sizeX),), "Incorrect Input."
        # Transform into a matrix of the correct size.
        Ainner = A.reshape(self.sizeY, self.sizeX)
        grad = np.zeros((self.sizeY, self.sizeX))
        aux = np.matmul(Ainner, self.X)
        for i in range(self.numSamples):
            grad += 2 * np.outer(self.Y[:, i] - aux[:, i], self.X[:, i])
        return -np.ravel(grad)

    # Evaluate stochastic gradient.
    def fEvalGradStoch(self, A, snapShot, snapPoint, m):
        assert A.shape == (int(self.sizeY * self.sizeX),), "Incorrect Input."
        # Transform into a matrix of the correct size.
        Aux = A.reshape(self.sizeY, self.sizeX) - snapPoint.reshape(
            self.sizeY, self.sizeX
        )
        grad = np.zeros((self.sizeY, self.sizeX))
        for i in range(m):
            index = np.random.randint(0, self.numSamples)
            grad += 2 * np.outer(np.dot(Aux, self.X[:, index],), self.X[:, index])
        grad *= self.numSamples / m
        return np.ravel(grad) + snapShot

    # Line Search.
    def lineSearch(self, grad, d, x, maxStep=None):
        assert d.shape == (int(self.sizeY * self.sizeX),), "Incorrect Input."
        # Transform into a matrix of the correct size.
        dinner = d.reshape(self.sizeY, self.sizeX)
        Ainner = x.reshape(self.sizeY, self.sizeX)
        matAux = np.dot(dinner.T, Ainner) + np.dot(Ainner.T, dinner)
        aux1 = 0
        aux2 = 0
        for i in range(0, self.numSamples):
            val = np.dot(dinner, self.X[:, i])
            aux1 += -np.dot(self.X[:, i], np.dot(matAux, self.X[:, i])) + 2 * np.dot(
                self.Y[:, i], val
            )
            aux2 += 2 * np.dot(val, val)
        alpha = aux1 / aux2
        if maxStep is None:
            return min(1.0, alpha)
        else:
            return min(maxStep, alpha)

    # Return largest eigenvalue.
    def largestEig(self):
        return self.L

    # Return smallest eigenvalue.
    def smallestEig(self):
        return self.Mu

    # Return the Hessian as a block diagonal matrix.
    # As if we had unraveled the A matrix.
    def returnM(self, x, omega=None, distance=None):
        from scipy.sparse import block_diag, csr_matrix, eye

        auxSparse = csr_matrix(self.Hessian)
        listMat = []
        for i in range(0, self.sizeY):
            listMat.append(auxSparse)
        matrix = block_diag(listMat)
        if omega is None:
            return matrix
        else:
            dim1, dim2 = matrix.shape
            val = np.random.uniform(
                -omega * distance * self.largestEig() / (omega * distance + 1),
                self.smallestEig() * omega * distance,
            )
            return block_diag(listMat) + val * eye(dim1, dim2)


from scipy.sparse.linalg import splu

# Graphical-Lasso type function.
# n is the dimension of the matrix, such that the matrices are nxn.
# S represents the second moment matrix about the mean of some data.
class GraphicalLasso:
    import autograd.numpy as np

    def __init__(self, n, S, lambaVal, delta=0.0):
        self.dim = n
        self.S = S
        self.lambdaVal = lambaVal
        self.largestEigenval = None
        self.delta = 0.0
        return

    # Evaluate function.
    def fEval(self, X):
        self.delta = 0.0
        val = X.reshape((self.dim, self.dim))
        return (
            -self.logdetFun(val + self.delta * np.identity(self.dim))
            + np.matrix.trace(np.matmul(self.S, val))
            + 0.5 * self.lambdaVal * np.sum(np.dot(X, X))
        )

    # Evaluate gradient.
    def fEvalGrad(self, X):
        val = X.reshape((self.dim, self.dim))
        # L2 penalty parameter.
        self.delta = 0.0
        return (
            -np.linalg.inv(val + self.delta * np.identity(self.dim)) + self.S
        ).flatten() + self.lambdaVal * X

    # Line Search.
    def lineSearch(self, grad, d, x, maxStep=None):
        options = {"xatol": 1e-12, "maxiter": 5000000, "disp": 0}

        def InnerFunction(t):  # Hidden from outer code
            return self.fEval(x + t * d)

        if maxStep is None:
            res = minimize_scalar(
                InnerFunction, bounds=(0, 1), method="bounded", options=options
            )
        else:
            res = minimize_scalar(
                InnerFunction, bounds=(0, maxStep), method="bounded", options=options
            )
        return res.x

    def logdetFun(self, X):
        lu = splu(X)
        diagL = lu.L.diagonal().astype(np.complex128)
        diagU = lu.U.diagonal().astype(np.complex128)
        logdet = np.log(diagL).sum() + np.log(diagU).sum()
        return logdet.real


# Graphical-Lasso type function.
# n is the dimension of the matrix, such that the matrices are nxn.
# S represents the second moment matrix about the mean of some data.
class LogisticRegressionSparse:
    def __init__(self, n, numSamples, samples, labels, mu=0.0):
        self.samples = samples.copy()
        self.labels = labels.copy()
        self.numSamples = numSamples
        self.dim = n
        self.mu = mu
        return

    def fEval(self, x):
        aux = np.sum(
            np.logaddexp(
                np.zeros(self.numSamples),
                np.multiply(self.samples.dot(-x), self.labels),
            )
        )
        return aux / self.numSamples + self.mu * np.dot(x, x) / 2.0

    def fEvalGrad(self, x):
        aux = -self.labels / (
            1.0 + np.exp(np.multiply(self.samples.dot(x), self.labels))
        )
        vectors = self.samples.T.multiply(aux).sum(axis=1)
        return np.squeeze(np.asarray(vectors)) / self.numSamples + self.mu * x

    # Line Search.
    def lineSearch(self, grad, d, x, maxStep=None):
        options = {"xatol": 1e-12, "maxiter": 50000, "disp": 0}

        def InnerFunction(t):  # Hidden from outer code
            return self.fEval(x + t * d)

        if maxStep is None:
            res = minimize_scalar(
                InnerFunction, bounds=(0, 1), method="bounded", options=options
            )
        else:
            res = minimize_scalar(
                InnerFunction, bounds=(0, maxStep), method="bounded", options=options
            )
        return res.x


# Graphical-Lasso type function.
# n is the dimension of the matrix, such that the matrices are nxn.
# S represents the second moment matrix about the mean of some data.
class LogisticRegression:
    def __init__(self, n, numSamples, samples, labels, mu=0.0):
        self.samples = samples.copy()
        self.labels = labels.copy()
        self.numSamples = numSamples
        self.dim = n
        self.mu = mu
        self.largestEigenval = None
        return

    def fEval(self, x):
        aux = 0.0
        for i in range(self.numSamples):
            aux += np.logaddexp(0.0, -float(self.labels[i] * self.samples[i].dot(x)))
        return aux / self.numSamples + self.mu * np.dot(x, x) / 2.0

    def fEvalGrad(self, x):
        aux = 0.0
        for i in range(self.numSamples):
            val = np.exp(self.labels[i] * self.samples[i].dot(x))
            aux += -self.labels[i] * self.samples[i] / (1 + val)
        return np.squeeze(np.asarray(aux)) / self.numSamples + self.mu * x

    # Line Search.
    def lineSearch(self, grad, d, x, maxStep=None):
        options = {"xatol": 1e-16, "maxiter": 500000, "disp": 0}

        def InnerFunction(t):  # Hidden from outer code
            return self.fEval(x + t * d)

        if maxStep is None:
            res = minimize_scalar(
                InnerFunction, bounds=(0, 1), method="bounded", options=options
            )
        else:
            res = minimize_scalar(
                InnerFunction, bounds=(0, maxStep), method="bounded", options=options
            )
        return res.x


# Takes a random PSD matrix generated by the functions above and uses them as a function.
class QuadApprox:
    import numpy as np

    def __init__(self):
        self.alpha = 1.0
        return

    # Evaluate function.
    def fEval(self, x):
        return np.dot(self.g, x - self.x_k) + 0.5 / self.alpha * np.dot(
            x - self.x_k, self.H.dot(x - self.x_k)
        )

    # Evaluate gradient.
    def fEvalGrad(self, x):
        return self.g + self.H.dot(x - self.x_k) / self.alpha

    # Line Search.
    def lineSearch(self, grad, d, x, maxStep=None):
        alpha = -np.dot(grad, d) / np.dot(d, self.H.dot(d))
        if maxStep is None:
            return min(alpha, 1.0)
        else:
            return min(alpha, maxStep)

    # Is the approximation linear.
    def isLinear(self):
        return False

    # Return the Hessian of the function.
    def fEvalHessian(self):
        return self.H

    # Return the Hessian of the function.
    def fEvalHessianNorm(self, x):
        return np.sqrt(np.dot(x, self.H.dot(x)))

    # Update gradient vector
    def updateApprox(self, gradient, x, hessian=None):
        self.g = gradient.copy()
        self.x_k = x.copy()
        self.H = hessian.copy()
        return


class QuadApproxLogReg:
    def __init__(self, n, numSamples, samples, labels, mu=0.0):
        self.samples = csr_matrix(samples.copy())
        self.labels = labels.copy()
        self.numSamples = numSamples
        self.dim = n
        self.g = np.zeros(n)
        self.x_k = np.zeros(n)
        self.quotient = np.zeros(numSamples)
        self.mu = mu
        return

    def fEval(self, x):
        aux = np.dot(np.square(self.samples.dot(x - self.x_k)), self.quotient)
        return (
            np.dot(self.g, x - self.x_k)
            + aux / (2.0 * self.numSamples)
            + self.mu * np.dot(x - self.x_k, x - self.x_k) / 2.0
        )

    def fEvalGrad(self, x):
        aux = self.samples.T.multiply(
            self.samples.dot(x - self.x_k) * self.quotient
        ).sum(axis=1)
        return (
            self.g
            + np.squeeze(np.asarray(aux)) / self.numSamples
            + self.mu * (x - self.x_k)
        )

    def fEvalGradBackup(self, x):
        aux = 0.0
        for i in range(self.numSamples):
            aux += (
                self.samples[i]
                * self.quotient[i]
                * np.dot(x - self.x_k, self.samples[i])
            )
        return self.g + aux / self.numSamples + self.mu * (x - self.x_k)

    def lineSearch(self, grad, d, x, maxStep=None):
        aux = np.dot(np.square(self.samples.dot(d)), self.quotient)
        if maxStep is None:
            return -np.dot(grad, d) / (aux / self.numSamples + self.mu * np.dot(d, d))
        else:
            return min(
                -np.dot(grad, d) / (aux / self.numSamples + self.mu * np.dot(d, d)),
                maxStep,
            )

    def lineSearchBackup(self, grad, d, x):
        aux = 0.0
        for i in range(self.numSamples):
            aux += self.quotient[i] * np.dot(self.samples[i], d) ** 2
        return -np.dot(grad, d) / (aux / self.numSamples + self.mu * np.dot(d, d))

    # Return the Hessian of the function.
    def fEvalHessianNorm(self, x):
        aux = np.dot(np.square(self.samples.dot(x)), self.quotient)
        return np.sqrt(aux / self.numSamples + self.mu * np.dot(x, x))

    # Update gradient vector
    def updateApprox(self, gradient, x, hessian=None):
        self.g = gradient.copy()
        self.x_k = x.copy()
        aux = np.multiply(self.samples.dot(x), self.labels)
        self.quotient = 1.0 / (1.0 + np.exp(aux)) / (1.0 + np.exp(-aux))
        return


from numpy.core.umath_tests import inner1d


class QuadApproxGLasso:
    def __init__(self, n, lambdaValue, delta):
        self.dim = n
        self.lambdaVal = lambdaValue
        self.delta = delta
        return

    # Evaluate function.
    def fEval(self, X):
        val = X.reshape((self.dim, self.dim))
        aux = val - self.x_k
        return (
            0.5 * np.linalg.norm(np.matmul(self.inv_x_k, aux)) ** 2
            + 0.5 * self.lambdaVal * np.linalg.norm(val - self.x_k) ** 2
            + np.sum(inner1d(self.g, aux.T))
        )

    # Evaluate gradient.
    def fEvalGrad(self, X):
        val = X.reshape((self.dim, self.dim))
        aux = val - self.x_k
        return (
            self.g
            + self.lambdaVal * (val - self.x_k)
            + np.matmul(self.inv_x_k, np.matmul(aux, self.inv_x_k))
        ).flatten()

    # Line Search.
    def lineSearch(self, grad, d, x, maxStep=None):
        D = d.reshape((self.dim, self.dim))
        Gradient = grad.reshape((self.dim, self.dim))
        alpha = -np.sum(inner1d(Gradient, D.T)) / (
            self.lambdaVal * np.linalg.norm(D) ** 2
            + np.linalg.norm(np.matmul(self.inv_x_k, D)) ** 2
        )
        if maxStep is not None:
            return min(alpha, maxStep)
        else:
            return alpha

    # Update gradient vector
    def updateApprox(self, gradient, x, hessian=None):
        self.g = gradient.reshape((self.dim, self.dim)).copy()
        self.x_k = x.reshape((self.dim, self.dim)).copy()
        self.inv_x_k = np.linalg.inv(self.x_k + self.delta * np.identity(self.dim))
        return

    # Return the Hessian of the function.
    def fEvalHessianNorm(self, X):
        val = X.reshape((self.dim, self.dim))
        return np.sqrt(
            np.linalg.norm(np.matmul(self.inv_x_k, val)) ** 2
            + self.lambdaVal * np.linalg.norm(val) ** 2
        )


#        return np.sqrt(scipy.linalg.norm(np.matmul(self.inv_x_k, val))**2 + self.lambdaVal*scipy.linalg.norm(val)**2)

# Creates a compact Hessian Approximation.
# m is the number of elements we'll use to calculate the matrix.
class QuadApproxInexactHessianLBFGS:
    import numpy as np

    def __init__(self, dimension, m):
        self.dim = dimension
        self.m = m
        self.g = None
        self.x_k = None
        self.left = None
        self.center = None
        self.S = None
        self.Y = None
        self.delta = None
        self.I = sparse.eye(dimension)
        self.L = 1.0
        self.Mu = 1.0
        return

    # Update the function
    def updateApprox(self, gradient, xk):
        if self.g is None and self.x_k is None:
            self.gOld = gradient.copy()
            self.x_kOld = xk.copy()
            self.g = gradient.copy()
            self.x_k = xk.copy()
        else:
            self.gOld = self.g.copy()
            self.x_kOld = self.x_k.copy()
            self.g = gradient.copy()
            self.x_k = xk.copy()

            s = self.x_k - self.x_kOld
            y = self.g - self.gOld

            if self.S is None and self.Y is None:
                self.S = s.copy().reshape(self.dim, 1)
                self.Y = y.copy().reshape(self.dim, 1)
            else:
                self.S = np.hstack((self.S, s.reshape(self.dim, 1)))
                self.Y = np.hstack((self.Y, y.reshape(self.dim, 1)))
            self.delta = np.dot(y, y) / np.dot(s, y)
            if self.delta <= 0.0:
                print("The direction was not a descent direction.")
                quit()
            # Need to delete the first element in the matrix.
            if self.S.shape[1] >= self.m:
                self.S = np.delete(self.S, 0, 1)
                self.Y = np.delete(self.Y, 0, 1)
            self.left = np.hstack((self.delta * self.S, self.Y))
            # Build the L matrix.
            L = np.tril(np.matmul(self.S.T, self.Y), -1)
            N = self.S.shape[1]
            D = np.zeros((N, N))
            for i in range(N):
                D[i, i] = np.dot(self.S[:, i], self.Y[:, i])
            self.center = np.linalg.pinv(
                np.block([[self.delta * np.matmul(self.S.T, self.S), L], [L.T, -D]])
            )
            self.hessian = self.delta * np.identity(self.dim) - np.matmul(
                np.matmul(self.left, self.center), self.left.T
            )
        return

    # Evaluate function.
    def fEval(self, x):
        if self.S is not None:
            aux1 = np.dot(x - self.x_k, self.left)
            aux = 0.5 * self.delta * np.dot(x - self.x_k, x - self.x_k) - 0.5 * np.dot(
                np.dot(aux1, self.center), aux1
            )
            return np.dot(self.g, x - self.x_k) + aux
        else:
            return np.dot(self.g, x - self.x_k)

    # Evaluate gradient.
    def fEvalGrad(self, x):
        if self.S is not None:
            return (
                self.g
                + self.delta * (x - self.x_k)
                - np.dot(
                    np.dot(self.left, self.center), np.dot(x - self.x_k, self.left)
                )
            )
        else:
            return self.g

    # Line Search.
    def lineSearch(self, grad, d, x, maxStep=None):
        if self.S is not None:
            aux1 = np.dot(d, self.left)
            aux = self.delta * np.dot(d, d) - np.dot(np.dot(aux1, self.center), aux1)
            alpha = -np.dot(grad, d) / aux
        else:
            alpha = 100000.0
        if maxStep is not None:
            return min(maxStep, alpha)
        else:
            return alpha
