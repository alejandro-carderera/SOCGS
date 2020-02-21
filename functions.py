import numpy as np
from scipy.sparse import  issparse
from autograd import hessian
from scipy.optimize import minimize_scalar

"""# Objective Functions

"""
#Basis generator.
#Generates a set of n-orthonormal vectors.
def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H

#Generate a random PSD quadratic with eigenvalues between certain numbers.
def randomPSDGenerator(dim, Mu, L):
    eigenval = np.zeros(dim)
    eigenval[0] = Mu
    eigenval[-1] = L
    eigenval[1:-1] = np.random.uniform(Mu, L, dim - 2)
    M = np.zeros((dim, dim))
    A = rvs(dim)
    for i in range(dim):
        M += eigenval[i]*np.outer(A[i], A[i])
    return M

#Random PSD matrix with a given sparsity.
def randomPSDGeneratorSparse(dim, sparsity):
    mask = np.random.rand(dim,dim)> (1- sparsity)
    mat = np.random.normal(size = (dim,dim))
    Aux = np.multiply(mat, mask)
    return np.dot(Aux.T, Aux) + np.identity(dim)

def calculateEigenvalues(M):
    from scipy.linalg import eigvalsh
    dim = len(M)
    L = eigvalsh(M, eigvals = (dim - 1,dim - 1))[0]
    Mu = eigvalsh(M, eigvals = (0,0))[0]
    return L, Mu

#Custom Quadratic
#sizeVectorY denotes the dimension of the y subspace.
#sizeVectorX denotes the dimension of the x subspace.
#\Sum ||y_i - Ax_i||^2 where y_i and x_i are vectors.
#The dimension of the A vector is the dimension of the x vector times the y vector.
class funcQuadraticCustomv2:
    def __init__(self, n, sizeVectorY, sizeVectorX, AMat):
        self.AMat = AMat
        self.sizeY = sizeVectorY
        self.sizeX = sizeVectorX
        assert AMat.shape == (self.sizeY, self.sizeX), "Incorrect Input."
        self.numSamples = n
        #Generate the samples.
        self.Y = np.zeros((self.sizeY, self.numSamples))
        self.X = np.random.normal(size = (self.sizeX, self.numSamples))
        for i in range(0, self.numSamples):
            self.Y[:,i] = np.dot(self.AMat, self.X[:,i])
        
        #Find the largest eigenvalue.
        #The Hessian as a block diagonal matrix where each diagonal is this matrix.
        self.Hessian = 2.0*np.outer(self.X[:, 0], self.X[:, 0])
        for i in range(1, self.numSamples):
           self.Hessian += 2.0*np.outer(self.X[:, i], self.X[:, i])
            
        from scipy.linalg import eigvalsh
        dim = len(self.Hessian)
        self.L = eigvalsh(self.Hessian, eigvals = (dim - 1,dim - 1))[0]
        self.Mu = eigvalsh(self.Hessian, eigvals = (0,0))[0]
        return       
           
    #Evaluate function.
    def fEval(self, A):
        assert A.shape == (int(self.sizeY*self.sizeX), ), "Incorrect Input."
        #Transform into a matrix of the correct size.
        Ainner = A.reshape(self.sizeY, self.sizeX)
        aux = np.matmul(Ainner, self.X)
        val = 0.0
        for i in range(self.numSamples):
            val += np.dot(self.Y[:,i] - aux[:,i], self.Y[:,i] - aux[:,i])
        return val
    
    #Evaluate gradient.
    def fEvalGrad(self, A):
        assert A.shape == (int(self.sizeY*self.sizeX), ), "Incorrect Input."
        #Transform into a matrix of the correct size.
        Ainner = A.reshape(self.sizeY, self.sizeX)
        grad = np.zeros((self.sizeY, self.sizeX))
        aux = np.matmul(Ainner, self.X)
        for i in range(self.numSamples):
            grad += 2*np.outer(self.Y[:,i] - aux[:,i], self.X[:,i])
        return -np.ravel(grad)
            
    #Evaluate stochastic gradient.
    def fEvalGradStoch(self, A, snapShot, snapPoint, m):
        assert A.shape == (int(self.sizeY*self.sizeX), ), "Incorrect Input."
        #Transform into a matrix of the correct size.
        Aux =  A.reshape(self.sizeY, self.sizeX) - snapPoint.reshape(self.sizeY, self.sizeX)
        grad = np.zeros((self.sizeY, self.sizeX))
        for i in range(m):
            index = np.random.randint(0, self.numSamples)
            grad += 2*np.outer(np.dot(Aux, self.X[:,index],), self.X[:,index])
        grad *= self.numSamples/m
        return np.ravel(grad) + snapShot
    
    #Line Search.
    def lineSearch(self, grad, d, x):
        assert d.shape == (int(self.sizeY*self.sizeX), ), "Incorrect Input."
        #Transform into a matrix of the correct size.
        dinner = d.reshape(self.sizeY, self.sizeX)
        Ainner = x.reshape(self.sizeY, self.sizeX)
        matAux = np.dot(dinner.T, Ainner) + np.dot(Ainner.T, dinner)
        aux1 = 0
        aux2 = 0
        for i in range(0, self.numSamples):
            val = np.dot(dinner, self.X[:,i])
            aux1 += -np.dot(self.X[:,i], np.dot(matAux, self.X[:,i])) + 2*np.dot(self.Y[:,i], val)
            aux2 += 2*np.dot(val, val)
        return aux1/aux2
      
    #Return largest eigenvalue.
    def largestEig(self):
        return self.L

    #Return smallest eigenvalue.
    def smallestEig(self):
        return self.Mu
    
    #Return the Hessian as a block diagonal matrix.
    #As if we had unraveled the A matrix.
    def returnM(self):
        from scipy.sparse import block_diag, coo_matrix
        auxSparse = coo_matrix(self.Hessian)
        listMat = []
        for i in range(0, self.sizeY):
            listMat.append(auxSparse)
        return block_diag(listMat)
    
#Graphical-Lasso type function.
#n is the dimension of the matrix, such that the matrices are nxn.
#S represents the second moment matrix about the mean of some data.
class GraphicalLasso:
    def __init__(self, n, S):
        self.dim = n
        self.S = S
        self.lambdaVal = 0.05
        return       
           
    #Evaluate function.
    def fEval(self, X):
        val = X.reshape((self.dim, self.dim))
        (sign, logdet) = np.linalg.slogdet(val)
        return -logdet + np.matrix.trace(np.matmul(self.S, val)) + self.lambdaVal*np.sum(np.dot(X, X))
    
    #Evaluate gradient.
    def fEvalGrad(self, X):
        val = X.reshape((self.dim, self.dim))
        return (-np.linalg.inv(val) + self.S).flatten() + self.lambdaVal*X
    
    #Line Search.
    def lineSearch(self, grad, d, x):
        def InnerFunction(t):  # Hidden from outer code
                return self.fEval(x + t*d)
        res = minimize_scalar(InnerFunction, bounds=(0, 1), method='bounded')
        return res.x
    
    def fEvalHessian(self, X):
        hessianTest = hessian(self.fEval)
        hessianValue =  hessianTest(X)
        return hessianValue
    
#Graphical-Lasso type function.
#n is the dimension of the matrix, such that the matrices are nxn.
#S represents the second moment matrix about the mean of some data.
class LogisticRegression:
    def __init__(self, n, numSamples, samples, labels):
        self.samples = samples.copy()
        self.labels = labels.copy()
        self.numSamples = numSamples
        self.dim = n
        return       
           
    def fEval(self, x):
        aux = 0.0
        for i in range(self.numSamples):
            aux += np.logaddexp(0.0, -self.labels[i]*np.dot(self.samples[i], x))
        return aux/self.numSamples
    
    def fEvalGrad(self, x):
        aux = 0.0
        for i in range(self.numSamples):
            val =  np.exp(self.labels[i]*np.dot(self.samples[i], x))
            aux += -self.labels[i]*self.samples[i]/(1 +  val)
        return aux/self.numSamples
    
    def lineSearch(self, grad, d, x):
        def InnerFunction(t):  # Hidden from outer code
                return self.fEval(x + t*d)
        res = minimize_scalar(InnerFunction, bounds=(0, 1), method='bounded')
        return res.x

#Takes a random PSD matrix generated by the functions above and uses them as a function.
class funcQuadratic:
    import numpy as np
    def __init__(self, size, matrix, vector, Mu, L):
        self.len = size
        self.M = matrix.copy()
        self.b = vector.copy()
        self.L = L
        self.Mu = Mu
        self.InvHess = None
        return       
           
    def dim(self):
        return self.len
        
    #Evaluate function.
    def fEval(self, x):
        return 0.5*np.dot(x, self.M.dot(x)) + np.dot(self.b, x) 
    
    #Evaluate gradient.
    def fEvalGrad(self, x):
        return self.M.dot(x) + self.b
    
    #Evaluate the inverse of the Hessian
    def fEvalInvHess(self):
        if self.InvHess is None:
            if(issparse(self.M)):
                self.InvHess = np.linalg.pinv(self.M.todense())
            else:
                self.InvHess = np.linalg.pinv(self.M)
        return self.InvHess
    
    #Line Search.
    def lineSearch(self, grad, d):
        return -np.dot(grad, d)/np.dot(d, self.M.dot(d))
      
    #Return largest eigenvalue.
    def largestEig(self):
        return self.L

    #Return smallest eigenvalue.
    def smallestEig(self):
        return self.Mu
    
    #Return largest eigenvalue.
    def returnM(self):
        return self.M

    #Return smallest eigenvalue.
    def returnb(self):
        return self.b

class funcQuadraticDiag:
    import numpy as np
    def __init__(self, size, xOpt, Mu = 1.0, L = 2.0):
        self.len = size
        self.matdim = int(np.sqrt(size))
        self.eigenval = np.zeros(size)
        self.eigenval[0] = Mu
        self.eigenval[-1] = L
        self.eigenval[1:-1] = np.random.uniform(Mu, L, size - 2)
        self.L = L
        self.Mu = Mu
        self.xOpt = xOpt
        self.b = - np.multiply(self.xOpt, self.eigenval)
        self.InvHess = None
        return       
           
    def dim(self):
        return self.len
        
    #Evaluate function.
    def fEval(self, x):
        return 0.5*np.dot(x, np.multiply(self.eigenval,x)) + np.dot(self.b, x) 
    
    #Evaluate gradient.
    def fEvalGrad(self, x):
        return np.multiply(x, self.eigenval) + self.b
    
        #Evaluate the inverse of the Hessian
    def fEvalInvHess(self):
        if self.InvHess is None:
            self.InvHess = np.diag(np.reciprocal(self.eigenval))
        return self.InvHess
      
    #Return largest eigenvalue.
    def largestEig(self):
        return self.L

    #Return smallest eigenvalue.
    def smallestEig(self):
        return self.Mu
    
    #Line Search.
    def lineSearch(self, grad, d):
        return -np.dot(grad, d)/np.dot(d, np.multiply(self.eigenval, d))
    
        #Return largest eigenvalue.
    def returnM(self):
        return np.diag(self.eigenval)

    #Return smallest eigenvalue.
    def returnb(self):
        return self.b