import numpy as np
import time
ts = time.time()
from functions import rvs
from scipy.optimize import minimize_scalar

"""# Miscelaneous Functions

Functions include those used in the active set management and projection functions onto the L1 ball and simplex.
"""
#Defines the type of maximum vertex dot product that we'll return.
def maxVertex(grad, activeVertex):
      #See which extreme point in the active set gives greater inner product.
      maxProd = np.dot(activeVertex[0], grad)
      maxInd = 0
      for i in range(len(activeVertex)):
          if(np.dot(activeVertex[i], grad) > maxProd):
              maxProd = np.dot(activeVertex[i], grad)
              maxInd = i
      return activeVertex[maxInd], maxInd
  
#Finds the step with the maximum and minimum inner product.
def maxMinVertex(grad, activeVertex):
      #See which extreme point in the active set gives greater inner product.
      maxProd = np.dot(activeVertex[0], grad)
      minProd = np.dot(activeVertex[0], grad)
      maxInd = 0
      minInd = 0
      for i in range(len(activeVertex)):
          if(np.dot(activeVertex[i], grad) > maxProd):
              maxProd = np.dot(activeVertex[i], grad)
              maxInd = i
          else:
              if(np.dot(activeVertex[i], grad) < minProd):
                  minProd = np.dot(activeVertex[i], grad)
                  minInd = i
      return activeVertex[maxInd], maxInd, activeVertex[minInd], minInd

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

def newVertexFailFast(x, extremePoints):
    for i in range(len(extremePoints)):
        #Compare succesive indices.
        for j in range(len(extremePoints[i])):
            if(extremePoints[i][j] != x[j]):
                break
        if(j == len(extremePoints[i]) - 1):
            return False, i
    return True, np.nan

#Deletes the extremepoint from the representation.
def deleteVertexIndex(index, extremePoints, weights):
  del extremePoints[index]
  del weights[index]
  return

def performUpdate(function, x, gap, fVal, timing, gapVal):
    gap.append(gapVal)
    fVal.append(function.fEval(x))
    timing.append(time.time())
    return

#Pick a stepsize.
def stepSize(function, d, grad, x, typeStep = "EL", maxStep = None):
    if(typeStep == "SS"):
        return -np.dot(grad, d)/(function.largestEig()*np.dot(d, d))
    else:
        if(typeStep == "GS"):
            tolerance = 1e-08
            iterations = 500000
            while(True):
                options={'xatol': tolerance, 'maxiter': iterations, 'disp': 0}
                def InnerFunction(t):  # Hidden from outer code
                    return function.fEval(x + t*d)
                if(maxStep is None):
                    res = minimize_scalar(InnerFunction, bounds=(0, 1), method='bounded', options = options)
                else:
                    res = minimize_scalar(InnerFunction, bounds=(0, maxStep), method='bounded', options = options)
                if(function.fEval(x) > function.fEval(x + res.x*d)):
                    return res.x
                tolerance /= 100.0
                iterations *= 100.0
                if(tolerance < 1.0e-14):
                    return 1.0e-12
        else:
            return function.lineSearch(grad, d, x)
    
#Pick a stepsize for decomposition invariant strategy.
#Exact Linesearch: "EL"
#Fixed Step-size: "FSS"
#Note that the fixed stepsize requires knowledge of the cardinality of the solution
#which we do not know a priori.
def stepSizeDI(function, feasibleReg, it, d, grad, x, typeStep = "EL"):
    return function.lineSearch(grad, d, x)

#Used in the DICG algorithm.
def calculateStepsize(x, d):
    assert not np.any(x < 0.0), "There is a negative coordinate."
    index = np.where(x == 0)[0]
    if(np.any(d[index] < 0.0)):
        return 0.0
    index = np.where(x > 0)[0]
    coeff = np.zeros(len(x))
    for i in index:
        if(d[i] < 0.0):
            coeff[i] = -x[i]/d[i]
    val = coeff[coeff > 0]
    if(len(val) == 0):
        return 0.0
    else:
        return min(val)      
    
#Evaluate exit criterion. Evaluates to true if we must exit. Three posibilities:
# 1 - "PG": Evaluate primal gap.
# 2 - "DG": Evaluate dual gap.
# 3 - "IT": Evaluate number of iterations.
def exitCriterion(it, f, dualGap, criterion = "PG", numCriterion = 1.0e-3, critRef = 0.0):
    if(criterion == "DG"):
        print("Wolfe-Gap: " + str(dualGap))
        return dualGap < numCriterion
    else:
        if(criterion == "PG"):
            print("Primal gap: " + str(f - critRef))
            return f - critRef < numCriterion
        else:
            return it >= numCriterion