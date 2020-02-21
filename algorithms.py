import numpy as np
import datetime
import time
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar
import math

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S').replace(' ', '-').replace(':', '-')

#Import functions for the stepsizes and the updates.
from auxiliaryFunctions import performUpdate, exitCriterion, stepSize, calculateStepsize, stepSizeDI

#Import functions for active set management
from auxiliaryFunctions import newVertexFailFast, deleteVertexIndex, maxMinVertex

#Import the function used in the subproblems.

from scipy import sparse

"""## Conditional Gradient Sliding

Parameters:
1 --> criterion: Specify if the terminating criterion is the primal gap ("PG") or the
dual gap ("DG"). If anything else is specified will run for a given number of iterations.

2 --> criterionRef: Value to which the algorithm will run, according to the criterion choosen.
Will usually include the value with which we calculate the primal gap.
"""

class CGS:
    def __init__(self):
        self.iteration = 0
    def run(self, x0, function, feasibleReg, tolerance, maxTime, locOpt, criterion = "PG", criterionRef = 0.0):
        #Quantities we want to output.
        grad = function.fEvalGrad(x0)
        FWGap = [np.dot(grad, x0 - feasibleReg.LPOracle(grad))]
        fVal = [function.fEval(x0)]
        timing = [time.time()]
        xVal = [np.linalg.norm(x0 - locOpt)]
        iteration = [1]
        x = x0.copy()
        self.limit_time = maxTime
        self.initTime = timing[0]
        itCount = 1.0
        N = int(np.ceil(2*np.sqrt(6.0*function.largestEig()/function.smallestEig())))
        s = 1.0
        while(True): 
            x = self.CGSubroutine(function, feasibleReg, x0, FWGap[0], N, s)
            if(exitCriterion(itCount, fVal[-1], FWGap[-1], criterion = criterion, numCriterion = tolerance, critRef = criterionRef) or time.time() - timing[0] > maxTime):
                timing[:] = [t - timing[0] for t in timing]
                return x, FWGap, fVal, timing, iteration, xVal
            grad = function.fEvalGrad(x)
            performUpdate(function, x, FWGap, fVal, timing, np.dot(grad, x - feasibleReg.LPOracle(grad)))
            xVal.append(np.linalg.norm(x - locOpt))
            iteration.append(self.iteration)
            s += 1.0
            itCount += 1

    #Runs the subroutine with the stepsizes for the number of iterations depicted.
    def CGSubroutine(self, function, feasibleRegion, x0, delta0, N, s):
        L = function.largestEig()
        Mu = function.smallestEig()
        y = x0.copy()
        x = x0.copy()
        for k in range(1, N + 1):
            gamma = 2.0/(k + 1.0)
            nu = 8.0*L*delta0*np.power(2, -s)/(Mu*N*k)
            beta = 2.0*L/k
            z = (1 - gamma)*y + gamma*x
            x = self.CGSuProjection(function.fEvalGrad(z), x, beta, nu, feasibleRegion)
            if(time.time() - self.initTime > self.limit_time):
                return y
            y = (1 - gamma)*y + gamma*x
        return y
    
    #Subroutine used in CGS for str.cvx. smooth functions.
    def CGSuProjection(self, g, u, beta, nu, feasibleRegion):
        t = 1
        u_t = u
        while(True):
            grad = g + beta*(u_t - u)
            v = feasibleRegion.LPOracle(grad)      
            self.iteration += 1
            V = np.dot(g + beta*(u_t - u), u_t - v)
            if(time.time() - self.initTime > self.limit_time):
                return u_t
            if(V <= nu):
                return u_t 
            else:
                d = v - u_t
                alphaOpt = -np.dot(grad, d)/(beta*np.dot(d,d))
                alpha = min(1, alphaOpt)
                #alpha = min(1, np.dot(beta*(u - u_t) - g, v - u_t)/(beta*np.dot(v - u_t, v - u_t)))
                u_t = (1 - alpha)*u_t + alpha*v
                t += 1
                
class SVRFW:
    def __init__(self):
        return
    
    def run(self, x0, function, feasibleReg, tolerance, maxTime, locOpt, criterion = "PG", criterionRef = 0.0):
        #Quantities we want to output.
        grad = function.fEvalGrad(x0)
        FWGap = [np.dot(grad, x0 - feasibleReg.LPOracle(grad))]
        fVal = [function.fEval(x0)]
        timing = [time.time()]
        xVal = [np.linalg.norm(x0 - locOpt)]
        x = x0.copy()
        itCount_t = 1.0
        while(True): 
            N_t = int(math.ceil(np.power(2, itCount_t + 3) - 2))
            snapShot = function.fEvalGrad(x)
            snapPoint = x.copy()
            for k in range(0, N_t):
                m_k = int(math.ceil(96.0 * (k + 2)))
                StochGrad = function.fEvalGradStoch(x, snapShot, snapPoint, m_k)
                v = feasibleReg.LPOracle(StochGrad)
                gamma_k = 2.0 / (k + 2)
                x = x + gamma_k*(v - x)
                if(exitCriterion(itCount_t, fVal[-1], FWGap[-1], criterion = criterion, numCriterion = tolerance, critRef = criterionRef) or timing[-1] - timing[0] > maxTime):
                    timing[:] = [t - timing[0] for t in timing]
                    return x, FWGap, fVal, timing,  xVal
            grad = function.fEvalGrad(x)
            performUpdate(function, x, FWGap, fVal, timing, np.dot(grad, x - feasibleReg.LPOracle(grad)))
            xVal.append(np.linalg.norm(x - locOpt))
            if(exitCriterion(itCount_t, fVal[-1], FWGap[-1], criterion = criterion, numCriterion = tolerance, critRef = criterionRef) or timing[-1] - timing[0] > maxTime):
                timing[:] = [t - timing[0] for t in timing]
                return x, FWGap, fVal, timing,  xVal
            itCount_t += 1
            
"""## Away FW, Pairwise FW, AFW Lazy

Parameters:
1 --> FWVariant: Specifies if we want to run the Away-step FW ("AFW"), pairwise-step
FW ("PFW") or the lazified version of AFW ("Lazy").

2 --> typeStep: Specifies the type of step. Choosing "EL" performs exact line search
for the quadratic objective functions. Otherwise choosing "SS" chooses a step
that minimizes the smoothness equation and ensures progress in every iteration.

3 --> criterion: Specify if the terminating criterion is the primal gap ("PG") or the
dual gap ("DG"). If anything else is specified will run for a given number of iterations.

4 --> criterionRef: Value to which the algorithm will run, according to the criterion choosen.

5 --> returnVar: If a value is specified will return the active set and the 
barycentric coordinates.
"""
def runFW(x0, activeSet, lambdas, function, feasibleReg, tolerance, maxTime, locOpt, FWVariant = "AFW", typeStep = "SS", criterion = "PG", criterionRef = 0.0, returnVar = None):
    #Quantities we want to output.
    grad = function.fEvalGrad(x0)
    FWGap = [np.dot(grad, x0 - feasibleReg.LPOracle(grad))]
    fVal = [function.fEval(x0)]
    timing = [time.time()]
    xVal = [np.linalg.norm(x0 - locOpt)]
    activeSize = [1]
    x = x0.copy()
    active = activeSet.copy()
    lambdaVal = lambdas.copy()
    itCount = 1
    if(FWVariant == "Lazy"):
        phiVal = [FWGap[-1]]
    while(True):
        if(FWVariant == "AFW"):
            x, vertvar, gap = awayStepFW(function, feasibleReg, x, active, lambdaVal, typeStep)
        else:
            if(FWVariant == "PFW"):
                x, vertvar, gap = pairwiseStepFW(function, feasibleReg, x, active, lambdaVal, typeStep)
            if(FWVariant == "Lazy"):
                x, vertvar, gap = awayStepFWLazy(function, feasibleReg, x, active, lambdaVal, phiVal, typeStep)
            if(FWVariant == "Vanilla"):
                x, vertvar, gap = stepFW(function, feasibleReg, x, active, lambdaVal, typeStep)
        activeSize.append(len(active))
        performUpdate(function, x, FWGap, fVal, timing, gap)
        xVal.append(np.linalg.norm(x - locOpt))
        if(exitCriterion(itCount, fVal[-1], FWGap[-1], criterion = criterion, numCriterion = tolerance, critRef = criterionRef) or timing[-1] - timing[0] > maxTime):
            timing[:] = [t - timing[0] for t in timing]
            if(returnVar is not None):
                return x, FWGap, fVal, timing, lambdaVal[:], active[:], xVal
            else:
                return x, FWGap, fVal, timing, activeSize, xVal
        itCount += 1

#Takes a random PSD matrix generated by the functions above and uses them as a function.
class QuadApprox:
    import numpy as np
    def __init__(self, hessian, gradient, xk):
        self.H = hessian.copy()
        self.g = gradient.copy()
        self.x_k = xk.copy()
        self.alpha = 1.0
        return       
           
    #Evaluate function.
    def fEval(self, x):
        return np.dot(self.g, x  - self.x_k) + 0.5/self.alpha*np.dot(x - self.x_k, self.H.dot(x - self.x_k))
    
    #Evaluate gradient.
    def fEvalGrad(self, x):
        return self.g + self.H.dot(x - self.x_k)/self.alpha
    
    #Line Search.
    def lineSearch(self, grad, d, x):
        return -np.dot(grad, d)/np.dot(d, self.H.dot(d))
    
    #Update Hessian matrix.
    def updateHessian(self, hessian):
        self.H = hessian.copy()
        return
    
    #Update gradient vector
    def updateGradient(self, gradient):
        self.g = gradient.copy()
        return
    
    #Update point.
    def updatePoint(self, xk):
        self.x_k = xk.copy()
        return
    
    #Is the approximation linear.
    def isLinear(self):
        return False
    
#Creates a compact Hessian Approximation.
#m is the number of elements we'll use to calculate the matrix.
class QuadApproxInexactHessianLBFGS:
    import numpy as np
    def __init__(self, dimension, m, gradient, xk):
        self.dim = dimension
        self.m = m
        self.g = gradient.copy()
        self.x_k = xk.copy()
        self.left = None
        self.center = None
        self.S = None
        self.Y = None
        self.delta = None
        self.I = sparse.eye(dimension)
        self.L = 1.0
        self.Mu = 1.0
        return       
          
    #Update the function
    def updateRepresentation(self, s, y):
        if(self.S is None and self.Y is None):
            self.S = s.copy().reshape(self.dim,1)
            self.Y = y.copy().reshape(self.dim,1)
        else:
            self.S = np.hstack((self.S, s.reshape(self.dim, 1)))
            self.Y = np.hstack((self.Y, y.reshape(self.dim, 1)))
        self.delta = np.dot(y, y)/np.dot(s, y)
        if(self.delta <= 0.0):
            print("The direction was not a descent direction.")
            quit()
        #Need to delete the first element in the matrix.  
        if(self.S.shape[1] >= self.m):
            self.S = np.delete(self.S, 0, 1)
            self.Y = np.delete(self.Y, 0, 1)
        self.left = np.hstack((self.delta*self.S, self.Y))
        #Build the L matrix.
        L = np.tril(np.matmul(self.S.T, self.Y), -1)
        N = self.S.shape[1]
        D = np.zeros((N, N))
        for i in range(N):
            D[i, i] = np.dot(self.S[:, i], self.Y[:, i])
        self.center = np.linalg.pinv(np.block([[self.delta*np.matmul(self.S.T, self.S), L], [L.T, -D]]))
#        hessian = self.delta*np.identity(self.dim) - np.matmul(np.matmul(self.left, self.center), self.left.T)
#        #Estimate the local L-smoothness.
#        w,v = eigsh(hessian, 1, which = 'LA', maxiter = 10000000)
#        self.L = np.max(np.real(w))
#        #Estimate the local strong convexity.
#        w,v = eigsh(hessian, 1, which = 'SA', maxiter = 10000000)
#        self.Mu = np.min(np.real(w))
        return

    #Evaluate function.
    def fEval(self, x):
        if(self.S is not None):
            aux1 = np.dot(x - self.x_k, self.left)
            aux =  0.5*self.delta*np.dot(x - self.x_k, x - self.x_k) - 0.5*np.dot(np.dot(aux1, self.center), aux1)
            return np.dot(self.g, x  - self.x_k) + aux
        else:
            return np.dot(self.g, x  - self.x_k) 
    
    #Evaluate gradient.
    def fEvalGrad(self, x):
        if(self.S is not None):
            return self.g + self.delta*(x - self.x_k) - np.dot(np.dot(self.left, self.center), np.dot(x - self.x_k, self.left))
        else:
            return self.g 
    
    #Line Search.
    def lineSearch(self, grad, d, x):
        if(self.S is not None):
            aux1 = np.dot(d, self.left)
            aux = self.delta*np.dot(d, d) - np.dot(np.dot(aux1, self.center), aux1)
            return -np.dot(grad, d)/aux
        else:
            return 100000.0
    
    #Update gradient vector
    def updateGradient(self, gradient):
        self.g = gradient.copy()
        return
    
    #Update point.
    def updatePoint(self, xk):
        self.x_k = xk.copy()
        return
    
    #Is the approximation linear.
    def isLinear(self):
        return self.S is None
    
    def largestEig(self):
        if(self.S is not None):
#            return self.L
            return 1.0
        else:
            return 1.0

    def smallestEig(self):
        if(self.S is not None):
#            return self.Mu
            return 1.0
        else:
            return 1.0
    
#Creates a compact Hessian Approximation.
#m is the number of elements we'll use to calculate the matrix.
class QuadApproxInexactHessianBFGS:
    import numpy as np
    def __init__(self, dimension, gradient, xk):
        self.dim = dimension
        self.g = gradient.copy()
        self.x_k = xk.copy()
        self.Hessian = None
        self.L = 1.0
        self.Mu = 1.0
        return       
          
    #Update the function
    def updateRepresentation(self, s, y):
        if(np.dot(s, y) <= 0.0):
            print("The direction was not a descent direction.")
            quit()
        if self.Hessian is None:
            self.Hessian = np.dot(y,y)/np.dot(s,y)*np.identity(self.dim)
        else:
            self.Hessian = self.Hessian - np.outer(np.dot(self.Hessian, s), np.dot(s.T, self.Hessian))/np.dot(s.T, np.dot(self.Hessian, s)) + np.outer(y,y)/np.dot(s,y)
#        w, v = np.linalg.eig(self.Hessian)
#        print("Minimum Eigenvalues: " + str(np.min(np.real(w))))
#        print("Maximum Eigenvalues: " + str(np.max(np.real(w))))
#        self.Mu = np.min(np.real(w))
#        self.L = np.max(np.real(w)) 
        return

    #Evaluate function.
    def fEval(self, x):
        if self.Hessian is None:
            return np.dot(self.g, x - self.x_k)
        else:
            return np.dot(self.g, x - self.x_k) + 0.5*np.dot(x - self.x_k, np.dot(self.Hessian, x - self.x_k))
    
    #Evaluate gradient.
    def fEvalGrad(self, x):
        if self.Hessian is None:
            return self.g
        else:
            return self.g + np.dot(self.Hessian, x - self.x_k)
    
    #Line Search.
    def lineSearch(self, grad, d, x):
        if self.Hessian is None:
            return 100000.0
        else:
            return -np.dot(grad, d)/np.dot(d, np.dot(self.Hessian, d))
    
    #Update gradient vector
    def updateGradient(self, gradient):
        self.g = gradient.copy()
        return
    
    #Update point.
    def updatePoint(self, xk):
        self.x_k = xk.copy()
        return
    
    def greatestEig(self):
        return self.L

    def smallestEig(self):
        return self.Mu
    
    
#Projected Newton method with AFW to solve the subproblems.     
def runProjectedNewton(x0, activeSet, lambdas, function, feasibleReg, tolerance, maxTime, locOpt, criterion = "PG", criterionRef = 0.0, Hessian = "Exact", ExactLinesearch = True, TypeSolver = "Lazy", forcingParam = None, useDual = None, HessianParam = None):
    #Quantities we want to output.
    grad = function.fEvalGrad(x0)
    FWGap = [np.dot(grad, x0 - feasibleReg.LPOracle(grad))]
    fVal = [function.fEval(x0)]
    timing = [time.time()]
    xVal = [np.linalg.norm(x0 - locOpt)]
    activeSize = [1]
    x = x0.copy()
    activeProj  = activeSet.copy()
    lambdaValProj = lambdas.copy()
    proj = x0.copy()
    itCount = 1
    #Create the quadratic function we'll keep updating.
    if(Hessian == "Exact"):
        #Use exact Hessian
        fun = QuadApprox(function.returnM(), grad, x)
    else:
        if(Hessian == "LBFGS"):
            if(HessianParam is not None):
                fun = QuadApproxInexactHessianLBFGS(len(x0), HessianParam, grad, x)
            else:
                fun = QuadApproxInexactHessianLBFGS(len(x0), 20, grad, x)
        if(Hessian == "BFGS"):
            fun = QuadApproxInexactHessianBFGS(len(x0), grad, x)
    subprobTol = FWGap[-1]
    while(True):
        if(forcingParam is not None):
            #Use forcing sequence for the errors to go to zero.
            forcing = forcingParam
            subprobTol *= forcing
            subprobTol = max(tolerance, subprobTol)
        else:
            if(useDual is not None):
                #Use the FW Gap to drive convergence.
                subprobTol = max(tolerance, (FWGap[-1]/np.linalg.norm(grad))**2)
            else:
                #Original subproblem tolerance.
                eta = function.smallestEig()/np.sqrt(2*function.largestEig())
                subprobTol = max(tolerance, (eta*(fVal[-1] - criterionRef)/np.linalg.norm(grad))**2)          
        
        #Solve the subproblems to a given accuracy.
        if(TypeSolver == "DICG"):
            proj, gapSub, fValSub, timingSub, distanceSub = DIPFW(proj, fun, feasibleReg, subprobTol, maxTime, np.zeros(len(x)), typeStep = "EL", criterion = "DG")
        else:
            proj, gapSub, fValSub, timingSub, lambdaValProj, activeProj, distanceSub  = runFW(proj, activeProj, lambdaValProj, fun, feasibleReg, subprobTol, maxTime, np.zeros(len(x)), FWVariant = TypeSolver, typeStep = "EL", criterion = "DG", returnVar = True)
        #Choose which step to take.
        alpha = min(1.0, function.lineSearch(grad, proj - x, x))
        xOld = x.copy()
        x += alpha*(proj - x)
        gradOld = grad.copy()
        grad = function.fEvalGrad(x)
        performUpdate(function, x, FWGap, fVal, timing, np.dot(grad, x - feasibleReg.LPOracle(grad)))
        xVal.append(np.linalg.norm(x - locOpt))
        if(exitCriterion(itCount, fVal[-1], FWGap[-1], criterion = criterion, numCriterion = tolerance, critRef = criterionRef) or timing[-1] - timing[0] > maxTime):
            timing[:] = [t - timing[0] for t in timing]
            return x, FWGap, fVal, timing, activeSize, xVal
        itCount += 1
        #Update the function
        fun.updateGradient(grad)
        fun.updatePoint(x)
        #Update the inexact quadratic
        if(Hessian != "Exact"):
            fun.updateRepresentation(x - xOld, grad - gradOld)

#Perform one step of the AFW algorithm
#Also specifies if the number of vertices has decreased var = -1 or
#if it has increased var = +1. Otherwise 0.      
def awayStepFW(function, feasibleReg, x, activeSet, lambdas, typeStep):
    grad = function.fEvalGrad(x)
    v = feasibleReg.LPOracle(grad)
    a, indexMax = feasibleReg.AwayOracle(grad, activeSet)
    vertvar = 0
    #Choose FW direction, can overwrite index.
    FWGap = np.dot(grad, x - v)
    if(FWGap == 0.0):
        return x , vertvar, FWGap
    if(FWGap > np.dot(grad, a - x)):
      d = v - x
      alphaMax = 1.0
      optStep = stepSize(function, d, grad, x, typeStep)
      alpha = min(optStep, alphaMax)
      
      if(function.fEval(x + alpha*d) > function.fEval(x)):
          options={'xatol': 1e-12, 'maxiter': 500000, 'disp': 0}
          def InnerFunction(t):  # Hidden from outer code
              return function.fEval(x + t*d)
          res = minimize_scalar(InnerFunction, bounds=(0, alphaMax), method='bounded', options = options)
          alpha = min(res.x, alphaMax)
            
      
      if(alpha != alphaMax):
          #newVertex returns true if vertex is new.
          flag, index = newVertexFailFast(v, activeSet)
          lambdas[:] = [i * (1 - alpha) for i in lambdas]
          if(flag):
              activeSet.append(v)
              lambdas.append(alpha)
              vertvar = 1
          else:
              #Update existing weights
              lambdas[index] += alpha
      #Max step length away step, only one vertex now.
      else:
          activeSet[:] = [v]
          lambdas[:] = [alphaMax]
          vertvar = -1
    else:
      d = x - a
      alphaMax = lambdas[indexMax]/(1.0 - lambdas[indexMax])
      optStep = stepSize(function, d, grad, x, typeStep, maxStep = alphaMax)
      alpha = min(optStep, alphaMax)
      
      if(function.fEval(x + alpha*d) > function.fEval(x)):
          options={'xatol': 1e-12, 'maxiter': 500000, 'disp': 0}
          def InnerFunction(t):  # Hidden from outer code
              return function.fEval(x + t*d)
          res = minimize_scalar(InnerFunction, bounds=(0, alphaMax), method='bounded', options = options)
          alpha = min(res.x, alphaMax)
      
      lambdas[:] = [i * (1 + alpha) for i in lambdas]
      #Max step, need to delete a vertex.
      if(alpha != alphaMax):
          lambdas[indexMax] -= alpha
      else:
          deleteVertexIndex(indexMax, activeSet, lambdas)
          vertvar = -1
    return x + alpha*d, vertvar, FWGap

#Perform one step of the Pairwise FW algorithm
#Also specifies if the number of vertices has decreased var = -1 or
#if it has increased var = +1. Otherwise 0.
def pairwiseStepFW(function, feasibleReg, x, activeSet, lambdas, typeStep):
    grad = function.fEvalGrad(x)
    v = feasibleReg.LPOracle(grad)
    a, index = feasibleReg.AwayOracle(grad, activeSet)
    vertVar = 0
    #Find the weight of the extreme point a in the decomposition.
    alphaMax = lambdas[index]
    #Update weight of away vertex.
    d = v - a
    optStep = stepSize(function, d, grad, x, typeStep, maxStep = alphaMax)
    alpha = min(optStep, alphaMax)
    lambdas[index] -= alpha
    if(alpha == alphaMax):
        deleteVertexIndex(index, activeSet, lambdas)
        vertVar = -1
    #Update the FW vertex
    flag, index = newVertexFailFast(v, activeSet)
    if(flag):
        activeSet.append(v)
        lambdas.append(alpha)
        vertVar = 1
    else:
        lambdas[index] += alpha
    return x + alpha*d, vertVar, np.dot(grad, x - v)

#Perform one step of the Lazified AFW algorithm
#Also specifies if the number of vertices has decreased var = -1 or
#if it has increased var = +1. Otherwise 0.
def awayStepFWLazy(function, feasibleReg, x, activeSet, lambdas, phiVal, typeStep):
    grad = function.fEvalGrad(x)
    a, indexMax, v, indexMin = maxMinVertex(grad, activeSet)
    vertvar = 0
    #Use old FW vertex.
    if(np.dot(grad, x - v) >= np.dot(grad, a - x) and np.dot(grad, x - v) > phiVal[0]/2.0):
        d = v - x
        alphaMax = 1.0
        optStep = stepSize(function, d, grad, x, typeStep)
        alpha = min(optStep, alphaMax)
        if(alpha != alphaMax):
            lambdas[:] = [i * (1 - alpha) for i in lambdas]
            lambdas[indexMin] += alpha
        #Max step length away step, only one vertex now.
        else:
            activeSet[:] = [v]
            lambdas[:] = [alphaMax]
            vertvar = -1        
    else:
        #Use old away vertex.
        if(np.dot(grad, a - x) > np.dot(grad, x - v) and np.dot(grad, a - x) > phiVal[0]/2.0):
            d = x - a
            alphaMax = lambdas[indexMax]/(1.0 - lambdas[indexMax])
            optStep = stepSize(function, d, grad, x, typeStep, maxStep = alphaMax)
            alpha = min(optStep, alphaMax)
            lambdas[:] = [i * (1 + alpha) for i in lambdas]
            #Max step, need to delete a vertex.
            if(alpha != alphaMax):
                lambdas[indexMax] -= alpha
            else:
                deleteVertexIndex(indexMax, activeSet, lambdas)
                vertvar = -1            
        else:
            v = feasibleReg.LPOracle(grad)
            #New FW vertex.
            if(np.dot(grad, x - v) > phiVal[0]/2.0):
                d = v - x
                alphaMax = 1.0
                optStep = stepSize(function, d, grad, x, typeStep)
                alpha = min(optStep, alphaMax)
                #Less than maxStep
                if(alpha != alphaMax):
                    #newVertex returns true if vertex is new.
                    lambdas[:] = [i * (1 - alpha) for i in lambdas]
                    activeSet.append(v)
                    lambdas.append(alpha)
                    vertvar = 1
                #Max step length away step, only one vertex now.
                else:
                    activeSet[:] = [v]
                    lambdas[:] = [alphaMax]
                    vertvar = -1                
            #None of the vertices are satisfactory, halve phi.
            else:
                phiVal[0] = phiVal[0]/2.0
                alpha = 0.0
                d = v - x
    return x + alpha*d, vertvar, np.dot(grad, x - v)

#Perform one step of the vanilla FW algorithm
#Also specifies if the number of vertices has decreased var = -1 or
#if it has increased var = +1. Otherwise 0.
def stepFW(function, feasibleReg, x, activeSet, lambdas, typeStep):
    grad = function.fEvalGrad(x)
    v = feasibleReg.LPOracle(grad)
    vertvar = 0
    #Choose FW direction, can overwrite index.
    d = v - x
    alphaMax = 1.0
    optStep = stepSize(function, d, grad, x, typeStep)
    alpha = min(optStep, alphaMax)
    #Less than maxStep
    if(alpha != alphaMax):
        #newVertex returns true if vertex is new.
        flag, index = newVertexFailFast(v, activeSet)
        lambdas[:] = [i * (1 - alpha) for i in lambdas]
        if(flag):
            activeSet.append(v)
            lambdas.append(alpha)
            vertvar = 1
        else:
            #Update existing weights
            lambdas[index] += alpha
      #Max step length away step, only one vertex now.
    else:
        activeSet[:] = [v]
        lambdas[:] = [alphaMax]
        vertvar = -1
    return x + alpha*d, vertvar, np.dot(grad, x - v)

#Decomposition Invariant PFW. Only works for 0/1 polytopes in the unit
#hypercube which can be expressed in standard form.
#Can only either use linesearch or fixed step.    
"""## Decomposition Invariant CG (DICG)

Parameters:

1 --> typeStep: Specifies the type of step. Choosing "EL" performs exact line search
for the quadratic objective functions. Otherwise choosing "SS" chooses a step
that minimizes the smoothness equation and ensures progress in every iteration.

2 --> criterion: Specify if the terminating criterion is the primal gap ("PG") or the
dual gap ("DG"). If anything else is specified will run for a given number of iterations.

3 --> criterionRef: Value to which the algorithm will run, according to the criterion choosen.

"""

def DIPFW(x0, function, feasibleReg, tolerance, maxTime, locOpt, typeStep = "SS", criterion = "PG", criterionRef = 0.0):
    x = x0.copy()
    FWGap = []
    fVal = []
    timing = []
    xVal = []
    itCount = 1
    grad = function.fEvalGrad(x)
    v = feasibleReg.LPOracle(grad)
    timeRef = time.time()
    while(True):
        performUpdate(function, x, FWGap, fVal, timing, np.dot(grad, x - v))
        xVal.append(np.linalg.norm(x - locOpt))
        if(exitCriterion(itCount, fVal[-1], FWGap[-1], criterion = criterion, numCriterion = tolerance, critRef = criterionRef) or timing[-1] - timing[0] > maxTime):
            timing[:] = [t - timeRef for t in timing]
            return x, FWGap, fVal, timing, xVal
        gradAux = grad.copy()
        for i in range(len(gradAux)):
            if(x[i] == 0.0):
                gradAux[i] = -1.0e15
        a = feasibleReg.LPOracle(-gradAux)
        #Find the weight of the extreme point a in the decomposition.
        d = v - a
        alphaMax = calculateStepsize(x, d)
        optStep = stepSizeDI(function,feasibleReg, itCount, d, grad, x, typeStep)
        alpha = min(optStep, alphaMax)
        x += alpha*d
        grad = function.fEvalGrad(x)
        v = feasibleReg.LPOracle(grad)
        itCount += 1