import numpy as np
import time
from scipy.optimize import minimize_scalar
import math

# Import functions for the stepsizes and the updates.
from auxiliaryFunctions import (
    performUpdate,
    exitCriterion,
    stepSize,
    calculateStepsize,
    stepSizeDI,
)

# Import functions for active set management
from auxiliaryFunctions import newVertexFailFast, deleteVertexIndex, maxMinVertex


class CGS:
    """
    Run CGS

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize.
    feasibleReg : feasible region function.
        Returns LP oracles over feasible region.
    tolerance : float
        Tolerance to which we solve problem.
    maxTime : float
        Maximum number of seconds the algorithm is run.
    locOpt : numpy array
        Location of the optimal value (to keep track of distance to optimum)
    criterion : str
        Criterion for stopping: Dual gap or primal gap (DG, PG)
    criterionRef : float
        Value of the function evaluated at the optimum.
        
    Returns
    -------
    x
        Output point
    FWGap
        List containing FW gap for points along the run.
    fVal
        List containing primal gap for points along the run.
    timing
        List containing timing for points along the run.
    iteration
        List contains the number of cumulative LP oracle calls performed.
    distance
        List containing distance to optimum for points along the run.
        
    """

    def __init__(self):
        return

    def run(
        self,
        x0,
        function,
        feasibleReg,
        tolerance,
        maxTime,
        locOpt,
        criterion="PG",
        criterionRef=0.0,
    ):
        self.iteration = 0
        # Quantities we want to output.
        grad = function.fEvalGrad(x0)
        FWGap = [np.dot(grad, x0 - feasibleReg.LPOracle(grad))]
        fVal = [function.fEval(x0)]
        timing = [time.time()]
        distance = [np.linalg.norm(x0 - locOpt)]
        iteration = [1]
        x = x0.copy()
        self.limit_time = maxTime
        self.initTime = timing[0]
        itCount = 1.0
        N = int(
            np.ceil(2 * np.sqrt(6.0 * function.largestEig() / function.smallestEig()))
        )
        s = 1.0
        while True:
            x = self.CGSubroutine(function, feasibleReg, x0, FWGap[0], N, s)
            if (
                exitCriterion(
                    itCount,
                    fVal[-1],
                    FWGap[-1],
                    criterion=criterion,
                    numCriterion=tolerance,
                    critRef=criterionRef,
                )
                or time.time() - timing[0] > maxTime
            ):
                timing[:] = [t - timing[0] for t in timing]
                return "CGS", x, FWGap, fVal, timing, distance, iteration
            grad = function.fEvalGrad(x)
            performUpdate(
                function,
                x,
                FWGap,
                fVal,
                timing,
                np.dot(grad, x - feasibleReg.LPOracle(grad)),
            )
            itCount += 1
            distance.append(np.linalg.norm(x - locOpt))
            iteration.append(itCount)
            s += 1.0

    # Runs the subroutine with the stepsizes for the number of iterations depicted.
    def CGSubroutine(self, function, feasibleRegion, x0, delta0, N, s):
        L = function.largestEig()
        Mu = function.smallestEig()
        y = x0.copy()
        x = x0.copy()
        for k in range(1, N + 1):
            gamma = 2.0 / (k + 1.0)
            nu = 8.0 * L * delta0 * np.power(2, -s) / (Mu * N * k)
            beta = 2.0 * L / k
            z = (1 - gamma) * y + gamma * x
            x = self.CGSuProjection(function.fEvalGrad(z), x, beta, nu, feasibleRegion)
            if time.time() - self.initTime > self.limit_time:
                return y
            y = (1 - gamma) * y + gamma * x
        return y

    # Subroutine used in CGS for str.cvx. smooth functions.
    def CGSuProjection(self, g, u, beta, nu, feasibleRegion):
        t = 1
        u_t = u
        while True:
            grad = g + beta * (u_t - u)
            v = feasibleRegion.LPOracle(grad)
            self.iteration += 1
            V = np.dot(g + beta * (u_t - u), u_t - v)
            if time.time() - self.initTime > self.limit_time:
                return u_t
            if V <= nu:
                return u_t
            else:
                d = v - u_t
                alphaOpt = -np.dot(grad, d) / (beta * np.dot(d, d))
                alpha = min(1, alphaOpt)
                # alpha = min(1, np.dot(beta*(u - u_t) - g, v - u_t)/(beta*np.dot(v - u_t, v - u_t)))
                u_t = (1 - alpha) * u_t + alpha * v
                t += 1


def runSVRCG(
    x0,
    function,
    feasibleReg,
    tolerance,
    maxTime,
    locOpt,
    criterion="PG",
    criterionRef=0.0,
):
    """
    Run SVRFW

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize.
    feasibleReg : feasible region function.
        Returns LP oracles over feasible region.
    tolerance : float
        Tolerance to which we solve problem.
    maxTime : float
        Maximum number of seconds the algorithm is run.
    locOpt : numpy array
        Location of the optimal value (to keep track of distance to optimum)
    criterion : str
        Criterion for stopping: Dual gap or primal gap (DG, PG)
    criterionRef : float
        Value of the function evaluated at the optimum.
        
    Returns
    -------
    x
        Output point
    FWGap
        List containing FW gap for points along the run.
    fVal
        List containing primal gap for points along the run.
    timing
        List containing timing for points along the run.
    distance
        List containing distance to optimum for points along the run.
        
    """
    # Quantities we want to output.
    grad = function.fEvalGrad(x0)
    FWGap = [np.dot(grad, x0 - feasibleReg.LPOracle(grad))]
    fVal = [function.fEval(x0)]
    timing = [time.time()]
    distance = [np.linalg.norm(x0 - locOpt)]
    iteration = [1]
    x = x0.copy()
    itCount_t = 1
    while True:
        N_t = int(math.ceil(np.power(2, itCount_t + 3) - 2))
        snapShot = function.fEvalGrad(x)
        snapPoint = x.copy()
        for k in range(0, N_t):
            m_k = int(math.ceil(96.0 * (k + 2)))
            StochGrad = function.fEvalGradStoch(x, snapShot, snapPoint, m_k)
            v = feasibleReg.LPOracle(StochGrad)
            gamma_k = 2.0 / (k + 2)
            x = x + gamma_k * (v - x)
            if (
                exitCriterion(
                    itCount_t,
                    fVal[-1],
                    FWGap[-1],
                    criterion=criterion,
                    numCriterion=tolerance,
                    critRef=criterionRef,
                )
                or timing[-1] - timing[0] > maxTime
            ):
                timing[:] = [t - timing[0] for t in timing]
                return "SVRCG", x, FWGap, fVal, timing, distance, iteration
        grad = function.fEvalGrad(x)
        itCount_t += 1
        iteration.append(itCount_t)
        performUpdate(
            function,
            x,
            FWGap,
            fVal,
            timing,
            np.dot(grad, x - feasibleReg.LPOracle(grad)),
        )
        distance.append(np.linalg.norm(x - locOpt))
        if (
            exitCriterion(
                itCount_t,
                fVal[-1],
                FWGap[-1],
                criterion=criterion,
                numCriterion=tolerance,
                critRef=criterionRef,
            )
            or timing[-1] - timing[0] > maxTime
        ):
            timing[:] = [t - timing[0] for t in timing]
            return "SVRCG", x, FWGap, fVal, timing, distance, iteration


def runCG(
    x0,
    activeSet,
    lambdas,
    function,
    feasibleReg,
    tolerance,
    maxTime,
    locOpt,
    FWVariant="ACG",
    typeStep="EL",
    criterion="PG",
    criterionRef=0.0,
    returnVar=None,
    maxIter=None,
):
    """
    Run CG Variant.

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    activeSet : list of numpy arrays.
        Initial active set.
    lambdas : list of floats.
        Initial barycentric coordinates.
    function: function being minimized
        Function that we will minimize.
    feasibleReg : feasible region function.
        Returns LP oracles over feasible region.
    tolerance : float
        Tolerance to which we solve problem.
    maxTime : float
        Maximum number of seconds the algorithm is run.
    locOpt : numpy array
        Location of the optimal value (to keep track of distance to optimum)
    FWVariant : str
        Variant used to minimize function. (AFW, PFW, Vanilla, Lazy)
    typeStep : str
        Type of step used (EL, SS)
    criterion : str
        Criterion for stopping: Dual gap or primal gap (DG, PG)
    criterionRef : float
        Value of the function evaluated at the optimum.
   returnVar : Bool
        If function returns active set and lambda
   maxIter : int
        Maximum number of iterations run.      
        
    Returns
    -------
    x
        Output point
    FWGap
        List containing FW gap for points along the run.
    fVal
        List containing primal gap for points along the run.
    timing
        List containing timing for points along the run.
    lambdaVal
        List barycentric coordinates of final point
    active
        List containing numpy arrays with vertices in actice set.
    distance
        List containing distance to optimum for points along the run.
        
    """
    # Quantities we want to output.
    grad = function.fEvalGrad(x0)
    FWGap = [np.dot(grad, x0 - feasibleReg.LPOracle(grad))]
    fVal = [function.fEval(x0)]
    timing = [time.time()]
    distance = [np.linalg.norm(x0 - locOpt)]
    iteration = [1]
    x = x0.copy()
    active = activeSet.copy()
    lambdaVal = lambdas.copy()
    itCount = 1
    if FWVariant == "LazyACG":
        phiVal = [FWGap[-1]]
    while True:
        if FWVariant == "ACG":
            x, gap = awayStepFW(function, feasibleReg, x, active, lambdaVal, typeStep)
        else:
            if FWVariant == "PCG":
                x, gap = pairwiseStepFW(
                    function, feasibleReg, x, active, lambdaVal, typeStep
                )
            if FWVariant == "LazyACG":
                x, gap = awayStepFWLazy(
                    function, feasibleReg, x, active, lambdaVal, phiVal, typeStep
                )
            if FWVariant == "CG":
                x, gap = stepFW(function, feasibleReg, x, active, lambdaVal, typeStep)
        itCount += 1
        performUpdate(function, x, FWGap, fVal, timing, gap)
        iteration.append(itCount)
        distance.append(np.linalg.norm(x - locOpt))
        if (
            exitCriterion(
                itCount,
                fVal[-1],
                FWGap[-1],
                criterion=criterion,
                numCriterion=tolerance,
                critRef=criterionRef,
            )
            or timing[-1] - timing[0] > maxTime
        ):
            timing[:] = [t - timing[0] for t in timing]
            if returnVar is not None:
                return (
                    FWVariant,
                    x,
                    FWGap,
                    fVal,
                    timing,
                    lambdaVal[:],
                    active[:],
                    distance,
                    iteration,
                )
            else:
                return FWVariant, x, FWGap, fVal, timing, distance, iteration
        if maxIter is not None:
            if itCount > maxIter:
                timing[:] = [t - timing[0] for t in timing]
                if returnVar is not None:
                    return (
                        FWVariant,
                        x,
                        FWGap,
                        fVal,
                        timing,
                        lambdaVal[:],
                        active[:],
                        distance,
                        iteration,
                    )
                else:
                    return FWVariant, x, FWGap, fVal, timing, distance, iteration


def SOCGS(
    x0,
    activeSet,
    lambdas,
    function,
    QuadFunApprox,
    feasibleReg,
    tolerance,
    maxTime,
    locOpt,
    criterion="PG",
    criterionRef=0.0,
    TypeSolver="LazyACG",
    updateHessian=True,
    known_primal_gap = False,
    maxIter=100,
    omega=0.0,
):
    """
    Run SOCGS

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    activeSet : list of numpy arrays.
        Initial active set.
    lambdas : list of floats.
        Initial barycentric coordinates.
    function: function being minimized
        Function that we will minimize.
    QuadFunApprox: quadratic function.
        Quadratic function that will be used in the PVM steps.
    feasibleReg : feasible region function.
        Returns LP oracles over feasible region.
    tolerance : float
        Tolerance to which we solve problem.
    maxTime : float
        Maximum number of seconds the algorithm is run.
    locOpt : numpy array
        Location of the optimal value (to keep track of distance to optimum)
    criterion : str
        Criterion for stopping: Dual gap or primal gap (DG, PG)
    criterionRef : float
        Value of the function evaluated at the optimum. Used to stop the algorithm
        and optionally to compute the inner loop stopping accuracy
    TypeSolver : str
        Variant used to minimize function. (AFW, PFW, Vanilla, Lazy, DICG)
    updateHessian : bool
        If the quadratic approximation explicitly requires updating Hessian.
    known_primal_gap : bool
        If True, then criterionRef will be used to compute the primal gap for the
        accuracy criterion, otherwisae, alternative strategy will be used.
   maxIter : int
        Maximum number of inner iterations used per outer iteration.
   omega : float
        Value of omega when returning inexact Hessian.
   
        
    Returns
    -------
    x
        Output point
    FWGap
        List containing FW gap for points along the run.
    fVal
        List containing primal gap for points along the run.
    timing
        List containing timing for points along the run.
    lambdaVal
        List barycentric coordinates of final point
    active
        List containing numpy arrays with vertices in actice set.
    distance
        List containing distance to optimum for points along the run.
        
    """
    # Quantities we want to output.
    grad = function.fEvalGrad(x0)
    FWGap = [np.dot(grad, x0 - feasibleReg.LPOracle(grad))]
    fVal = [function.fEval(x0)]
    timing = [time.time()]
    distance = [np.linalg.norm(x0 - locOpt)]
    iteration = [1]
    # Initialize SOCGS iterates
    x = x0.copy()
    activeSet = activeSet.copy()
    lambdaVal = lambdas.copy()
    # Initialize PVM iterates
    xPVM = x0.copy()
    activeSetPVM = activeSet.copy()
    lambdaValPVM = lambdas.copy()
    # Initialize CG iterate
    xCG = x0.copy()
    activeSetCG = activeSet.copy()
    lambdaValCG = lambdas.copy()
    # Create the approximate quadratic function
    if updateHessian:
        QuadFunApprox.updateApprox(
            grad, x, function.returnM(x, omega, distance[-1] ** 2)
        )
    else:
        QuadFunApprox.updateApprox(grad, x)
    # Used for the Lazy AFW algorithm.
    phiVal = [FWGap[-1]]
    itCount = 0
    while True:
        #Compute the inner problem accuracy using primal gap
        if(known_primal_gap):
            subprobTol = max(
                    tolerance, ((fVal[-1] - criterionRef) / np.linalg.norm(grad)) ** 4
                    )
        #Compute the inner problem accuracy using alternative strategy
        else:
            xCG_aux = xCG.copy()
            activeSetCG_aux = activeSetCG.copy()
            lambdaValCG_aux = lambdaValCG.copy()
            while(function.fEval(xCG_aux) >= function.fEval(xCG)):
                xCG_aux, _ = awayStepFW(
                        function, feasibleReg, xCG_aux, activeSetCG_aux, lambdaValCG_aux, "EL"
                    )
            subprobTol = max(
                    tolerance, ((fVal[-1] - function.fEval(xCG_aux)) / np.linalg.norm(grad)) ** 4
                    )
        if TypeSolver == "DICG":
            _, xPVM, _, _, _, _, _ = DIPFW(
                x,
                QuadFunApprox,
                feasibleReg,
                subprobTol,
                maxTime,
                np.zeros(len(x)),
                typeStep="EL",
                criterion="DG",
                maxIter=maxIter,
            )
            xCG, _, _ = stepDICG(function, feasibleReg, xCG, "EL")
        else:
            _, xPVM, _, _, _, lambdaValPVM[:], activeSetPVM[:], _, _ = runCG(
                x,
                activeSet,
                lambdaVal,
                QuadFunApprox,
                feasibleReg,
                subprobTol,
                maxTime,
                np.zeros(len(x)),
                FWVariant=TypeSolver,
                typeStep="EL",
                criterion="DG",
                returnVar=True,
                maxIter=maxIter,
            )
            if TypeSolver == "LazyACG":
                xCG, _ = awayStepFWLazy(
                    function, feasibleReg, xCG, activeSetCG, lambdaValCG, phiVal, "EL"
                )
            if TypeSolver == "PCG":
                xCG, _ = pairwiseStepFW(
                    function, feasibleReg, xCG, activeSetCG, lambdaValCG, "EL"
                )
            if TypeSolver == "CG":
                xCG, _ = stepFW(
                    function, feasibleReg, xCG, activeSetCG, lambdaValCG, "EL"
                )
            if TypeSolver == "ACG":
                xCG, _ = awayStepFW(
                    function, feasibleReg, xCG, activeSetCG, lambdaValCG, "EL"
                )
        if function.fEval(xCG) <= function.fEval(xPVM):
            x = xCG.copy()
            activeSet = activeSetCG.copy()
            lambdaVal = lambdaValCG.copy()
        else:
            x = xPVM.copy()
            activeSet = activeSetPVM.copy()
            lambdaVal = lambdaValPVM.copy()
        grad = function.fEvalGrad(x)
        itCount += 1
        iteration.append(itCount)
        performUpdate(
            function,
            x,
            FWGap,
            fVal,
            timing,
            np.dot(grad, x - feasibleReg.LPOracle(grad)),
        )
        distance.append(np.linalg.norm(x - locOpt))
        # Check the exit criterion.
        if (
            exitCriterion(
                itCount,
                fVal[-1],
                FWGap[-1],
                criterion=criterion,
                numCriterion=tolerance,
                critRef=criterionRef,
            )
            or timing[-1] - timing[0] > maxTime
        ):
            timing[:] = [t - timing[0] for t in timing]
            return "SOCGS", x, FWGap, fVal, timing, distance, iteration

        # Update the approximation.
        if updateHessian:
            QuadFunApprox.updateApprox(
                grad, x, function.returnM(x, omega, distance[-1] ** 2)
            )
        else:
            QuadFunApprox.updateApprox(grad, x)


class NCG:
    """
    Run NCG

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    activeSet : list of numpy arrays.
        Initial active set.
    lambdas : list of floats.
        Initial barycentric coordinates.
    function: function being minimized
        Function that we will minimize.
    QuadFunApprox: quadratic function.
        Quadratic function that will be used in the PVM steps.
    feasibleReg : feasible region function.
        Returns LP oracles over feasible region.
    tolerance : float
        Tolerance to which we solve problem.
    maxTime : float
        Maximum number of seconds the algorithm is run.
    locOpt : numpy array
        Location of the optimal value (to keep track of distance to optimum)
    criterion : str
        Criterion for stopping: Dual gap or primal gap (DG, PG)
    criterionRef : float
        Value of the function evaluated at the optimum.
    TypeSolver : str
        Variant used to minimize function. (AFW, PFW, Vanilla, Lazy, DICG)
    updateHessian : bool
        If the quadratic approximation explicitly requires updating Hessian.
   maxIter : int
        Maximum number of inner iterations used per outer iteration.
   
    Returns
    -------
    x
        Output point
    FWGap
        List containing FW gap for points along the run.
    fVal
        List containing primal gap for points along the run.
    timing
        List containing timing for points along the run.
    lambdaVal
        List barycentric coordinates of final point
    active
        List containing numpy arrays with vertices in actice set.
    distance
        List containing distance to optimum for points along the run.
        
    """

    def __init__(self, sigma, beta, C):
        self.sigma = sigma
        self.beta = beta
        self.C = C
        # Verify that we satisfy the conditions.
        assert (
            1 / (C * (1 - beta)) + beta / ((1 - 2 * beta) * (1 - beta) ** 2) <= sigma
        ), "First condition not satisfied."
        assert 1 / C + 1 / (1 - 2 * beta) <= 2, "Second condition not satisfied."
        return

    def run(
        self,
        x0,
        activeSet,
        lambdas,
        function,
        QuadFunApprox,
        feasibleReg,
        tolerance,
        maxTime,
        locOpt,
        criterion="PG",
        criterionRef=0.0,
        TypeSolver="Vanilla",
        updateHessian=True,
        maxIter=100,
    ):
        # Quantities we want to output.
        grad = function.fEvalGrad(x0)
        FWGap = [np.dot(grad, x0 - feasibleReg.LPOracle(grad))]
        fVal = [function.fEval(x0)]
        timing = [time.time()]
        distance = [np.linalg.norm(x0 - locOpt)]
        iteration = [1]
        x = x0.copy()
        activeSet = activeSet.copy()
        lambdasValues = lambdas.copy()
        itCount = 1.0
        C_1 = 0.25
        delta = 0.99
        lambdaVal = self.beta / self.sigma
        hValBeta = (
            self.beta
            * (1 - 2.0 * self.beta + 2.0 * self.beta ** 2)
            / ((1 - 2.0 * self.beta) * (1 - self.beta) ** 2 - self.beta ** 2)
        )
        etaVal = min(self.beta / self.C, C_1 / hValBeta)
        if updateHessian:
            QuadFunApprox.updateApprox(grad, x, function.returnM(x))
        else:
            QuadFunApprox.updateApprox(grad, x)
        while True:
            if TypeSolver == "CG":
                _, z, _, _, _, _, _ = runCG(
                    x,
                    activeSet,
                    lambdasValues,
                    QuadFunApprox,
                    feasibleReg,
                    etaVal ** 2,
                    maxTime,
                    np.zeros(len(x)),
                    FWVariant="CG",
                    typeStep="EL",
                    criterion="DG",
                    maxIter=maxIter,
                )
            else:
                _, z, _, _, _, _, _ = DIPFW(
                    x,
                    QuadFunApprox,
                    feasibleReg,
                    etaVal ** 2,
                    maxTime,
                    np.zeros(len(x)),
                    typeStep="EL",
                    criterion="DG",
                    maxIter=maxIter,
                )
            d = z - x
            gamma = QuadFunApprox.fEvalHessianNorm(d)

            # Take a full step.
            if gamma + etaVal <= 1 / hValBeta or lambdaVal <= self.beta:
                lambdaVal = self.sigma * lambdaVal
                etaVal = self.sigma * etaVal
                alpha = 1.0
                x = x + alpha * d
            else:
                alpha = min(
                    delta
                    * (gamma ** 2 - etaVal ** 2)
                    / (gamma ** 3 + gamma ** 2 - gamma * etaVal ** 2),
                    1.0,
                )
                x = x + alpha * d
            grad = function.fEvalGrad(x)
            itCount += 1
            iteration.append(itCount)
            performUpdate(
                function,
                x,
                FWGap,
                fVal,
                timing,
                np.dot(grad, x - feasibleReg.LPOracle(grad)),
            )
            distance.append(np.linalg.norm(x - locOpt))
            if (
                exitCriterion(
                    itCount,
                    fVal[-1],
                    FWGap[-1],
                    criterion=criterion,
                    numCriterion=tolerance,
                    critRef=criterionRef,
                )
                or timing[-1] - timing[0] > maxTime
            ):
                timing[:] = [t - timing[0] for t in timing]
                return "NCG", x, FWGap, fVal, timing, distance, iteration
            if updateHessian:
                QuadFunApprox.updateApprox(grad, x, function.returnM(x))
            else:
                QuadFunApprox.updateApprox(grad, x)


def DIPFW(
    x0,
    function,
    feasibleReg,
    tolerance,
    maxTime,
    locOpt,
    typeStep="EL",
    criterion="PG",
    criterionRef=0.0,
    maxIter=None,
):
    """
    Run DIPFW for 0-1 polytopes.

    Parameters
    ----------
    x0 : numpy array.
        Initial point.
    function: function being minimized
        Function that we will minimize.
    feasibleReg : feasible region function.
        Returns LP oracles over feasible region.
    tolerance : float
        Tolerance to which we solve problem.
    maxTime : float
        Maximum number of seconds the algorithm is run.
    locOpt : numpy array
        Location of the optimal value (to keep track of distance to optimum)
    typeStep : str
        Type of step size used.
    criterion : str
        Criterion for stopping: Dual gap or primal gap (DG, PG)
    criterionRef : float
        Value of the function evaluated at the optimum.
   maxIter : int
        Maximum number of inner iterations used per outer iteration.
   
    Returns
    -------
    x
        Output point
    FWGap
        List containing FW gap for points along the run.
    fVal
        List containing primal gap for points along the run.
    timing
        List containing timing for points along the run.
    lambdaVal
        List barycentric coordinates of final point
    active
        List containing numpy arrays with vertices in actice set.
    distance
        List containing distance to optimum for points along the run.
        
    """
    x = x0.copy()
    FWGap, fVal, timing, distance, iteration = ([] for i in range(5))
    itCount = 1
    timeRef = time.time()
    while True:
        x, xOld, oldGap = stepDICG(function, feasibleReg, x, typeStep)
        distance.append(np.linalg.norm(xOld - locOpt))
        performUpdate(function, xOld, FWGap, fVal, timing, oldGap)
        iteration.append(itCount)
        itCount += 1
        if (
            exitCriterion(
                itCount,
                fVal[-1],
                FWGap[-1],
                criterion=criterion,
                numCriterion=tolerance,
                critRef=criterionRef,
            )
            or timing[-1] - timeRef > maxTime
        ):
            distance.append(np.linalg.norm(x - locOpt))
            grad = function.fEvalGrad(x)
            performUpdate(
                function,
                x,
                FWGap,
                fVal,
                timing,
                np.dot(grad, x - feasibleReg.LPOracle(grad)),
            )
            iteration.append(itCount)
            timing[:] = [t - timeRef for t in timing]
            return "DICG", x, FWGap, fVal, timing, distance, iteration
        if maxIter is not None:
            if itCount > maxIter:
                distance.append(np.linalg.norm(x - locOpt))
                grad = function.fEvalGrad(x)
                performUpdate(
                    function,
                    x,
                    FWGap,
                    fVal,
                    timing,
                    np.dot(grad, x - feasibleReg.LPOracle(grad)),
                )
                iteration.append(itCount)
                timing[:] = [t - timeRef for t in timing]
                return "DICG", x, FWGap, fVal, timing, distance, iteration


def stepDICG(function, feasibleReg, x, typeStep):
    """
    Performs a single step of the DICG/DIPFW algorithm.

    Parameters
    ----------
    function: function being minimized
        Function that we will minimize.
    feasibleReg : feasible region function.
        Returns LP oracles over feasible region.
    x : numpy array.
        Point.
    typeStep : str
        Type of step size used.
   
    Returns
    -------
    x + alpha*d
        Output point
    x
        Input point
    oldGap
        FW gap at initial point.
        
    """
    grad = function.fEvalGrad(x)
    v = feasibleReg.LPOracle(grad)
    oldGap = np.dot(grad, x - v)
    gradAux = grad.copy()
    for i in range(len(gradAux)):
        if x[i] == 0.0:
            gradAux[i] = -1.0e15
    a = feasibleReg.LPOracle(-gradAux)
    # Find the weight of the extreme point a in the decomposition.
    d = v - a
    alphaMax = calculateStepsize(x, d)
    optStep = stepSizeDI(function, feasibleReg, 1, d, grad, x, typeStep)
    alpha = min(optStep, alphaMax)
    return x + alpha * d, x, oldGap


def awayStepFW(function, feasibleReg, x, activeSet, lambdas, typeStep):
    """
    Performs a single step of the ACG/AFW algorithm.

    Parameters
    ----------
    function: function being minimized
        Function that we will minimize.
    feasibleReg : feasible region function.
        Returns LP oracles over feasible region.
    x : numpy array.
        Point.
    activeSet : list of numpy arrays.
        Initial active set.
    lambdas : list of floats.
        Initial barycentric coordinates.
    typeStep : str
        Type of step size used.
   
    Returns
    -------
    x + alpha*d
        Output point
    FWGap
        FW gap at initial point.
        
    """
    grad = function.fEvalGrad(x)
    v = feasibleReg.LPOracle(grad)
    a, indexMax = feasibleReg.AwayOracle(grad, activeSet)
    # Choose FW direction, can overwrite index.
    FWGap = np.dot(grad, x - v)
    if FWGap == 0.0:
        return x, FWGap
    if FWGap > np.dot(grad, a - x):
        d = v - x
        alphaMax = 1.0
        optStep = stepSize(function, d, grad, x, typeStep)
        alpha = min(optStep, alphaMax)
        if function.fEval(x + alpha * d) > function.fEval(x):
            options = {"xatol": 1e-12, "maxiter": 500000, "disp": 0}

            def InnerFunction(t):  # Hidden from outer code
                return function.fEval(x + t * d)

            res = minimize_scalar(
                InnerFunction, bounds=(0, alphaMax), method="bounded", options=options
            )
            alpha = min(res.x, alphaMax)
        if alpha != alphaMax:
            # newVertex returns true if vertex is new.
            flag, index = newVertexFailFast(v, activeSet)
            lambdas[:] = [i * (1 - alpha) for i in lambdas]
            if flag:
                activeSet.append(v)
                lambdas.append(alpha)
            else:
                # Update existing weights
                lambdas[index] += alpha
        # Max step length away step, only one vertex now.
        else:
            activeSet[:] = [v]
            lambdas[:] = [alphaMax]
    else:
        d = x - a
        alphaMax = lambdas[indexMax] / (1.0 - lambdas[indexMax])
        optStep = stepSize(function, d, grad, x, typeStep, maxStep=alphaMax)
        alpha = min(optStep, alphaMax)
        if function.fEval(x + alpha * d) > function.fEval(x):
            options = {"xatol": 1e-12, "maxiter": 500000, "disp": 0}

            def InnerFunction(t):  # Hidden from outer code
                return function.fEval(x + t * d)

            res = minimize_scalar(
                InnerFunction, bounds=(0, alphaMax), method="bounded", options=options
            )
            alpha = min(res.x, alphaMax)
        if alpha < 1.0e-9:
            alpha = alphaMax
        lambdas[:] = [i * (1 + alpha) for i in lambdas]
        # Max step, need to delete a vertex.
        if alpha != alphaMax:
            lambdas[indexMax] -= alpha
        else:
            deleteVertexIndex(indexMax, activeSet, lambdas)
    return x + alpha * d, FWGap


def pairwiseStepFW(function, feasibleReg, x, activeSet, lambdas, typeStep):
    """
    Performs a single step of the PCG/PFW algorithm.

    Parameters
    ----------
    function: function being minimized
        Function that we will minimize.
    feasibleReg : feasible region function.
        Returns LP oracles over feasible region.
    x : numpy array.
        Point.
    activeSet : list of numpy arrays.
        Initial active set.
    lambdas : list of floats.
        Initial barycentric coordinates.
    typeStep : str
        Type of step size used.
   
    Returns
    -------
    x + alpha*d
        Output point
    FWGap
        FW gap at initial point.
        
    """
    grad = function.fEvalGrad(x)
    v = feasibleReg.LPOracle(grad)
    a, index = feasibleReg.AwayOracle(grad, activeSet)
    # Find the weight of the extreme point a in the decomposition.
    alphaMax = lambdas[index]
    # Update weight of away vertex.
    d = v - a
    alpha = stepSize(function, d, grad, x, typeStep, maxStep=alphaMax)
    lambdas[index] -= alpha
    # Before this was an equality.
    if alphaMax - alpha < 1.0e-8:
        deleteVertexIndex(index, activeSet, lambdas)
    # Update the FW vertex
    flag, index = newVertexFailFast(v, activeSet)
    if flag:
        activeSet.append(v)
        lambdas.append(alpha)
    else:
        lambdas[index] += alpha
    return x + alpha * d, np.dot(grad, x - v)


def awayStepFWLazy(function, feasibleReg, x, activeSet, lambdas, phiVal, typeStep):
    """
    Performs a single step of the Lazy ACG/AFW algorithm.

    Parameters
    ----------
    function: function being minimized
        Function that we will minimize.
    feasibleReg : feasible region function.
        Returns LP oracles over feasible region.
    x : numpy array.
        Point.
    activeSet : list of numpy arrays.
        Initial active set.
    lambdas : list of floats.
        Initial barycentric coordinates.
    phiVal : List with a single float
        Value of phi used in a Lazy AFW step.
    typeStep : str
        Type of step size used.
   
    Returns
    -------
    x + alpha*d
        Output point
    FWGap
        FW gap at initial point.
        
    """
    grad = function.fEvalGrad(x)
    a, indexMax, v, indexMin = maxMinVertex(grad, activeSet)
    # Use old FW vertex.
    if (
        np.dot(grad, x - v) >= np.dot(grad, a - x)
        and np.dot(grad, x - v) > phiVal[0] / 2.0
    ):
        d = v - x
        alphaMax = 1.0
        optStep = stepSize(function, d, grad, x, typeStep)
        alpha = min(optStep, alphaMax)
        if alpha != alphaMax:
            lambdas[:] = [i * (1 - alpha) for i in lambdas]
            lambdas[indexMin] += alpha
        # Max step length away step, only one vertex now.
        else:
            activeSet[:] = [v]
            lambdas[:] = [alphaMax]
    else:
        # Use old away vertex.
        if (
            np.dot(grad, a - x) > np.dot(grad, x - v)
            and np.dot(grad, a - x) > phiVal[0] / 2.0
        ):
            d = x - a
            alphaMax = lambdas[indexMax] / (1.0 - lambdas[indexMax])
            optStep = stepSize(function, d, grad, x, typeStep, maxStep=alphaMax)
            alpha = min(optStep, alphaMax)
            if alpha < 1.0e-9:
                alpha = alphaMax
            lambdas[:] = [i * (1 + alpha) for i in lambdas]
            # Max step, need to delete a vertex.
            if alpha != alphaMax:
                lambdas[indexMax] -= alpha
            else:
                deleteVertexIndex(indexMax, activeSet, lambdas)
        else:
            v = feasibleReg.LPOracle(grad)
            # New FW vertex.
            if np.dot(grad, x - v) > phiVal[0] / 2.0:
                d = v - x
                alphaMax = 1.0
                optStep = stepSize(function, d, grad, x, typeStep)
                alpha = min(optStep, alphaMax)
                # Less than maxStep
                if alpha != alphaMax:
                    # newVertex returns true if vertex is new.
                    lambdas[:] = [i * (1 - alpha) for i in lambdas]
                    activeSet.append(v)
                    lambdas.append(alpha)
                # Max step length away step, only one vertex now.
                else:
                    activeSet[:] = [v]
                    lambdas[:] = [alphaMax]
            # None of the vertices are satisfactory, halve phi.
            else:
                phiVal[0] = min(np.dot(grad, x - v), phiVal[0] / 2.0)
                alpha = 0.0
                d = v - x
    return x + alpha * d, np.dot(grad, x - v)


def stepFW(function, feasibleReg, x, activeSet, lambdas, typeStep):
    """
    Performs a single step of the Vanilla CG/FW algorithm.

    Parameters
    ----------
    function: function being minimized
        Function that we will minimize.
    feasibleReg : feasible region function.
        Returns LP oracles over feasible region.
    x : numpy array.
        Point.
    activeSet : list of numpy arrays.
        Initial active set.
    lambdas : list of floats.
        Initial barycentric coordinates.
    typeStep : str
        Type of step size used.
   
    Returns
    -------
    x + alpha*d
        Output point
    FWGap
        FW gap at initial point.
        
    """
    grad = function.fEvalGrad(x)
    v = feasibleReg.LPOracle(grad)
    # Choose FW direction, can overwrite index.
    d = v - x
    alphaMax = 1.0
    optStep = stepSize(function, d, grad, x, typeStep)
    alpha = min(optStep, alphaMax)
    # Less than maxStep
    if alpha != alphaMax:
        # newVertex returns true if vertex is new.
        flag, index = newVertexFailFast(v, activeSet)
        lambdas[:] = [i * (1 - alpha) for i in lambdas]
        if flag:
            activeSet.append(v)
            lambdas.append(alpha)
        else:
            # Update existing weights
            lambdas[index] += alpha
    # Max step length away step, only one vertex now.
    else:
        activeSet[:] = [v]
        lambdas[:] = [alphaMax]
    return x + alpha * d, np.dot(grad, x - v)
