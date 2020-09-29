if __name__ == "__main__":

    import os

    # Computing parameters.
    os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

    # General imports
    import numpy as np
    import os, sys
    import time
    import datetime
    import matplotlib.pyplot as plt
    from algorithms import runCG, SOCGS, NCG
    from auxiliaryFunctions import randomPSDGenerator

    """
    ----------------------------Graphical Lasso experiment--------------------
    """
    ts = time.time()
    timestamp = (
        datetime.datetime.fromtimestamp(ts)
        .strftime("%Y-%m-%d %H:%M:%S")
        .replace(" ", "-")
        .replace(":", "-")
    )

    from feasibleRegions import PSDUnitTrace
    from functions import (
        GraphicalLasso,
        QuadApproxGLasso,
        QuadApproxInexactHessianLBFGS,
    )

    # Parse the arguments of the function.
    import argparse

    parser = argparse.ArgumentParser("Parse algorithm settings")
    parser.add_argument(
        "--max_time",
        type=int,
        required=True,
        help="Maximum time the algorithms are run in seconds.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        required=True,
        help="Dimensionality of the problem n. This results in matrices of size nxn.",
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        required=True,
        help="Accuracy to which the problem is solved.",
    )
    parser.add_argument(
        "--lambda_value",
        type=float,
        required=True,
        help="Lambda value for l2 regularization.",
    )
    parser.add_argument(
        "--delta_value",
        type=float,
        required=True,
        help="Delta value for smoothing of logdet.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        required=True,
        help="Maximum number of inner iterations in second-order algorithms.",
    )
    parser.add_argument(
        "--type_solver",
        type=str,
        required=True,
        help="CG subsolver to use in SOCGS: CG, ACG, PCG, LazyACG.",
    )
    parser.add_argument(
        "--known_primal_gap",
        type=str,
        required=True,
        help="True if the primal gap is known, false otherwise.",
    )

    args = parser.parse_args()
    TIME_LIMIT = args.max_time
    TIME_LIMIT_REFERENCE_SOL = int(2.0 * args.max_time)
    dimension = args.dimension
    tolerance = args.accuracy
    lambdaVal = args.lambda_value
    deltaVal = args.delta_value
    maxIter = args.max_iter
    type_of_solver = args.type_solver
    if(args.known_primal_gap == 'True'):
        known_primal_gap = True
    else:
        if(args.known_primal_gap == 'False'):
            known_primal_gap = False
        else:
            assert False, 'Invalid known_primal_gap argument'

    #    #Problem dimension n for the matrix. The resulting matrix will have size nxn.
    #    dimension = int(sys.argv[1])
    #    #Regularization parameter used in the 2 norm.
    #    lambdaVal = float(sys.argv[2])
    #    #Small delta used to make the problem smooth.
    #    deltaVal = float(sys.argv[3])
    #    #Time limit spent calculating the reference solution and running the algorithm.
    #    TIME_LIMIT_REFERENCE_SOL = int(5*int(sys.argv[4]))
    #    TIME_LIMIT = int(sys.argv[4])
    #    #Tolerance to which we will solve the problem.
    #    tolerance = float(sys.argv[5])
    #    #Maximum number of inner iterations that we will have in the SOCGS and NCG algorithm.
    #    maxIter = int(sys.argv[6])

    # Declare function.
    solution = randomPSDGenerator(dimension, 0.5, 1.0)
    solution /= np.trace(solution)
    covariance = np.linalg.inv(solution)
    fun = GraphicalLasso(dimension, covariance, lambdaVal)

    # Declare feasible region
    feasibleRegion = PSDUnitTrace(int(dimension * dimension))

    # Declare quadratic approximations.
    funQuadApprox = QuadApproxGLasso(dimension, lambdaVal, delta=deltaVal)

    # Number of samples that are going to be used to build the LBFGS approximation to the Hessian.
    numSamplesHessian = 10
    funQuadApproxLBFGS = QuadApproxInexactHessianLBFGS(
        int(dimension * dimension), numSamplesHessian
    )

    # Line search used. In this case exact line search.
    typeOfStep = "EL"

    # Create random starting point. Completely random.
    alpha_0 = np.linspace(1, 10, dimension)
    alpha_0 /= np.sum(alpha_0)
    x_0 = np.diag(alpha_0).flatten()
    S_0 = []
    for i in range(dimension):
        auxMat = np.zeros((dimension, dimension))
        auxMat[i, i] = 1.0
        S_0.append(auxMat.flatten())
    alpha_0 = alpha_0.tolist()

    print("Solving the problem over the Spectrahedron polytope.")
    if not os.path.exists(os.path.join(os.getcwd(), "Spectrahedron")):
        os.makedirs(os.path.join(os.getcwd(), "Spectrahedron"))

    ##Run to a high Frank-Wolfe primal gap accuracy for later use?
    from auxiliaryFunctions import exportsolution, dump_pickled_object

    print("\nFinding optimal solution to high accuracy using ACG.")
    nameAlg, xTest, FWGapTest, fValTest, timingTest, distTest, iterationTest = runCG(
        x_0,
        S_0,
        alpha_0,
        fun,
        feasibleRegion,
        tolerance / 2.0,
        TIME_LIMIT_REFERENCE_SOL,
        np.zeros(len(x_0)),
        FWVariant="ACG",
        typeStep=typeOfStep,
        criterion="DG",
    )
    fValOpt = fValTest[-1]
    tolerance = max(tolerance, min(np.asarray(FWGapTest)))
    if not os.path.exists(os.path.join(os.getcwd(), "Spectrahedron", "Solutions")):
        os.makedirs(os.path.join(os.getcwd(), "Spectrahedron", "Solutions"))
    # Saving solution for future use
    exportsolution(
        os.path.join(
            os.getcwd(),
            "Spectrahedron",
            "Solutions",
            "Solution_Spectrahedron_"
            + str(timestamp)
            + "_size"
            + str(dimension)
            + "_TypeStep_"
            + typeOfStep
            + ".txt",
        ),
        sys.argv,
        fValOpt,
        xTest,
        min(np.asarray(FWGapTest)),
        dimension,
    )
    dump_pickled_object(
        os.path.join(
            os.getcwd(),
            "Spectrahedron",
            "Solutions",
            "function_" + str(timestamp) + ".pickle",
        ),
        fun,
    )

    #    #Importing solution
    #    from auxiliaryFunctions import importSolution, load_pickled_object
    #    fValOpt, xTest, importTolerance, sizeSol = importSolution(os.path.join(os.getcwd(), "GLasso", "Solution_GLassoPSD_2020-05-29-05-10-31_size100_TypeStep_EL.txt"))
    #    tolerance = max(tolerance, importTolerance)
    #    fun = load_pickled_object(os.path.join(os.getcwd(), "GLasso", "function_2020-05-29-05-10-31.pickle"))

    # Create list to store all the results.
    results = []

    # Run SOCGS
    print("\nSOCGS.")
    resultsSOCGS1 = SOCGS(
        x_0,
        S_0,
        alpha_0,
        fun,
        funQuadApproxLBFGS,
        feasibleRegion,
        tolerance,
        TIME_LIMIT,
        xTest,
        criterion="PG",
        criterionRef=fValOpt,
        TypeSolver=type_of_solver,
        updateHessian=False,
        known_primal_gap = known_primal_gap,
        maxIter=maxIter,
    )
    resultsSOCGS1 = list(resultsSOCGS1)
    resultsSOCGS1[0] = "SOCGS-LBFGS"

    # Run SOCGS with LBFGS updates
    print("\nSOCGS with LBFGS updates.")
    resultsSOCGS = SOCGS(
        x_0,
        S_0,
        alpha_0,
        fun,
        funQuadApprox,
        feasibleRegion,
        tolerance,
        TIME_LIMIT,
        xTest,
        criterion="PG",
        criterionRef=fValOpt,
        TypeSolver=type_of_solver,
        updateHessian=False,
        maxIter=maxIter,
    )

    # Run Newton CG
    print("\nRunning NCG.")
    FrankWolfeProjNewton = NCG(0.96, 1 / 6.0, 2.0)
    resultsNCG = FrankWolfeProjNewton.run(
        x_0,
        S_0,
        alpha_0,
        fun,
        funQuadApprox,
        feasibleRegion,
        tolerance,
        TIME_LIMIT,
        xTest,
        criterion="PG",
        criterionRef=fValOpt,
        TypeSolver="CG",
        maxIter=maxIter,
        updateHessian=False,
    )

    # Run Lazy ACG
    print("\nRunning Lazy ACG.")
    resultsAFWLazy = runCG(
        x_0,
        S_0,
        alpha_0,
        fun,
        feasibleRegion,
        tolerance,
        TIME_LIMIT,
        xTest,
        FWVariant="LazyACG",
        typeStep=typeOfStep,
        criterion="PG",
        criterionRef=fValOpt,
    )

    # CG
    print("\nRunning CG.")
    resultsFW = runCG(
        x_0,
        S_0,
        alpha_0,
        fun,
        feasibleRegion,
        tolerance,
        TIME_LIMIT,
        xTest,
        FWVariant="CG",
        typeStep=typeOfStep,
        criterion="PG",
        criterionRef=fValOpt,
    )

    # ACG
    print("\nRunning ACG.")
    resultsAFW = runCG(
        x_0,
        S_0,
        alpha_0,
        fun,
        feasibleRegion,
        tolerance,
        TIME_LIMIT,
        xTest,
        FWVariant="ACG",
        typeStep=typeOfStep,
        criterion="PG",
        criterionRef=fValOpt,
    )

    # PCG
    print("\nRunning PCG.")
    resultsPFW = runCG(
        x_0,
        S_0,
        alpha_0,
        fun,
        feasibleRegion,
        tolerance,
        TIME_LIMIT,
        xTest,
        FWVariant="PCG",
        typeStep=typeOfStep,
        criterion="PG",
        criterionRef=fValOpt,
    )

    # Store all the results.
    results = [
        resultsSOCGS1,
        resultsSOCGS,
        resultsNCG,
        resultsAFWLazy,
        resultsFW,
        resultsAFW,
        resultsPFW,
    ]

    # Export results
    # Save the data from the run.
    from auxiliaryFunctions import export_results

    export_results(
        os.path.join(os.getcwd(), "Spectrahedron"),
        results,
        sys.argv,
        timestamp,
        fValOpt,
    )

    # Plot the results.
    from auxiliaryFunctions import plot_results

    plot_results(
        os.path.join(os.getcwd(), "Spectrahedron"),
        results,
        sys.argv,
        timestamp,
        fValOpt,
        save_images=True,
    )
