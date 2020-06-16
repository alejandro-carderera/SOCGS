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
    from algorithms import runCG, SOCGS, NCG
    from auxiliaryFunctions import (
        exportsolution,
        importSolution,
        get_data_realsim,
        get_data_gisette,
    )
    from functions import LogisticRegressionSparse, LogisticRegression, QuadApproxLogReg

    """
    ------------------------------Logistic Regression L1 Ball----------------------------
    """
    ts = time.time()
    timestamp = (
        datetime.datetime.fromtimestamp(ts)
        .strftime("%Y-%m-%d %H:%M:%S")
        .replace(" ", "-")
        .replace(":", "-")
    )

    from feasibleRegions import L1UnitBallPolytope

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
        "--dataset",
        type=str,
        required=True,
        help="Dataset that will be used. Either gisette or real-sim.",
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

    args = parser.parse_args()
    dataset = args.dataset
    TIME_LIMIT = args.max_time
    TIME_LIMIT_REFERENCE_SOL = int(2.0 * args.max_time)
    tolerance = args.accuracy
    lambdaVal = args.lambda_value
    maxIter = args.max_iter
    type_of_solver = args.type_solver

    if not os.path.exists(os.path.join(os.getcwd(), "Dataset")):
        os.makedirs(os.path.join(os.getcwd(), "Dataset"))
    if dataset == "gisette":
        samples, labels, numSamples, dimension = get_data_gisette(mu=lambdaVal)
        fun = LogisticRegression(dimension, numSamples, samples, labels, mu=lambdaVal)
        funQuadApprox = QuadApproxLogReg(
            dimension, numSamples, samples, labels, mu=lambdaVal
        )
    else:
        samples, labels, numSamples, dimension = get_data_realsim(mu=lambdaVal)
        fun = LogisticRegressionSparse(
            dimension, numSamples, samples, labels, mu=lambdaVal
        )
        funQuadApprox = QuadApproxLogReg(
            dimension, numSamples, samples, labels, mu=lambdaVal
        )

    # Initialize the feasible region.
    feasibleRegion = L1UnitBallPolytope(dimension, 1.0)
    typeOfStep = "EL"

    # Initial starting point by calling the LPOracle.
    x_0 = feasibleRegion.initialPoint()
    S_0 = [x_0]
    alpha_0 = [1]

    print("Solving the problem over the l1Ball polytope.")
    if not os.path.exists(os.path.join(os.getcwd(), "l1Ball")):
        os.makedirs(os.path.join(os.getcwd(), "l1Ball"))

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

    if not os.path.exists(os.path.join(os.getcwd(), "l1Ball", "Solutions")):
        os.makedirs(os.path.join(os.getcwd(), "l1Ball", "Solutions"))
    # Saving solution.
    exportsolution(
        os.path.join(
            os.getcwd(), "l1Ball", "Solutions", "Solution_" + str(timestamp) + ".txt"
        ),
        sys.argv,
        fValOpt,
        xTest,
        min(np.asarray(FWGapTest)),
        dimension,
    )

    #    #Importing solution
    #    fValOpt, xTest, importTolerance, sizeSol = importSolution(os.path.join(os.getcwd(), "LogReg", "Solution_2020-06-03-16-55-30_size20958_TypeStep_EL_Mu_0.05.txt"))
    #    tolerance = max(tolerance, importTolerance)

    # Create list to store all the results.
    results = []

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

    # Run SOCGS
    print("\nSOCGS.")
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

    # Store all the results.
    results = [
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
        os.path.join(os.getcwd(), "l1Ball"), results, sys.argv, timestamp, fValOpt
    )

    # Plot the results.
    from auxiliaryFunctions import plot_results

    plot_results(
        os.path.join(os.getcwd(), "l1Ball"),
        results,
        sys.argv,
        timestamp,
        fValOpt,
        save_images=True,
    )
