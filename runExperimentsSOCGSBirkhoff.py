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
    from algorithms import CGS, runCG, DIPFW, SOCGS, runSVRCG

    """
    ------------------------------Birkhoff Polytope experiment---------------------------
    """

    ts = time.time()
    timestamp = (
        datetime.datetime.fromtimestamp(ts)
        .strftime("%Y-%m-%d %H:%M:%S")
        .replace(" ", "-")
        .replace(":", "-")
    )

    from feasibleRegions import BirkhoffPolytope
    from functions import funcQuadraticCustom, QuadApprox

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
        "--num_samples",
        type=int,
        required=True,
        help="Number of samples to artificially generate.",
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
        "--accuracy_Hessian",
        type=float,
        required=True,
        help="Accuracy parameter for the Hessian.",
    )
    parser.add_argument(
        "--type_solver",
        type=str,
        required=True,
        help="CG subsolver to use in SOCGS: CG, ACG, PCG, LazyACG, DICG.",
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
    numSamples = args.num_samples
    sizeVectorY = args.dimension
    sizeVectorX = args.dimension
    tolerance = args.accuracy
    omega = args.accuracy_Hessian
    type_of_solver = args.type_solver
    if(args.known_primal_gap == 'True'):
        known_primal_gap = True
    else:
        if(args.known_primal_gap == 'False'):
            known_primal_gap = False
        else:
            assert False, 'Invalid known_primal_gap argument'

    # Generate a function where we know the matrix.
    AMat = np.random.normal(size=(sizeVectorY, sizeVectorX))
    fun = funcQuadraticCustom(numSamples, sizeVectorY, sizeVectorX, AMat)

    # Initialize the quadratic approximation function.
    funQuadApprox = QuadApprox()

    # Initialize the function that will return the feasible region oracles.
    size = int(sizeVectorY * sizeVectorX)
    feasibleRegion = BirkhoffPolytope(size)
    x_0 = feasibleRegion.initialPoint()
    S_0 = [x_0]
    alpha_0 = [1]

    typeOfStep = "EL"

    print("Solving the problem over the Birkhoff polytope.")

    if not os.path.exists(os.path.join(os.getcwd(), "Birkhoff")):
        os.makedirs(os.path.join(os.getcwd(), "Birkhoff"))

    ##Run to a high Frank-Wolfe primal gap accuracy for later use?
    from auxiliaryFunctions import exportsolution, dump_pickled_object

    print("\nFinding optimal solution to high accuracy using DIPFW.")
    (
        nameAlg,
        xTest,
        FWGapTest,
        fValTest,
        timingTest,
        distanceTest,
        iterationTest,
    ) = DIPFW(
        x_0,
        fun,
        feasibleRegion,
        tolerance / 2.0,
        TIME_LIMIT_REFERENCE_SOL,
        np.zeros(len(x_0)),
        criterion="DG",
    )
    fValOpt = fValTest[-1]
    tolerance = max(tolerance, min(np.asarray(FWGapTest)))
    if not os.path.exists(os.path.join(os.getcwd(), "Birkhoff", "Solutions")):
        os.makedirs(os.path.join(os.getcwd(), "Birkhoff", "Solutions"))
    # Saving solution.
    exportsolution(
        os.path.join(
            os.getcwd(),
            "Birkhoff",
            "Solutions",
            "Solution_Birkhoff_"
            + str(timestamp)
            + "_sizeVector"
            + str(sizeVectorY)
            + "_TypeStep_"
            + typeOfStep
            + ".txt",
        ),
        sys.argv,
        fValOpt,
        xTest,
        min(np.asarray(FWGapTest)),
        sizeVectorY,
    )
    dump_pickled_object(
        os.path.join(
            os.getcwd(),
            "Birkhoff",
            "Solutions",
            "function_" + str(timestamp) + ".pickle",
        ),
        fun,
    )

    #    #Importing solution
    #    from auxiliaryFunctions import importSolution, load_pickled_object
    #    fValOpt, xTest, importTolerance, sizeSol = importSolution(os.path.join(os.getcwd(), "Birkhoff", "Solution_Birkhoff_2020-06-01-16-13-27_sizeVector20_TypeStep_EL.txt"))
    #    tolerance = max(tolerance, importTolerance)
    #    fun = load_pickled_object(os.path.join(os.getcwd(), "Birkhoff", "function_2020-06-01-16-13-27.pickle"))

    # Create list to store all the results.
    results = []

    # Run the projected Newton method.
    print("\nRunning SOCGS.")
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
        known_primal_gap = known_primal_gap,
        omega=omega,
    )

    # CGS
    print("\nRunning CGS.")
    CSGAlg = CGS()
    resultsCGS = CSGAlg.run(
        x_0,
        fun,
        feasibleRegion,
        tolerance,
        TIME_LIMIT,
        xTest,
        criterion="PG",
        criterionRef=fValOpt,
    )

    # SVRFW
    print("\nRunning SVRCG.")
    resultsSVRCG = runSVRCG(
        x_0,
        fun,
        feasibleRegion,
        tolerance,
        TIME_LIMIT,
        xTest,
        criterion="PG",
        criterionRef=fValOpt,
    )

    # Decomposition Invariant CG
    print("\nRunning DICG.")
    resultsDICG = DIPFW(
        x_0,
        fun,
        feasibleRegion,
        tolerance,
        TIME_LIMIT,
        xTest,
        typeStep=typeOfStep,
        criterion="PG",
        criterionRef=fValOpt,
    )

    #  Lazy AFW
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

    # Vanilla FW
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
        resultsSOCGS,
        resultsCGS,
        resultsSVRCG,
        resultsDICG,
        resultsAFWLazy,
        resultsFW,
        resultsAFW,
        resultsPFW,
    ]

    # Export results
    # Save the data from the run.
    from auxiliaryFunctions import export_results

    export_results(
        os.path.join(os.getcwd(), "Birkhoff"), results, sys.argv, timestamp, fValOpt
    )

    # Plot the results.
    from auxiliaryFunctions import plot_results

    plot_results(
        os.path.join(os.getcwd(), "Birkhoff"),
        results,
        sys.argv,
        timestamp,
        fValOpt,
        save_images=True,
    )
