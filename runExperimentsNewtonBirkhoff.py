if __name__== "__main__":
    
    import os
    #Computing parameters.
    os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
    
    #General imports
    import numpy as np
    import os, sys
    import time
    import datetime
    import matplotlib.pyplot as plt
    from algorithms import CGS, runFW, DIPFW, runProjectedNewton, SVRFW

    """
    ------------------------------Birkhoff POLYTOPE----------------------------
    """ 
    
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S').replace(' ', '-').replace(':', '-')
    
    from feasibleRegions import BirkhoffPolytope
    from functions import funcQuadraticCustomv2
    
    #Time limit spent calculating the reference solution and running the algorithm.
    TIME_LIMIT = int(sys.argv[1])
    TIME_LIMIT_REFERENCE_SOL = int(2*TIME_LIMIT)
    
    
    #Function parameters.
    numSamples = int(sys.argv[2])
    sizeVectorY = int(sys.argv[3])
    sizeVectorX = int(sys.argv[3])
    
    #Generate a function where we know the matrix.
    AMat = np.random.normal(size = (sizeVectorY, sizeVectorX))
    fun = funcQuadraticCustomv2(numSamples, sizeVectorY, sizeVectorX, AMat)
    #Print out eigenvalues of the Hessian.
    print("Smoothness parameter: " + str(fun.largestEig()))
    print("Strong convexity parameter: " + str(fun.smallestEig()))
    
    
    size = int(sizeVectorY*sizeVectorX)
    
    feasibleRegion = BirkhoffPolytope(size)
    x_0 = feasibleRegion.initialPoint()
    S_0 = [x_0]
    alpha_0 = [1]
    tolerance = float(sys.argv[4])
    typeOfStep = "EL"
    
    numberSamplesHessian = int(sys.argv[5])

    print("Solving the problem over the Birkhoff polytope.")
    ##Run to a high Frank-Wolfe primal gap accuracy for later use?
    print("\nFinding optimal solution to high accuracy using DIPFW.")
    xTest, FWGapTest, fValTest, timingTest, distanceTest = DIPFW(x_0, fun, feasibleRegion, tolerance/2.0, TIME_LIMIT_REFERENCE_SOL, np.zeros(len(x_0)), criterion = "DG")
    fValOpt = fValTest[-1]
    tolerance = min(np.asarray(FWGapTest))
 
    #Decomposition CGS
    print("\nRunning CGS.")
    CSGAlg = CGS()
    xCGS, FWGapCGS, fValCGS, timingCGS, iterationCGS, distanceCGS = CSGAlg.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, criterion = "PG", criterionRef = fValOpt)

    #Decomposition SVRFW
    print("\nRunning SVRFW.")
    StochasticVarRedFW = SVRFW()
    xSVRFW, FWGapSVRFW, fValSVRFW, timingSVRFW, distanceSVRFW = StochasticVarRedFW.run(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, criterion = "PG", criterionRef = fValOpt)
    
    #Run the projected Newton method.
    print("\nRunning FW Newton (LBFGS).")
    xFWNLBFGS, FWGapFWNLBFGS, fValFWNLBFGS, timingFWNLBFGS, activeSetFWNLBFGS, distanceFWNLBFGS = runProjectedNewton(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, criterion = "PG", criterionRef = fValOpt,  Hessian = "LBFGS", ExactLinesearch = True, TypeSolver = "DICG",  HessianParam = numberSamplesHessian)

    #Run the projected Newton method.
    print("\nRunning FW Newton (BFGS).")
    xFWNBFGS, FWGapFWNBFGS, fValFWNBFGS, timingFWNBFGS, activeSetFWNBFGS, distanceFWNBFGS = runProjectedNewton(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, criterion = "PG", criterionRef = fValOpt,  Hessian = "BFGS", ExactLinesearch = True, TypeSolver = "DICG")

    #Run the projected Newton method.
    print("\nRunning FW Newton (Exact Hessian).")
    xFWN, FWGapFWN, fValFWN, timingFWN, activeSetFWN, distanceFWN = runProjectedNewton(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, criterion = "PG", criterionRef = fValOpt,  Hessian = "Exact", ExactLinesearch = True, TypeSolver = "DICG")

    #Decomposition Invariant CG
    print("\nRunning DICG.")
    xDICG, FWGapDICG, fValDICG, timingDICG, distanceDICG = DIPFW(x_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)

    #Run Lazy AFW
    print("\nRunning Lazy AFW.")
    xAFWLazy, FWGapAFWLazy, fValAFWLazy, timingAFWLazy, activeSetLazy, distanceAFWLazy  = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, FWVariant = "Lazy", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)

    #Vanilla FW
    print("\nRunning FW.")
    xFW, FWGapFW, fValFW, timingFW, activeSetFW, distanceFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, FWVariant = "Vanilla", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Vanilla AFW
    print("\nRunning AFW.")
    xAFW, FWGapAFW, fValAFW, timingAFW, activeSetAFW, distanceAFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, FWVariant = "AFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)

    #Vanilla PFW
    print("\nRunning PFW.")
    xPFW, FWGapPFW, fValPFW, timingPFW, activeSetPFW, distancePFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, FWVariant = "PFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)


    #Generate a timestamp for the example and save data
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S').replace(' ', '-').replace(':', '-')
    with open(os.path.join(os.getcwd(), "BirkhoffNewton_" + str(timestamp) + "_numSamples" + str(numSamples) + "_sizeY" + str(sizeVectorY)+ "_sizeX" + str(sizeVectorX)  + "_TypeStep_" + typeOfStep  + ".txt"), 'w') as f:
        f.write("size:\t" + str(len(x_0)) + "\n")
        f.write("Mu:\t" + str(fun.smallestEig())+ "\n")
        f.write("LVal:\t" + str(fun.largestEig())+ "\n")
        f.write("Tolerance:\t" + str(tolerance)+ "\n")
        f.write("Optimum:\t" + str(fValOpt)+ "\n")
        f.write("FW"+ "\n")
        f.write(str([x - fValOpt for x in fValFW]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceFW).replace("[", "").replace("]", "") + "\n")
        f.write("FW Newton"+ "\n")
        f.write(str([x - fValOpt for x in fValFWN]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValFWN).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapFWN).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingFWN).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceFWN).replace("[", "").replace("]", "") + "\n")
        f.write("FW Newton LBFGS"+ "\n")
        f.write(str([x - fValOpt for x in fValFWNLBFGS]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValFWNLBFGS).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapFWNLBFGS).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingFWNLBFGS).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceFWNLBFGS).replace("[", "").replace("]", "") + "\n")
        f.write("FW Newton BFGS"+ "\n")
        f.write(str([x - fValOpt for x in fValFWNBFGS]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValFWNBFGS).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapFWNBFGS).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingFWNBFGS).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceFWNBFGS).replace("[", "").replace("]", "") + "\n")
        #Output the FW Gap.
        f.write("AFW"+ "\n")
        f.write(str([x - fValOpt for x in fValAFW]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValAFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapAFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingAFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceAFW).replace("[", "").replace("]", "") + "\n")
        #Output the PFW Gap.
        f.write("PFW"+ "\n")
        f.write(str([x - fValOpt for x in fValPFW]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValPFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapPFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingPFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(distancePFW).replace("[", "").replace("]", "") + "\n")
        #Output the FW Gap.
        f.write("DICG"+ "\n")
        f.write(str([x - fValOpt for x in fValDICG]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValDICG).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapDICG).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingDICG).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceDICG).replace("[", "").replace("]", "") + "\n")
        #Output the FW Gap.
        f.write("AFW Lazy"+ "\n")
        f.write(str([x - fValOpt for x in fValAFWLazy]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValAFWLazy).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapAFWLazy).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingAFWLazy).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceAFWLazy).replace("[", "").replace("]", "") + "\n")
        #Output the FW Gap.
        f.write("CGS"+ "\n")
        f.write(str([x - fValOpt for x in fValCGS]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValCGS).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapCGS).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingCGS).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceCGS).replace("[", "").replace("]", "") + "\n")
        f.write(str(iterationCGS).replace("[", "").replace("]", "") + "\n")
        #Output the FW Gap.
        f.write("SVRFW"+ "\n")
        f.write(str([x - fValOpt for x in fValSVRFW]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValSVRFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapSVRFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingSVRFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceSVRFW).replace("[", "").replace("]", "") + "\n")
        
        
    plt.loglog(np.arange(len(FWGapAFW)) + 1, FWGapAFW, '-', color = 'b',  label = 'AFW')
    plt.loglog(np.arange(len(FWGapPFW)) + 1, FWGapPFW,  label = 'PFW')
    plt.loglog(np.arange(len(FWGapFW)) + 1, FWGapFW, '-', color = 'y',  label = 'FW')
    plt.loglog(np.arange(len(FWGapFWN) - 1) + 1, FWGapFWN[:-1], '-', color = 'c', label = 'FW Newton')
    plt.loglog(np.arange(len(FWGapFWNLBFGS)) + 1, FWGapFWNLBFGS, '--', color = 'c', label = 'FW Newton LBFGS')
    plt.loglog(np.arange(len(FWGapFWNBFGS)) + 1, FWGapFWNBFGS, ':', color = 'c', label = 'FW Newton BFGS')
    plt.loglog(np.arange(len(FWGapFWN)) + 1, FWGapFWN, '-', color = 'c', label = 'FW Newton Exact')
    plt.loglog(np.arange(len(FWGapTest) - 1) + 1, FWGapTest[:-1], '-', color = 'k', label = 'Test')
    plt.loglog(np.arange(len(FWGapAFWLazy)), FWGapAFWLazy, color = 'k', label = 'AFW (L)')
    plt.loglog(np.arange(len(FWGapDICG)), FWGapDICG, color = 'r', label = 'DICG')

    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$g(x_k)$')
    plt.grid()
#    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "NewtonFW_Birkhoff_DG_Iteration_" + str(timestamp) + "_numSamples" + str(numSamples) + "_sizeY" + str(sizeVectorY)+ "_sizeX" + str(sizeVectorX) + "_tolerance" + str(tolerance) + ".pdf"), format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    plt.semilogy(timingAFW, FWGapAFW, '-', color = 'b',  label = 'AFW')
    plt.semilogy(timingPFW, FWGapPFW,  label = 'PFW')
    plt.semilogy(timingFW, FWGapFW, '-', color = 'y',  label = 'FW')
    plt.semilogy(timingFWNLBFGS, FWGapFWNLBFGS, '--', color = 'c', label = 'FW Newton LBFGS')
    plt.semilogy(timingFWNBFGS, FWGapFWNBFGS, ':', color = 'c', label = 'FW Newton BFGS')
    plt.semilogy(timingFWN, FWGapFWN, '-', color = 'c', label = 'FW Newton Exact')
    plt.semilogy(timingAFWLazy, FWGapAFWLazy, color = 'k', label = 'AFW (L)')
    plt.semilogy(timingDICG, FWGapDICG, color = 'r', label = 'DICG')
    plt.legend()
    plt.xlabel(r'$Time$')
    plt.ylabel(r'$g(x_k)$')
    plt.grid()
#    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "NewtonFW_Birkhoff_DG_Time_" + str(timestamp) + "_numSamples" + str(numSamples) + "_sizeY" + str(sizeVectorY)+ "_sizeX" + str(sizeVectorX) + "_tolerance" + str(tolerance) + ".pdf"), format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.loglog(np.arange(len(fValAFW)) + 1, distanceAFW, '-', color = 'b',  label = 'AFW')
    plt.loglog(np.arange(len(fValPFW)) + 1, distancePFW, label = 'PFW')
    plt.loglog(np.arange(len(fValFW)) + 1, distanceFW, '-', color = 'y',  label = 'FW')
    plt.loglog(np.arange(len(fValFWNLBFGS)) + 1, distanceFWNLBFGS, '--', color = 'c', label = 'FW Newton LBFGS')
    plt.loglog(np.arange(len(fValFWNBFGS)) + 1, distanceFWNBFGS, ':', color = 'c', label = 'FW Newton BFGS')
    plt.loglog(np.arange(len(fValFWN)) + 1, distanceFWN, '-', color = 'c', label = 'FW Newton Exact')
    plt.loglog(np.arange(len(fValAFWLazy)) + 1, distanceAFWLazy, color = 'k', label = 'AFW (L)')
    plt.loglog(np.arange(len(fValDICG)) + 1, distanceDICG, color = 'r', label = 'DICG')
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$||x_k - x^*||$')
    plt.grid()
#    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "NewtonFW_Birkhoff_Distance_Iteration_" + str(timestamp) + "_numSamples" + str(numSamples) + "_sizeY" + str(sizeVectorY)+ "_sizeX" + str(sizeVectorX) + "_tolerance" + str(tolerance) + ".pdf"), format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    plt.semilogy(timingAFW, distanceAFW, '-', color = 'b',  label = 'AFW')
    plt.semilogy(timingPFW, distancePFW,  label = 'PFW')
    plt.semilogy(timingFW, distanceFW, '-', color = 'y',  label = 'FW')
    plt.semilogy(timingFWNLBFGS, distanceFWNLBFGS, '--', color = 'c', label = 'FW Newton LBFGS')
    plt.semilogy(timingFWNBFGS, distanceFWNBFGS, ':', color = 'c', label = 'FW Newton BFGS')
    plt.semilogy(timingFWN, distanceFWN, '-', color = 'c', label = 'FW Newton Exact')
    plt.semilogy(timingAFWLazy, distanceAFWLazy, color = 'k', label = 'AFW (L)')
    plt.semilogy(timingDICG, distanceDICG, color = 'r', label = 'DICG')
    plt.legend()
    plt.xlabel(r'$Time$')
    plt.ylabel(r'$||x_k - x^*||$')
    plt.grid()
#    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "NewtonFW_Birkhoff_Distance_Time_" + str(timestamp) + "_numSamples" + str(numSamples) + "_sizeY" + str(sizeVectorY)+ "_sizeX" + str(sizeVectorX) + "_tolerance" + str(tolerance) + ".pdf"), format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    
    plt.loglog(np.arange(len(fValAFW)) + 1, [(x - fValOpt) for x in fValAFW], '-', color = 'b',  label = 'AFW')
    plt.loglog(np.arange(len(fValPFW)) + 1, [(x - fValOpt) for x in fValPFW],  label = 'PFW')
    plt.loglog(np.arange(len(fValFW)) + 1, [(x - fValOpt) for x in fValFW], '-', color = 'y',  label = 'FW')
    plt.loglog(np.arange(len(fValFWNLBFGS)) + 1, [(x - fValOpt) for x in fValFWNLBFGS], '--', color = 'c', label = 'FW Newton LBFGS')
    plt.loglog(np.arange(len(fValFWNBFGS)) + 1, [(x - fValOpt) for x in fValFWNBFGS], ':', color = 'c', label = 'FW Newton BFGS')
    plt.loglog(np.arange(len(fValFWN)) + 1, [(x - fValOpt) for x in fValFWN], '-', color = 'c', label = 'FW Newton Exact')
    plt.loglog(np.arange(len(fValAFWLazy)) + 1, [(x - fValOpt) for x in fValAFWLazy], color = 'k', label = 'AFW (L)')
    plt.loglog(np.arange(len(fValDICG)) + 1, [(x - fValOpt) for x in fValDICG], color = 'r', label = 'DICG')
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$f(x_{k}) - f^{*}$')
    plt.grid()
#    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(),"NewtonFW_Birkhoff_PG_Iteration_" + str(timestamp) + "_numSamples" + str(numSamples) + "_sizeY" + str(sizeVectorY)+ "_sizeX" + str(sizeVectorX) + "_tolerance" + str(tolerance) + ".pdf"), format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    #Plot Primal gap in terms of time.
    plt.semilogy(timingAFW, [(x - fValOpt) for x in fValAFW], '-', color = 'b',  label = 'AFW')
    plt.semilogy(timingPFW, [(x - fValOpt) for x in fValPFW],  label = 'PFW')
    plt.semilogy(timingFW, [(x - fValOpt) for x in fValFW], '-', color = 'y',  label = 'FW')
    plt.semilogy(timingFWNLBFGS, [(x - fValOpt) for x in fValFWNLBFGS], '--', color = 'c', label = 'FW Newton LBFGS')
    plt.semilogy(timingFWNBFGS, [(x - fValOpt) for x in fValFWNBFGS], ':', color = 'c', label = 'FW Newton BFGS')
    plt.semilogy(timingFWN, [(x - fValOpt) for x in fValFWN], '-', color = 'c', label = 'FW Newton Exact')
    plt.semilogy(timingAFWLazy, [(x - fValOpt) for x in fValAFWLazy], color = 'k', label = 'AFW (L)')
    plt.semilogy(timingDICG, [(x - fValOpt) for x in fValDICG], color = 'r', label = 'DICG')
    plt.legend()
    plt.ylabel(r'$f(x_{k}) - f^{*}$')
    plt.xlabel(r't[s]')
    plt.grid()
#    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "NewtonFW_Birkhoff_PG_Time_" + str(timestamp) + "_numSamples" + str(numSamples) + "_sizeY" + str(sizeVectorY)+ "_sizeX" + str(sizeVectorX) + "_tolerance" + str(tolerance) + ".pdf"), format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()