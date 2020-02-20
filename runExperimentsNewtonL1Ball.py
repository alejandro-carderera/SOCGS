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
    from algorithms import runFW, runProjectedNewton

    """
    ------------------------------Logistic Regression L1 Ball----------------------------
    """ 
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S').replace(' ', '-').replace(':', '-')
    
    from feasibleRegions import L1UnitBallPolytope
    from functions import LogisticRegression
    
    size = int(sys.argv[5])
    
    samples = np.loadtxt('gisette_train.data')
    labels = np.loadtxt('gisette_train.labels')
    samples = samples[:, 0:size]
    #Stack them up to get more samples
    (numSamples, dimension) = samples.shape
    
    print(samples.shape)
    
    fun = LogisticRegression(dimension, numSamples, samples, labels)
    
    #Time limit spent calculating the reference solution and running the algorithm.
    TIME_LIMIT_REFERENCE_SOL = int(2*int(sys.argv[1]))
    TIME_LIMIT = int(sys.argv[1])
    
    feasibleRegion = L1UnitBallPolytope(size, 1.0)
   
    #Initial starting point by calling the LPOracle.
    x_0 = feasibleRegion.initialPoint()
    S_0 = [x_0]
    alpha_0 = [1]
    
    tolerance = float(sys.argv[2])
    typeOfStep = sys.argv[3]
    numberHessianSamples = int(sys.argv[4])

    print("Solving the problem over the L1 Unit Ball polytope.")
    
    ##Run to a high Frank-Wolfe primal gap accuracy for later use?
    print("\nFinding optimal solution to high accuracy using Lazy AFW.")
    xTest, FWGapTest, fValTest, timingTest, activeSetTest, distTest = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance/2.0, TIME_LIMIT_REFERENCE_SOL, np.zeros(len(x_0)), FWVariant = "AFW", typeStep = typeOfStep, criterion = "DG")
    fValOpt = fValTest[-1]
    
    print("\nRunning FW Newton (LBFGS).")
    xFWNLBFGS3, FWGapFWNLBFGS3, fValFWNLBFGS3, timingFWNLBFGS3, activeSetFWNLBFGS3, distanceFWNLBFGS3 = runProjectedNewton(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, criterion = "PG", criterionRef = fValOpt,  Hessian = "LBFGS", ExactLinesearch = True, TypeSolver = "Lazy", forcingParam = 0.9, HessianParam = numberHessianSamples)

    print("\nRunning FW Newton (LBFGS).")
    xFWNLBFGS4, FWGapFWNLBFGS4, fValFWNLBFGS4, timingFWNLBFGS4, activeSetFWNLBFGS4, distanceFWNLBFGS4 = runProjectedNewton(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, criterion = "PG", criterionRef = fValOpt,  Hessian = "LBFGS", ExactLinesearch = True, TypeSolver = "Lazy", forcingParam = 0.8, HessianParam = numberHessianSamples)
    
    #Run Lazy AFW
    print("\nRunning Lazy AFW.")
    xAFWLazy, FWGapAFWLazy, fValAFWLazy, timingAFWLazy, activeSetLazy, distanceAFWLazy  = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, FWVariant = "Lazy", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)

    #Vanilla FW
    print("\nRunning FW.")
    xFW, FWGapFW, fValFW, timingFW, activeSetFW, distanceFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, FWVariant = "Vanilla", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)
    
    #Vanilla AFW
    print("\nRunning AFW.")
    xAFW, FWGapAFW, fValAFW, timingAFW, activeSetAFW, distanceAFW = runFW(x_0, S_0, alpha_0, fun, feasibleRegion, tolerance, TIME_LIMIT, xTest, FWVariant = "AFW", typeStep = typeOfStep, criterion = "PG", criterionRef = fValOpt)

    #Generate a timestamp for the example and save data
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S').replace(' ', '-').replace(':', '-')
    with open(os.path.join(os.getcwd(), "LogisticRegL1Unit_" + str(timestamp) + "_size" + str(dimension) + "_TypeStep_" + typeOfStep  + ".txt"), 'w') as f:
        f.write(str(sys.argv) + "\n")
        f.write("size:\t" + str(len(x_0)) + "\n")
        f.write("Tolerance:\t" + str(tolerance)+ "\n")
        f.write("Optimum:\t" + str(fValOpt)+ "\n")
        f.write("FW"+ "\n")
        f.write(str([x - fValOpt for x in fValFW]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceFW).replace("[", "").replace("]", "") + "\n")
        f.write("FW Newton LBFGS 0.9"+ "\n")
        f.write(str([x - fValOpt for x in fValFWNLBFGS3]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValFWNLBFGS3).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapFWNLBFGS3).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingFWNLBFGS3).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceFWNLBFGS3).replace("[", "").replace("]", "") + "\n")
        f.write("FW Newton LBFGS 0.8"+ "\n")
        f.write(str([x - fValOpt for x in fValFWNLBFGS4]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValFWNLBFGS4).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapFWNLBFGS4).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingFWNLBFGS4).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceFWNLBFGS4).replace("[", "").replace("]", "") + "\n")
        #Output the FW Gap.
        f.write("AFW"+ "\n")
        f.write(str([x - fValOpt for x in fValAFW]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValAFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapAFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingAFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(activeSetAFW).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceAFW).replace("[", "").replace("]", "") + "\n")
        #Output the FW Gap.
        f.write("AFW Lazy"+ "\n")
        f.write(str([x - fValOpt for x in fValAFWLazy]).replace("[", "").replace("]", "") + "\n")
        f.write(str(fValAFWLazy).replace("[", "").replace("]", "") + "\n")
        f.write(str(FWGapAFWLazy).replace("[", "").replace("]", "") + "\n")
        f.write(str(timingAFWLazy).replace("[", "").replace("]", "") + "\n")
        f.write(str(activeSetLazy).replace("[", "").replace("]", "") + "\n")
        f.write(str(distanceAFWLazy).replace("[", "").replace("]", "") + "\n")
    
    plt.loglog(np.arange(len(FWGapAFW)) + 1, FWGapAFW, '-', color = 'b',  label = 'AFW')
    plt.loglog(np.arange(len(FWGapFW)) + 1, FWGapFW, '-', color = 'y',  label = 'FW')
    plt.loglog(np.arange(len(FWGapFWNLBFGS3)) + 1, FWGapFWNLBFGS3, label = 'FW Newton LBFGS 0.9')
    plt.loglog(np.arange(len(FWGapFWNLBFGS4)) + 1, FWGapFWNLBFGS4, label = 'FW Newton LBFGS 0.8')
    plt.loglog(np.arange(len(FWGapAFWLazy)), FWGapAFWLazy, color = 'k', label = 'AFW (L)')
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$g(x_k)$')
    plt.tight_layout()
    plt.grid()
#    plt.show()
    plt.savefig(os.path.join(os.getcwd(),"NewtonFW_LogisticRegL1Unit__DG_Iteration_" + str(timestamp) + "_size" + str(dimension)+ "_tolerance" + str(tolerance) + "_numHessianSamples" + str(numberHessianSamples) + ".pdf"), format='pdf')
    plt.close()
    
    plt.semilogy(timingAFW, FWGapAFW, '-', color = 'b',  label = 'AFW')
    plt.semilogy(timingFW, FWGapFW, '-', color = 'y',  label = 'FW')
    plt.semilogy(timingFWNLBFGS3, FWGapFWNLBFGS3, label = 'FW Newton LBFGS 0.9')
    plt.semilogy(timingFWNLBFGS4, FWGapFWNLBFGS4, label = 'FW Newton LBFGS 0.8')
    plt.semilogy(timingAFWLazy, FWGapAFWLazy, color = 'k', label = 'AFW (L)')
    plt.legend()
    plt.xlabel(r'$Time$')
    plt.ylabel(r'$g(x_k)$')
    plt.tight_layout()
    plt.grid()
#    plt.show()
    plt.savefig(os.path.join(os.getcwd(),"NewtonFW_LogisticRegL1Unit_DG_Time_" + str(timestamp) + "_size" + str(dimension)+ "_tolerance" + str(tolerance) + "_numHessianSamples" + str(numberHessianSamples) + ".pdf"), format='pdf')
    plt.close()

    plt.loglog(np.arange(len(fValAFW)) + 1, distanceAFW, '-', color = 'b',  label = 'AFW')
    plt.loglog(np.arange(len(fValFW)) + 1, distanceFW, '-', color = 'y',  label = 'FW')
    plt.loglog(np.arange(len(fValFWNLBFGS3)) + 1, distanceFWNLBFGS3, label = 'FW Newton LBFGS 0.9')
    plt.loglog(np.arange(len(fValFWNLBFGS4)) + 1, distanceFWNLBFGS4, label = 'FW Newton LBFGS 0.8')
    plt.loglog(np.arange(len(fValAFWLazy)) + 1, distanceAFWLazy, color = 'k', label = 'AFW (L)')
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$||x_k - x^*||$')
    plt.tight_layout()
    plt.grid()
#    plt.show()
    plt.savefig(os.path.join(os.getcwd(),"NewtonFW_LogisticRegL1Unit_Distance_Iteration_" + str(timestamp) + "_size" + str(dimension)+ "_tolerance" + str(tolerance) + "_numHessianSamples" + str(numberHessianSamples) + ".pdf"), format='pdf')
    plt.close()
    
    plt.semilogy(timingAFW, distanceAFW, '-', color = 'b',  label = 'AFW')
    plt.semilogy(timingFW, distanceFW, '-', color = 'y',  label = 'FW')
    plt.semilogy(timingFWNLBFGS3, distanceFWNLBFGS3, label = 'FW Newton LBFGS 0.9')
    plt.semilogy(timingFWNLBFGS4, distanceFWNLBFGS4, label = 'FW Newton LBFGS 0.8')
    plt.semilogy(timingAFWLazy, distanceAFWLazy, color = 'k', label = 'AFW (L)')
    plt.legend()
    plt.xlabel(r'$Time$')
    plt.ylabel(r'$||x_k - x^*||$')
    plt.tight_layout()
    plt.grid()
#    plt.show()
    plt.savefig(os.path.join(os.getcwd(),"NewtonFW_LogisticRegL1Unit_Distance_Time_" + str(timestamp) + "_size" + str(dimension)+ "_tolerance" + str(tolerance) + "_numHessianSamples" + str(numberHessianSamples) + ".pdf"), format='pdf')
    plt.close()
    
    plt.loglog(np.arange(len(fValAFW)) + 1, [(x - fValOpt) for x in fValAFW], '-', color = 'b',  label = 'AFW')
    plt.loglog(np.arange(len(fValFW)) + 1, [(x - fValOpt) for x in fValFW], '-', color = 'y',  label = 'FW')
    plt.loglog(np.arange(len(fValFWNLBFGS3)) + 1, [(x - fValOpt) for x in fValFWNLBFGS3], label = 'FW Newton LBFGS 0.9')
    plt.loglog(np.arange(len(fValFWNLBFGS4)) + 1, [(x - fValOpt) for x in fValFWNLBFGS4], label = 'FW Newton LBFGS 0.8')
    plt.loglog(np.arange(len(fValAFWLazy)) + 1, [(x - fValOpt) for x in fValAFWLazy], color = 'k', label = 'AFW (L)')
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'$f(x_{k}) - f^{*}$')
    plt.tight_layout()
    plt.grid()
#    plt.show()
    plt.savefig(os.path.join(os.getcwd(),"NewtonFW_LogisticRegL1Unit_PG_Iteration_" + str(timestamp) + "_size" + str(dimension)+ "_tolerance" + str(tolerance) + "_numHessianSamples" + str(numberHessianSamples) + ".pdf"), format='pdf')
    plt.close()
    
    #Plot Primal gap in terms of time.
    plt.semilogy(timingAFW, [(x - fValOpt) for x in fValAFW], '-', color = 'b',  label = 'AFW')
    plt.semilogy(timingFW, [(x - fValOpt) for x in fValFW], '-', color = 'y',  label = 'FW')
    plt.semilogy(timingFWNLBFGS3, [(x - fValOpt) for x in fValFWNLBFGS3], label = 'FW Newton LBFGS 0.9')
    plt.semilogy(timingFWNLBFGS4, [(x - fValOpt) for x in fValFWNLBFGS4], label = 'FW Newton LBFGS 0.8')
    plt.semilogy(timingAFWLazy, [(x - fValOpt) for x in fValAFWLazy], color = 'k', label = 'AFW (L)')
    plt.legend()
    plt.ylabel(r'$f(x_{k}) - f^{*}$')
    plt.xlabel(r't[s]')
    plt.tight_layout()
    plt.grid()
#    plt.show()
    plt.savefig(os.path.join(os.getcwd(),"NewtonFW_LogisticRegL1Unit_PG_Time_" + str(timestamp) + "_size" + str(dimension)+ "_tolerance" + str(tolerance) + "_numHessianSamples" + str(numberHessianSamples) + ".pdf"), format='pdf')
    plt.close()