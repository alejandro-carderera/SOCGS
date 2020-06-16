import numpy as np
import time, os

ts = time.time()
from scipy.optimize import minimize_scalar
from sklearn.datasets import load_svmlight_file
import pickle
from functions import LogisticRegressionSparse, LogisticRegression, QuadApproxLogReg
import matplotlib.pyplot as plt

"""# Miscelaneous Functions

"""


def load_pickled_object(filepath):
    with open(filepath, "rb") as f:
        loaded_object = pickle.load(f)
    return loaded_object


def dump_pickled_object(filepath, target_object):
    with open(filepath, "wb") as f:
        pickle.dump(target_object, f)


def get_data(filepath):
    data = load_svmlight_file(filepath)
    return data[0], data[1]


import requests


def get_data_realsim(mu=0.0):
    file_directory = os.path.join(os.getcwd(), "Dataset")
    if not os.path.isfile(os.path.join(file_directory, "real-sim")):
        if not os.path.isfile(os.path.join(file_directory, "real-sim.bz2")):
            print("Downloading the dataset.")
            r = requests.get(
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
                allow_redirects=True,
            )
            open(os.path.join(file_directory, "real-sim.bz2"), "wb").write(r.content)
        print("Decompressing the dataset.")
        import bz2

        newfilepath = os.path.join(file_directory, "real-sim")
        with open(newfilepath, "wb") as new_file, bz2.BZ2File(
            os.path.join(file_directory, "real-sim.bz2"), "rb"
        ) as file:
            for data in iter(lambda: file.read(100 * 1024), b""):
                new_file.write(data)

    data = load_svmlight_file(os.path.join(file_directory, "real-sim"))
    (numSamples, dimension) = data[0].shape
    return (
        data[0],
        data[1],
        numSamples,
        dimension,
    )


def get_data_gisette(mu=0.0):
    # Download the samples
    gisette_train_data_path = os.path.join(os.getcwd(), "Dataset", "gisette_train.data")
    gisette_train_labels_path = os.path.join(
        os.getcwd(), "Dataset", "gisette_train.labels"
    )
    if not os.path.isfile(gisette_train_data_path) or not os.path.isfile(
        gisette_train_labels_path
    ):
        print("Downloading the labels and the data for the experiment.")
        # Save the data.
        r = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data",
            allow_redirects=True,
        )
        open(gisette_train_data_path, "wb").write(r.content)
        # Save the labels
        s = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels",
            allow_redirects=True,
        )
        open(gisette_train_labels_path, "wb").write(s.content)
    samples = np.loadtxt(gisette_train_data_path)
    labels = np.loadtxt(gisette_train_labels_path)
    (numSamples, dimension) = samples.shape
    return (
        samples,
        labels,
        numSamples,
        dimension,
    )


# Defines the type of maximum vertex dot product that we'll return.
def maxVertex(grad, activeVertex):
    # See which extreme point in the active set gives greater inner product.
    maxProd = np.dot(activeVertex[0], grad)
    maxInd = 0
    for i in range(len(activeVertex)):
        if np.dot(activeVertex[i], grad) > maxProd:
            maxProd = np.dot(activeVertex[i], grad)
            maxInd = i
    return activeVertex[maxInd], maxInd


# Finds the step with the maximum and minimum inner product.
def maxMinVertex(grad, activeVertex):
    # See which extreme point in the active set gives greater inner product.
    maxProd = np.dot(activeVertex[0], grad)
    minProd = np.dot(activeVertex[0], grad)
    maxInd = 0
    minInd = 0
    for i in range(len(activeVertex)):
        if np.dot(activeVertex[i], grad) > maxProd:
            maxProd = np.dot(activeVertex[i], grad)
            maxInd = i
        else:
            if np.dot(activeVertex[i], grad) < minProd:
                minProd = np.dot(activeVertex[i], grad)
                minInd = i
    return activeVertex[maxInd], maxInd, activeVertex[minInd], minInd


def newVertexFailFast(x, extremePoints):
    for i in range(len(extremePoints)):
        # Compare succesive indices.
        for j in range(len(extremePoints[i])):
            if extremePoints[i][j] != x[j]:
                break
        if j == len(extremePoints[i]) - 1:
            return False, i
    return True, np.nan


# Basis generator.
# Generates a set of n-orthonormal vectors.
def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = np.eye(dim - n + 1) - 2.0 * np.outer(x, x) / (x * x).sum()
        mat = np.eye(dim)
        mat[n - 1 :, n - 1 :] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


# Generate a random PSD quadratic with eigenvalues between certain numbers.
def randomPSDGenerator(dim, Mu, L):
    eigenval = np.zeros(dim)
    eigenval[0] = Mu
    eigenval[-1] = L
    eigenval[1:-1] = np.random.uniform(Mu, L, dim - 2)
    M = np.zeros((dim, dim))
    A = rvs(dim)
    for i in range(dim):
        M += eigenval[i] * np.outer(A[i], A[i])
    return M


# Random PSD matrix with a given sparsity.
def randomPSDGeneratorSparse(dim, sparsity):
    mask = np.random.rand(dim, dim) > (1 - sparsity)
    mat = np.random.normal(size=(dim, dim))
    Aux = np.multiply(mat, mask)
    return np.dot(Aux.T, Aux) + np.identity(dim)


def calculateEigenvalues(M):
    from scipy.linalg import eigvalsh

    dim = len(M)
    L = eigvalsh(M, eigvals=(dim - 1, dim - 1))[0]
    Mu = eigvalsh(M, eigvals=(0, 0))[0]
    return L, Mu


# Deletes the extremepoint from the representation.
def deleteVertexIndex(index, extremePoints, weights):
    del extremePoints[index]
    del weights[index]
    return


def performUpdate(function, x, gap, fVal, timing, gapVal):
    gap.append(gapVal)
    fVal.append(function.fEval(x))
    timing.append(time.time())
    return


# Pick a stepsize.
def stepSize(function, d, grad, x, typeStep="EL", maxStep=None):
    if typeStep == "SS":
        return -np.dot(grad, d) / (function.largestEig() * np.dot(d, d))
    else:
        if typeStep == "GS":
            options = {"xatol": 1e-08, "maxiter": 500000, "disp": 0}

            def InnerFunction(t):  # Hidden from outer code
                return function.fEval(x + t * d)

            if maxStep is None:
                res = minimize_scalar(
                    InnerFunction, bounds=(0, 1), method="bounded", options=options
                )
            else:
                res = minimize_scalar(
                    InnerFunction,
                    bounds=(0, maxStep),
                    method="bounded",
                    options=options,
                )
            return res.x
        else:
            if maxStep is None:
                return function.lineSearch(grad, d, x, maxStep=1.0)
            else:
                return function.lineSearch(grad, d, x, maxStep=maxStep)


def stepSizeDI(function, feasibleReg, it, d, grad, x, typeStep="EL"):
    return function.lineSearch(grad, d, x)


# Used in the DICG algorithm.
def calculateStepsize(x, d):
    assert not np.any(x < 0.0), "There is a negative coordinate."
    index = np.where(x == 0)[0]
    if np.any(d[index] < 0.0):
        return 0.0
    index = np.where(x > 0)[0]
    coeff = np.zeros(len(x))
    for i in index:
        if d[i] < 0.0:
            coeff[i] = -x[i] / d[i]
    val = coeff[coeff > 0]
    if len(val) == 0:
        return 0.0
    else:
        return min(val)


# Evaluate exit criterion. Evaluates to true if we must exit. Three posibilities:
# 1 - "PG": Evaluate primal gap.
# 2 - "DG": Evaluate dual gap.
# 3 - "IT": Evaluate number of iterations.
def exitCriterion(it, f, dualGap, criterion="PG", numCriterion=1.0e-3, critRef=0.0):
    if criterion == "DG":
        print("Wolfe-Gap: " + str(dualGap))
        return dualGap < numCriterion
    else:
        if criterion == "PG":
            print("Primal gap: " + str(f - critRef))
            return f - critRef < numCriterion
        else:
            return it >= numCriterion


# Once the problem has been solved to a high accuracy, solve the problem.
def exportsolution(filepath, formatString, fOpt, xOpt, tolerance, size):
    with open(filepath, "wb") as f:
        np.savetxt(f, [np.array(formatString)], fmt="%s", delimiter=",")
        np.savetxt(f, np.array([fOpt]), fmt="%.15f")
        np.savetxt(f, [xOpt.T], fmt="%.11f", delimiter=",")
        np.savetxt(f, np.array([tolerance]), fmt="%.15f")
        np.savetxt(f, np.array([size]), fmt="%.15f")
    return


# Once the problem has been solved to a high accuracy, solve the problem.
def importSolution(filepath):
    with open(filepath) as f:
        _ = f.readline()
        fOpt = float(f.readline().rstrip())
        xOpt = np.asarray(f.readline().rstrip().split(",")).astype(float)
        tolerance = float(f.readline().rstrip())
        size = int(float(f.readline().rstrip()))
    return fOpt, xOpt, tolerance, size


def export_results(filepath, results, arguments, timestamp, fValOpt):
    # Save the data from the run.
    if not os.path.exists(os.path.join(filepath, "Results")):
        os.makedirs(os.path.join(filepath, "Results"))
    with open(
        os.path.join(filepath, "Results", "SOCGS_" + str(timestamp) + ".txt"), "w"
    ) as f:
        f.write(str(arguments).replace("[", "").replace("]", "") + "\n")
        for i in range(len(results)):
            algType, x, FWGap, fVal, timing, distance, iteration = results[i]
            f.write(algType + "\n")
            f.write(
                str([x - fValOpt for x in fVal]).replace("[", "").replace("]", "")
                + "\n"
            )
            f.write(str(fVal).replace("[", "").replace("]", "") + "\n")
            f.write(str(FWGap).replace("[", "").replace("]", "") + "\n")
            f.write(str(timing).replace("[", "").replace("]", "") + "\n")
            f.write(str(distance).replace("[", "").replace("]", "") + "\n")
            f.write(str(iteration).replace("[", "").replace("]", "") + "\n")
    return


def plot_results(filepath, results, arguments, timestamp, fValOpt, save_images=True):
    # Plot the data from the run.
    if not os.path.exists(os.path.join(filepath, "Images")):
        os.makedirs(os.path.join(filepath, "Images"))
    # Plot Frank-Wolfe gap in terms of iteration.
    for i in range(len(results)):
        plt.semilogy(
            np.asarray(results[i][6], dtype=int), results[i][2], label=results[i][0]
        )
    plt.legend()
    plt.xlabel(r"$k$")
    plt.ylabel("Frank-Wolfe gap")
    plt.grid()
    plt.tight_layout()
    if save_images is False:
        plt.show()
    else:
        plt.savefig(
            os.path.join(
                filepath, "Images", "SOCGS_DG_Iteration_" + str(timestamp) + ".pdf"
            ),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()

    # Plot Frank-Wolfe gap in terms of time.
    for i in range(len(results)):
        plt.semilogy(results[i][4], results[i][2], label=results[i][0])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Frank-Wolfe gap")
    plt.grid()
    plt.tight_layout()
    if save_images is False:
        plt.show()
    else:
        plt.savefig(
            os.path.join(
                filepath, "Images", "SOCGS_DG_Time_" + str(timestamp) + ".pdf"
            ),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()

    # Plot primal gap in terms of iteration.
    for i in range(len(results)):
        plt.semilogy(
            np.asarray(results[i][6], dtype=int),
            [(x - fValOpt) for x in results[i][3]],
            label=results[i][0],
        )
    plt.legend()
    plt.xlabel(r"$k$")
    plt.ylabel("Primal gap")
    plt.grid()
    plt.tight_layout()
    if save_images is False:
        plt.show()
    else:
        plt.savefig(
            os.path.join(
                filepath, "Images", "SOCGS_PG_Iteration_" + str(timestamp) + ".pdf"
            ),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()

    # Plot primal gap in terms of time.
    for i in range(len(results)):
        plt.semilogy(
            results[i][4], [(x - fValOpt) for x in results[i][3]], label=results[i][0]
        )
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Primal gap")
    plt.grid()
    plt.tight_layout()
    if save_images is False:
        plt.show()
    else:
        plt.savefig(
            os.path.join(
                filepath, "Images", "SOCGS_PG_Time_" + str(timestamp) + ".pdf"
            ),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()

    # Plot distance in terms of iteration.
    for i in range(len(results)):
        plt.semilogy(
            np.asarray(results[i][6], dtype=int), results[i][5], label=results[i][0]
        )
    plt.legend()
    plt.xlabel(r"$k$")
    plt.ylabel("Distance to optimum")
    plt.grid()
    plt.tight_layout()
    if save_images is False:
        plt.show()
    else:
        plt.savefig(
            os.path.join(
                filepath,
                "Images",
                "SOCGS_Distance_Iteration_" + str(timestamp) + ".pdf",
            ),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()

    # Plot primal gap in terms of time.
    for i in range(len(results)):
        plt.semilogy(results[i][4], results[i][5], label=results[i][0])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance to optimum")
    plt.grid()
    plt.tight_layout()
    if save_images is False:
        plt.show()
    else:
        plt.savefig(
            os.path.join(
                filepath, "Images", "SOCGS_Distance_Time_" + str(timestamp) + ".pdf"
            ),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()

    return
