# Second-order Conditional Gradient Sliding

This is the code to reproduce all the experiments in our 
[paper](https://arxiv.org/pdf/2002.08907.pdf):

```
Second-order Conditional Gradient Sliding
```

## Abstract

Constrained second-order convex optimization algorithms are the method of choice when a high accuracy solution to a problem is needed, due to their local quadratic convergence. These algorithms require the solution of a constrained quadratic subproblem at every iteration. We present the Second-Order Conditional Gradient Sliding (SOCGS) algorithm, which uses a projection-free algorithm to solve the constrained quadratic subproblems inexactly. When the feasible region is a polytope the algorithm converges quadratically in primal gap after a finite number of linearly convergent iterations. Once in the quadratic regime the SOCGS algorithm requires <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(\log(\log 1/\varepsilon))"> first-order and Hessian oracle calls and <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(\log (1/\varepsilon) \log(\log1/\varepsilon))"> linear minimization oracle calls to achieve an <img src="https://render.githubusercontent.com/render/math?math=\varepsilon">-optimal solution. This algorithm is useful when the feasible region can only be accessed efficiently through a linear optimization oracle, and computing first-order information of the function, although possible, is costly. 

## Content

This repository contains the code to compare the performance of the **Second-order Conditional Gradient Sliding** ([SOCGS](https://arxiv.org/pdf/2002.08907.pdf)) **Conditional Gradients** (CG), **Away-step Conditional Gradients** ([ACG](http://www.iro.umontreal.ca/~marcotte/ARTIPS/1986_MP.pdf)), **Pairwise-step Conditional Gradients** ([PCG](https://arxiv.org/pdf/1511.05932.pdf)), **Lazy Away-step Conditional Gradients** ([Lazy-ACG](https://arxiv.org/pdf/1610.05120.pdf)), **Decomposition-invariant Conditional Gradients** ([DICG](https://arxiv.org/pdf/1605.06492.pdf)), **Conditional Gradient Sliding** ([CGS](http://www.optimization-online.org/DB_FILE/2014/10/4605.pdf)) and the **Newton Conditional Gradient** ([NCG](https://arxiv.org/pdf/2002.07003.pdf)) algorithms on the following instances:

1. Sparse coding over the Birkhoff polytope.
2. Inverse covariance estimation over spectrahedron.
3. Structured logistic regression over <img src="https://render.githubusercontent.com/render/math?math=\ell_1"> unit ball.

The code runs in Python (was tested on Windows). The files include the following:
* `algorithms.py` contains the implementations of the SOCGS, CG, ACG, PCG, Lazy ACG, DICG, CGS, and NCG algorithms.
* `functions.py` contains the objective functions being used in the numerical experiments, as well as the quadratic approximations to those functions.
* `feasibleRegions.py` contains the linear minimization oracles for the feasible regions used in the numerical experiments.
* `auxiliaryFunctions.py` contains miscelaneous functions used throughout the code.
* `feasibleRegions.py` contains the linear minimization oracles for the feasible regions used in the numerical experiments.
* `runExperimentsSOCGSBirkhoff.py` contains code to reproduce the **Sparse Coding over the Birkhoff polytope** experiment.
* `runExperimentsSOCGSGLasso.py` contains code to reproduce the **Inverse covariance estimation over spectrahedron** experiment.
* `runExperimentsSOCGSL1Ball.py` contains code to reproduce the **Structured logistic regression** experiment.


## Reproducing the results from the paper.

In order to reproduce the results from the paper:

### Sparse coding over the Birkhoff polytope with <img src="https://render.githubusercontent.com/render/math?math=m = 10,000"> (Figure 7)

```
python runExperimentsSOCGSBirkhoff.py --max_time 600 --type_solver DICG --num_samples 10000 --dimension 80 --accuracy 1.0e-5 --accuracy_Hessian 0.1
```

### Sparse coding over the Birkhoff polytope with <img src="https://render.githubusercontent.com/render/math?math=m = 100,000"> (Figure 8)

```
python runExperimentsSOCGSBirkhoff.py --max_time 600 --type_solver DICG --num_samples 100000 --dimension 80 --accuracy 1.0e-5 --accuracy_Hessian 0.1
```

### Inverse covariance estimation over spectrahedron with <img src="https://render.githubusercontent.com/render/math?math=n = 100"> (Figure 9)

```
python runExperimentsSOCGSGLasso.py --max_time 600 --type_solver PCG --dimension 100 --accuracy 1.0e-5 --lambda_value 0.05 --delta_value 1.0e-5 --max_iter 100
```

### Inverse covariance estimation over spectrahedron with <img src="https://render.githubusercontent.com/render/math?math=n = 50"> (Figure 10)

```
python runExperimentsSOCGSGLasso.py --max_time 600 --type_solver PCG --dimension 50 --accuracy 1.0e-5 --lambda_value 0.05 --delta_value 1.0e-5 --max_iter 100
```
### Structured logistic regression over <img src="https://render.githubusercontent.com/render/math?math=\ell_1"> unit ball with the *gisette* dataset (Figure 11)

```
python runExperimentsSOCGSL1Ball.py --max_time 3600 --type_solver ACG --dataset gisette --accuracy 1.0e-4 --lambda_value 0.05 --max_iter 100
```
### Structured logistic regression over <img src="https://render.githubusercontent.com/render/math?math=\ell_1"> unit ball with the *real-sim* dataset (Figure 12)

```
python runExperimentsSOCGSL1Ball.py --max_time 120 --type_solver PCG --dataset real-sim --accuracy 1.0e-4 --lambda_value 0.05 --max_iter 100
```

## Citation

Please use the following BibTeX entry to cite this software in your work:
    
    @article{carderera2020second,
  title={Second-order Conditional Gradient Sliding},
  author={Carderera, Alejandro and Pokutta, Sebastian},
  journal={arXiv preprint arXiv:2002.08907},
  year={2020}
}
    
## Authors

* [Alejandro Carderera](https://alejandro-carderera.github.io/)
* [Sebastian Pokutta](http://www.pokutta.com/)
