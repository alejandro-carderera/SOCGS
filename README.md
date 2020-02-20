# Second-order-Conditional-Gradients
Second-order Conditional Gradients Code

#Download the Gisette dataset for the L1 ball experiment at:
https://archive.ics.uci.edu/ml/datasets/Gisette

#To run the Birkhoff experiment:
python runExperimentsNewtonBirkhoff.py 1600 100000 80 1.0e-5 5

#To run the Graphical-Lasso Experiment:
python runExperimentsNewtonGLasso.py 50 600 1.0e-7 GS 10

#To run the L1 ball experiment:
python runExperimentsNewtonL1Ball.py 600 1.0e-7 GS 15 5000
