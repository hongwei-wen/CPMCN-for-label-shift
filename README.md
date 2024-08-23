# CPMCN-for-label-shift

This is the code for the paper: Class Probability Matching with Calibrated Networks for Label Shift Adaption. Paper link is https://openreview.net/forum?id=mliQ2huFrZ&noteId=fwyTilHYlH

(I) Main Empirical Comparison for CPMCN:
For comparing the empirical results of our CPMCN, please first enter the directory "Empirical_performance_comparison".
1. For CIFAR100 under tweak one shift or dirichlet shift:
Run "./notebook_CIFAR100_dirichlet/CIFAR100_exp1225run1.py" or "./notebook_CIFAR100_tweakone/CIFAR100_exp1225run1.py"

2. For CIFAR10 under tweak one shift or dirichlet shift:
Run "./notebook_CIFAR10_dirichlet/CIFAR10_exp1221run1.py" or "./notebook_CIFAR10_tweakone/CIFAR10_exp1221run1.py"

3. For MNIST under tweak one shift or dirichlet shift:
Run "./notebook_MNIST_dirichlet/MNIST_exp1225run1.py" or "./notebook_MNIST_tweakone/MNIST_exp1225run1.py"

We basically follow the code structure of the paper [1]. To be specific, we train the network on the first 50,000 training points of the source domain data. The training code and parameters can be found in the {\tt obtaining\_predictions} folder at https://github.com/kundajelab/labelshiftexperiments/tree/master/notebooks/obtaining_predictions. The MNIST, CIFAR10, and CIFAR100 folders correspond to the networks for the respective datasets. We use the codes and hyperparameter values from the above github repository to train the network. In addition, the calibration method Bias-Corrected Temperature Scaling (BCTS) is implemented based on the code in https://github.com/kundajelab/abstention/blob/master/abstention/calibration.py. It is invoked using the {\tt TempScaling(bias\_positions='all')} method. The last 10,000 data points in the training set are used as a validation set to train the calibration parameters of BCTS.

[1] Alexandari, Amr, Anshul Kundaje, and Avanti Shrikumar. "Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation." International Conference on Machine Learning. PMLR, 2020.

(II) Other_experiments_for_illustration:
1. For plotting the training curve, please run "./Other_experiments_for_illustration/training_curve/notebook_CIFAR100_tweakone/CIFAR100_exp1225run1.py".

2. To compare the computational complexity of CPMCN solved by the BFGS optimizer to FPM-based methods, please run  "./Other_experiments_for_illustration/runtime_of_solving_by_BFGS/notebook_CIFAR100_dirichlet/CIFAR100_exp1225run1.py".

