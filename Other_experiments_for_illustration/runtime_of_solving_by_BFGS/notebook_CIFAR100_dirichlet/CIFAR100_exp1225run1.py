import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("../abstentionmaster")
sys.path.append("..")

from importlib import reload
import abstention
reload(abstention)
reload(abstention.calibration)
reload(abstention.label_shift)
reload(abstention.figure_making_utils)
from abstention.calibration import (TempScaling, VectorScaling,
                                    NoBiasVectorScaling, softmax)
from abstention.label_shift import (
    WenImbalanceAdapterRun,
    EMImbalanceAdapter,
    BBSEImbalanceAdapter,
    RLLSImbalanceAdapter,
)
import glob
import gzip
import numpy as np
from collections import defaultdict, OrderedDict

import labelshiftexperiments
reload(labelshiftexperiments)
reload(labelshiftexperiments.cifarandmnist)
from labelshiftexperiments import cifarandmnist
# import cifarandmnist
# reload(cifarandmnist)

test_labels = cifarandmnist.read_preds(open("am_cifar100_test_labels.txt"))
valid_labels = cifarandmnist.read_preds(open("am_cifar100_valid_labels.txt"))

imbalanceadaptername_to_imbalanceadapter = {
    'wenRun':WenImbalanceAdapterRun(),
    'em': EMImbalanceAdapter(),
    #"wen15B": WenImbalanceAdapter15B(),
    'bbse-soft': BBSEImbalanceAdapter(soft=True),
    'rlls-soft': RLLSImbalanceAdapter(soft=True),
}

calibname_to_calibfactory = OrderedDict([
    ('None', abstention.calibration.Softmax()),
    ('TS', TempScaling(verbose=False)),
    ('NBVS', NoBiasVectorScaling(verbose=False)),
    ('BCTS', TempScaling(verbose=False, bias_positions='all')),
    ('VS', VectorScaling(verbose=False))
])

adaptncalib_pairs = [
('em', 'BCTS'),
('wenRun', 'BCTS'),
('bbse-soft', 'BCTS'),
('rlls-soft', 'BCTS')
]

num_trials = 10
seeds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
dirichlet_alphas_and_samplesize = [
(0.1, 2000),
]

print("Dirichlet shift")

(dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
 dirichlet_alpha_to_samplesize_to_baselineacc,
 metric_to_samplesize_to_calibname_to_unshiftedvals) =\
  cifarandmnist.run_experiments(
    num_trials=num_trials,
    seeds=seeds,
    alphas_and_samplesize = dirichlet_alphas_and_samplesize,
    shifttype='dirichlet',
    calibname_to_calibfactory=calibname_to_calibfactory,
    imbalanceadaptername_to_imbalanceadapter=
      imbalanceadaptername_to_imbalanceadapter,
    adaptncalib_pairs=adaptncalib_pairs,
    validglobprefix="cifar100_validpreacts_seed",
    testglobprefix="cifar100_testpreacts_seed",
    valid_labels=valid_labels,
    test_labels=test_labels)

