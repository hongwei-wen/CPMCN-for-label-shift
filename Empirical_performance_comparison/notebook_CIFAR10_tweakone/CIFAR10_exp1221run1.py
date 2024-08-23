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
    EMImbalanceAdapter,
    WenImbalanceAdapter15B,
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

test_labels = cifarandmnist.read_labels(open("test_labels.txt"))
valid_labels = cifarandmnist.read_labels(open("valid_labels.txt"))

imbalanceadaptername_to_imbalanceadapter = {
    'em': EMImbalanceAdapter(),
    "wen15B": WenImbalanceAdapter15B(),
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
    ('bbse-soft', 'BCTS'),
    #('rlls-soft', 'BCTS'),
    ('wen15B', 'BCTS'),
]

num_trials = 10
seeds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
tweakone_alphas_and_samplesize = [
    (0.01, 2000),
    (0.9, 2000),
    (0.02,2000),
    (0.8, 2000),
    (0.05,2000),
    (0.6, 2000),
]

print("Tweak one shift")

(tweakone_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
 tweakone_alpha_to_samplesize_to_baselineacc,
 _) = cifarandmnist.run_experiments(
     num_trials=num_trials,
     seeds=seeds,
     alphas_and_samplesize=tweakone_alphas_and_samplesize,
     shifttype='tweakone',
     calibname_to_calibfactory=calibname_to_calibfactory,
     imbalanceadaptername_to_imbalanceadapter=
     imbalanceadaptername_to_imbalanceadapter,
     adaptncalib_pairs=adaptncalib_pairs,
     validglobprefix="validpreacts_model_cifar10_balanced_seed-",
     testglobprefix="testpreacts_model_cifar10_balanced_seed-",
     valid_labels=valid_labels,
     test_labels=test_labels)
