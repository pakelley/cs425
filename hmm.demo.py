import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from scipy import stats

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from hmmlearn.hmm import GaussianHMM, GMMHMM

from parse_csv import parse_s1_csv
#############################################################
########################## Globals ##########################
#############################################################
# DATA_CSV = "../AllFromMarchDataPatrick.csv"
DATA_CSV = "../Data_Normal_4_3_17.csv"

N_COMPONENTS = 100

roll_classes = ['ramp_up', 'ramp_down', 'slow_roll', 'fast_roll']

#############################################################
########################### Main ############################
#############################################################
# Read in data
print "Parsing Data..."
globs = parse_s1_csv(DATA_CSV, "normal")
X = np.array([glob['samples'] for glob in globs])
y = np.array([glob['roll_class'] for glob in globs])

# Split data
X_train = X[:,:int( 0.7*len(X[0]) )]
y_train = y[:,:int( 0.7*len(X[0]) )]
X_test = X[:,int( 0.7*len(y[0]) ):]
y_test = y[:,int( 0.7*len(y[0]) ):]
# X_train = X
# y_train = y
# X_test = X
# y_test = y

# Train
ruhmms = []
rdhmms = []
srhmms = []
frhmms = []
scores = np.zeros((4, 4, y_test.shape[1]))
for i in range(4):
    print "#############################################"
    print ("TRAINING ON SENSOR %d" % i)
    print "#############################################"
    print "Training Ramp-Up Model"
    ru_mask = (y_train[i] == 'ramp_up')
    ruhmms.append( GMMHMM(n_components=N_COMPONENTS,
                              covariance_type="diag",
                              n_iter=1000).fit(X_train[i][ru_mask]) )
    print "Training Ramp-Down Model"
    rd_mask = (y_train[i] == 'ramp_down')
    rdhmms.append( GMMHMM(n_components=N_COMPONENTS,
                              covariance_type="diag",
                              n_iter=1000).fit(X_train[i][rd_mask]) )
    print "Training Slow-Roll Model"
    sr_mask = (y_train[i] == 'slow_roll')
    srhmms.append( GMMHMM(n_components=N_COMPONENTS,
                              covariance_type="diag",
                              n_iter=1000).fit(X_train[i][sr_mask]) )
    print "Training Fast-Roll Model"
    fr_mask = (y_train[i] == 'fast_roll')
    frhmms.append( GMMHMM(n_components=N_COMPONENTS,
                              covariance_type="diag",
                              n_iter=1000).fit(X_train[i][fr_mask]) )
    scores[i][0] = [ruhmms[-1].score(x) for x in X_test[i]]
    scores[i][1] = [rdhmms[-1].score(x) for x in X_test[i]]
    scores[i][2] = [srhmms[-1].score(x) for x in X_test[i]]
    scores[i][3] = [frhmms[-1].score(x) for x in X_test[i]]

# Evaluate
sensor_fused_scores = np.sum(scores, axis=0)
class_ids = np.nanargmax(sensor_fused_scores, axis=0)
class_names = [roll_classes[id] for id in class_ids]

class_id_by_sensor = np.nanargmax(scores, axis=1)
class_ids2 = stats.mode(class_id_by_sensor, axis=0).mode[0]
class_names2 = [roll_classes[id] for id in class_ids2]

class_id_by_sensor = np.nanmax(scores, axis=1)
class_ids3 = np.argmax(class_id_by_sensor, axis=0)
class_names3 = [roll_classes[id] for id in class_ids2]

sensor_max_scores = np.nanmax(scores, axis=0)
class_ids4 = np.argmax(sensor_max_scores, axis=0)
class_names4 = [roll_classes[id] for id in class_ids3]

print "Average score(pre-mean-fused sensors): %f" % np.mean(class_names == y_test)
print "Average score(pre-max-fused sensors): %f" % np.mean(class_names4 == y_test)
print "Average score(post-mean-fused sensors): %f" % np.mean(class_names2 == y_test)
print "Average score(post-max-fused sensors): %f" % np.mean(class_names3 == y_test)



