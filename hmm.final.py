#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import datetime
from random import uniform, seed
random.seed(42)

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from hmmlearn.hmm import GaussianHMM, GMMHMM

from parse_csv import parse_s1_csv


#############################################################
########################## Globals ##########################
#############################################################
# DATA_CSV = "../AllFromMarchDataPatrick.csv"
# DATA_CSV = "../Data_Normal_4_3_17.csv"
# DATA_CSV = "../data/combined_allClasses_test.csv"
# DATA_CSV = "../data/combined_allClasses.csv"
# DATA_CSV = "../data/combined_4-17.csv"
DATA_CSV = "../data/combined_4-29.csv"

N_SENSORS    = 4
SAMPLE_LEN   = 2048
DPI = 80

MAX_SAMPLES      = "This value is set after reading in the data"

ROLL_CLASSES      = ['ramp_up', 'ramp_down', 'slow_roll', 'fast_roll' ]
ALL_CONDITION_CLASSES = ['Normal',  'Unbalance', 'Preload',   'BearingRub']
CONDITION_CLASSES = ['Normal',  'Unbalance', 'Preload', 'BearingRub']
# EXCLUDED_CLASSES = list( set(ALL_CONDITION_CLASSES) ^ set(CONDITION_CLASSES) )


# Set the classification type here
CLASS_LABELS = CONDITION_CLASSES

N_CLASSES    = len(CLASS_LABELS)


#############################################################
######################### Functions #########################
#############################################################
def prec_rec(pred, truth):
    TP = np.sum((pred == "Normal") and (truth == "Normal"))
    FP = np.sum((pred != "Normal") and (truth == "Normal"))
    TN = np.sum((pred != "Normal") and (truth != "Normal"))
    FN = np.sum((pred == "Normal") and (truth != "Normal"))
    precision = TP / (TP + FP)
    recall    = TP / (TP + TN)
    return precision, recall

def calc_startprob(length):
    startprob = np.zeros(length)
    startprob[0] = 2
    return startprob

def calc_transmat(length):
    # transmat = np.zeros((length, length))
    # for i in xrange(length):
    #     transmat[i,i] = 1.0 + (1.0 / length)
    #     if i != (length - 1):
    #         transmat[i,i+1] = 3 - transmat[i,i]
    #     else:
    #         transmat[i,i] = 2.0
    transmat = ( np.ones((length, length)) ) + random.uniform(0.05, 0.8)
    return transmat


#############################################################
###################### Meta-Parameters ######################
#############################################################
N_COMPONENTS    = N_SENSORS
N_MIX           = 2
N_ITERS         = 50
COV_TYPE        = "diag"
ALGORITHM       = "map"
TOL             = 0.1

STARTPROB_PRIOR = calc_startprob(N_COMPONENTS)
TRANSMAT_PRIOR  = calc_transmat(N_COMPONENTS)

# INIT_PARAMS     = 'stmcw'
# PARAMS          = 'stmcw'
INIT_PARAMS     = 'stmcw'
PARAMS          = 'tmcw'

VERBOSE         = True

N_FOLDS         = 3

RUN_ID = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
metaparam_file = open(('metaparams/metaparams_%s.txt' % RUN_ID), 'w')
metaparam_file.write("Number of Components: %d" % N_COMPONENTS)
metaparam_file.write("Number of Mix Models: %d" % N_MIX)
metaparam_file.write("Number of Iterations: %d" % N_ITERS)
metaparam_file.write("Covariance Type     : %s" % COV_TYPE)
metaparam_file.write("Algorithm           : %s" % ALGORITHM)

#############################################################
########################### Main ############################
#############################################################
# Read in data
print "Parsing Data..."
globs = parse_s1_csv(DATA_CSV, "normal")

samples          = np.array([np.array(glob['samples'])         for glob in globs])
roll_labels      = np.array([np.array(glob['roll_class'])      for glob in globs])
condition_labels = np.array([np.array(glob['condition_class']) for glob in globs])

# Training on conditions
if( CLASS_LABELS == CONDITION_CLASSES ):
    print "Training on Condition Classes"
    labels      = condition_labels
    class_names = CONDITION_CLASSES
elif (CLASS_LABELS == ROLL_CLASSES):
    print "Training on Roll Classes"
    labels      = condition_labels
else:
    print "WARNING: Classes not set, defaulting to condition classes"
    labels      = condition_labels
# class_names = CONDITION_CLASSES


# Get one sensor channel of samples/labels for use in cross-val
samples_ref  = samples[0]
ground_truth = labels[0]

# Since the HMM works slightly differently than the other models, we need to transpose the data
# In particular, we want all sensor values to be included in each sample
samples = np.transpose(samples, (1, 2, 0))
labels  = np.transpose(labels,  (1, 0))

# red dashes, blue squares and green triangles
one_sample = np.transpose(samples, (0,2,1))[0]
fig1 = plt.figure(0, figsize=(4.8, 3.2))
plt.plot(one_sample[0])
plt.xlim(0,2048)
plt.title("Outer-Side X Plot")

fig2 = plt.figure(1, figsize=(4.8, 3.2))
plt.plot(one_sample[1])
plt.xlim(0,2048)
plt.title("Outer-Side Y Plot")

fig3 = plt.figure(2, figsize=(4.8, 3.2))
plt.plot(one_sample[0], one_sample[1])
plt.title("Outer-Side Orbit Plot")

fig4 = plt.figure(3, figsize=(4.8, 3.2))
two_sample = np.transpose(samples, (0,2,1))[1]
plt.plot(two_sample[0])
plt.xlim(0,2048)
plt.title("Motor-Side X Plot")

fig5 = plt.figure(4, figsize=(4.8, 3.2))
plt.plot(two_sample[1])
plt.xlim(0,2048)
plt.title("Motor-Side Y Plot")

fig6 = plt.figure(5, figsize=(4.8, 3.2))
plt.plot(two_sample[0], two_sample[1])
plt.title("Motor-Side Orbit Plot")
plt.show()

# Now that our data is almost situated, enforce a uniform prior for all classes for training
MAX_SAMPLES = min([ np.sum(samples[ground_truth == class_name]) for class_name in CLASS_LABELS ])

skf = StratifiedKFold(n_splits=N_FOLDS, random_state=42)
# Note here that we're getting one fold mask to be used for all sensors(for sensor agreement)

summaries = [{}] * N_FOLDS
# scores = [[]] * MAX_SAMPLES # FIXME (MAX_SAMPLES)
fold_index = 0

for train, test in skf.split(samples_ref, ground_truth): 
    # Train
    X_train = samples[train]
    X_test  = samples[test]
    # reshape for use in testing later
    # X_test = np.reshape(X_test, (-1,N_SENSORS))
    # X_test = np.transpose(X_test, (2,1,0))[0]

    # samples = np.transpose(samples, (1, 0, 2))
    # labels  = np.transpose(labels,  (1, 0))

    # Reset this for the current class
    MAX_SAMPLES = min([ np.sum(X_train[ground_truth[train] == class_name]) for class_name in CLASS_LABELS ])
    MAX_SAMPLES = min(MAX_SAMPLES, 50)
    # scores = np.zeros((N_CLASSES, MAX_SAMPLES)) # FIXME (MAX_SAMPLES)
    
    print "#############################################"
    print ("TRAINING ON FOLD %d" % fold_index)
    print "#############################################"

    hmm = {}
    model = {}
        
    for class_name in CLASS_LABELS:
        class_mask = (ground_truth[train] == class_name)
        hmm[class_name] = GMMHMM(
            n_components    = N_COMPONENTS,
            n_mix           = N_MIX,
            n_iter          = N_ITERS,
            covariance_type = COV_TYPE,
            algorithm       = ALGORITHM,
            tol             = TOL,
            transmat_prior  = TRANSMAT_PRIOR,
            startprob_prior = STARTPROB_PRIOR,
            init_params     = INIT_PARAMS,
            params          = PARAMS,
            verbose         = VERBOSE
            )
        
        print "Training %s Model" % class_name

        X_raw = X_train[class_mask]
        X_raw = X_train[:MAX_SAMPLES] # FIXME (MAX_SAMPLES)
        X_raw = np.reshape(X_raw, (-1, N_COMPONENTS)) # N_COMPONENTS should == N_SENSORS!!!
        # lengths = [SAMPLE_LEN] * MAX_SAMPLES # X_train[class_mask].shape[0]
        lengths = [SAMPLE_LEN] * (X_raw.shape[0] / SAMPLE_LEN)
        # ^tell hmmlearn to split data in N_SENSORS(so probably 4) equal parts
        model[class_name] = hmm[class_name].fit(X_raw , lengths)
            

        filename = "models/hmm/%s_fold-%d__run%s.pkl" % (class_name, fold_index, RUN_ID)
        print "Saving %s Model to %s" % (class_name, filename)
        joblib.dump(model[class_name], filename)

        
    # X_test = X_test[:(MAX_SAMPLES)] # FIXME (MAX_SAMPLES)
    scores = [None] * N_CLASSES
    for class_ind, class_name in enumerate(CLASS_LABELS):
        print "Evaluating %s Model" % class_name
        scores[class_ind] = [model[class_name].score(x) for x in X_test]
        # scores[class_ind] = model.score(X_test, lengths)


    # Evaluate
    class_ids   = np.nanargmax(scores, axis=0)
    classifications = [CLASS_LABELS[id] for id in class_ids]
    fold_ground_truth = ground_truth[test] # [:MAX_SAMPLES] #FIXME (MAX_SAMPLES)
    summaries[fold_index]['acc']      = np.mean(classifications == fold_ground_truth)
    summaries[fold_index]['conf_mat'] = confusion_matrix(fold_ground_truth, classifications)
    
    precision, recall = prec_rec(classifications, fold_ground_truth)
    summaries[fold_index]['precision'] = precision
    summaries[fold_index]['recall'] = recall

    print "Average Score (Sensor Mean): %f" % summaries[fold_index]['acc']
    print "Precision on Safe vs Unsafe: %f" % precision
    print "Recall    on Safe vs Unsafe: %f" % recall
    print("Sensor Mean Confusion Matrix: \n%s\n" % summaries[fold_index]['conf_mat'])
    fold_index += 1

# Evaluate cross-fold    
total_accuracy  = np.mean([output['acc']       for output in summaries], axis=0)
total_precision = np.mean([output['precision'] for output in summaries], axis=0)
total_recall    = np.mean([output['recall']    for output in summaries], axis=0)
total_conf_mat  = np.sum([output['conf_mat']   for output in summaries], axis=0)

print("Total Accuracy: %s" % total_accuracy)
print("Total Precision: \n%s\n" % total_precision)
print("Total Recall: \n%s\n" % total_recall)
print("Total Confusion Matrix: \n%s\n" % total_conf_mat)
