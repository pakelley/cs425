#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import datetime
from random import uniform, seed
from scipy.stats.mode import mode
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
N_ESTIMATORS      = 20
MAX_DEPTH         = None
MIN_SAMPLES_SPLIT = 2
RANDOM_STATE      = 42
VERBOSE           = True

N_FOLDS           = 5

# RUN_ID = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# metaparam_file = open(('metaparams_%s.txt' % RUN_ID), 'w')
# metaparam_file.write("Number of Components: %d" % N_COMPONENTS)
# metaparam_file.write("Number of Mix Models: %d" % N_MIX)
# metaparam_file.write("Number of Iterations: %d" % N_ITERS)
# metaparam_file.write("Covariance Type     : %s" % COV_TYPE)
# metaparam_file.write("Algorithm           : %s" % ALGORITHM)

class RF:
    def __init__(self, n_estimators=N_ESTIMATORS,
                     max_depth=MAX_DEPTH,
                     min_samples_split=MIN_SAMPLES_SPLIT,
                     random_state=RANDOM_STATE,
                     verbose=VERBOSE):
        
        self.rf_clfs = [RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state, verbose=verbose).fit(X[s_id][train], y[s_id][train])] * N_SENSORS
        
        self.et_clfs = [ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state, verbose=verbose).fit(X[s_id][train], y[s_id][train])] * N_SENSORS


    #############################################################
    ########################### Main ############################
    #############################################################
    def train(self):
        rf_preds = [None] * N_SENSORS
        et_preds = [None] * N_SENSORS
        for s_id in xrange(N_SENSORS):
            print "### Training on sensor id %d ###" % s_id

            self.rf_clfs[s_id] = self.rf_clfs[s_id].fit(X[s_id][train], y[s_id][train])
            self.et_clfs[s_id] = self.et_clfs[s_id].fit(X[s_id][train], y[s_id][train])

        save(rf_clfs, et_clss)

        return (rf_clfs, et_clfs)

    def evaluate(self):
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

        # # Exclude some classes for now :/ FIXME (Excluded Classes)
        # class_mask = ground_truth == ground_truth # np.ma.array(ground_truth, mask=True)
        # for class_name in EXCLUDED_CLASSES:
        #     class_mask   = np.logical_and(class_mask, (ground_truth != class_name))

        # samples      = samples[class_mask]
        # labels       = labels[class_mask]
        # ground_truth = ground_truth[class_mask]
        # samples_ref   = samples_ref[class_mask]

        # Now that our data is almost situated, enforce a uniform prior for all classes for training
        # MAX_SAMPLES = min([ np.sum(samples[ground_truth == class_name]) for class_name in CLASS_LABELS ])

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

            # Reset this for the current class

            print "#############################################"
            print ("TRAINING ON FOLD %d" % fold_index)
            print "#############################################"


            for class_name in CLASS_LABELS:
                class_mask = (ground_truth[train] == class_name)
                # n_class_samples = np.sum(class_mask) / N_CLASSES
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
                # X_raw = X_train[:MAX_SAMPLES] # FIXME (MAX_SAMPLES)
                X_raw = np.reshape(X_raw, (-1, N_COMPONENTS)) # N_COMPONENTS should == N_SENSORS!!!
                # lengths = [SAMPLE_LEN] * MAX_SAMPLES # X_train[class_mask].shape[0]
                lengths = [SAMPLE_LEN] * X_train[class_mask].shape[0]
                # ^tell hmmlearn to split data in N_SENSORS(so probably 4) equal parts
                model[class_name] = hmm[class_name].fit(X_raw , lengths)

                # for samp_id in range( n_class_samples ):


                filename = "models/hmm/%s_fold-%d__run%s.pkl" % (class_name, fold_index, RUN_ID)
                print "Saving %s Model to %s" % (class_name, filename)
                joblib.dump(model, filename)

            scores = [None] * N_CLASSES
            # X_test = X_test[:(MAX_SAMPLES)] # FIXME (MAX_SAMPLES)
            for class_ind, class_name in enumerate(CLASS_LABELS):
                print "Evaluating %s Model" % class_name
                scores[class_ind] = [model[class_name].score(x) for x in X_test]
                # scores[class_ind] = model.score(X_test, lengths)

                # for index, x in enumerate(X_test):
                #     scores[class_ind].append(model.score(x, lengths))
                #     if(index % 100 == 0):
                #         print "Finished with %d of %d samples" % ()



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

    def save(self, rf_clfs, et_clfs):
        print "Saving Random Forest Model"
        joblib.dump(rf_clfs[s_id], "models/rf/rf_fold%d_s%d.pkl" % (fold_count, s_id))
        print "Saving Extra Trees Model"
        joblib.dump(et_clfs[s_id], "models/rf/et_fold%d_s%d.pkl" % (fold_count, s_id))


    def classify(self, cracked_data):
        prob = [[]] * len(cracked_data)
        for i, clf in enumerate(self.et_clfs):
            prob[i] = clf.predict_proba(cracked_data)
        classification = mode( np.argmax(prob, axis=0) )
        return {
            "condition_class": ,
            "roll_class":      classification,
            "condition_vec":   {
                "Safe":       prob[0],
                "Unbalance":  prob[1],
                "Preload":    prob[2],
                "BearingRub": prob[3]
                },
                "roll_vec": {
                    "SlowRoll": 0.0,
                    "FastRoll": 0.9,
                    "RampUp":   0.07,
                    "RampDown": 0.03
                    }
            }


# # Evaluate models with respect to n_estimators
# scores = []
# rf_outputs = [{}] * N_SENSORS
# et_outputs = [{}] * N_SENSORS
# rf_total = [{}] * N_FOLDS
# et_total = [{}] * N_FOLDS

# skf = StratifiedKFold(n_splits=N_FOLDS, random_state=42)
# # Note here that we're getting one fold mask to be used for all sensors(for sensor agreement)

# for i in xrange(6):
#     print "#############################################"
#     print ("USING %d ESTIMATORS:" % (5*(i+1)))
#     print "#############################################"

#     fold_count = 0
#     for train, test in skf.split(X[0],y[0]): 

#         # print "Calculating DT Cross-Val"
#         # dt_scores = cross_val_score(dt_clf, X, y)
#         # print "Calculating RF Cross-Val"
#         # rf_pred = rf_clfs[s_id].predict(X[s_id][test])
#         # rf_outputs[s_id]['conf_mat'] = confusion_matrix(y[s_id][test], rf_pred)
#         # rf_outputs[s_id]['acc'] = np.mean(rf_pred == y[s_id][test])
#         # rf_preds[s_id] = rf_pred
#         # precision, recall = prec_rec(rf_pred, y[s_id][test])
#         # rf_outputs[s_id]['precision'] = precision
#         # rf_outputs[s_id]['recall'] = recall
#         # # rf_scores = cross_val_score(rf_clf, X, y, cv=5)

#         # print "Calculating ET Cross-Val"
#         # et_pred = et_clfs[s_id].predict(X[s_id][test])
#         # et_outputs[s_id]['conf_mat'] = confusion_matrix(y[s_id][test], et_pred)
#         # et_outputs[s_id]['acc'] = np.mean(et_pred == y[s_id][test])
#         # et_preds[s_id] = et_pred
#         # precision, recall = prec_rec(et_pred, y[s_id][test])
#         # et_outputs[s_id]['precision'] = precision
#         # et_outputs[s_id]['recall'] = recall
#         # et_scores = cross_val_score(et_clf, X, y, cv=5)

#         # print("Decision Tree Score: %s" % np.array_str(dt_scores))
#         # print("Random Forest Accuracy: %s" % rf_outputs[s_id]['acc'])
#         # print("Random Forest Confusion Matrix: \n%s\n" % rf_outputs[s_id]['conf_mat'])
#         # print("Extra Trees Accuracy: %s" % et_outputs[s_id]['acc'])
#         # print("Extra Trees Confusion Matrix: \n%s\n" % et_outputs[s_id]['conf_mat'])

#         # Store scores for plotting
#         # scores.append((5*(i+1), np.mean(rf_scores), np.mean(et_scores)))

#         rf_pred = stats.mode(rf_preds).mode[0]
#         rf_total[fold_count]['conf_mat'] = confusion_matrix(y[s_id][test], rf_pred)
#         rf_total[fold_count]['acc'] = np.mean(rf_pred == y[s_id][test])

#         et_pred = stats.mode(et_preds).mode[0]
#         et_total[fold_count]['conf_mat'] = confusion_matrix(y[0][test], et_pred)
#         et_total[fold_count]['acc'] = np.mean(et_pred == y[0][test])

#         print("Random Forest Accuracy: %s" % rf_total[fold_count]['acc'])
#         print("Random Forest Confusion Matrix: \n%s\n" % rf_total[fold_count]['conf_mat'])
#         print("Extra Trees Accuracy: %s" % et_total[fold_count]['acc'])
#         print("Extra Trees Confusion Matrix: \n%s\n" % et_total[fold_count]['conf_mat'])

#         fold_count += 1
        
        # for samp_id in range( n_class_samples ):
            

