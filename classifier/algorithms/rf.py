import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import datetime
from random import uniform, seed
seed(42)


#############################################################
########################## Globals ##########################
#############################################################
DATA_CSV = "../data/combined_4-29.csv"

ROLL_CLASSES      = ['ramp_up', 'ramp_down', 'slow_roll', 'fast_roll' ]
CONDITION_CLASSES = ['Normal',  'Unbalance', 'Preload', 'BearingRub']

N_SENSORS    = 4
N_CLASSES    = 4
SAMPLE_LEN   = 2048

RUN_ID = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#############################################################
######################### Functions #########################
#############################################################
from parse_csv import parse_s1_csv

def prec_rec(pred, truth):
    TP = np.sum((pred == "Normal") and (truth == "Normal"))
    FP = np.sum((pred != "Normal") and (truth == "Normal"))
    TN = np.sum((pred != "Normal") and (truth != "Normal"))
    FN = np.sum((pred == "Normal") and (truth != "Normal"))
    precision = TP / (TP + FP)
    recall    = TP / (TP + TN)
    return precision, recall


def write_metaparams():
    metaparam_file = open(('metaparams/rf/mp_%s.txt' % RUN_ID), 'w')
    
    metaparam_file.write("Number of Components: %d" % N_COMPONENTS)
    metaparam_file.write("Number of Mix Models: %d" % N_MIX)
    metaparam_file.write("Number of Iterations: %d" % N_ITERS)
    metaparam_file.write("Covariance Type     : %s" % COV_TYPE)
    metaparam_file.write("Algorithm           : %s" % ALGORITHM)

#############################################################
###################### Meta-Parameters ######################
#############################################################
N_ESTIMATORS      = 20
MAX_DEPTH         = None
MIN_SAMPLES_SPLIT = 2
RANDOM_STATE      = 42
VERBOSE           = True

N_FOLDS           = 5


#############################################################
########################### Main ############################
#############################################################
class RF:
    def __init__(self, n_estimators    = N_ESTIMATORS,
                     max_depth         = MAX_DEPTH,
                     min_samples_split = MIN_SAMPLES_SPLIT,
                     random_state      = RANDOM_STATE,
                     verbose           = VERBOSE,
                     class_names       = CONDITION_CLASSES):
        
        self.rf_clfs = [RandomForestClassifier(n_estimators          = n_estimators,
                                                   max_depth         = max_depth,
                                                   min_samples_split = min_samples_split,
                                                   random_state      = random_state,
                                                   verbose           = verbose
                                                   )] * N_SENSORS
        
        self.et_clfs = [ExtraTreesClassifier(n_estimators          = n_estimators,
                                                 max_depth         = max_depth,
                                                 min_samples_split = min_samples_split,
                                                 random_state      = random_state,
                                                 verbose           = verbose
                                                 )] * N_SENSORS

        self.class_names = class_names


    # X[s_id][train], y[s_id][train]
    def train(self, X, y):
        rf_preds = [None] * N_SENSORS
        et_preds = [None] * N_SENSORS
        for s_id in xrange(N_SENSORS):
            print "### Training on sensor id %d ###" % s_id

            self.rf_clfs[s_id] = self.rf_clfs[s_id].fit(X[s_id], y[s_id])
            self.et_clfs[s_id] = self.et_clfs[s_id].fit(X[s_id], y[s_id])

        self.save(rf_clfs, et_clss)


    def read_data(self):
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


    # models/rf/%s/fold%d % (RUN_ID, fold_count)
    def save(self, path):
        for s_id in range(N_SENSORS):
            print "Saving Random Forest Model"
            joblib.dump(rf_clfs[s_id], "%s/rf_s%d.pkl" % (path, s_id))
            print "Saving Extra Trees Model"
            joblib.dump(et_clfs[s_id], "%s/et_s%d.pkl" % (path, s_id))
    

    def classify(self, cracked_data):
        probs = [None] * N_SENSORS
        for s_id in range(N_SENSORS):
            # rf_pred = rf_clfs[s_id].predict(X[s_id][test])
            # et_pred = et_clfs[s_id].predict(X[s_id][test])
            probs[s_id] = np.random.rand(N_CLASSES)  # clf.predict_proba(cracked_data)
            
        # print probs
        ens_probs = np.mean(probs, axis=0)
        classification = np.argmax(ens_probs, axis=0)
        
        
        return {
            "classification": classification,
            "confidence_vec": { k:v for k,v in (zip(self.class_names, ens_probs)) }
            }

    
    def evaluate(self, X, y):
        (et_probs, rf_probs) = self.classify(X)

         
    def detailed_evaluate(self):
        rf_outputs = [{}] * N_SENSORS
        et_outputs = [{}] * N_SENSORS
        
        precision, recall = prec_rec(rf_pred, y[s_id][test])
        rf_outputs[s_id]['precision'] = precision
        rf_outputs[s_id]['recall'] = recall
        rf_outputs[s_id]['conf_mat'] = confusion_matrix(y[s_id][test], rf_pred)
        rf_outputs[s_id]['acc'] = np.mean(rf_pred == y[s_id][test])
        
        precision, recall = prec_rec(et_pred, y[s_id][test])
        et_outputs[s_id]['precision'] = precision
        et_outputs[s_id]['recall'] = recall
        et_outputs[s_id]['conf_mat'] = confusion_matrix(y[s_id][test], et_pred)
        et_outputs[s_id]['acc'] = np.mean(et_pred == y[s_id][test])

        return rf_outputs, et_outputs
        

    def cross_val(self, X, y):
        for (fold_count, (train, test)) in enumerate( skf.split(X[0],y[0]) ): 
            rf_total = [{}] * N_FOLDS
            et_total = [{}] * N_FOLDS

            # rf_pred = stats.mode(rf_preds).mode[0]
            rf_total[fold_count]['conf_mat'] = confusion_matrix(y[s_id][test], rf_pred)
            rf_total[fold_count]['acc'] = np.mean(rf_pred == y[s_id][test])

            # et_pred = stats.mode(et_preds).mode[0]
            et_total[fold_count]['conf_mat'] = confusion_matrix(y[0][test], et_pred)
            et_total[fold_count]['acc'] = np.mean(et_pred == y[0][test])

            print("Random Forest Accuracy: %s" % rf_total[fold_count]['acc'])
            print("Random Forest Confusion Matrix: \n%s\n" % rf_total[fold_count]['conf_mat'])
            print("Extra Trees Accuracy: %s" % et_total[fold_count]['acc'])
            print("Extra Trees Confusion Matrix: \n%s\n" % et_total[fold_count]['conf_mat'])


    def roc_nTrees(self):
        for i in xrange(6):
            print "#############################################"
            print ("USING %d ESTIMATORS:" % (5*(i+1)))
            print "#############################################"

            (X, y) = self.read_data()
            self.cross_val(X, y)


            rf_accuracy = np.mean([output['acc'] for output in rf_totals])
            rf_conf_mat = np.sum([output['conf_mat'] for output in rf_totals], axis=0)
            et_accuracy = np.mean([output['acc'] for output in et_totals])
            et_conf_mat = np.sum([output['conf_mat'] for output in et_totals], axis=0)
            print("Random Forest Accuracy: %s" % rf_accuracy)
            print("Random Forest Confusion Matrix: \n%s\n" % rf_conf_mat)
            print("Extra Trees Accuracy: %s" % et_accuracy)
            print("Extra Trees Confusion Matrix: \n%s\n" % et_conf_mat)


    def sk_evaluate(self, X, y):
        for s_id in range(N_SENSORS):
            rf_scores[s_id] = self.rf_clfs[s_id].score(X,y)
            et_scores[s_id] = self.et_clfs[s_id].score(X,y)

        return rf_scores, et_scores


    def sk_cross_val(self, X, y):
        rf_scores = cross_val_score(rf_clf, X, y, cv=5)
        et_scores = cross_val_score(et_clf, X, y, cv=5)

        return rf_scores, et_scores


    
# Evaluate models with respect to n_estimators
scores = []

skf = StratifiedKFold(n_splits=N_FOLDS, random_state=42)
# Note here that we're getting one fold mask to be used for all sensors(for sensor agreement)

