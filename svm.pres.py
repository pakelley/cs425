import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

np.random.seed(0)

class SVM:

    N_SENSORS = 4
    N_FOLDS = 5

    # def prec_rec(pred, truth):
    #     TP = np.sum((pred == "Normal") and (truth == "Normal"))
    #     FP = np.sum((pred != "Normal") and (truth == "Normal"))
    #     TN = np.sum((pred != "Normal") and (truth != "Normal"))
    #     FN = np.sum((pred == "Normal") and (truth != "Normal"))
    #     precision = TP / (TP + FP)
    #     recall = TP / (TP + TN)
    #     recall precision, recall

    def train():
        # Read in data
        from parse_csv import parse_s1_csv
        # DATA_CSV = "../AllFromMarchDataPatrick.csv"
        # DATA_CSV = "../Data_Normal_4_3_17.csv"
        # DATA_CSV = "../data/combined_allClasses_test.csv"
        DATA_CSV = "../data/combined_4-17.csv"

        print "Parsing Data..."

        globs = parse_s1_csv(DATA_CSV, "normal")
        # X = np.array([glob['samples'] for glob in globs])
        # X = X.reshape(-1, X.shape[-1])
        # y = np.array([glob['roll_class'] for glob in globs]).reshape(-1)
        X = np.array([np.array(glob['samples']) for glob in globs])
        y = np.array([np.array(glob['condition_class']) for glob in globs])


        svm_outputs = [{}] * N_SENSORS
        svm_totals = [{}] * N_FOLDS
        svm_clfs = [None] * N_SENSORS

        skf = StratifiedKFold(n_splits=5, random_state=42)
        # Note here that we're getting one fold mask to be used for all sensors(for sensor agreement)

        fold_index = 0
        for train, test in skf.split(X[0],y[0]):
            print "#############################################"
            print ("Training on fold %d" % fold_index)
            print "#############################################"


            svm_preds = [None] * N_SENSORS
            for s_id in xrange(N_SENSORS):
                print("##### Training model for sensor %d #####" % s_id)

                # fit the model
                print "Training SVM"
                kernel = 'rbf'
                svm_clfs[s_id] = svm.SVC(kernel=kernel, gamma='auto', verbose=True).fit(X[s_id][train], y[s_id][train])

                # print "Saving Model"
                # joblib.dump(svm_clfs[s_id], "models/svm/svm_fold%d_s%d.pkl" % (fold_count, s_id))


                print "Calculating SVM Cross-Val"
                svm_pred = svm_clfs[s_id].predict(X[s_id][test])
                svm_outputs[s_id]['conf_mat'] = confusion_matrix(y[s_id][test], svm_pred)
                svm_outputs[s_id]['acc'] = np.mean(svm_pred == y[s_id][test])
                svm_preds[s_id] = svm_pred
                # precision, recall = prec_rec(svm_pred, y[s_id][test])
                # svm_outputs[s_id]['precision'] = precision
                # svm_outputs[s_id]['recall'] = recall

                # print("SVM Accuracy: %s" % svm_outputs[s_id]['acc'])
                # print("SVM Confusion Matrix: \n%s\n" % svm_outputs[s_id]['conf_mat'])

                # Store scores for plotting
                # scores.append((5*(i+1), np.mean(svm_scores), np.mean(et_scores)))



            svm_pred = stats.mode(svm_preds).mode[0]
            svm_totals[fold_index]['conf_mat'] = confusion_matrix(y[s_id][test], svm_pred)
            svm_totals[fold_index]['acc'] = np.mean(svm_pred == y[s_id][test])

            # print("SVM Accuracy: %s" % svm_totals[fold_index]['acc'])
            # print("SVM Confusion Matrix: \n%s\n" % svm_totals[fold_index]['conf_mat'])

            fold_index += 1


        accuracy = np.mean([output['acc'] for output in svm_totals])
        conf_mat = np.sum([output['conf_mat'] for output in svm_totals], axis=0)
        print("SVM Accuracy: %s" % accuracy)
        print("SVM Confusion Matrix: \n%s\n" % conf_mat)

    def classify(self, cracked_data):
        return {
            "condition_class": "Safe",
            "roll_class":      "SlowRoll",
            "condition_vec":   {
                "Safe":       0.7,
                "Preload":    0.1,
                "Unbalance":  0.05,
                "BearingRub": 0.15
                },
                "roll_vec": {
                    "SlowRoll": 0.6,
                    "FastRoll": 0.2,
                    "RampUp":   0.2,
                    "RampDown": 0.0
                    }
            }

