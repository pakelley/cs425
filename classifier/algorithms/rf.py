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
DATA_CSV          = "../data/combined_4-29.csv"

ROLL_CLASSES      = ['ramp_up', 'ramp_down', 'slow_roll', 'fast_roll']
CONDITION_CLASSES = ['Normal', 'Unbalance', 'Preload', 'BearingRub']

N_SENSORS         = 4
N_CLASSES         = 4
SAMPLE_LEN        = 2048

RUN_ID            = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOAD_PATH         = ""

FILENAME          = "data/combined_4-29.csv"
SENSOR_IDS        = [218, 244, 270, 296]
COND_CLASS_NAMES  = ['Normal', 'Unbalance', 'Preload', 'BearingRub']
TOP_THRESH        = 2800
BOT_THRESH        = 600

#############################################################
######################### Functions #########################
#############################################################
# from parse_csv import parse_s1_csv


def prec_rec(pred, truth):
    TP = np.sum((pred == "Normal") and (truth == "Normal"))
    FP = np.sum((pred != "Normal") and (truth == "Normal"))
    TN = np.sum((pred != "Normal") and (truth != "Normal"))
    FN = np.sum((pred == "Normal") and (truth != "Normal"))
    precision = TP / (TP + FP)
    recall = TP / (TP + TN)
    return precision, recall


def write_metaparams():
    metaparam_file = open(('metaparams/rf/mp_%s.txt' % RUN_ID), 'w')

    metaparam_file.write("Number of Components: %d" % N_COMPONENTS)
    metaparam_file.write("Number of Mix Models: %d" % N_MIX)
    metaparam_file.write("Number of Iterations: %d" % N_ITERS)
    metaparam_file.write("Covariance Type     : %s" % COV_TYPE)
    metaparam_file.write("Algorithm           : %s" % ALGORITHM)


def isChangingVelocity(d_speed, index):
    forward_idx = 0
    for i in xrange(index + 1, min(index + 4, len(d_speed) - 1)):
        if (d_speed[index] * d_speed[i] > 0):
            forward_idx += 1

    backward_idx = 0
    for i in xrange(index - 1, max(index - 4, 0), -1):
        if (d_speed[index] * d_speed[i] > 0):
            backward_idx += 1

    return (forward_idx >= 3) or (backward_idx >= 3)


def isAccelerating(d_speed, index):
    return isChangingVelocity(d_speed, index) and (d_speed[index] > 0)


def isDecelerating(d_speed, index):
    return isChangingVelocity(d_speed, index) and (d_speed[index] < 0)


def classify_roll(speed):
    d_speed = [x - speed[max(0, index - 1)] for index, x in enumerate(speed)]
    last_yield = 'unknown'  # Note that this will make classification funny for weird initial values
    for index, s in enumerate(speed):
        if (isAccelerating(d_speed, index)):
            last_yield = 'ramp_up'
            yield 'ramp_up'
        elif (isDecelerating(d_speed, index)):
            last_yield = 'ramp_down'
            yield 'ramp_down'
        elif (speed[index] > TOP_THRESH):
            last_yield = 'fast_roll'
            yield 'fast_roll'
        elif (speed[index] < BOT_THRESH):
            last_yield = 'slow_roll'
            yield 'slow_roll'
        else:
            yield last_yield


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

    # Constructor
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 max_depth=MAX_DEPTH,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 random_state=RANDOM_STATE,
                 verbose=VERBOSE,
                 class_names=CONDITION_CLASSES,
                 filepath=None):

        self.class_names = class_names

        # Either make blank classifiers for training, or read from file
        if filepath != None:
            self.rf_clfs = [None] * N_SENSORS
            self.et_clfs = [None] * N_SENSORS
            self.load(filepath)
            print filepath

        else:
            self.rf_clfs = [
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=random_state,
                    verbose=verbose)
            ] * N_SENSORS

            self.et_clfs = [
                ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=random_state,
                    verbose=verbose)
            ] * N_SENSORS

    # Classify # TODO: add check for trained model
    def classify(self, cracked_data):
        probs = [None] * N_SENSORS
        X = np.transpose(cracked_data, (2, 0, 1))
        for s_id in range(N_SENSORS):
            # rf_pred = rf_clfs[s_id].predict(X[s_id][test])
            et_pred = self.et_clfs[s_id].predict(X[s_id])
            # probs[s_id] = np.random.rand(N_CLASSES)  # clf.predict_proba(cracked_data)
            probs[s_id] = et_pred

        # print probs
        ens_probs = np.mean(probs, axis=0)
        class_id = np.argmax(ens_probs, axis=0)
        classification = self.class_names[class_id]

        return {
            "classification": classification,
            "confidence_vec": list(ens_probs),
            "class_names": self.class_names
        }

    def probs(self, X):
        rf_probs = [None] * N_SENSORS
        et_probs = [None] * N_SENSORS
        for s_id in range(N_SENSORS):
            rf_probs = self.rf_clfs[s_id].predict_proba(X[s_id])
            et_probs = self.rf_clfs[s_id].predict_proba(X[s_id])

        rf_ens_probs = np.argmax(rf_probs, axis=1)
        et_ens_probs = np.argmax(et_probs, axis=1)

        return rf_ens_probs, et_ens_probs

    def evaluate(self, X, y):
        (et_probs, rf_probs) = self.probs(X)

        rf_decision = np.argmax(rf_probs)
        et_decision = np.argmax(et_probs)

        return rf_decision, et_decision

    # X[s_id][train], y[s_id][train]
    def train(self, X, y):
        rf_preds = [None] * N_SENSORS
        et_preds = [None] * N_SENSORS
        for s_id in xrange(N_SENSORS):
            print "### Training on sensor id %d ###" % s_id

            self.rf_clfs[s_id] = self.rf_clfs[s_id].fit(X[s_id], y)
            self.et_clfs[s_id] = self.et_clfs[s_id].fit(X[s_id], y)

        self.save("./models/rf")

    # models/rf/%s/fold%d % (RUN_ID, fold_count)
    def save(self, path):
        for s_id in range(N_SENSORS):
            rf_filename = "%s/rf_%s_s%d.pkl" % (path, RUN_ID, s_id)
            print "Saving Random Forest Model to %s" % rf_filename
            joblib.dump(self.rf_clfs[s_id], rf_filename)
            print "Saving Extra Trees Model"
            joblib.dump(self.et_clfs[s_id], "%s/et_%s_s%d.pkl" % (path, RUN_ID,
                                                                  s_id))

    def load(self, path):
        for s_id in range(N_SENSORS):
            rf_filename = "%s/rf_s%d.pkl" % (path, s_id)
            print "Saving Random Forest Model from %s" % rf_filename
            self.rf_clfs[s_id] = joblib.load(rf_filename)
            print "Saving Extra Trees Model"
            self.et_clfs[s_id] = joblib.load("%s/et_s%d.pkl" % (path, s_id))

    def read_data(self):
        ### Read from file
        summ_data = np.genfromtxt(
            FILENAME, dtype=None, delimiter=',', names=True)
        cracked_data = summ_data['Cracked']
        s_ids = summ_data['segment_id']
        rpms = summ_data['start_rev_rpm']
        condition_classes = summ_data['condition_class']

        ### Get masks for sensors
        s_mask = [None] * 4
        s_mask[0] = [ids == SENSOR_IDS[0] for ids in s_ids]
        s_mask[1] = [ids == SENSOR_IDS[1] for ids in s_ids]
        s_mask[2] = [ids == SENSOR_IDS[2] for ids in s_ids]
        s_mask[3] = [ids == SENSOR_IDS[3] for ids in s_ids]

        ### Get class labels
        roll_classes = np.array(list(classify_roll(rpms[s_mask[0]])))
        condition_classes = condition_classes[s_mask[0]]

        ### Split up blobs
        X_i = np.array(
            [cracked_data[s_mask[i]] for i in range(len(SENSOR_IDS))])
        X_spl = np.array([[x.split('_') for x in x_i] for x_i in X_i])

        ### Fast mask all data
        fast_mask = (roll_classes == 'fast_roll')
        X_f = np.array([x[fast_mask] for x in X_spl])
        # X_ft = np.transpose(X_f, (1, 2, 0))
        roll_classes = np.array(roll_classes[fast_mask])
        r_classes = [
            ROLL_CLASSES.index(class_name) for class_name in roll_classes
        ]
        condition_classes = np.array(condition_classes[fast_mask])
        cond_classes = [
            COND_CLASS_NAMES.index(class_name)
            for class_name in condition_classes
        ]

        ### Stratified selection of data
        # MAX_LEN = 200
        # cl_masks = [condition_classes == class_name for class_name in COND_CLASS_NAMES]
        # X_cl = [None] * N_SENSORS
        # for s_id in range(N_SENSORS):
        # X_cl = np.array([X_ft[mask] for mask in cl_masks])
        # X_cl = np.array([X_ft[mask][:MAX_LEN] for mask in cl_masks])
        # X_cl = np.transpose(X_cl, (0, 3, 1, 2))
        # roll_cl = np.array([roll_classes[mask] for mask in cl_masks])
        # roll_cl = np.array([roll_classes[mask][:MAX_LEN] for mask in cl_masks])
        # cond_cl = np.array([condition_classes[mask] for mask in cl_masks])
        # cond_cl = np.array([cond_classes[mask][:MAX_LEN] for mask in cl_masks])

        ### FINAL FORMatting
        # X = np.reshape(X_cl, (N_SENSORS, -1, SAMPLE_LEN)) # np.transpose(X_cl, (1, 2, 0))
        # y = np.reshape(cond_cl, (-1) )
        print X_f.shape
        print np.array(cond_classes).shape
        X = X_f
        y = cond_classes
        return X, y

    def detailed_evaluate(self):
        rf_outputs = [{}] * N_SENSORS
        et_outputs = [{}] * N_SENSORS

        precision, recall             = prec_rec(rf_pred, y[s_id][test])
        rf_outputs[s_id]['precision'] = precision
        rf_outputs[s_id]['recall']    = recall
        rf_outputs[s_id]['conf_mat']  = confusion_matrix(y[s_id][test], rf_pred)
        rf_outputs[s_id]['acc']       = np.mean(rf_pred = = y[s_id][test])

        precision, recall             = prec_rec(et_pred, y[s_id][test])
        et_outputs[s_id]['precision'] = precision
        et_outputs[s_id]['recall']    = recall
        et_outputs[s_id]['conf_mat']  = confusion_matrix(y[s_id][test], et_pred)
        et_outputs[s_id]['acc']       = np.mean(et_pred = = y[s_id][test])

        return rf_outputs, et_outputs

    def cross_val(self, X, y):
        for (fold_count, (train, test)) in enumerate(skf.split(X[0], y[0])):
            rf_total = [{}] * N_FOLDS
            et_total = [{}] * N_FOLDS

            # rf_pred = stats.mode(rf_preds).mode[0]
            rf_total[fold_count]['conf_mat'] = confusion_matrix(
                y[s_id][test], rf_pred)
            rf_total[fold_count]['acc'] = np.mean(rf_pred == y[s_id][test])

            # et_pred = stats.mode(et_preds).mode[0]
            et_total[fold_count]['conf_mat'] = confusion_matrix(
                y[0][test], et_pred)
            et_total[fold_count]['acc'] = np.mean(et_pred == y[0][test])

            print("Random Forest Accuracy: %s" % rf_total[fold_count]['acc'])
            print("Random Forest Confusion Matrix: \n%s\n" %
                  rf_total[fold_count]['conf_mat'])
            print("Extra Trees Accuracy: %s" % et_total[fold_count]['acc'])
            print("Extra Trees Confusion Matrix: \n%s\n" %
                  et_total[fold_count]['conf_mat'])

    def roc_nTrees(self):
        for i in xrange(6):
            print "#############################################"
            print("USING %d ESTIMATORS:" % (5 * (i + 1)))
            print "#############################################"

            (X, y) = self.read_data()
            self.cross_val(X, y)

            rf_accuracy = np.mean([output['acc'] for output in rf_totals])
            rf_conf_mat = np.sum(
                [output['conf_mat'] for output in rf_totals], axis=0)
            et_accuracy = np.mean([output['acc'] for output in et_totals])
            et_conf_mat = np.sum(
                [output['conf_mat'] for output in et_totals], axis=0)
            print("Random Forest Accuracy: %s" % rf_accuracy)
            print("Random Forest Confusion Matrix: \n%s\n" % rf_conf_mat)
            print("Extra Trees Accuracy: %s" % et_accuracy)
            print("Extra Trees Confusion Matrix: \n%s\n" % et_conf_mat)

    def sk_evaluate(self, X, y):
        for s_id in range(N_SENSORS):
            rf_scores[s_id] = self.rf_clfs[s_id].score(X, y)
            et_scores[s_id] = self.et_clfs[s_id].score(X, y)

        return rf_scores, et_scores

    def sk_cross_val(self, X, y):
        rf_scores = cross_val_score(rf_clf, X, y, cv=5)
        et_scores = cross_val_score(et_clf, X, y, cv=5)

        return rf_scores, et_scores


# Evaluate models with respect to n_estimators
# scores = []

# skf = StratifiedKFold(n_splits=N_FOLDS, random_state=42)
# Note here that we're getting one fold mask to be used for all sensors(for sensor agreement)

# rf = RF()
# X, y = rf.read_data()
# rf.train(X,y)
# rf_d, et_d = rf.evaluate(X,y)
# rf_preds = [None] * N_SENSORS

# rt_p, et_p = rf.probs(X)
# print (sum(rt_p == y) * 1.0) / rt_p.shape[0]
