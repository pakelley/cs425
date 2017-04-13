
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import random

# Read in data
from parse_csv import parse_s1_csv
# DATA_CSV = "../AllFromMarchDataPatrick.csv"
DATA_CSV = "../Data_Normal_4_3_17.csv"

print "Parsing Data..."

globs = parse_s1_csv(DATA_CSV, "normal")

X = np.array([glob['samples'] for glob in globs])
X = X.reshape(-1, X.shape[-1])
y = np.array([glob['roll_class'] for glob in globs])
y = y.reshape(-1)

# Evaluate models with respect to n_estimators
scores = []
for i in xrange(20):
    print "#############################################"
    print ("USING %d ESTIMATORS:" % (5*(i+1)))
    print "#############################################"
    # dt_clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    rf_clf = RandomForestClassifier(n_estimators=5*(i+1), max_depth=None, min_samples_split=2, random_state=0, verbose=1)
    et_clf = ExtraTreesClassifier(n_estimators=5*(i+1), max_depth=None, min_samples_split=2, random_state=0, verbose=1)

    # print "Calculating DT Cross-Val"
    # dt_scores = cross_val_score(dt_clf, X, y)
    print "Calculating RF Cross-Val"
    rf_scores = cross_val_score(rf_clf, X, y, cv=5)
    print "Calculating ET Cross-Val"
    et_scores = cross_val_score(et_clf, X, y, cv=5)

    # print("Decision Tree Score: %s" % np.array_str(dt_scores))
    print("Random Forest Score: %s" % np.array_str(rf_scores))
    print("Random Forest Average: %s" % np.mean(rf_scores))
    print("Extra Trees Score: %s" % np.array_str(et_scores))
    print("Extra Trees Average: %s" % np.mean(et_scores))

    # Store scores for plotting
    scores.append((5*(i+1), np.mean(rf_scores), np.mean(et_scores)))
    
print "#############################################"
print "ALL SCORES:"
print "#############################################"
print scores
