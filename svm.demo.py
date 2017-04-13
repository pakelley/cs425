import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score
import pickle
import random

np.random.seed(0)

# Read in data
from parse_csv import parse_s1_csv
# DATA_CSV = "../AllFromMarchDataPatrick.csv"
DATA_CSV = "../Data_Normal_4_3_17.csv"

print "Parsing Data..."

globs = parse_s1_csv(DATA_CSV, "normal")
X = np.array([glob['samples'] for glob in globs])
X = X.reshape(-1, X.shape[-1])
y = np.array([glob['roll_class'] for glob in globs]).reshape(-1)

# fit the model
print "Training..."
kernel = 'rbf'
clf = svm.SVC(kernel=kernel, gamma='auto', verbose=True)
clf.fit(X, y)

# Evaluate the model
print "Evaluating Model..."
pred = clf.predict(X)
print "%s accuracy: %s" % (kernel, np.array_str(cross_val_score(clf, X, y)))

