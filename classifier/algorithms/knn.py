from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import random

class kNN:
    def train():
        samples = pickle.load( open( "synth_small_samples.pkl", "rb" ) )
        random.shuffle(samples)

        X = map((lambda x: np.array(x['data'])), samples)
        y = map((lambda x: x['label']), samples)

        clf = KNeighborsClassifier(n_neighbors=3)

        print("kNN Score: %s" % np.array_str(cross_val_score(clf, X, y)))

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
                    "RampUp":   0.18,
                    "RampDown": 0.02
                    }
            }

