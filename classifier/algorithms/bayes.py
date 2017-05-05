from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import random

class Bayes:
    def train():
        samples = pickle.load( open( "synth_small_samples.pkl", "rb" ) )
        random.shuffle(samples)

        X = map((lambda x: np.array(x['data'])), samples)
        y = map((lambda x: x['label']), samples)

        clf = BernoulliNB()

        print("Naive Bayes Score: %s" % np.array_str(cross_val_score(clf, X, y)))

    def classify(self, cracked_data):
        return {
            "condition_class": "Preload",
            "roll_class":      "SlowRoll",
            "condition_vec":   {
                "Safe":       0.2,
                "Preload":    0.6,
                "Unbalance":  0.05,
                "BearingRub": 0.15
                },
                "roll_vec": {
                    "SlowRoll": 0.5,
                    "FastRoll": 0.4,
                    "RampUp":   0.0,
                    "RampDown": 0.1
                    }
            }
