class Classifier:
    def __init__(self):
        print "Something is happening... :O"

    # Classify: Return a dictionary with the classification and cracked data
    # The algorithm used is determined by the string passed in. Accepted values are
    #      'HMM', 'SVM', 'RandomForest', 'Bayes', and 'kNN'
    def classify(self, cracked_data, algorithm):
        classifications = {
                 "HMM":          self.hmm(cracked_data),
                 "SVM":          self.svm(cracked_data),
                 "RandomForest": self.rf(cracked_data),
                 "Bayes":        self.bayes(cracked_data),
                 "kNN":          self.knn(cracked_data)
                }.get(algorithm, None )
            
        if(classifications == None):
            raise ValueError("Unrecognized input for 'algorithm'")

        return {
            "classifications": classifications,
            "cracked_data":    cracked_data
            }
        

    ### Classifiers ###
    # Each returns a (currently arbitrary) dictionary of classes and confidence vectors
    def hmm(self, cracked_vector):
        return {
            "condition_class": "Safe",
            "roll_class":      "FastRoll",
            "condition_vec":   {
                "Safe":       0.7,
                "Preload":    0.1,
                "Unbalance":  0.05,
                "BearingRub": 0.15
                },
                "roll_vec": {
                    "SlowRoll": 0.0,
                    "FastRoll": 0.9,
                    "RampUp":   0.07,
                    "RampDown": 0.03
                    }
            }

    def svm(self, cracked_vector):
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

    def rf(self, cracked_vector):
        return {
            "condition_class": "Safe",
            "roll_class":      "FastRoll",
            "condition_vec":   {
                "Safe":       0.8,
                "Preload":    0.01,
                "Unbalance":  0.04,
                "BearingRub": 0.15
                },
                "roll_vec": {
                    "SlowRoll": 0.0,
                    "FastRoll": 0.9,
                    "RampUp":   0.07,
                    "RampDown": 0.03
                    }
            }

    def bayes(self, cracked_vector):
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

    def knn(self, cracked_vector):
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

    def plot_stuff():
        one_sample = samples[0]
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
        two_sample = samples[1]
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



c = Classifier()
print c.classify("sum data", "SVM")
# c.classify("other data", "unknown class") # Will raise an error
# c.classify("other data", "hmm")           # Will raise an error

