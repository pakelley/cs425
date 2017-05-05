from algorithms import SVM, kNN, Bayes, RF

class Classifier:
    def __init__(self):
        #self.hmm   = HMM()
        self.svm   = SVM()
        self.rf    = RF()
        self.bayes = Bayes()
        self.knn   = kNN()

    # Classify: Return a dictionary with the classification and cracked data
    # The algorithm used is determined by the string passed in. Accepted values are
    #      'HMM', 'SVM', 'RandomForest', 'Bayes', and 'kNN'
    def classify(self, cracked_data, algorithm):
        classifications = {
                 #"HMM":          self.hmm.classify(cracked_data),
                 "SVM":          self.svm.classify(cracked_data),
                 "RandomForest": self.rf.classify(cracked_data),
                 "Bayes":        self.bayes.classify(cracked_data),
                 "kNN":          self.knn.classify(cracked_data)
                }.get(algorithm, None )
            
        if(classifications == None):
            raise ValueError("Unrecognized input for 'algorithm'")

        return {
            "classifications": classifications,
            "cracked_data":    cracked_data[0]
            }



# FILENAME = "demo.csv"

# summ_data = np.genfromtxt(FILENAME, dtype=None, delimiter=',', names=True)
# cracked_data_str = np.array(summ_data['Cracked'])
# cracked_data = [cr_str.split('_') for cr_str in cracked_data_str]
# cracked_data = np.transpose(cracked_data, (1, 0))

# c = Classifier()
# print c.classify(cracked_data, "RandomForest")
# c.classify("other data", "unknown class") # Will raise an error
# c.classify("other data", "hmm")           # Will raise an error

