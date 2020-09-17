import numpy as np
from sklearn.preprocessing import Normalizer
from scipy import stats



class KNN():
    
    """ Lets initials our KNN model with 3 atribiutes:
        K : Number of neighbors to use
        X : Training Data
        Y : Training Label
        distance : All distances 
    """
    def __init__(self, k, X, Y):
        self.k = k
        self.X = X
        self.Y = Y
    
    """ One of the brillient ways to find distance is to 
         calculate using cosin similarity.
         KNN is just uses the first K nearest labels to find the test label.
    """   
    def forward(self,  X_test):
        
        transformer = Normalizer().fit(self.X)  
        train_normalized = transformer.transform(self.X)
        transformer = Normalizer().fit(X_test) 
        test_normalized = transformer.transform(X_test)
        distance = np.dot(train_normalized,test_normalized.transpose())
        nearest = np.argsort(distance, axis=0)
        # USE K nearest 
        labels = self.Y[nearest[-(self.k):]]
        # Which one happens more
        final_labels = stats.mode(labels,axis=0)[0]
        
        return final_labels[0]

