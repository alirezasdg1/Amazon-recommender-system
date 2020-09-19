import pandas as pd
import numpy as np
import sys
from io import StringIO
import surprise
from sklearn.model_selection import train_test_split, GridSearchCV

import random

from surprise import SVD, NMF
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import GridSearchCV

class CollaborativeRecommender():
    
    def __init__(self,df,reader,model):
        self.data = surprise.Dataset.load_from_df(df,reader)
        self.trainset = self.data.build_full_trainset()
        self.model = model
        self.alog = None
        
    
    
    def grid(self,param_grid,cv=3):
        raw_ratings = self.data.raw_ratings
        random.shuffle(raw_ratings)
        # A = 90% of the data, B = 10% of the data
        threshold = int(.9 * len(raw_ratings))
        A_raw_ratings = raw_ratings[:threshold]
        B_raw_ratings = raw_ratings[threshold:]
        self.data.raw_ratings = A_raw_ratings  # data is now the set A

        grid_search = GridSearchCV(NMF, param_grid, measures=['rmse'], cv=3)
        grid_search.fit(self.data)
        self.algo = grid_search.best_estimator['rmse']

        return self

    def fit(self):
        
        self.algo.fit(self.trainset)
        return self

    def similarity(self):
        sim_options = {'name': 'cosine', 'user_based': False}
        return self.algo.compute_similarities(sim_options)    

    def accuracy(self):
        predictions = self.algo.test(self.trainset.build_testset())
        
        return accuracy.rmse(predictions)




if __name__ == "__main__":
    # spark = SparkSession.builder.getOrCreate()
    pass
