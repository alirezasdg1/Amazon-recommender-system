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

    def predictions(self):
        self.predictions = self.algo.test(self.trainset.build_testset())
        acc = accuracy.rmse(self.predictions)
        
        return self.predictions, acc
    
    def get_Iu(self,uid):
        """ return the number of items rated by given user
        args: 
        uid: the id of the user
        returns: 
        the number of items rated by the user
        """
        try:
            return len(self.trainset.ur[self.trainset.to_inner_uid(uid)])
        except ValueError: # user was not part of the trainset
            return 0
    
    def get_Ui(self,iid):
        """ return number of users that have rated given item
        args:
        iid: the raw id of the item
        returns:
        the number of users that have rated the item.
        """
        try: 
            return len(self.trainset.ir[self.trainset.to_inner_iid(iid)])
        except ValueError:
            return 0
    
    # def get_top_n(predictions, n=10):
    # """Return the top-N recommendation for each user from a set of predictions.

    # Args:
    #     predictions(list of Prediction objects): The list of predictions, as
    #         returned by the test method of an algorithm.
    #     n(int): The number of recommendation to output for each user. Default
    #         is 10.

    # Returns:
    # A dict where keys are user (raw) ids and values are lists of tuples:
    #     [(raw item id, rating estimation), ...] of size n.
    # """

    # # First map the predictions to each user.
    # top_n = defaultdict(list)
    # for uid, iid, true_r, est, _ in predictions:
    #     top_n[uid].append((iid, est))

    # # Then sort the predictions for each user and retrieve the k highest ones.
    # for uid, user_ratings in top_n.items():
    #     user_ratings.sort(key=lambda x: x[1], reverse=True)
    #     top_n[uid] = user_ratings[:n]

    # return top_n




if __name__ == "__main__":
    # spark = SparkSession.builder.getOrCreate()
    pass
