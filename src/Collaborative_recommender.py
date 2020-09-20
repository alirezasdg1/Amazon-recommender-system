import pandas as pd
import numpy as np
import sys
from io import StringIO
import surprise
from sklearn.model_selection import train_test_split, GridSearchCV
from collections import defaultdict
import random

from surprise import SVD, NMF, KNNBaseline
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import GridSearchCV

class CollaborativeRecommender():
    
    def __init__(self,df,reader,model):
        self.data = surprise.Dataset.load_from_df(df,reader)
        self.trainset = self.data.build_full_trainset()
        self.algo = None
        
    
    
    def grid(self,param_grid,cv=3):

        sim_options = {'name': 'pearson_baseline', 'user_based': False}
        raw_ratings = self.data.raw_ratings
        random.shuffle(raw_ratings)
        # A = 90% of the data, B = 10% of the data
        threshold = int(.9 * len(raw_ratings))
        A_raw_ratings = raw_ratings[:threshold]
        B_raw_ratings = raw_ratings[threshold:]
        self.data.raw_ratings = A_raw_ratings  # data is now the set A

        grid_search = GridSearchCV(KNNBaseline(sim_options=sim_options), param_grid, measures=['rmse'], cv=3)
        grid_search.fit(self.data)
        self.algo = grid_search.best_estimator['rmse']

        return self

    def fit(self):
        self.algo.fit(self.trainset)
   

    def pred(self):
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
    
    def get_top_n(self,n=10):
        """Return the top-N recommendation for each user from a set of predictions.
        """
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n



if __name__ == "__main__":
    
    pass
