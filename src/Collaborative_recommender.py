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
        self.df_pred = pd.DataFrame.from_dict(self.predictions)
        self.df_pred['err'] = abs(self.df_pred.est - self.df_pred.r_ui)
        acc = accuracy.rmse(self.predictions)
        
        return self.df_pred, acc

    def utility_matrix(self):
        self.um_model = self.df_pred.pivot_table(index='uid',columns='iid', values='est')
        self.um_err = self.df_pred.pivot_table(index='uid',columns='iid', values='err')

        return self.um_model
        

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

            
    
    def get_top_n(self,UI,n=10):
        recommended_items = pd.DataFrame(self.um_model.loc[UI])
        recommended_items.columns = ["predicted_rating"]
        top_n = recommended_items.sort_values('predicted_rating', ascending=False).head(n)

        return top_n

if __name__ == "__main__":
    
    pass
