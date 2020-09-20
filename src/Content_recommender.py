from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class ContentRecommender():
    
    def __init__(self):
        self.similarity_matrix = None
        self.item_names = None
        self.similarity_measure = cosine_similarity
    
    def fit(self, X, items=None):
   
        if isinstance(X, pd.DataFrame):
            self.item_counts = X
            self.item_counts.reindex(items)
            self.item_names = items
            self.similarity_df = pd.DataFrame(self.similarity_measure(X.values, X.values),
                 index = self.item_names)
        else:
            self.item_counts = pd.DataFrame(X, index = items)
            self.similarity_df = pd.DataFrame(self.similarity_measure(X, X),
                 index = items)
            self.item_names = self.similarity_df.index
        return self.similarity_df  

        
    def get_recommendations(self, item, n=5):

        return self.item_names[self.similarity_df.loc[item].values.argsort()[-(n+1):-1]].values[::-1]


    def get_user_preference(self, items):

        user_profile = np.zeros(self.item_counts.shape[1])
        for item in items:
            user_profile += self.item_counts.loc[item].values

        return user_profile


    def get_user_recommendation(self, items, n=5):

        num_items = len(items)
        user_profile = self.get_user_preference(items)

        user_sim =  self.similarity_measure(self.item_counts, user_profile.reshape(1,-1))

        return self.item_names[user_sim[:,0].argsort()[-(num_items+n):-num_items]].values[::-1]
