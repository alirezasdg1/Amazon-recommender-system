
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def nlp_process(df,max_features=None):

    vectorizer = TfidfVectorizer(min_df=0.005, stop_words='english', max_features = max_features)
    m_nlp = vectorizer.fit_transform(df).toarray()
    df_nlp = pd.DataFrame(m_nlp, columns=vectorizer.get_feature_names())
    
    return vectorizer, df_nlp