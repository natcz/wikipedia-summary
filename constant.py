import nltk
import pandas as pd
from nltk.corpus import stopwords

STOPLIST = stopwords.words('polish')
ARTICLES_DF = pd.read_csv('data/articles_limited.csv')
CATEGORIES_DF = pd.read_csv('data/categories.csv')
