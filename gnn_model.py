import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from constant import ARTICLES_DF, CATEGORIES_DF
from constant import STOPLIST as stoplist


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        return x


def create_graph_category():
    articles_categories_merged_df = pd.merge(ARTICLES_DF, CATEGORIES_DF, on = 'article_id').reset_index()
    articles_categories_merged_df['node_index'] = articles_categories_merged_df.groupby('title').cumcount()
    edges = []
    for _, group in articles_categories_merged_df.groupby('category'):
        node_indices = group['node_index'].dropna().astype(int).tolist()
        edges.extend([(i, j) for i in node_indices for j in node_indices])
    edges = list(set(edges))

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000, stop_words=stoplist)
    tfidf_matrix = tfidf_vectorizer.fit_transform(ARTICLES_DF["lemmatized"])
    x = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data