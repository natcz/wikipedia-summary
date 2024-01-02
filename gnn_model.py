import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

def create_graph_similarity(df):
    # Assuming each document is a node and there's an edge between nodes based on similarity
    texts = df['lemmatized'].tolist()
    similarity_matrix = text_to_tfidf_similarity(texts)

    edges = []
    edge_weights = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            edges.append((i, j))
            edge_weights.append(similarity_matrix[i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    x = torch.arange(len(df), dtype=torch.float).view(-1, 1)  # Node features (just an example)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    return data

def create_graph_category(df):