import torch
from torch_geometric.data import  DataLoader

from gnn_model import  create_graph_category, GNNModel


def compute_embeddings_graph(model, data_loader):
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            output = model(data)
    return output

data_graph = create_graph_category()

data_loader = DataLoader([data_graph], batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = data_graph.num_node_features
hidden_dim = 64
model = GNNModel(input_dim, hidden_dim)

model.train()

model.eval()
with torch.no_grad():
    for data in data_loader:
        output = model(data)
print(output)