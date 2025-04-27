import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric_temporal.nn.attention import STGCN
from torch_geometric.nn import ChebConv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
velocity_df = pd.read_csv("PeMSD7_V_228.csv")
adjacency_df = pd.read_csv("PeMSD7_W_228.csv")

# Convert adjacency matrix to edge_index and edge_weight format for PyG
adjacency_matrix = adjacency_df.values
edge_index = []
edge_weight = []

# Convert adjacency matrix to COO format
for i in range(adjacency_matrix.shape[0]):
    for j in range(adjacency_matrix.shape[1]):
        if adjacency_matrix[i, j] > 0:  # If there's a connection
            edge_index.append([i, j])
            edge_weight.append(adjacency_matrix[i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight, dtype=torch.float)

# Prepare feature sequences (assuming we want to use velocity as the feature)
features = velocity_df.values
num_timesteps = features.shape[0]
num_nodes = features.shape[1]

# Create sequences for STGCN (using history to predict future)
sequence_length = 12  # Example: using 1-hour data (12 * 5min) to predict
target_offset = 1  # Example: predicting the next time step

# Prepare features and targets
feature_sequences = []
target_sequences = []

for i in range(num_timesteps - sequence_length - target_offset + 1):
    # Sequence of features for each node
    feature_sequences.append(features[i : i + sequence_length])
    # Target values for each node
    target_sequences.append(
        features[i + sequence_length : i + sequence_length + target_offset]
    )

# Convert to tensors
feature_sequences = torch.FloatTensor(np.array(feature_sequences))
target_sequences = torch.FloatTensor(np.array(target_sequences))

# Create a dataset object
dataset = StaticGraphTemporalSignal(
    edge_index=edge_index,
    edge_weight=edge_weight,
    features=feature_sequences,
    targets=target_sequences,
)
