import pandas as pd
import numpy as np
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import torch
import numpy as np


def prepare_data_for_stgcn():
    import numpy as np


import pandas as pd
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


def load_pemsd7_data(window_size=12):
    """
    Load PeMSD7 traffic dataset into a StaticGraphTemporalSignal object.

    Args:
        window_size (int): Number of time steps to use as features before predicting next step
                        Default is 12 (representing 1 hour with 5-min intervals)

    Returns:
        StaticGraphTemporalSignal: Temporal graph data loader
    """
    # Load velocity data (speeds at sensor stations)
    velocity_df = pd.read_csv("dataset/PeMSD7_V_228.csv", header=None)
    velocity_matrix = velocity_df.values  # Shape: (288, 12672)

    # Load adjacency matrix (data structure of the graph)
    adj_df = pd.read_csv("dataset/PeMSD7_W_228.csv", header=None)
    adj_matrix = adj_df.values  # Shape: (288, 288)

    # Create edge_index and edge_weight from adjacency matrix
    edge_indices = []
    edge_weights = []

    # Convert adjacency matrix to edge_index and edge_weight format
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0:  # If there is an edge, otherwise skip
                edge_indices.append([i, j])
                edge_weights.append(adj_matrix[i, j])

    edge_index = np.array(edge_indices).T  # Shape: (2, num_edges)
    edge_weight = np.array(edge_weights)  # Shape: (num_edges,)

    # Create temporal features and targets
    num_nodes = velocity_matrix.shape[0]  # 288 nodes
    num_time_steps = velocity_matrix.shape[
        1
    ]  # 12672 time steps (5-min intervals representing 44 days in total)

    # We'll create sequences where we use window_size previous time steps as features
    # and predict the next time step
    features = []
    targets = []

    # For each valid time window
    for t in range(num_time_steps - window_size):
        # Features: window_size previous time steps
        feature_window = []
        for w in range(window_size):
            # Get node features at time t+w
            node_features = velocity_matrix[:, t + w].reshape(num_nodes, 1)
            feature_window.append(node_features)

        # Stack time steps to create node features with temporal dimension
        # Shape: (num_nodes, window_size)
        time_features = np.hstack(feature_window)
        features.append(time_features)

        # Target: next time step after the window
        target = velocity_matrix[:, t + window_size].reshape(num_nodes, 1)
        targets.append(target)

    # Convert to StaticGraphTemporalSignal
    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weight,
        features=features,
        targets=targets,
    )

    return dataset


def batch_to_df(batch_idx, dataset):
    """
    Converts a specific batch of features and targets from the dataset to DataFrames.

    Args:
        batch_idx (int): The index of the batch to convert.
        dataset (StaticGraphTemporalSignal): The dataset object.

    Returns:
        feature_df (pd.DataFrame): DataFrame containing the features of the batch.
        target_df (pd.DataFrame): DataFrame containing the targets of the batch.
    """

    # Get the feature sequence and target for a specific batch
    feature_seq = dataset[batch_idx][0].cpu().numpy()  # Features shape: [12, 228]
    target_seq = dataset[batch_idx][1].cpu().numpy()  # Target shape: [1, 228]

    # Create DataFrames
    feature_df = pd.DataFrame(
        feature_seq,
        columns=[f"Sensor_{i}" for i in range(228)],
        index=[f"T-{12-i}" for i in range(12)],
    )

    target_df = pd.DataFrame(
        target_seq, columns=[f"Sensor_{i}" for i in range(228)], index=["T+1"]
    )

    return feature_df, target_df


def save_example_batch(batch_idx, dataset):
    """
    Extracts a sample batch from the dataset and saves it to CSV files.

    Args:
        dataset (StaticGraphTemporalSignal): The dataset object.
        batch_idx (int): The index of the batch to extract.

    Returns:
        None
    """

    # Example: Extract and save sample 100
    feature_df, target_df = batch_to_df(batch_idx, dataset)

    # Save to CSV
    feature_df.to_csv("sample_100_features.csv")
    target_df.to_csv("sample_100_targets.csv")

    # Print summary statistics
    print("Feature statistics:")
    print(feature_df.describe())
    print("\nTarget statistics:")
    print(target_df.describe())
