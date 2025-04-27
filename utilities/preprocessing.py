import pandas as pd
import numpy as np
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import torch
import numpy as np


def prepare_data():
    """
    Prepares the data for the STGCN model.

    Loads the velocity and adjacency data, converts the adjacency matrix to edge_index and edge_weight format,
    and creates sequences of features and targets for training.

    Returns:
        dataset (StaticGraphTemporalSignal): The dataset object containing the graph structure and temporal features.
    """

    # Load the data
    # To do: Allow for dynamic data loading
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
    feature_seq = dataset[batch_idx][0].numpy()  # Features shape: [12, 228]
    target_seq = dataset[batch_idx][1].numpy()  # Target shape: [1, 228]

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
