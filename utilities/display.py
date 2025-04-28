# Author: Reef Lakin
# Last Modified: 28.04.2025
# Description: Contains functions for displaying STGCN data in a very human-friendly way.

from torch_geometric_temporal.signal import StaticGraphTemporalSignal

def get_target_velocity(dataset: StaticGraphTemporalSignal, time_index, node_index):
	"""
	Get the target velocity for a specific time index and node index.

	Args:
		dataset: The dataset in a StaticGraphTemporalSignal format.
		time_index: The time index for which to get the target velocity.
		node_index: The node index (i.e., station) for which to get the target velocity.

	Returns:
		The target velocity for the specified time and node index.
	"""
	
	targets = dataset._get_target(time_index)
	return targets