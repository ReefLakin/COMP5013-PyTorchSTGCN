from utilities.preprocessing import load_dataset_for_stgcn
from utilities.display import get_target_velocities, get_target_velocity_for_station

# Get a dataset
dataset = load_dataset_for_stgcn(window_size=12)

# Get the target velocity for a specific time index and node index
time_index = 0
node_index = 0

# Get the target velocity
target_velocity = get_target_velocity_for_station(dataset, time_index, node_index)

print(target_velocity)