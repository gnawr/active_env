import numpy as np
import pickle
import os

from utils.environment import Environment

def generate_kl_env_set(feat_list, num_envs=20, set_size=10):
	"""
	Generate a set of environments by randomly sampling a set of them 
	and taking the pair that maximize KL divergence, then repeating.
	"""
	assert 'human' in feat_list and 'laptop' in feat_list, 'Feature list incompatible, does not have either human or laptop'
	# TODO: FILL OUT
	final_env_set = {}
	while len(final_env_set.keys()) < num_envs:
		pass
		# generate env object centers, size set_size

		# generate the environments

		# Generate


def generate_random_env_set(feat_list, set_size):
	"""Generate a random environment set given the features in the feature list.
	Regions are defined by bottom left and top right coordinates.
	Returns:
		- object_centers_dict [dict]: a dictionary that contains index:object_centers pairs 
			that can be used to generate a RunChoice instance
	"""
	assert 'human' in feat_list and 'laptop' in feat_list, 'Feature list incompatible, does not have either human or laptop'

	human_regions = [[[-0.2, -0.55, 0.0], [-0.7, -0.55, 0.0]], # top side left
					 [[-0.7, -0.55, 0.0], [-1.3, -0.55, 0.0]], # top side right
					 [[-1.3, 0.0, 0.0], [-1.3, -0.55, 0.0]], # right side top
					 [[-1.3, 0.55, 0.0], [-1.3, 0.0, 0.0]], # right side bottom
					 [[-0.2, 0.55, 0.0], [-0.7, 0.55, 0.0]], # bottom side left
					 [[-0.7, 0.55, 0.0], [-1.3, 0.55, 0.0]] # bottom side right
					 ]

	laptop_regions = [[[-1.05, -0.1, 0.0], [-0.675, -0.3, 0.0]], # top left quadrant
					  [[-0.675, -0.1, 0.0], [-1.05, -0.3, 0.0]], # top right quadrant
					  [[-0.3, 0.1, 0.0], [-0.675, -0.1, 0.0]], # bottom left quadrant
					  [[-0.675, 0.1, 0.0], [-1.05, -0.1, 0.0]], # bottom right quadrant 
					  ]
	num_human_regions = len(human_regions)
	num_laptop_regions = len(laptop_regions)

	# Sample the parametrized objects
	human_object_centers = sample_centers(human_regions)
	laptop_object_centers = sample_centers(laptop_regions)

	# sample from the set of object locations to create a list of object combinations
	env_set_indices = []
	while len(env_set_indices) < set_size:
		# Sample from the different object sets
		human_idx = np.random.randint(low=0, high=num_human_regions)
		laptop_idx = np.random.randint(low=0, high=num_laptop_regions)
		object_indices = [human_idx, laptop_idx]
		# Skip duplicates
		if object_indices not in env_set_indices:
			env_set_indices.append(object_indices)
	
	# generate the object centers dict from `env_set_indices` (translating from indices to coordinates)
	object_centers_dict = {}
	for env_idx, object_indices in enumerate(env_set_indices):
		object_centers = {'HUMAN_CENTER': human_object_centers[object_indices[0]],
						  'LAPTOP_CENTER': laptop_object_centers[object_indices[1]]}
		object_centers_dict[env_idx] = object_centers
	
	here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
	savefile = "/data/env_sets/env_set_human_laptop_table.p"
	pickle.dump(object_centers_dict, open( here + savefile, "wb" ))
	print "Saved in: ", savefile

	return object_centers_dict



def sample_centers(regions):
	"""Given a regions list, sample an object location within each region.
	Params:
		- regions [list]: a list of the bounds for an object
	Returns:
		- centers [list]: a list of the same length as regions, where every entry 
			is a uniformly sampled coordinate point within the corresponding region
	"""
	centers = []
	for region in regions:
		bot_left_corner = region[0]
		top_right_corner = region[1]
		object_center = []
		for coord_idx in np.arange(3):
			# sample a coordinate within the range of the region bounds
			bounds = [bot_left_corner[coord_idx], top_right_corner[coord_idx]]
			bounds.sort()
			lower_bound, upper_bound = bounds
			sampled_coord = np.random.uniform(lower_bound, upper_bound)
			object_center.append(sampled_coord)
		centers.append(object_center)
	return centers


if __name__ == '__main__':
	feat_list = ["human", "table", "laptop"]
	set_size = 20
	generate_random_env_set(feat_list, set_size)