from utils.environment import Environment

def generate_env_set(feat_list, num_envs=20, set_size=10):
	"""
	Generate a set of environments by randomly sampling a set of them and taking the pair that maximize KL divergence, then repeating.
	"""
	# TODO: FILL OUT
	final_env_set = {}
	while len(final_env_set.keys()) < num_envs:
		# generate env object centers, size set_size

		# generate the environments

		# Generate


def generate_random_env_set(feat_list, set_size):
	"""Generate a random environment given the features in the feature list."""
	human_regions = {'top side left': [[-0.2, -0.55, 0.0], [-0.7, -0.55, 0.0]],
					 'top side right': [[-0.7, -0.55, 0.0], [-1.3, -0.55, 0.0]],
					 'right side top': [[-1.3, -0.55, 0.0], [-1.3, 0.0, 0.0]],
					 'right side bottom': [[-1.3, 0.0, 0.0], [-1.3, 0.55, 0.0]],
					 'bottom side left': [[-0.2, 0.55, 0.0], [-0.7, 0.55, 0.0]],
					 'bottom side right': [[-0.7, 0.55, 0.0], [-1.3, 0.55, 0.0]]
					 }

	laptop_regions = {'top left quadrant': [[-1.05, -0.1, 0.0], [-0.675, -0.3, 0.0]],
					  'top right quadrant': [[-0.675, -0.1, 0.0], [-1.05, -0.3, 0.0]],
					  'bottom left quadrant': [[-0.3, 0.1, 0.0], [-0.675, -0.1, 0.0]],
					  'bottom right quadrant': [[-0.675, 0.1, 0.0], [-1.05, -0.1, 0.0]]
					  }