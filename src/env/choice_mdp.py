from utils.environment import Environment
from learners.demo_learner import DemoLearner

"""Class that keeps track of all environments that can be used."""

class ChoiceMDP(object):

	def __init__(self, model_filename, object_centers_dict, control_idx, feat_list, constants):
		"""
		- control_idx [int]: the index of the environment to be used for learning. 
							 if not control, this is -1
		"""
		# generate environments to be chosen between. if control, only generate one
		self.model_filename = model_filename
		self.object_centers_dict = object_centers_dict
		self.is_control = control_idx != -1
		self.control_idx = control_idx
		self.envs = self.gen_envs(control_idx)

		# learners
		self.learners = [DemoLearner(feat_list, env, constants) for env in self.envs]

	def gen_envs(self, control_idx):
		"""Generate the environment options. """
		if self.is_control:
			return [Environment(self.model_filename, self.object_centers_dict[control_idx])]
		else:
			num_envs = len(self.object_centers_dict.keys())
			return [Environment(self.model_filename, self.object_centers_dict[i]) for i in np.arange(num_envs)]


	def choose_env(self, P_bt):
		"""Using info gain metric, choose an environment from the options to learn from.
		inputs:
			- P_bt: current belief
		outputs:
			- chosen_env: env with highest info gain
			- chosen_env_idx: index of previous env
			- learner: learner for the chosen env
		"""
		if self.is_control:
			return self.envs[0], 0, self.learners[0]
		else:
			# TODO: handle info gain computation
			# ...
			return chosen_env, learner

