
"""Class that keeps track of all environments that can be used."""

class ChoiceMDP(object):

	def __init__(self, control_idx):
		"""
		- control_idx [int]: the index of the environment to be used for learning. 
							 if not control, this is -1
		"""
		# generate environments to be chosen between
		self.envs = self.gen_envs()
		self.is_control = control_idx != -1
		self.control_idx = control_idx

		# learners

	def gen_envs(self):
		"""Generate the environment options. """

	def choose_envs(self, P_bt):
		"""Using info gain metric, choose an environment from the options to learn from.
		inputs:
			- P_bt: current belief
		outputs:
			- chosen_env: env with highest info gain
			- learner: learner for the chosen env
		"""
		return chosen_env, learner

