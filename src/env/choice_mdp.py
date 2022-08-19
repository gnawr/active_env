import numpy as np
import time
import ast


from utils.environment import Environment
from learners.demo_learner import DemoLearner

"""Class that keeps track of all environments that can be used."""

class ChoiceMDP(object):

	def __init__(self, model_filename, object_centers_dict, control_idx, feat_list, constants, show=True):
		"""
		- control_idx [int]: the index of the environment to be used for learning. 
							 if not control, this is -1
		"""
		# generate environments to be chosen between. if control, only generate one
		self.model_filename = model_filename
		self.object_centers_dict = object_centers_dict
		self.is_control = control_idx != -1
		self.control_idx = control_idx
		self.envs = self.gen_envs(control_idx, show)
		self.show = show

		# learners
		self.learners = [DemoLearner(feat_list, env, constants) for env in self.envs]

	def gen_envs(self, control_idx, show):
		"""Generate the environment options. """
		if self.is_control:
			return [Environment(self.model_filename, self.object_centers_dict[control_idx])]
		else:
			num_envs = len(self.object_centers_dict.keys())
			return [Environment(self.model_filename, self.object_centers_dict[i], show=False) for i in np.arange(num_envs)]


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
			# Right now, just calculate for one env and measure how long it takes
			start = time.time()
			print("STARTING INFO GAIN CALCULATION")

			best_env_idx = -1
			info_gains = []

			# for env_idx in np.arange(self.num_envs):
			for env_idx in np.arange(1):
				env = self.envs[env_idx]
				learner = self.learners[env_idx]
				num_trajs = len(learner.traj_rand.keys())
				info_gain = 0
				print "NUM_TRAJS: ", num_trajs		

				for traj_i, traj_str in enumerate(learner.traj_rand.keys()):
					curr_traj = np.array(ast.literal_eval(traj_str))
					# get P(xi | theta)
					obs_model = learner.calc_obs_model(curr_traj)
					obs_model.reshape(learner.num_weights)

					# create the denominator
					new_belief = obs_model * self.P_bt
					denominator = np.sum(new_belief)

					# log term
					log_term = np.log(obs_model / denominator)

					# weigh log term by belief and likelihood
					weighted =  log_term * self.P_bt * obs_model
					info_gain += np.sum(weighted)

					if traj_i < 3:
						print '\nNext three should be the same..'
						print 'OBS MODEL SHAPE: ', obs_model.shape
						print 'LOG TERM SHAPE:  ', log_term.shape
						print 'WEIGHTED SHAPE:  ', weighted.shape

				info_gains.append(info_gain)

			best_env_idx = np.argmax(np.array(info_gain))
			print '\nINFO GAINS: ', info_gains

			end = time.time()
			print 'TIME FOR INFO GAIN CALC: ', end - start


			return self.envs[best_env_idx], best_env_idx, self.learners[best_env_idx]

