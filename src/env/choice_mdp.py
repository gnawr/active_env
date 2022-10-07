import numpy as np
import time
import ast


from utils.environment import Environment
from utils.trajectory import Trajectory
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
		self.num_envs = len(self.envs)
		self.show = show
		self.eps = 1e-100

		# learners
		if self.control_idx == -1: # experiment
			self.learners = [DemoLearner(feat_list, env, constants, precompute=True) for env in self.envs]
		else:
			self.learners = [DemoLearner(feat_list, env, constants) for env in self.envs]

	def gen_envs(self, control_idx, show):
		"""Generate the environment options. """
		print control_idx, self.object_centers_dict

		if self.control_idx >= 0: # single env baselines
			return [Environment(self.model_filename, self.object_centers_dict[control_idx], show=False)]
		else: # random env baseline + experiment
			num_envs = len(self.object_centers_dict.keys())
			print "Generating environments..."
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
			if self.control_idx >= 0: # single env baseline
				return self.envs[0], 0, self.learners[0], None
			elif self.control_idx == -2:
				chosen_env_idx = np.random.randint(low=0, high=self.num_envs)
				return self.envs[chosen_env_idx], chosen_env_idx, self.learners[chosen_env_idx], None
		else:
			# TODO: handle info gain computation
			# Right now, just calculate for one env and measure how long it takes
			start = time.time()
			print("STARTING INFO GAIN CALCULATION")
			info_gains = []

			for env_idx in np.arange(self.num_envs):
			# for env_idx in np.arange(1):
				env = self.envs[env_idx]
				learner = self.learners[env_idx]
				num_trajs = len(learner.traj_rand.keys())
				info_gain = 0
				print "NUM_TRAJS: ", num_trajs		

				for traj_i, traj_str in enumerate(learner.traj_rand.keys()):
					curr_traj = np.array(ast.literal_eval(traj_str))
					# get P(xi | theta)
					# obs_model = learner.calc_obs_model([curr_traj])
					obs_model = learner.get_obs_model(traj_i)
					# TODO: USE TRAJ_I ^
					obs_model.reshape(learner.num_weights)

					# create the denominator
					new_belief = obs_model * P_bt
					denominator = np.sum(new_belief)

					# log term
					log_term = np.log((obs_model / denominator) + self.eps)

					# weigh log term by belief and likelihood
					weighted =  log_term * P_bt * obs_model
					summed = np.sum(weighted)
					info_gain += summed
					print 'OBS MODEL: ', obs_model
					print 'LOG TERM:  ', log_term
					print 'WEIGHTED:  ', weighted
					print 'SUMMED:    ', summed
					print 'info gain (should be float): ', info_gain

					# assert summed >= 0, 'Info gain must be positive'
					assert not np.isnan(info_gain), 'Info gain miscalculation' 

				info_gains.append(info_gain)

			best_env_idx = np.argmax(np.array(info_gains))
			print '\nINFO GAINS: ', info_gains


			end = time.time()
			print 'TIME FOR INFO GAIN CALC: ', end - start
			print '\n'

			return self.envs[best_env_idx], best_env_idx, self.learners[best_env_idx], info_gains

	def give_best_traj(self, env_idx, theta, num_demos=1, T=20.0):
		# Give the best trajectory from the denominator trajs under this theta
		env = self.envs[env_idx]
		learner = self.learners[env_idx]
		theta = np.array(theta) / np.linalg.norm(np.array(theta))

		rewards = np.dot(np.array(learner.Phi_rands), theta)
		# best_traj_idx = np.argmin(rewards)
		best_traj_idxs = rewards.argsort()[:num_demos]
		print 'BEST TRAJ INDICES: ', best_traj_idxs

		best_trajs = []

		for best_traj_idx in best_traj_idxs:

			print 'BEST TRAJ REWARD: ', rewards[best_traj_idx]
			print 'features for best traj: ', learner.Phi_rands[best_traj_idx]
			best_traj_str = learner.traj_rand.keys()[best_traj_idx]
			best_traj_waypts = np.array(ast.literal_eval(best_traj_str))
			print 'BEST TRAJ LENGTH: ', best_traj_waypts.shape


			waypts_time = np.linspace(0.0, T, best_traj_waypts.shape[0])
			traj = Trajectory(best_traj_waypts, waypts_time)
			best_trajs.append(traj)
		return best_trajs




