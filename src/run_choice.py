#! /usr/bin/env python

import sys, os
import numpy as np
import time

from utils.openrave_utils import *
from env.choice_mdp import ChoiceMDP
from env.opt_human import OptHuman

"""
This class is meant to run a simulation of demonstration learning, 
where the agent can select the environment to receive feedback from.
"""

class RunChoice(object):

	def __init__(self, control_idx):
		"""Initialize parameters for this simulation."""
		#--- ARGUMENTS --- (TODO: yaml later)

		num_rounds = 10
		model_filename = "jaco_dynamics"
		# feat_list = ["table", "human", "laptop"]
		# feat_list = ["efficiency", "human", "laptop"]
		feat_list = ["human", "table", "laptop"]
		# feat_list = ["efficiency", "table", "laptop"]

		max_iter = 50
		num_waypts = 5
		start = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
		goal = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
		goal_pose = None
		T = 20.0
		timestep = 0.5 
		# object_centers_dict = {0: {'HUMAN_CENTER': [-0.6,-0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.1,0.0]},
		# 					   1: {'HUMAN_CENTER': [-0.6,0.55,0.0], 'LAPTOP_CENTER': [-0.4,-0.1,0.0]},
		# 					   # 2: {'HUMAN_CENTER': [-0.9,-0.55,0.0], 'LAPTOP_CENTER': [-1.0,-0.3,0.0]},
		# 					   # 3: {'HUMAN_CENTER': [-0.3, 0.55,0.0], 'LAPTOP_CENTER': [-1.0,-0.3,0.0]},
		# 					   }

		# Sanity check environments
		object_centers_dict = {
							   0: {'HUMAN_CENTER': [-0.4, 0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.15,0.0]},
							   1: {'HUMAN_CENTER': [-0.9, -0.55,0.0], 'LAPTOP_CENTER': [-1.0,-0.3,0.0]},
							   2: {'HUMAN_CENTER': [-0.3,-0.55,0.0], 'LAPTOP_CENTER': [-0.3,-0.3,0.0]},
							   3: {'HUMAN_CENTER': [-0.5, 0.55,0.0], 'LAPTOP_CENTER': [-0.6, 0.1,0.0]},
							   4: {'HUMAN_CENTER': [-0.4, 0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.15,0.0]},
							   5: {'HUMAN_CENTER': [-0.9, -0.55,0.0], 'LAPTOP_CENTER': [-1.0,-0.3,0.0]},
							   6: {'HUMAN_CENTER': [-0.3,-0.55,0.0], 'LAPTOP_CENTER': [-0.3,-0.3,0.0]},
							   7: {'HUMAN_CENTER': [-0.5, 0.55,0.0], 'LAPTOP_CENTER': [-0.6, 0.1,0.0]},
							   8: {'HUMAN_CENTER': [-0.4, 0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.15,0.0]},
							   9: {'HUMAN_CENTER': [-0.9, -0.55,0.0], 'LAPTOP_CENTER': [-1.0,-0.3,0.0]},
							   10: {'HUMAN_CENTER': [-0.3,-0.55,0.0], 'LAPTOP_CENTER': [-0.3,-0.3,0.0]},
							   11: {'HUMAN_CENTER': [-0.5, 0.55,0.0], 'LAPTOP_CENTER': [-0.6, 0.1,0.0]},
							   12: {'HUMAN_CENTER': [-0.4, 0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.15,0.0]},
							   13: {'HUMAN_CENTER': [-0.9, -0.55,0.0], 'LAPTOP_CENTER': [-1.0,-0.3,0.0]},
							   14: {'HUMAN_CENTER': [-0.3,-0.55,0.0], 'LAPTOP_CENTER': [-0.3,-0.3,0.0]},
							   15: {'HUMAN_CENTER': [-0.5, 0.55,0.0], 'LAPTOP_CENTER': [-0.6, 0.1,0.0]},
							   16: {'HUMAN_CENTER': [-0.4, 0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.15,0.0]},
							   17: {'HUMAN_CENTER': [-0.9, -0.55,0.0], 'LAPTOP_CENTER': [-1.0,-0.3,0.0]},
							   18: {'HUMAN_CENTER': [-0.3,-0.55,0.0], 'LAPTOP_CENTER': [-0.3,-0.3,0.0]},
							   19: {'HUMAN_CENTER': [-0.5, 0.55,0.0], 'LAPTOP_CENTER': [-0.6, 0.1,0.0]},
							   }

		constants = {'trajs_path': "/data/traj_sets/traj_rand_merged_H.p",
					 'betas_list': [10.0],
					 'weight_vals': [0.0, 0.5, 1.0], # Per feature theta options.
					 'FEAT_RANGE': {'table':1.0, 'coffee':1.0, 'laptop':1.6, 'human':1.6, 'efficiency':0.01},
					 }
		# feat_weights = [0.0, 1.0, 1.0]
		feat_weights = [1.0, 0.0, 1.0]

		#--- Initialize parameters ---#
		self.feat_weights = feat_weights
		self.constants = constants
		start = np.array(start) * math.pi / 180.0
		goal = np.array(goal) * math.pi / 180.0
		# convert to radians?
		self.num_rounds = num_rounds
		self.control_idx = control_idx
		self.is_control = control_idx != -1

		self.cmdp = ChoiceMDP(model_filename=model_filename, object_centers_dict=object_centers_dict, control_idx=control_idx, feat_list=feat_list, constants=constants)
		envs = self.cmdp.envs
		self.num_envs = len(envs)

		self.human = OptHuman(feat_list=feat_list, max_iter=max_iter, num_waypts=num_waypts, environments=envs, start=start, goal=goal, goal_pose=goal_pose, T=T, timestep=timestep, weights=feat_weights, seed=None)
		self.P_bt = self.initial_belief()

	def initial_belief(self):			
		return self.cmdp.learners[0].P_bt


	def run(self):
		"""Run the simulation."""
		beliefs = [self.P_bt]
		envs_chosen = []
		info_gain_options = []
		start = time.time()
		for i in np.arange(self.num_rounds):
			print('ITERATION ', i)

			env, env_idx, learner, info_gains = self.cmdp.choose_env(self.P_bt)

			# # Use TrajOpt for giving a demonstration
			# xi_d = self.human.give_demo(env_idx)

			# Use the best trajectory from the denominator as the demonstration
			xi_d = self.cmdp.give_best_traj(env_idx, theta=self.feat_weights, num_demos=5)

			# if self.is_control:
			for traj in xi_d:
				color = np.random.rand(3)
				plotTraj(env.env, env.robot, env.bodies, traj.waypts, size=0.015, color=color)
			plotCupTraj(env.env, env.robot, env.bodies, [xi_d[0].waypts[-1]], color=[0,1,0])

			new_belief = learner.learn_posterior(trajs=xi_d, P_bt=self.P_bt)
			self.P_bt = new_belief
			beliefs.append(self.P_bt)
			envs_chosen.append(env_idx)
			info_gain_options.append(info_gains)

		beliefs = np.array(beliefs).reshape((self.num_rounds + 1, 19))
		print 'BELIEFS OVER TIME: ', beliefs
		if not self.is_control:
			print 'ENVS CHOSEN: ', envs_chosen

		if info_gains: print 'INFO GAIN OPTIONS :', info_gain_options

		if self.is_control:
			title_suffix = 'control, ENV {}'.format(str(self.control_idx))
		else:
			title_suffix = 'experiment, 4 choices'

		# Saving time taken
		end = time.time()
		print 'TIME FOR 10 ITERATIONS: ', end - start
		time_taken = end-start
		file_path = os.path.join(os.getcwd(), 'data/time_0913.txt')
		with open(file_path, 'w') as f:
			f.write(str(time_taken))


		file_path = os.path.join(os.getcwd(), 'data/exp_0914_metadata.npz')
		np.savez(file_path, envs_chosen=np.array(envs_chosen), info_gain_options=np.array(info_gain_options), beliefs=beliefs)

		

		learner.visualize_stacked_posterior(beliefs, title=title_suffix, save='')

		# learner.visualize_stacked_posterior(beliefs, title=title_suffix, save='data/exp_0909_5demo')

		



if __name__ == "__main__":
	# # #--- Run controls --- #
	# control_idx = 0

	# simulation = RunChoice(control_idx=control_idx)
	# simulation.run()


	#--- Run experiment --- #
	simulation = RunChoice(control_idx=-1)
	simulation.run()

	# #--- Print metadata --- #
	# path = '../data/exp_cost_0908_metadata.npz'
	# data = np.load(path)

	# print path
	# print 'ENVS CHOSEN', data['envs_chosen']
	# print 'INFO GAINS', data['info_gain_options']

