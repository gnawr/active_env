#! /usr/bin/env python

import sys, os
import numpy as np

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
		feat_list = ["efficiency", "table", "laptop"]
		max_iter = 50
		num_waypts = 5
		start = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
		goal = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
		goal_pose = None
		T = 20.0
		timestep = 0.5 
		object_centers_dict = {0: {'HUMAN_CENTER': [-0.6,-0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.1,0.0]},
							   1: {'HUMAN_CENTER': [-0.6,0.55,0.0], 'LAPTOP_CENTER': [-0.4,-0.1,0.0]}
							   }

		constants = {'trajs_path': "/data/traj_sets/traj_rand_merged_H.p",
					 'betas_list': [100.0],
					 'weight_vals': [0.0, 0.5, 1.0], # Per feature theta options.
					 'FEAT_RANGE': {'table':1.0, 'coffee':1.0, 'laptop':1.6, 'human':1.6, 'efficiency':0.01},
					 }
		# feat_weights = [1.0, 1.0, 0.0]
		feat_weights = [1.0, 1.0, 0.0]

		#--- Initialize parameters ---#
		start = np.array(start) * math.pi / 180.0
		goal = np.array(goal) * math.pi / 180.0
		# convert to radians?
		self.num_rounds = num_rounds
		self.control_idx = control_idx

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
		for i in np.arange(self.num_rounds):
			print('ITERATION ', i)

			env, env_idx, learner = self.cmdp.choose_env(self.P_bt)
			xi_d = self.human.give_demo(env_idx)

			plotTraj(env.env, env.robot, env.bodies, xi_d[0].waypts, size=0.015,color=[0, 0, 1])
			plotCupTraj(env.env, env.robot, env.bodies, [xi_d[0].waypts[-1]], color=[0,1,0])

			new_belief = learner.learn_posterior(trajs=xi_d, P_bt=self.P_bt)
			self.P_bt = new_belief
			beliefs.append(self.P_bt)

		beliefs = np.array(beliefs).reshape((self.num_rounds + 1, 19))
		print 'BELIEFS OVER TIME: ', beliefs
		title_suffix = 'control, ENV {}'.format(str(self.control_idx))
		learner.visualize_stacked_posterior(beliefs, title=title_suffix)



if __name__ == "__main__":

	control_idx = 0
	simulation = RunChoice(control_idx=control_idx)
	simulation.run()