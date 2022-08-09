#! /usr/bin/env python

import numpy as np

from choice_mdp import ChoiceMDP
from opt_human import OptHuman

"""
This class is meant to run a simulation of demonstration learning, 
where the agent can select the environment to receive feedback from.
"""

class RunChoice(object):

	def __init__(self, control_idx):
		"""Initialize parameters for this simulation."""
		#--- ARGUMENTS --- (TODO: yaml later)
		num_rounds = 10
		feat_list = ["table", "laptop"]
		max_iter = 50
		num_waypts = 5
		start = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
		goal = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
		goal_pose = None
		T = 20.0
		timestep = 0.5 
		seed = 0

		#--- Initialize parameters ---
		self.num_rounds = num_rounds

		self.cmdp = ChoiceMDP()
		environments = self.cmdp.environments
		self.num_environments = len(environments)

		self.human = OptHuman(feat_list=feat_list, max_iter=max_iter, num_waypts=num_waypts, environments=environments, start=start, goal=goal, goal_pose=goal_pose, T=T, timestep=timestep, seed=seed)
		self.P_bt = self.initial_belief()


	def initial_belief():
		...
		return 


	def run():
		"""Run the simulation."""
		for _ in np.arange(self.num_rounds):
			env, learner = cmdp.choose_envs(...)
			xi_d = human.give_demo(...)
			new_belief = learner.learn_posterior(...)


	def save_output():
		"""Save needed metrics"""
		...

if __name__ == "main":
	control_idx = 0
	simulation = RunChoice(control_idx=control_idx)
	...