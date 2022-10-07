#! /usr/bin/env python

import sys, os

import numpy as np 
import yaml
import unittest
import ast

from utils.openrave_utils import *
from utils.environment import Environment
from utils.generate_env_set import *
from demo_inference import DemoInference
from catkin.find_in_workspaces import find_in_workspaces
from learners.demo_learner import DemoLearner
from planners.trajopt_planner import TrajoptPlanner
from env.choice_mdp	import ChoiceMDP
from env.opt_human import OptHuman

import time


class DevTest(unittest.TestCase):



	def test_env_gen(self):
		model_filename = "jaco_dynamics"
		
		# object_centers_dict = {
		# 					   # 0: {'HUMAN_CENTER': [-0.4, 0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.15,0.0]},
		# 					   1: {'HUMAN_CENTER': [-0.7, -0.55,0.0], 'LAPTOP_CENTER': [-1.05, -0.1, 0.0]},
		# 					   # 2: {'HUMAN_CENTER': [-0.3,-0.55,0.0], 'LAPTOP_CENTER': [-0.3,-0.3,0.0]},
		# 					   # 3: {'HUMAN_CENTER': [-0.5, 0.55,0.0], 'LAPTOP_CENTER': [-0.6, 0.1,0.0]},
		# 					   }

		object_centers_path = "/data/env_sets/env_set_human_laptop_table.p"
		# Load in object centers
		here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../'))
		object_centers_dict = pickle.load(open(here + object_centers_path, "rb"))


		for object_centers in object_centers_dict.values():
			env = Environment(model_filename, object_centers)
			
			# #sphere
			# objects_path = '../data'
			
			# env.env.Load('{:s}/sphere.xml'.format(objects_path))
			# mug = env.env.GetKinBody('sphere')
			# body = mug
			# body.SetName("pt"+str(len(env.bodies)))
			# env.env.Add(body, True)
			# env.bodies.append(body)

			
			raw_input("Press Enter to continue...")
			env.kill_environment()

	def viz_positions(self):
		model_filename = 'jaco_dynamics'
		human_center = [-0.6,0.55,0.0]
		laptop_center = [-0.7929,-0.1,0.0]
		object_centers = {'HUMAN_CENTER': human_center, 'LAPTOP_CENTER': laptop_center}
		
		env = Environment(model_filename, object_centers)
		start = np.array([104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 225.0, 0., 0., 0.])
		goal = np.array([210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0, 0., 0., 0.])
		candlestick = np.array([180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 0., 0., 0.])

		env.robot.SetDOFValues(candlestick * (math.pi / 180.0))
		ee_xyz = manipToCartesian(env.robot, 0.0)
		print('Cartesian of manip:', ee_xyz)
		raw_input("Visualizing CANDLESTICK. Press Enter to continue...")

		dof_radians = start * (math.pi / 180.0)
		dof_radians[2] += math.pi
		env.robot.SetDOFValues(dof_radians)
		ee_xyz = manipToCartesian(env.robot, 0.0)
		print('Cartesian of manip:', ee_xyz)
		raw_input("Visualizing START. Press Enter to continue...")

		dof_radians = goal * (math.pi / 180.0)
		dof_radians[2] += math.pi
		env.robot.SetDOFValues(dof_radians)
		ee_xyz = manipToCartesian(env.robot, 0.0)
		print('Cartesian of manip:', ee_xyz)
		raw_input("Visualizing GOAL. Press Enter to continue...")

		env.kill_environment

	def test_all_trajs(self):
		model_filename = 'jaco_dynamics'
		human_center = [-0.6,0.55,0.0]
		laptop_center = [-0.7929,-0.1,0.0]
		object_centers = {'HUMAN_CENTER': human_center, 'LAPTOP_CENTER': laptop_center}
		
		env = Environment(model_filename, object_centers)

		# Trajectory paths.
		here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../'))
		self.traj_rand = pickle.load(open(here + "/data/traj_sets/traj_rand_merged_H.p", "rb" ))

		# Visualize each trajectory
		for rand_i, traj_str in enumerate(self.traj_rand.keys()):
			curr_traj = np.array(ast.literal_eval(traj_str))
			
			color = np.random.rand(3)
			plotTraj(env.env, env.robot, env.bodies, curr_traj, size=0.01,color=color)
		raw_input("Visualizing all trajectories. Press Enter to continue...")


	def test_demos(self):
		num_rounds = 10
		model_filename = "jaco_dynamics"
		feat_list = ["human", "table", "laptop"]

		max_iter = 50
		num_waypts = 5
		start = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
		goal = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]
		start = np.array(start) * math.pi / 180.0
		goal = np.array(goal) * math.pi / 180.0
		goal_pose = None
		T = 20.0
		timestep = 0.5 
		# Sanity check environments
		object_centers_dict = {
							   0: {'HUMAN_CENTER': [-0.4, 0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.15,0.0]},
							   1: {'HUMAN_CENTER': [-0.9, -0.55,0.0], 'LAPTOP_CENTER': [-1.0,-0.3,0.0]},
							   2: {'HUMAN_CENTER': [-0.3,-0.55,0.0], 'LAPTOP_CENTER': [-0.3,-0.3, 0.0]},
							   3: {'HUMAN_CENTER': [-0.5, 0.55,0.0], 'LAPTOP_CENTER': [-0.6, 0.1,0.0]},
							   }

		constants = {'trajs_path': "/data/traj_sets/traj_rand_merged_H.p",
					 'betas_list': [10.0],
					 'weight_vals': [0.0, 0.5, 1.0], # Per feature theta options.
					 'FEAT_RANGE': {'table':1.0, 'coffee':1.0, 'laptop':1.6, 'human':1.6, 'efficiency':0.01},
					 }
		feat_weights = [1.0, 0.0, 1.0]

		# EDIT THIS
		env_idx = 3

		cmdp = ChoiceMDP(model_filename=model_filename, object_centers_dict=object_centers_dict, control_idx=env_idx, feat_list=feat_list, constants=constants)
		human = OptHuman(feat_list=feat_list, max_iter=max_iter, num_waypts=num_waypts, environments=cmdp.envs, start=start, goal=goal, goal_pose=goal_pose, T=T, timestep=timestep, weights=feat_weights, seed=None)

		env = cmdp.envs[0]

		# plot bump distance spheres
		objects_path = '../data'
		env.env.Load('{:s}/sphere.xml'.format(objects_path))
		mug = env.env.GetKinBody('sphere')
		body = mug
		body.SetName("pt"+str(len(env.bodies)))
		env.env.Add(body, True)
		env.bodies.append(body)

		# Best demonstration
		best_denom_demo = cmdp.give_best_traj(0, theta=feat_weights)
		plotTraj(env.env, env.robot, env.bodies, best_denom_demo[0].waypts, size=0.015, color=[0, 0, 1])

		# Trajopt demonstration
		learner = cmdp.learners[0]
		traj_opt_demo = human.give_demo(0)
		traj_opt_features = np.array(env.featurize(traj_opt_demo[0].waypts, feat_list))
		traj_opt_features = np.sum(traj_opt_features, axis=1)
		traj_opt_features = (traj_opt_features - learner.min_features) / (learner.max_features - learner.min_features)
		theta = np.array(feat_weights) / np.linalg.norm(np.array(feat_weights))
		print 'trajopt traj feats: ', traj_opt_features
		trajopt_reward = np.dot(traj_opt_features, theta)
		print 'TRAJOPT TRAJ REWARD: ', trajopt_reward
		print 'TRAJOPT TRAJ LENGTH: ', traj_opt_demo[0].waypts.shape

		plotTraj(env.env, env.robot, env.bodies, traj_opt_demo[0].waypts, size=0.015,color=[1, 0, 0])


		raw_input("Visualizing best traj in blue, trajopt in red. Press Enter to continue...")




	def test_feat_range(self):

		model_filename = 'jaco_dynamics'
		human_center = [-0.6,0.55,0.0]
		laptop_center = [-0.7929,-0.1,0.0]
		object_centers = {'HUMAN_CENTER': human_center, 'LAPTOP_CENTER': laptop_center}
		feat_list = ["efficiency", "human", "laptop"]
		constants = {'trajs_path': "/data/traj_sets/traj_rand_merged_H.p",
					 'betas_list': [10.0],
					 'weight_vals': [0.0, 0.5, 1.0], # Per feature theta options.
					 'FEAT_RANGE': {'table':0.98, 'coffee':1.0, 'laptop':0.4, 'human':0.4, 'efficiency':0.22},
					 }
		FEAT_RANGE = {'table':0.98, 'coffee':1.0, 'laptop':0.4, 'human':0.4, 'efficiency':0.22}
		
		env = Environment(model_filename, object_centers)
		learner = DemoLearner(feat_list, env, constants)

		features = np.array(learner.Phi_rands)
		print 'FEAT MAX: ', np.max(features, axis=0)
		print 'FEAT MIN: ', np.min(features, axis=0)

	def test_obs_model(self):
		model_filename = 'jaco_dynamics'
		human_center = [-0.6,0.55,0.0]
		laptop_center = [-0.7929,-0.1,0.0]
		object_centers = {'HUMAN_CENTER': human_center, 'LAPTOP_CENTER': laptop_center}
		feat_list = ["efficiency", "human", "laptop"]
		constants = {'trajs_path': "/data/traj_sets/traj_rand_merged_H.p",
					 'betas_list': [10.0],
					 'weight_vals': [0.0, 0.5, 1.0], # Per feature theta options.
					 'FEAT_RANGE': {'table':0.98, 'coffee':1.0, 'laptop':0.4, 'human':0.4, 'efficiency':0.22},
					 }
		FEAT_RANGE = {'table':0.98, 'coffee':1.0, 'laptop':0.4, 'human':0.4, 'efficiency':0.22}

		
		env = Environment(model_filename, object_centers)
		learner = DemoLearner(feat_list, env, constants)
		true_weight = learner.weights_list[-1] # 1, 0, 1


		features = np.array(learner.Phi_rands)
		self.assertTrue((np.max(features, axis=0) <= 1).all())
		self.assertTrue((np.min(features, axis=0) >= 0).all())

		# Get a demo to test obs model
		max_iter = 50
		num_waypts = 5
		start = np.array([104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]) * math.pi / 180.0
		goal = np.array([210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]) * math.pi / 180.0
		goal_pose = None
		T = 20.0
		timestep = 0.5 

		planner = TrajoptPlanner(feat_list, max_iter, num_waypts, env)
		planned_traj = [planner.replan(start, goal, goal_pose, true_weight, T, timestep)]
		P_obs = learner.calc_obs_model(planned_traj)

	def test_random_env_gen(self):
		feat_list = ["human", "table", "laptop"]
		set_size = 10
		object_centers = generate_random_env_set(feat_list, set_size)
		print object_centers

if __name__ == '__main__':
    unittest.main()
		
