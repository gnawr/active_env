#! /usr/bin/env python

import sys, os

import numpy as np 
import yaml
import unittest
import ast

from utils.openrave_utils import *
from utils.environment import Environment
from demo_inference import DemoInference
from catkin.find_in_workspaces import find_in_workspaces
from learners.demo_learner import DemoLearner
from planners.trajopt_planner import TrajoptPlanner

import time


class DevTest(unittest.TestCase):


	# def test_demo_inference(self):
	# 	loadfile = '../config/demo_inference.yaml'
	# 	demo_inf = DemoInference(loadfile)
	# 	print('post', demo_inf.learner.P_bt)
	# 	expected = np.array(
	# 		[[0.00000000e+000, 7.18048503e-155, 7.12646977e-144,
 #        	8.63608124e-063, 8.77103520e-001, 1.22896480e-001,
 #        	1.95965048e-215, 0.00000000e+000, 0.00000000e+000,
 #        	2.60659839e-272, 3.05391785e-142, 7.36857027e-243,
 #        	5.35314651e-128, 0.00000000e+000, 0.00000000e+000,
 #        	2.46860748e-082, 9.89941688e-187, 0.00000000e+000,
 #        	1.29038981e-055]])
	# 	differences = demo_inf.learner.P_bt - expected
	# 	self.assertAlmostEqual(0, np.sum(differences))

	def test_env_gen(self):
		model_filename = "jaco_dynamics"
		

		human_center_options = [[-0.6,0.55,0.0]]
		# human_center_options = [[-0.6,-0.55,0.0]]
		# human_center_options = [[-0.9,-0.55,0.0]]
		# human_center_options = [[-0.3, 0.55,0.0]]


		# human_center_options = [[-0.6, -0.55, 0.0],
								# [-0.6, 0.0, 0.0],
								# [0.0, -0.55, 0.0]

		# laptop_center_options = [[-0.4,-0.1,0.0]]						
		laptop_center_options = [[-0.7929,-0.1,0.0]]

		# laptop_center_options = [[-1.0,-0.3,0.0]]

		
								
		# laptop_center_options = [[-0.7929,-0.1,0.0],
								 # [-0.7929,-0.2,0.0]]
		object_centers_dict = {
							   0: {'HUMAN_CENTER': [-0.4, 0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.15,0.0]},
							   # 1: {'HUMAN_CENTER': [-0.9, -0.55,0.0], 'LAPTOP_CENTER': [-1.0,-0.3,0.0]},
							   # 2: {'HUMAN_CENTER': [-0.3,-0.55,0.0], 'LAPTOP_CENTER': [-0.3,-0.3,0.0]},
							   # 3: {'HUMAN_CENTER': [-0.5, 0.55,0.0], 'LAPTOP_CENTER': [-0.6, 0.1,0.0]},
							   }


		for object_centers in object_centers_dict.values():
				env = Environment(model_filename, object_centers)
				
				#sphere
				objects_path = '../data'
				
				env.env.Load('{:s}/sphere.xml'.format(objects_path))
				mug = env.env.GetKinBody('sphere')
				body = mug
				body.SetName("pt"+str(len(env.bodies)))
				env.env.Add(body, True)
				env.bodies.append(body)

				# env.robot.SetDOFValues([math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, 0.,0.,0.])
				# env.robot.SetDOFValues([283.13, 162.59, 0.0, 43.45, 265.25, 257.59, 288.29, 0.,0.,0.])
				# env.robot.SetDOFValues([286.77, 100.91, 159.53, 77.13, 122.34, 109.7, 345.56, 0.,0.,0.])
				# # env.robot.SetDOFValues([141.44, 88.25, 207.99, 126.77, -59.24, 133.23, 375.59, 0.,0.,0.])
				# ee_xyz = manipToCartesian(env.robot, 0.0)
				# plotSphere(env.env, [], ee_xyz, size=10, color=[0, 0, 1])
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

	def test_demo(self):

		model_filename = 'jaco_dynamics'
		object_centers = {'HUMAN_CENTER': [-0.3,-0.55,0.0], 'LAPTOP_CENTER': [-0.3,-0.3,0.0]}
		feat_list = ["human", "table", "laptop"]
		constants = {'trajs_path': "/data/traj_sets/traj_rand_merged_H.p",
					 'betas_list': [10.0],
					 'weight_vals': [0.0, 0.5, 1.0], # Per feature theta options.
					 'FEAT_RANGE': {'table':1.0, 'coffee':1.0, 'laptop':0.4, 'human':0.4, 'efficiency':0.01},
					 }
		FEAT_RANGE = {'table':1.0, 'coffee':1.0, 'laptop':0.4, 'human':0.4, 'efficiency':0.01}
		
		env = Environment(model_filename, object_centers)
		learner = DemoLearner(feat_list, env, constants)
		true_weight = learner.weights_list[-1]


		# Trajectory paths.
		here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../'))
		traj_rand = pickle.load(open(here + "/data/traj_sets/traj_rand_merged_H.p", "rb" ))

		# Featurize each traj
		Phi_rands = []
		feat_range = [FEAT_RANGE[feat_list[feat]] for feat in range(3)]
		for rand_i, traj_str in enumerate(traj_rand.keys()):
			curr_traj = np.array(ast.literal_eval(traj_str))
			
			rand_features = env.featurize(curr_traj, feat_list)
			Phi_rand = np.array([sum(x)/feat_range[i] for i,x in enumerate(rand_features)])
			Phi_rands.append(Phi_rand)

		# Plot best traj
		rewards = np.dot(np.array(Phi_rands), true_weight)
		print "DEBUG: rewards shape should be ~1400", rewards.shape 
		best_traj_idx = np.argmax(rewards)
		best_traj_str = traj_rand.keys()[best_traj_idx]
		best_traj = np.array(ast.literal_eval(best_traj_str))
		plotTraj(env.env, env.robot, env.bodies, best_traj, size=0.015, color=[0, 0, 1])

		# Plot demo
		max_iter = 50
		num_waypts = 5
		start = np.array([104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]) * math.pi / 180.0
		goal = np.array([210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0]) * math.pi / 180.0
		goal_pose = None
		T = 20.0
		timestep = 0.5 

		planner = TrajoptPlanner(feat_list, max_iter, num_waypts, env)
		planned_traj = [planner.replan(start, goal, goal_pose, true_weight, T, timestep)]
		plotTraj(env.env, env.robot, env.bodies, planned_traj[0].waypts, size=0.015,color=[1, 0, 0])


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

if __name__ == '__main__':
    unittest.main()
		
