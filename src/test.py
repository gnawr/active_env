#! /usr/bin/env python

import sys, os

import numpy as np 
import yaml
import unittest

from utils.openrave_utils import *
from utils.environment import Environment
from demo_inference import DemoInference

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
		

		# human_center_options = [[-0.6,0.55,0.0]]
		human_center_options = [[-0.6,-0.55,0.0]]

		# human_center_options = [[-0.6, -0.55, 0.0],
								# [-0.6, 0.0, 0.0],
								# [0.0, -0.55, 0.0]

		# laptop_center_options = [[-0.4,-0.1,0.0]]						
		laptop_center_options = [[-0.7929,-0.1,0.0]]
		
								
		# laptop_center_options = [[-0.7929,-0.1,0.0],
								 # [-0.7929,-0.2,0.0]]


		for human_center in human_center_options:
			for laptop_center in laptop_center_options:
				object_centers = {'HUMAN_CENTER': human_center, 'LAPTOP_CENTER': laptop_center}
				env = Environment(model_filename, object_centers)

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




		
if __name__ == '__main__':
	unittest.main()

