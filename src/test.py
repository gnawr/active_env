#! /usr/bin/env python

import sys, os

import numpy as np 
import yaml
import unittest

from utils.openrave_utils import *
from utils.environment import Environment
from demo_inference import DemoInference


class DevTest(unittest.TestCase):


	def test_demo_inference(self):
		loadfile = '../config/demo_inference.yaml'
		demo_inf = DemoInference(loadfile)
		print('post', demo_inf.learner.P_bt)
		expected = np.array(
			[[0.00000000e+000, 7.18048503e-155, 7.12646977e-144,
        	8.63608124e-063, 8.77103520e-001, 1.22896480e-001,
        	1.95965048e-215, 0.00000000e+000, 0.00000000e+000,
        	2.60659839e-272, 3.05391785e-142, 7.36857027e-243,
        	5.35314651e-128, 0.00000000e+000, 0.00000000e+000,
        	2.46860748e-082, 9.89941688e-187, 0.00000000e+000,
        	1.29038981e-055]])
		differences = demo_inf.learner.P_bt - expected
		self.assertAlmostEqual(0, np.sum(differences))

		
if __name__ == '__main__':
	unittest.main()

