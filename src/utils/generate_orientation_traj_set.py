import itertools
import math
import numpy as np
import pickle
import sys, os
import time

from utils.environment import Environment
from planners.trajopt_planner import TrajoptPlanner
from generate_env_set import sample_single_environment

def generate_orientation_traj_set(feat_list):
	# Before calling this function, you need to decide what features you care
	# about, from a choice of table, coffee, human, origin, and laptop.
	# pick0 = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 310.8]
	pick1 = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 135.0] # upside down
	# pick2 = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 45.0]
	pick3 = [104.2, 151.6, 183.8, 101.8, 224.2, 216.9, 315.0] # upright

	place0 = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 322.0] # upright
	# place1 = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 232.0]
	place2 = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 142.0] # upside down
	# place3 = [210.8, 101.6, 192.0, 114.7, 222.2, 246.1, 52.0]

	# starts = np.array([pick0, pick1, pick2, pick3])*(math.pi/180.0)
	# goals = np.array([place0, place1, place2, place3])*(math.pi/180.0)

	starts = np.array([pick3, pick1])*(math.pi/180.0)
	goals = np.array([place0, place2])*(math.pi/180.0)
	goal_pose = None
	T = 20.0
	timestep = 0.5
	
	# Openrave parameters for the environment.
	model_filename = "jaco_dynamics"
	object_centers = {'HUMAN_CENTER': [-0.6,-0.55,0.0], 'LAPTOP_CENTER': [-0.7929,-0.1,0.0]}
	environment = Environment(model_filename, object_centers, show=False)

	# Planner Setup
	max_iter = 50
	num_waypts = 5
	# feat_list = [x.strip() for x in feat_list.split(',')]
	num_features = len(feat_list)
	planner = TrajoptPlanner(feat_list, max_iter, num_waypts, environment)

	# Construct set of weights of interest.  
	weight_vals = [0.0, 0.5, 1.0]
	weights_list = list(itertools.product(weight_vals, repeat=len(feat_list)))
	if (0.0,)*len(feat_list) in weights_list:
		weights_list.remove((0.0,) * len(feat_list))
	weights_list = [w / np.linalg.norm(w) for w in weights_list]
	weights_list = set([tuple(i) for i in weights_list])
	weights_list = [list(i) for i in weights_list]

	# Load in the object_centers_candidates
	object_centers_path = "/data/env_sets/env_set_human_laptop_table.p"
	here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
	object_centers_dict = pickle.load(open(here + object_centers_path, "rb"))
	
	# num_envs_to_sample = 20
	# num_envs_to_sample = 5 # for small trajset



	# Call the planner for each case
	start_time = time.time()
	print('start time', start_time)

	traj_rand = {}
	coffee_features = []

	print 'Expected number of trajs', len(object_centers_dict) * len(starts) * len(goals) * len(weights_list)

	for start in starts:
		for goal in goals:
			for env_idx in np.arange(len(object_centers_dict)):
				# FIXED ENVS CASE
				centers = object_centers_dict[env_idx]
				# SAMPLED ENVS CASE
				# centers = sample_single_environment()

				# Update the environment to have these object centers
				environment.object_centers = centers
				for (w_i, weights) in enumerate(weights_list):
					traj = planner.replan(start, goal, goal_pose, weights, T, timestep)
					Phi = environment.featurize(traj.waypts, feat_list)
					# Getting rid of bad, out-of-bounds trajectories
					if any(phi < 0.0 for phi in Phi):
						continue

					# Check orientation feature variance in here 
					coffee_feat_idx = feat_list.index('coffee')
					coffee_feature = np.array(Phi[coffee_feat_idx]) # this is an array with the coffee feature for each waypt
					assert len(coffee_feature) >= 20, 'coffee feature checking calculation incorrect!'
					coffee_features.append(np.sum(coffee_feature))


					traj = traj.waypts.tolist()
					if repr(traj) not in traj_rand:
						traj_rand[repr(traj)] = weights

	here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
	savefile = "/data/traj_sets/traj_set_fixed_envs_human_coffee_laptop.p"
	pickle.dump(traj_rand, open( here + savefile, "wb" ))

	print 'Number of trajs: ', len(traj_rand)
	print "Saved in: ", savefile
	print "Used the following number of weight-combos: ", len(weights_list)
	print 'Coffee feature max: ', np.max(coffee_features)
	print 'Coffee feature min: ', np.min(coffee_features)


	end_time = time.time()
	print 'TIME FOR GENERATION: ', end_time - start_time
	time_taken = end_time - start_time
	file_path = os.path.join(os.getcwd(), 'time_0223.txt')
	with open(file_path, 'w') as f:
		f.write(str(time_taken))


if __name__ == '__main__':
	feat_list = ["human", "coffee", "laptop"]
	generate_orientation_traj_set(feat_list)



