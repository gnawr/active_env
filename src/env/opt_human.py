from planners.trajopt_planner import TrajoptPlanner

class OptHuman(object):
	"""
	This class simulates a human giving an optimal demonstration in a number of environments. 
	"""

	def __init__(self, feat_list, max_iter, num_waypts, environments, start, goal, goal_pose, T, timestep, seed=None):
		self.planners = [TrajoptPlanner(feat_list, max_iter, num_waypts, environment) for environment in environments]

		self.start = start
		self.goal = goal
		self.goal_pose = goal_pose
		self.T = T
		self.timestep = timestep
		self.seed = seed
		

	def give_demo(self, env_index, weights):
		planner = self.planners[env_index]
		traj = [planner.replan(self.start, self.goal, self.goal_pose, weights, self.T, self.timestep, self.seed)]
		return traj