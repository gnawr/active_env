from planners.trajopt_planner import TrajoptPlanner

class OptHuman(object):
	"""
	This class simulates a human giving an optimal demonstration. 
	"""

	def __init__(self, feat_list, max_iter, num_waypts, environment):
		self.planner = TrajoptPlanner(feat_list, max_iter, num_waypts, environment)

	def give_demo(self, start, goal, goal_pose, weights, T, timestep, seed=None):
		traj = [self.planner.replan(start, goal, goal_pose, weights, T, timestep, seed)]
		return traj
