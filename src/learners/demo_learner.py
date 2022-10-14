import numpy as np
import os
import itertools
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ast
import time

class DemoLearner(object):
	"""
	This class performs demonstration inference given human trajectories.
	"""

	def __init__(self, feat_list, environment, constants, precompute=False):
		# ---- Important internal variables ---- #
		self.feat_list = feat_list
		self.num_features = len(self.feat_list)
		self.environment = environment

		# FEAT_RANGE = constants["FEAT_RANGE"]
		# self.feat_range = [FEAT_RANGE[self.feat_list[feat]] for feat in range(self.num_features)]

		# Set up discretization of theta and beta space.
		self.betas_list = constants["betas_list"]
		self.betas_list.reverse()
		weight_vals = constants["weight_vals"]
		self.weights_list = list(itertools.product(weight_vals, repeat=self.num_features))
		if (0.0,)*len(self.feat_list) in self.weights_list:
			self.weights_list.remove((0.0,) * self.num_features)
		self.weights_list = [w / np.linalg.norm(w) for w in self.weights_list]
		self.weights_list = set([tuple(i) for i in self.weights_list])
		self.weights_list = [list(i) for i in self.weights_list]
		print 'WEIGHTS LIST: ', self.weights_list
		self.num_betas = len(self.betas_list)
		self.num_weights = len(self.weights_list)

		# Construct uninformed prior
		self.P_bt = np.ones((self.num_betas, self.num_weights)) / (self.num_betas * self.num_weights)

		# Trajectory paths.
		here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
		self.traj_rand = pickle.load(open(here + constants["trajs_path"], "rb" ))
		self.traj_strs = self.traj_rand.keys()

		# Compute features for the normalizing trajectories.
		self.Phi_rands = []
		for rand_i, traj_str in enumerate(self.traj_strs):
			curr_traj = np.array(ast.literal_eval(traj_str))
			rand_features = self.environment.featurize(curr_traj, self.feat_list)
			# Phi_rand = np.array([sum(x)/self.feat_range[i] for i,x in enumerate(rand_features)])

			Phi_rand = np.array([sum(x) for i,x in enumerate(rand_features)])
			self.Phi_rands.append(Phi_rand)

		self.Phi_rands = np.array(self.Phi_rands)

		# Create scaling coeffs
		self.max_features = np.max(self.Phi_rands, axis=0)
		self.min_features = np.min(self.Phi_rands, axis=0)
		self.scaling_coeffs = [{"min": self.min_features[i], "max": self.max_features[i]} for i in np.arange(self.num_features)]

		# Apply scaling coeffs
		self.eps = 1e-5 # To avoid divide by zero
		self.Phi_rands = (self.Phi_rands - self.min_features) / (self.max_features - self.min_features + self.eps) 

		# Pre-compute the observation model. Might want to save this for later
		# To get obs model for specific trajectory: self.full_obs_model[traj_index]
		
		self.full_obs_model = []
		if precompute:
			print 'Pre-computing observation model...'
			start = time.time()
			for traj_i, traj_str in enumerate(self.traj_strs):
				curr_traj = np.array(ast.literal_eval(traj_str))
				# get P(xi | theta)
				obs_model = self.calc_obs_model([curr_traj])
				# TODO: USE TRAJ_I ^
				obs_model.reshape(self.num_weights)
				self.full_obs_model.append(obs_model)
			self.full_obs_model = np.array(self.full_obs_model)
			end = time.time()
			print 'Time to build obs model', end - start
			print 'OBS MODEL SHAPE', self.full_obs_model.shape

	def get_obs_model(self, traj_index):
		if len(self.full_obs_model) > 0:
			return self.full_obs_model[traj_index]
		else:
			# Get traj from traj_index
			traj_str = self.traj_strs[traj_index]
			traj = np.array(ast.literal_eval(traj_str))
			return self.calc_obs_model([traj])


	def calc_obs_model(self, trajs):
		"""
		Calculate the observation model P(xi|theta) for given demonstration trajectories.
		Params:
			- trajs [list]: list of Trajectory objects or waypoints
		Returns:
			- List of size len(weight_list), where each entry is P(xi | theta)
		"""
		if isinstance(trajs[0], np.ndarray):
			new_features = [np.sum(self.environment.featurize(traj, self.feat_list), axis=1) for traj in trajs]
		else: 
			new_features = [np.sum(self.environment.featurize(traj.waypts, self.feat_list), axis=1) for traj in trajs]
			
		# Apply scaling coefficients
		summed = np.array(np.sum(np.matrix(new_features), axis=0))
		Phi_H = (summed - self.min_features) / (self.max_features - self.min_features + self.eps)
		Phi_H = Phi_H.T
		# print "Phi_H: ", Phi_H

		# Now compute probabilities for each beta and theta pair.
		num_trajs = len(self.traj_strs)
		P_xi = np.zeros((self.num_betas, self.num_weights))
		for (weight_i, weight) in enumerate(self.weights_list):
			# print "Initiating inference with the following weights: ", weight
			for (beta_i, beta) in enumerate(self.betas_list):
				# Compute -beta*(weight^T*Phi(xi_H))
				numerator = -beta * np.dot(weight, Phi_H)

				# Calculate the integral in log space
				logdenom = np.zeros((num_trajs + 1,1))
				logdenom[-1] = -beta * np.dot(weight, Phi_H)

				# Compute costs for each of the random trajectories
				for rand_i in range(num_trajs):
					Phi_rand = self.Phi_rands[rand_i]

					# Compute each denominator log
					logdenom[rand_i] = -beta * np.dot(weight, Phi_rand)

				# Compute the sum in log space
				A_max = max(logdenom)
				expdif = logdenom - A_max
				denom = A_max + np.log(sum(np.exp(expdif)))

				# Get P(xi_H | beta, weight) by dividing them
				P_xi[beta_i][weight_i] = np.exp(numerator - denom * len(trajs))

		# print P_xi, sum(sum(P_xi))
		P_obs = P_xi / sum(sum(P_xi))
		return P_obs

	def learn_posterior(self, trajs, P_bt):
		# P_obs = P_xi / sum(sum(P_xi))
		P_obs = self.calc_obs_model(trajs)

		# Compute P(weight, beta | xi_H) via Bayes rule
		posterior = np.multiply(P_obs, P_bt)
		
		# Normalize posterior
		posterior = posterior / sum(sum(posterior))
		return posterior


	def learn_weights(self, trajs, P_bt=False):
		if not P_bt: #update in-place
			posterior = self.learn_posterior(trajs, self.P_bt)
			self.P_bt = posterior
		else: 
			posterior = self.learn_posterior(trajs, P_bt)

		# Compute optimal expected weight
		P_weight = sum(posterior, 0)
		weights = np.sum(np.transpose(self.weights_list)*P_weight, 1)
		P_beta = np.sum(posterior, axis=1)
		beta = np.dot(self.betas_list,P_beta)
		# print("observation model: ", P_obs)
		print("posterior: ", self.P_bt)
		print("theta marginal: " + str(P_weight))
		print("beta marginal: " + str(P_beta))
		print("theta average: " + str(weights))
		print("beta average: " + str(beta))

		self.visualize_posterior(posterior)
		return weights

	def visualize_posterior(self, post):
		matplotlib.rcParams['font.sans-serif'] = "Arial"
		matplotlib.rcParams['font.family'] = "Times New Roman"
		matplotlib.rcParams.update({'font.size': 15})


		print(post.shape)
		plt.imshow(post, cmap='Blues', interpolation='nearest')
		plt.colorbar()

		weights_rounded = [[round(i,2) for i in j] for j in self.weights_list]
		plt.xticks(range(len(self.weights_list)), weights_rounded, rotation = 'vertical')
		plt.yticks(range(len(self.betas_list)), list(self.betas_list))
		plt.xlabel(r'$\theta$', fontsize=15)
		plt.ylabel(r'$\beta$',fontsize=15)
		plt.title(r'Joint Posterior Belief b($\theta$, $\beta$)')
		plt.tick_params(length=0)
		plt.show()

	def visualize_stacked_posterior(self, beliefs, title='', save = ''):
		matplotlib.rcParams['font.sans-serif'] = "Arial"
		matplotlib.rcParams['font.family'] = "Times New Roman"
		matplotlib.rcParams.update({'font.size': 10})


		print(beliefs.shape)
		plt.imshow(beliefs, cmap='Blues', interpolation='nearest')
		plt.colorbar()

		weights_rounded = [[round(i,2) for i in j] for j in self.weights_list]
		plt.xticks(range(len(self.weights_list)), weights_rounded, rotation = 'vertical')
		plt.yticks(range(len(beliefs)))
		plt.xlabel(r'$\theta$', fontsize=15)
		plt.ylabel('Iteration',fontsize=15)
		plt.title(r'Belief b($\theta$) for ' + title)
		plt.tick_params(length=0)
		
		if save:
			plt.savefig(save)
		else:
			plt.show()
		
		return