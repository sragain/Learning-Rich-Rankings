import numpy as np
from itertools import permutations,combinations
from math import factorial
from scipy.optimize import minimize
from ctmc_utils import solve_ctmc
import pcmc_utils

def comp_alpha_gamma(x,n):
	"""unfolds a single parameter vector x
	of all parameters of the mmnl into
	the K mixture probabilities (first K entries)
	and K gamma vectors (final nK entries)

	Arguments:
	x- vector of all mmnl parameters
	n- number of of items ranked
	"""

	K = n*(n-1)/(n+1)
	alpha=x[:K]
	gamma = np.reshape(x[K:],(n,K))
	return alpha,gamma

def compute_probs(alpha,gamma):
	"""returns the selection probabilities of an MMNL model

	Arguments:
	gamma - i-th column are parameters to i-th mnl
	alpha - alpha[i] is weight of i-th mnl
	"""
	K = len(alpha)
	n = gamma.shape[0]
	p = np.zeros(n)
	#normalize
	alpha/=np.sum(alpha)
	gamma/=np.sum(gamma,axis=0)
	return np.sum(np.multiply(alpha,gamma),axis=1)

def RS_prob(x,sigma,n):
	"""computes the probability of sigma
	under repeated selection from an MMNL

	Arguments:
	x- parameter vector of mmnl
	sigma- ranking of n items
	"""
	alpha,gamma =comp_alpha_gamma(x,n)
	S = range(n)
	p=1
	for i in range(n):
		choice_p = compute_probs(alpha,gamma[S,:])
		p*=choice_p[S.index(sigma[i])]
		S.remove(sigma[i])
	return p

def log_RS_prob(x,sigma,n):
	"""computes the log probability of sigma
	under repeated selection from an MMNL

	Arguments:
	x- parameter vector of mmnl
	sigma- ranking of n items
	"""
	alpha,gamma =comp_alpha_gamma(x,n)
	S = range(n)
	l=0
	for i in range(len(sigma)):
		choice_p = compute_probs(alpha,gamma[S,:])
		l+= np.log(choice_p[S.index(sigma[i])])
		S.remove(sigma[i])
	return l

def greedy_selection_perm(x,n):
	"""	returns the permutation given
	by greedily selecting from the MMNL
	with the given parameters

	Arguments:
	x- parameters of mmnl
	n- number of items being ranked
	"""

	alpha,gamma =comp_alpha_gamma(x,n)
	sigma = []
	S = range(n)
	for i in range(n):
		p = compute_probs(alpha,gamma[S,:])
		winner = S[np.argmax(p)]
		sigma.append(winner)
		S.remove(winner)
	return sigma

def greedy_sequential_prediction_error(x,score,L,n):
	"""returns the error from predicting the next
	entry of each sigma given the prefix

	Arguments:
	x- model parameters
	score- scoring function
	L- list of test perms
	n- number of alternatives
	"""

	alpha,gamma =comp_alpha_gamma(x,n)
	errors = [[] for _ in range(n)]


	for sigma in L:
		S = range(n)
		for i in range(len(sigma)):


			p = compute_probs(alpha,gamma[S,:])
			errors[len(S)-1].append(score(p[S.index(sigma[i])],p))
			S.remove(sigma[i])

	return errors

def sample_sigma(alpha,gamma):
	n = len(alpha)
	S = range(n)
	sigma = []
	for i in range(n):
		winner = np.random.choice(S,p=compute_probs(alpha,gamma[S,:]))
		sigma.append(winner)
		S.remove(winner)
	return sigma
