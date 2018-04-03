import numpy as np
from itertools import permutations,combinations
from math import factorial
from scipy.optimize import minimize
from ctmc_utils import solve_ctmc
import pcmc_utils

def comp_Q(x):
	"""reshapes PCMC parameter vector into rate matrix

	input:
	x- parameter vector of off-diagnoal entries of Q
	"""
	n = int(1+np.sqrt(4*len(x)+1))/2
	Q = np.empty((n,n))
	for i in range(n):
		row = np.insert(x[i*(n-1):(i+1)*(n-1)],i,0)
		Q[i,:]=row
	return Q

def greedy_selection_perm(x):
	"""	returns the permutation given
	by greedily selecting from the pcmc
	with the given parameters

	Arguments:
	x- parameters of pcmc
	n- number of items being ranked
	"""
	Q = comp_Q(x)
	n  = Q.shape[0]
	sigma = []
	S = range(n)
	for i in range(n):
		p = solve_ctmc(Q[S,:][:,S])
		winner = S[np.argmax(p)]
		sigma.append(winner)
		S.remove(winner)
	return sigma

def greedy_sequential_prediction_error(x,score,L):
	"""returns the error from predicting the next
	entry of each sigma given the first

	Arguments:
	x- pcmc params
	score- scoring function
	L- list of test perms
	"""
	Q = comp_Q(x)
	n  = Q.shape[0]
	errors = [[] for _ in range(n)]

	for sigma in L:
		S = range(n)
		for i in range(len(sigma)):
			p = solve_ctmc(Q[S,:][:,S])
			errors[len(S)-1].append(score(p[S.index(sigma[i])],p))
			S.remove(sigma[i])
	return errors

def sample_sigma(x):
	"""Returns a permutation based by recursively sampling
	via PCMC from the unused items

	Arguments:
	x- pcmc parameters
	"""
	Q = comp_Q(x)
	n = Q.shape[0]
	sigma = []
	S = range(n)
	for i in range(n):
		winner = np.random.choice(S,p = solve_ctmc(Q[S,:][:,S]))
		sigma.append(winner)
		S.remove(winner)
	return sigma

def comp_error(x,C):
	"""computes ell_1 distance between model @ params x and
	empirical dist in dictionary C
	"""
	nsamp = np.sum(C.values()).astype(float)
	Q= comp_Q(x)
	err=0
	for sigma in C:
		err+=np.abs(C[sigma]/nsamp-RS_prob(sigma,Q))
	return err

def RS_prob(x,sigma):
	"""returns the probability of sigma under RS from pcmc
	Arguments:
	x- pcmc params
	"""
	Q = comp_Q(x)
	n = len(sigma)
	p=1
	S = range(n)
	for i in range(n):
		pi = solve_ctmc(Q[S,:][:,S])
		p*= pi[S.index(sigma[i])]
		S.remove(sigma[i])
	return p

def RS_prob_Q(Q,sigma):
	"""returns the probability of sigma under RS from pcmc
	Arguments:
	x- pcmc params
	"""
	n = len(sigma)
	p=1
	S = range(n)
	for i in range(n):
		print S, pi
		pi = solve_ctmc(Q[S,:][:,S])
		p*= pi[S.index(sigma[i])]
		S.remove(sigma[i])
	return p

def log_RS_prob(x,sigma):
	"""returns the log probability of sigma under RS from pcmc
	Arguments:
	x- pcmc params
	"""
	Q = comp_Q(x)
	n = Q.shape[0]
	l = 0
	S = range(n)
	for i in range(len(sigma)):
		pi = solve_ctmc(Q[S,:][:,S])
		l+= np.log(pi[S.index(sigma[i])])
		S.remove(sigma[i])
	return l
