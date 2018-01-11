import numpy as np
from itertools import permutations,combinations
from math import factorial
from scipy.optimize import minimize
from ctmc_utils import solve_ctmc
import pcmc_utils

def inversions(elt,head):
	if elt not in head:
		return len(head)
	else:
		return head.index(elt)

def infer_greedy(train_lists,n):
	sigma = []
	S = range(n)
	comparisons = np.sum(map(lambda l: len(l)*(len(l)-1)/2 + (n-len(l))*len(l),train_lists))

	L = map(list,train_lists)
	inv_tot = 0

	for i in range(n):
		inv = np.zeros(len(S))
		for idx in range(len(S)):
			inv[idx] = np.sum(map(lambda l: inversions(S[idx],list(l)),L))
		win_idx = np.argmin(np.array(inv))
		sigma.append(S[win_idx])
		for l in L:
			if S[win_idx] in l:
				l.remove(S[win_idx])
		inv_tot+= inv[win_idx]
		S.remove(S[win_idx])

	theta = np.log(float(inv_tot)/float(comparisons))

	return sigma,-theta

def greedy_selection_perm(sigma):
	return sigma

def mallows_choice_prob(S,sigma,theta):
	"""returns a vector of the selection probabilities of S under a mallows
	model with sigma and theta"""
	sig = [x for x in sigma if x in S]
	loc = map(lambda i: sig.index(i),S)
	p = np.exp(-theta* np.array(loc))
	return p/np.sum(p)

def RS_prob(sigma_0,theta,sigma):
	"""returns the probability of sigma under Mallows
	Arguments:
	sigma_0- reference perm
	theta- concentration parameter
	sigma- perm to compute probability of
	"""
	n = len(sigma)
	p=1
	S = range(n)
	for i in range(n):
		probs = mallows_choice_prob(S,sigma_0,theta)
		p*= probs[S.index(sigma[i])]
		S.remove(sigma[i])
	return p

def log_RS_prob(sigma_0,theta,sigma):
	"""returns the log probability of sigma under Mallows
	Arguments:
	sigma_0- reference perm
	theta- concentration parameter
	sigma- perm to compute probability of
	"""
	n = len(sigma_0)
	l = 0
	S = range(n)
	for i in range(len(sigma)):
		probs = mallows_choice_prob(S,sigma_0,theta)
		l+= np.log(probs[S.index(sigma[i])])
		S.remove(sigma[i])
	return l

def greedy_sequential_prediction_error(sigma_0,theta,score,L):
	"""returns the error from predicting the next
	entry of each sigma given the first

	Arguments:
	sigma_0- reference perm
	theta- concentration parameter
	score- scoring function
	L- list of test perms
	"""
	n = len(sigma_0)
	errors = [[] for _ in range(n)]

	for sigma in L:
		S = range(n)
		for i in range(len(sigma)):
			p = mallows_choice_prob(S,sigma_0,theta)
			errors[len(S)-1].append(score(p[S.index(sigma[i])],p))
			S.remove(sigma[i])
	return errors
