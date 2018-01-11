import numpy as np
from itertools import permutations,combinations
from math import factorial
from scipy.optimize import minimize
from ctmc_utils import solve_ctmc

def greedy_selection_perm(gamma):
	"""returns the ranking of items given by
	repeatedly selecting the most likely to be chosen next
	"""
	return np.argsort(-gamma)

def ILSR(C,x,n):
	"""computes the optimal MNL parameters for some choice data

	Arguments:
	C: choice data
	x: intial parameter guess
	n: number of items
	"""
	if x is None:
		x = np.ones(n).astype(float)/n

	pi=x
	diff = 1
	epsilon = 10**(-6)
	np.set_printoptions(suppress=True,precision=3)
	while diff>epsilon:
		pi_ = pi
		lam = np.zeros((n,n))+epsilon
		for S in C:
			gamma = np.sum([pi[x] for x in S])
			pairs = [(i,j) for i in range(len(S)) for j in range(len(S)) if j!=i]
			for i,j in pairs:
				lam[S[j],S[i]]+=C[S][i]/gamma
		pi = solve_ctmc(lam)
		diff = np.linalg.norm(pi_-pi)
	return pi

def add_choices(C,sigma,n):
	"""adds the choices given under RS from sigma into C

	Arguments:
	C- preexisting choice data
	sigma- new ranking to add
	n- number of items
	"""
	for i in range(n-1):
		S = tuple(np.sort(sigma[i:]))
		if S not in C:
			C[S]=np.zeros(len(S))
		C[S][S.index(sigma[i])]+=1.0
	return C

def ILSR_perms(C_sig,x,n):
	"""
	returns the PL parameters best fitting the permutation counts in C_sig

	Arguments:
	C_sig- permutations data
	x- inital guess
	n- number of items to rank
	"""
	C={}
	for sigma in C_sig:
		for i in range(n-1):
			S = tuple(np.sort(sigma[i:]))
			if S not in C:
				C[S]=np.zeros(len(S))
			C[S][S.index(sigma[i])]+=C_sig[sigma]

	if x is None:
		x = np.ones(n).astype(float)/n

	pi=x
	diff = 1
	epsilon = 10**(-6)
	np.set_printoptions(suppress=True,precision=3)
	while diff>epsilon:
		pi_ = pi
		lam = np.zeros((n,n))+epsilon
		for S in C:
			gamma = np.sum([pi[x] for x in S])
			pairs = [(i,j) for i in range(len(S)) for j in range(len(S)) if j!=i]
			for i,j in pairs:
				lam[S[j],S[i]]+=C[S][i]/gamma
		pi = solve_ctmc(lam)
		diff = np.linalg.norm(pi_-pi)
	return pi

def greedy_sequential_prediction_error(x,score,L):
	"""returns the error from predicting the next
	entry of each sigma given the first

	Arguments:
	x- parameters
	score- scoring function
	L- list of test rankings
	"""
	gamma = x
	n = len(gamma)
	errors = [[] for _ in range(n)]
	for sigma in L:
		S = range(n)
		for i in range(len(sigma)):


			p = gamma[S]/np.sum(gamma[S])
			errors[len(S)-1].append(score(p[S.index(sigma[i])],p))
			S.remove(sigma[i])

	return errors

def sample_sigma(gamma):
	"""Samples from Plackett Luce distribution

	Arguments:
	gamma- PL parameters
	"""
	n = len(gamma)
	sigma = []
	S = range(n)
	for i in range(n):
		prob = gamma[S]/np.sum(gamma[S])
		winner = np.random.choice(S,p = prob)
		sigma.append(winner)
		S.remove(winner)
	return sigma

def PL_dist(gamma):
	"""returns a vector of n! probabilities
	whose i-th entry is the probability under plackett-luce
	of the i-th permutation in permutations(range(n))
	"""
	n = len(gamma)
	d = []
	for sigma in permutations(range(n)):
		d.append(PL_prob(sigma,gamma))
	return np.array(d)

def comp_error(x,C):
	"""computes the TV distance between PL and an empirical distribution

	Arguments:
	x- PL params
	C- empirical data
	"""
	gamma=x
	nsamp = np.sum(C.values()).astype(float)
	err=0
	for sigma in C:
		err+=np.abs(C[sigma]/nsamp-PL_prob(sigma,gamma))
	return err

def PL_prob(sigma,gamma):
	"""computes probabilities under PL model
	Arguments:
	sigma- permutation
	gamma- PL parameters
	"""
	n = len(sigma)
	p=1
	S = range(n)
	for i in range(n):
		p*= gamma[sigma[i]]/np.sum(gamma[S])
		S.remove(sigma[i])
	return p

def log_PL_prob(sigma,gamma):
	"""computes probabilities under PL model
	Arguments:
	sigma- permutation
	gamma- PL parameters
	"""
	n = len(gamma)
	l=0
	S = range(n)
	for i in range(len(sigma)):
		l+= np.log(gamma[sigma[i]]/np.sum(gamma[S]))
		S.remove(sigma[i])
	return l
