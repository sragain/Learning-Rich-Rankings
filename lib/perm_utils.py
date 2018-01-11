import numpy as np
from itertools import permutations
from math import factorial

def borda(L):
	"""Applies Borda's method to a list of permutations
	
	Arguments:
	L- list of permutations
	
	Returns:
	sigma- permutation given by Borda scores according to L
	B- list of Borda counts
	"""
	n = len(L[0])
	B = np.zeros(n)
	for tau in L:
		for i in range(n):
			B[tau[i]]+=(n-i-1) #tau_i comes before the n-i-1 in the tail of tau
	sigma=np.argsort(-B)
	return sigma,B
	
def kemenization_N(sigma_0,N):
	"""
	Outputs the Kemenization of sigma with respect to a matrix of preference counts	
	Arguments:
	sigma_0- starting point
	N- matrix with n_ij:= #times i preferred to j
	
	Returns:
	sigma- locally Kemeny optimal ranking stable w.r.t. sigma_0
	"""
	n = len(sigma_0)
	sigma = []
	for i in sigma_0:
		idx = 0 
		while idx<len(sigma) and N[sigma[idx],i]>N[i,sigma[idx]]:
			idx+=1
		sigma.insert(idx,i)
	return sigma
		
def kemenization(sigma_0,Tau):
	"""
	Outputs the Kemenization of sigma with respect to the permuations in Tau
	
	Arguments:
	sigma_0- starting point
	Tau- list of permutations
	
	Returns:
	sigma- locally Kemeny optimal ranking stable w.r.t. sigma_0
	"""
	n = len(sigma_0)
	N = np.zeros((n,n))
	for tau in Tau:
		for j in range(len(tau)):
			x = tau[j]
			for y in tau[j:]:
				N[x,y]+=1
	sigma = []
	for i in sigma_0:
		idx = 0 
		while idx<len(sigma) and N[sigma[idx],i]>N[i,sigma[idx]]:
			idx+=1
		sigma.insert(idx,i)
	return sigma
	
def Plackett_Luce(gamma):
	"""
	returns a Plackett-Luce distribution from MNL parameters gamma
	"""
	gamma/=np.sum(gamma)
	n = len(gamma)
	S_n = [sigma for sigma in permutations(range(n))]
	pi = np.zeros(len(S_n))
	for idx in range(len(S_n)):
		sigma = S_n[idx]
		prob = 1
		gamma_left = 1
		for i in range(n):
			prob*=gamma[sigma[i]]/gamma_left
			gamma_left -= gamma[sigma[i]]
		pi[idx]=prob
	return pi
	
		