import numpy as np
from itertools import permutations
from math import factorial
from scipy.linalg import eig
from kendall_utils import kt
from numpy.linalg import solve

def solve_ctmc(Q):
	"""Solves the stationary distribution of the CTMC whose rate matrix matches
	the input on off-diagonal entries.
	Arguments:
	Q- rate matrix
	"""
	A=np.copy(Q)
	for i in range(Q.shape[0]):
		A[i,i] = -np.sum(Q[i,:])
	n=Q.shape[0]
	A[:,-1]=np.ones(n)
	b= np.zeros(n)
	b[n-1] = 1
	if np.linalg.matrix_rank(A)<Q.shape[0]:
		print Q
		print A
	return np.linalg.solve(A.T,b)


if __name__ == '__main__':
	X = np.reshape(np.random.rand(9),(3,3))
	print X
	print MX_Q(X)
	print solve_ctmc(Q)
