import numpy as np
from scipy.optimize import minimize
from pcmc_utils import *
from PL_utils import ILSR

def pcmc_grad(x,C):
	""" returns the gradient of pcmc at x for data C"""
	Q = comp_Q(x)
	n = Q.shape[0]
	D = np.zeros((n,n))

	for S in C:
		Ainv = get_inv(Q[S,:][:,S]) #gets the matrix whose final column is p
		p = Ainv[:,-1]
		#assert np.sum(np.abs(p))==np.sum(p)
		#assert np.abs(1-np.sum(p))<.001
		#assert np.sum(np.abs(solve_ctmc(Q[S,:][:,S])-p))<.001
		pairs = [(i,j) for i in range(len(S)) for j in range(len(S)) if j!=i]
		for (idx_i,idx_j) in pairs:
				Delta = np.zeros(Q.shape)
				Delta[S[idx_i],S[idx_j]]=1
				B = np.zeros(Ainv.shape)
				if idx_i< len(S)-1:
					B[idx_i,idx_i]=-1
				if idx_j < len(S)-1:
					B[idx_j,idx_i]=1

				d = np.dot(Ainv,np.dot(B,Ainv))[:,-1]

				#assert np.abs(np.sum(d))<.001
				D[S[idx_i],S[idx_j]]+=np.dot(C[S],np.divide(d,p))

	return flatten(D)

def get_inv(Q):
	"""Solves the stationary distribution of the CTMC whose rate matrix matches
	the input on off-diagonal entries.
	Arguments:
	Q- rate matrix
	"""
	A=np.copy(Q)
	for i in range(Q.shape[0]):
		A[i,i]=0
		A[i,i] = -np.sum(A[i,:])
	n=Q.shape[0]
	A[:,-1]=np.ones(n)
	return np.linalg.inv(A.T)

def flatten(D):
	n = D.shape[0]
	#x = np.empty(n*(n-1))
	x = np.zeros	(n*(n-1))
	for i in range(n):
		for j in range(n):
			if j<i:
				x[i*(n-1)+j]=D[i,j]
			if j>i:
				x[i*(n-1)+j-1]=D[i,j]
	return x

def infer_grad(C,n,x=None,epsilon=10**(-3),maxiter=25,ILSR_init=True):
	"""infers the parameters of a PCMC model using scipy.minimize to do MLE

	Arguments:
	C- training data
	n- number of elements in universe
	x- starting parameters
	delta- parameter of constraint q_ij+q_ji>=delta
	maxiter- number of iterations allowed to optimizer
	ILSR_init- whether to initilaize at Q that gives the MLE for MNL
	"""
	bounds=[(epsilon,None)]*(n*(n-1))
	if x is None and ILSR_init:
		gamma = ILSR(C,None,n)
		x = np.zeros(n*(n-1))
		for i in range(n):
			x[i*n:(i+1)*n] = gamma[i]
	elif x is None:
		x = np.ones(n*(n-1))
	res = minimize(neg_L,x,args=(C),bounds = bounds,jac = pcmc_grad,options={'disp':False,'maxiter':maxiter})
	return res.x
