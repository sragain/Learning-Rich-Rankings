import numpy as np
from itertools import permutations
from math import factorial
from ctmc_utils import solve_ctmc
from scipy.optimize import minimize

def predict_next_error_list(x,C):
	"""returns the error from predicting the next
	entry of each sigma given the first"""
	Q = comp_Q(x)
	pi = solve_pcmc(Q)
	n  = Q.shape[0]
	err = np.zeros(n)
	nsamp = np.sum(C.values())
	P = Q_reversal_P(Q)
	for sigma in C:
		assert pi[sigma[0]]<1
		err[0]+=C[sigma]*(1-pi[sigma[0]])/nsamp #error guessing sigma[0]
		S = range(n)
		S.remove(sigma[0])
		for i in range(1,n):
			h=hit_prob(sigma[i-1],sigma[i],S,P)
			S.remove(sigma[i])
			e=(1-h)
			err[i]+=C[sigma]/nsamp*e
	return err
	
def predict_next_error(x,C):
	"""returns the error from predicting the next
	entry of each sigma given the first"""
	err = 0
	Q = comp_Q(x)
	pi = solve_pcmc(Q)
	n  = Q.shape[0]
	nsamp = np.sum(C.values())
	P = Q_reversal_P(Q)
	print P
	for sigma in C:
		e=(1-pi[sigma[0]]) #error guessing sigma[0]
		S = range(n)
		S.remove(sigma[0])
		for i in range(1,n):
			h=hit_prob(sigma[i-1],sigma[i],S,P)
			S.remove(sigma[i])
			e+=(1-h)
		err+=C[sigma]/nsamp*e
	return err
	
def comp_error_brute(x,C):
	"""computes ell_1 distance between model @ params x and 
	empirical dist of counts C by brute force
	"""
	Q=comp_Q(x)	
	pi = MTF_dist(Q)
	n= Q.shape[0]
	err=0
	nsamp = np.sum(C.values())
	S_n = [sigma for sigma in permutations(range(n))]
	emp = np.zeros(len(S_n))
	for idx in range(len(S_n)):
		sigma = S_n[idx]
		if sigma in C:
			emp[idx]=C[sigma]/nsamp
			
	return np.sum(np.abs(pi-emp))

def comp_error(x,C):
	Q = comp_Q(x)
	n = Q.shape[0]
	nsamp = np.sum(C.values())
	err = 0
	for sigma in C:
		err+=np.abs(C[sigma]/nsamp-	MTF_prob(sigma,Q))
	return err		
	
def infer_MTF_brute(C,n,x=None,epsilon=.001,maxiter=25):
	"""performs MLE for MTF by solving the stationary distribution of the chain
	directly rather than computing probabilities only for sample sigma
	This method is O(n!^3) regardless of the sparsity of the input data in S_n
	"""
	bounds=[(epsilon,None)]*(n*(n-1))
	if x is None:
		x = np.random.rand(n*(n-1))+epsilon

	res = minimize(neg_L_MTF_brute,x,args=(C),bounds = bounds,options={'disp':False,'maxiter':maxiter})
	return res.x

def neg_L_MTF_brute(x,C):
	"""computes the negative log likelihood of Q given 
	permutations in C exactly
	
	Arguments:
	Q- PCMC rate matrix
	"""
	Q = comp_Q(x)
	pi = MTF_dist(Q)
	n = Q.shape[0]
	S_n = [sigma for sigma in permutations(range(n))]
	emp = np.zeros(len(S_n))
	nsamp = np.sum(C.values())
	for idx in range(len(S_n)):
		sigma = S_n[idx]
		if sigma in C:
			emp[idx]=C[sigma]/nsamp	
	L = -np.dot(emp,np.log(pi))
	return L

def infer_MTF(C,n,x=None,epsilon=.001,maxiter=25):
	"""performs MLE for MTF by solving the stationary distribution of the chain
	directly rather than computing probabilities only for sample sigma
	This method is O(n!^3) regardless of the sparsity of the input data in S_n
	"""
	bounds=[(epsilon,None)]*(n*(n-1))
	if x is None:
		x = np.random.rand(n*(n-1))+epsilon
	res = minimize(neg_L_MTF,x,args=(C),bounds = bounds,options={'disp':True,'maxiter':maxiter})
	return res.x

def neg_L_MTF(x,C):
	L=0
	Q = comp_Q(x)
	for sigma in C:
		L-=C[sigma]*np.log(MTF_prob(sigma,Q))
	return L
	
def sim_hit_prob(start,t,S,P,nsamp=1000):
	n = P.shape[0]
	count = 0
	for _ in xrange(nsamp):
		s=start
		while s not in S:
			s = np.random.choice(range(n),p=P[s,:])
		if s==t:
			count+=1.0
	return count/nsamp
				
def hit_prob(start,target,S,P):
	"""returns the probability that the first state hit in S in the time
	reversal of Q is target"""
	A = np.copy(P)
	A[S,:] = np.zeros(A[S,:].shape)
	I = np.identity(P.shape[0])
	return np.linalg.inv(I-A)[start,target]
		
def MTF_prob(sigma,Q):
	pi = solve_pcmc(Q)
	a = pi[sigma[0]]
	n = Q.shape[0]
	S = range(n)
	S.remove(sigma[0])
	state = sigma[0]
	P=Q_reversal_P(Q)
	for i in range(1,n):
		a*=hit_prob(start=state,target=sigma[i],S=S,P=P)
		S.remove(sigma[i])
		state=sigma[i]
	return a
	
def solve_pcmc(Q):
	for i in range(Q.shape[0]):
		Q[i,i]=0
		Q[i,i]=-np.sum(Q[i,:])
	return solve_ctmc(Q)
	
def Q_reversal_P(Q):
	"""transition matrix for the discrete time chain embedded in the time
	reversal of the """ 
	pi = solve_pcmc(Q)
	n=Q.shape[0]
	P = np.zeros((n,n))
	for i in range(Q.shape[0]):
		for j in range(Q.shape[0]):
			P[i,j] = Q[j,i]*pi[j]			
		P[i,i]=0
		P[i,:]/=np.sum(P[i,:])
	return P
	
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

def MTF_dist(Q):
	#print MTF_Q(Q)
	return solve_ctmc(MTF_Q(Q))
	
def MTF(sig,k):
	"""moves the k-th entry of sigma to the front of sig"""
	sigma = np.copy(sig)
	tmp = sigma[k]
	sigma[1:k+1]=sigma[0:k]
	#print sigma
	sigma[0]=tmp
	return sigma
	
def MTF_Q(Q_PCMC):
	"""returns the MTF transition rate matrix on S_n affiliated with 
	a PCMC model
	
	Arguments:
	Q_PCMC: transition rate matrix of a PCMC model
	
	Returns:
	Q: transition rate matrix on S_n
	"""
	n = Q_PCMC.shape[0]
	Q = np.zeros((factorial(n),factorial(n)))
	S_n = [sigma for sigma in permutations(range(n))]
	for idx1 in range(factorial(n)):
		sigma1 = S_n[idx1]
		for k in range(1,n):
			sigma2 = MTF(sigma1,k)
			#print sigma1,tuple(sigma2)
			idx2 = S_n.index(tuple(sigma2))
			Q[idx1,idx2] = Q_PCMC[sigma1[0],sigma2[0]]
		Q[idx1,idx1]=-np.sum(Q[idx1,:])
	return Q
	