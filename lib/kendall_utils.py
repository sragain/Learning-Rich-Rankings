import numpy as np
from itertools import permutations
from math import factorial
from multiprocessing import Pool

def kt_sum(a):
	sigma,lists = a[0],a[1]
	return np.sum(map(lambda x: kt_head(sigma,list(x))[0], lists))
	
def kt_min_parallel(lists,n,threads=32):
	p = Pool(threads)
	comparisons = float(np.sum(map(lambda x: len(x)*(len(x)-1)/2+ len(x)*(n-len(x)),lists)))
	S = [(x,lists) for x in permutations(range(n))]
	invs = p.map(kt_sum,S)
	p.close()
	p.join()
	idx = np.argmin(invs)
	best_val = invs[idx]
	best_sigma = S[idx][0]
	theta = np.log(float(best_val)/float(comparisons))
	return best_sigma,-theta
	

def kt_min(lists,n):
	best = None
	best_val = n*(n-1)/2*len(lists)
	comparisons = float(np.sum(map(lambda x: len(x)*(len(x)-1)/2+ len(x)*(n-len(x)),lists)))
	for sigma in permutations(range(n)):
		k = np.sum(map(lambda x: kt_head(sigma,list(x))[0], lists))
		
		if k<best_val:
			best_val = k
			best_sigma = sigma

	theta = 0
	theta = np.log(float(best_val)/float(comparisons))
	return best_sigma,-theta
	
def pairs(sigma):
	return [(sigma[j],sigma[i]) for i in range(len(sigma)) for j in range(i)]
	
def pairs_head(sigma,n):
	L = pairs(sigma)
	L.extend([(x,y) for x in sigma for y in range(n) if y not in sigma])
	return L
	
def kt_head(sigma,head):
	"""
	returns the number of discordant pairs between full ranking sigma
	and any completion of head to a full list. Note that all items ranked by
	sigma but not head would be listed below every element of head
	"""
	n = len(sigma)
	pr = pairs(sigma)
	hdpr = pairs_head(head,n)
	swaps = [x for x in hdpr if x not in pr]
	return len(swaps),swaps	

def kt(sigma,sigma_0):
	"""
	returns the Kendall-Tau distance of the input permutations
	
	Arguments:
	sigma,sigma_0 - permutations whose Kendall-Tau distance is computed
	"""
	pairs = [(i,j) for i in sigma_0 for j in sigma_0 if i<j]
	swaps = [(i,j) for (i,j) in pairs if (sigma.index(i)<sigma.index(j)) != (sigma_0.index(i)<sigma_0.index(j))]
	return len(swaps),swaps
	
def kt_D(sigma,sigma_0,D):
	"""
	returns Kendall-Tau distance weighted by matrix D
	
	Arguments:
	sigma,sigma_0 - permutations whose weighted Kendall-Tau distance is computed
	D - matrix with D[i,j] as the penalty for placing i and j out of order
	"""
	n = len(sigma)
	pairs = [(i,j) for j in range(n) for i in range(j)]
	swaps = [(i,j) for (i,j) in pairs if (sigma.index(i)<sigma.index(j)) != (sigma_0.index(i)<sigma_0.index(j))]	
	return np.sum(np.array([D[i,j] for (i,j) in swaps])),swaps
	
def gen_kt(sigma,sigma_0,w,delta,D):
	"""
	returns Generalized Kendall-Tau 
	
	Arguments:
	sigma,sigma_0 - permutations whose generalized Kendall-Tau distance is computed
	w- element weights
	delta- position weights
	D- pairwise weights
	"""
	n = len(sigma)
	swaps = [(i,j) for j in range(n) for i in range(j) if (sigma.index(i)>sigma.index(j)) and (sigma_0.index(i)<sigma_0.index(j))]
	d = 0
	for (i,j) in swaps:
		pbar[i]=np.sum(delta[:i]-delta[:sigma(i)])
		pbar[j]=np.sum(delta[:j]-delta[:sigma(j)])
		d+=w[i]*w[j]*D[i,j]*pbar[i]*pbar[j]
	return d,swaps

def mallows(theta,sigma_0):
	"""returns the probability distribution implied by the Mallows model with
	parameters theta and sigma_0
	
	pr(theta) \propto exp(-theta*kt(sigma,sigma_0))
	Arguments:
	theta - concentration parameter
	sigma_0 - reference permutation
	"""
	n = len(sigma_0)
	S_n = permutations(range(n))
	pi = np.zeros(factorial(n))
	for k in xrange(factorial(n)):
		sigma = S_n.next()
		pi[k] = np.exp(-theta*kt(sigma,sigma_0)[0])
	
	return pi/np.sum(pi)

def mallows_mixture(alpha,theta,sigma):
	"""returns a mixture of Mallows probabilities distributions:
	
	pr(theta) \propto \sum_k alpha[k]*exp(-theta[k]*kt(sigma,sigma[k]))
	
	Arguments:
	alpha - weights for each Mallows distribution
	theta - vector of concentration parameters for the Mallows distributions
	sigma - vector of reference permutations for the Mallows distributions
	"""
	n = len(sigma[0])
	pi = np.zeros(factorial(n))
	for k in range(len(alpha)):
		pi+=alpha[k]*mallows(theta[k],sigma[k])[0]
	
	return pi
	
def mallows_product(theta,sigma):
	"""returns a product of Mallows distributions, i.e. a Mallows distribution on a
	sum of distances from different reference parameters:
	
	pr(sigma) \propto exp(- sum_k theta[k] kt(sigma,sigma[k]))
	
	Arguments:
	theta - vector of concentration parameters
	sigma - vector of reference permutations
	"""
	n = len(sigma[0])
	pi = np.ones(factorial(n))
	for k in range(len(theta)):
		pi*=mallows(theta[k],sigma[k])
	return pi/np.sum(pi)

