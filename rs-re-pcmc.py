import numpy as np
from lib.RS_pcmc_utils import *
from lib.pcmc_utils import infer
from itertools import permutations
from data.soi_scrape import RS_choices
from lib.ctmc_utils import solve_ctmc

def RS_prob_Q(Q,sigma):
	"""returns the probability of sigma under RS from pcmc
	Arguments:
	x- pcmc params
	"""
	n = len(sigma)
	p=1
	S = range(n)
	for i in range(n):
		pi = solve_ctmc(Q[S,:][:,S])
		p*= pi[S.index(sigma[i])]
		S.remove(sigma[i])
	return p

def choices(probs,n,RE=True):
    C = {}
    perms = permutations(range(n))
    for p in probs:
        sigma = perms.next()
        if RE:
            sigma=sigma[::-1]
        S = range(n)
        for i in range(len(sigma)):
            if tuple(S) not in C:
                C[tuple(S)] = np.zeros(len(S))#np.ones(len(S))*alpha
            C[tuple(S)][S.index(sigma[i])] += p
            S.remove(sigma[i])
    return C

n=4
S_n = [sigma for sigma in permutations(range(n))]
#Q = np.random.rand(n**2).reshape((n,n))
#Q = np.array([[0,.4,.4],[.1,0,.4],[.1,.4,0]])
L=[]
for i in range(4):
    L.append([.1,.2,.3,.4])
Q = np.array(L)

np.fill_diagonal(Q,0)
Q
P_RS = map(lambda sigma: RS_prob_Q(Q,sigma), S_n)
P_RS
C_RE = choices(P_RS,n)
Q_RE = comp_Q(infer(C_RE,n))
P_RE_try = map(lambda sigma: RS_prob_Q(Q_RE,sigma[::-1]),S_n)
print np.sum(np.abs(np.array(P_RS)-np.array(P_RE_try)))
Q_RE
