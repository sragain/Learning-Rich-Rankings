import numpy as np
from matplotlib import pyplot as plt
import pickle
from multiprocessing import Pool
import os
import lib.pcmc_utils_grad,lib.RS_mmnl_utils,lib.PL_utils,lib.RS_pcmc_utils,lib.RS_mallows_utils
from matplotlib import pyplot as plt
from scipy.stats import sem


	
def AP_G(sigma,g):
	n = len(g)
	
	#G contains 1's where g>0, then is shuffled by sigma
	G = np.copy(g).astype(float)
	G[np.where(G>0)]=1.0
	G=G[sigma]
	
	p = np.zeros(n)
	R = float(np.sum(G))
	r = np.zeros(n)
	p[0] = G[0]
	r[0] = G[0]/R
	AP=0
	for i in range(1,n):
		p[i]= np.sum(G[:(i+1)])/(i+1)
		r[i]= np.sum(G[:(i+1)])/R
		AP += p[i]*(r[i]-r[i-1])
	return AP

def NDCG(sigma,g):
	assert len(sigma)==len(g)
	n = len(sigma)
	if np.sum(g)==0:
		return np.zeros(n) #letor script appears to do this
	idcg = np.zeros(n)
	dcg = np.zeros(n)		
	idx = np.argsort(-g)

	dcg[0] = 2**g[sigma[0]]-1
	idcg[0] = 2**g[idx[0]]-1
	for i in range(1,n):
		dcg[i]=dcg[i-1]+(2**g[sigma[i]]-1)/np.log2(i+2)
		idcg[i]=idcg[i-1]+(2**g[idx[i]]-1)/np.log2(i+2)

	ndcg = np.divide(dcg,idcg)
	
	return ndcg
	
def NDCG_2(sigma,g):
	assert len(sigma)==len(g)
	n = len(sigma)
	if np.sum(g)==0:
		return np.zeros(n) #letor script appears to do this
	idcg = np.zeros(n)
	dcg = np.zeros(n)		
	idx = np.argsort(-g)

	dcg[0] = g[sigma[0]]
	idcg[0] = g[idx[0]]
	for i in range(1,n):
		dcg[i]=dcg[i-1]+g[sigma[i]]/np.log2(i+2)
		idcg[i]=idcg[i-1]+g[idx[i]]/np.log2(i+2)

	ndcg = np.divide(dcg,idcg)
	
	return ndcg	

def compute_error(M,N,G,max_N,skip_zeros=False):
	ndcg = np.zeros(max_N)
	counts = np.zeros(max_N)
	for qid in M:
		if skip_zeros and np.sum(G[qid])==0:
			continue
		ndcg[:N[qid]]+=M[qid]
		counts[:N[qid]]+=1
	return np.divide(ndcg,counts)

def ndcg_tests(model_perm,test_lists,n):
	gain = 0
	L = map(list,test_lists)
	for l in L:
		g = np.zeros(n)
		for i in range(n):
			if i in l:
				g[i] = n-l.index(i)
			else:
				g[i] = 0
		gain += NDCG(model_perm,g)
	return gain/len(test_lists)

def map_tests(model_perm,test_lists,n):
	gain = 0
	L = map(list,test_lists)
	for l in L:
		g = np.zeros(n)
		for i in range(n):
			if i in l:
				g[i] = n-l.index(i)
			else:
				g[i] = 0
		gain += MAP(model_perm,g)
	return gain/len(test_lists)
	
def supervised_metrics(max_N):
	dir = os.getcwd() 
	rel_path = 'results'+os.sep+'MQ'+os.sep#
	path = dir+os.sep+rel_path
	G=pickle.load(open(path+'MQ-ground-truth.p','rb'))	
	N=pickle.load(open(path+'MQ-N.p','rb'))
	skip_zeros = False
	
	PL = pickle.load(open(path+'MQ-all-RS-PL-'+str(max_N)+'.p','rb'))
	M_PL = PL_NDCG_greedy(PL,G)
	print 'PL:'
	print compute_error(M_PL,N,G,max_N,skip_zeros)
	
	MMNL = pickle.load(open(path+'MQ-all-RS-MMNL-'+str(max_N)+'.p','rb'))
	M_MMNL = mmnl_NDCG_greedy(MMNL,G,N)
	print 'MMNL:'
	print compute_error(M_MMNL,N,G,max_N,skip_zeros)
	
	PCMC = pickle.load(open(path+'MQ-all-RS-PCMC-'+str(max_N)+'.p','rb'))
	M_PCMC = pcmc_NDCG_greedy(PCMC,G)
	print 'PCMC:'
	print compute_error(M_PCMC,N,G,max_N,skip_zeros)
	
	MC1 = pickle.load(open(path+'MQ-all-mc1-'+str(max_N)+'.p','rb'))
	print 'MC1:'
	print compute_error_full(MC1,N,G,max_N,skip_zeros)	
	
	MC2 = pickle.load(open(path+'MQ-all-mc2-'+str(max_N)+'.p','rb'))
	print 'MC2:'
	print compute_error_full(MC2,N,G,max_N,skip_zeros)	
	
	MC3 = pickle.load(open(path+'MQ-all-mc3-'+str(max_N)+'.p','rb'))
	print 'MC3:'
	print compute_error_full(MC3,N,G,max_N,skip_zeros)	
	
	MC4 = pickle.load(open(path+'MQ-all-mc4-'+str(max_N)+'.p','rb'))
	print 'MC4:'
	print compute_error_full(MC4,N,G,max_N,skip_zeros)		
	
	borda = pickle.load(open(path+'MQ-all-borda-'+str(max_N)+'.p','rb'))
	print 'borda:'
	print compute_error_full(borda,N,G,max_N,skip_zeros)	

def unsupervised_metrics(max_N):
	
	#get problem info
	dir = os.getcwd() 
	rel_path = 'results'+os.sep+'MQ'+os.sep#
	path = dir+os.sep+rel_path
	N=pickle.load(open(path+'MQ-N.p','rb'))
	test_lists=pickle.load(open(path+'MQ-test-lists-'+str(max_N)+'.p','rb'))
	skip_zeros = False
	
	#load in learned params
	PL = pickle.load(open(path+'MQ-train-RS-PL-'+str(max_N)+'.p','rb'))
	MMNL = pickle.load(open(path+'MQ-train-RS-MMNL-'+str(max_N)+'.p','rb'))
	PCMC = pickle.load(open(path+'MQ-train-RS-PCMC-'+str(max_N)+'.p','rb'))
	if max_N<=8:
		mallows = pickle.load(open(path+'MQ-train-mallows-'+str(max_N)+'.p','rb'))
	mallows_greedy = pickle.load(open(path+'MQ-train-greedy-mallows-'+str(max_N)+'.p','rb'))

	#structs for storing errors
	PL_RS_scores =  [[] for _ in range(max_N+1)]
	MMNL_RS_scores = [[] for _ in range(max_N+1)]
	PCMC_RS_scores =  [[] for _ in range(max_N+1)]
	Mallows_scores =  [[] for _ in range(max_N+1)]
	Mallows_greedy_scores = [[] for _ in range(max_N+1)]

		
	#counts number of problem instances
	cnt = np.zeros(max_N+1)
		
		

	for qid in PL:
		cnt[N[qid]]+=1
		pl_perm = lib.PL_utils.greedy_selection_perm(PL[qid])
		mmnl_perm = lib.RS_mmnl_utils.greedy_selection_perm(MMNL[qid],N[qid])
		pcmc_perm = lib.RS_pcmc_utils.greedy_selection_perm(PCMC[qid])
		mallows_greedy_perm = mallows_greedy[qid][0]
  
		PL_RS_scores[N[qid]].extend(ndcg_tests(pl_perm,test_lists[qid],N[qid]))
		MMNL_RS_scores[N[qid]].extend(ndcg_tests(mmnl_perm,test_lists[qid],N[qid]))
		PCMC_RS_scores[N[qid]].extend(ndcg_tests(pcmc_perm,test_lists[qid],N[qid]))
		Mallows_greedy_scores[N[qid]].extend(ndcg_tests(mallows_greedy_perm,test_lists[qid],N[qid]))
		
		if max_N<=8:
			mallows_perm = mallows[qid][0]
			Mallows_scores[N[qid]].extend(ndcg_tests(mallows_perm,test_lists[qid],N[qid]))
	
	print 'PL:'
	print np.mean(PL_RS_scores[-1]),sem(PL_RS_scores[-1])
	
	print 'MMNL:'
	print np.mean(MMNL_RS_scores[-1]),sem(MMNL_RS_scores[-1])	
	
	print 'PCMC:'
	print np.mean(PCMC_RS_scores[-1]),sem(PCMC_RS_scores[-1])		
	
	if max_N<=8:
		print 'Mallows:'
		print np.mean(Mallows_scores[-1]),sem(Mallows_scores[-1])		
	print 'Mallows approx:'
	print np.mean(Mallows_greedy_scores[-1]),sem(Mallows_greedy_scores[-1])				

if __name__=='__main__':
	np.set_printoptions(suppress=True,precision=3)

	max_N = 16
	unsupervised_metrics(max_N)
