import numpy as np
import lib.PL_utils,lib.RS_mmnl_utils,lib.RS_pcmc_utils,lib.RS_mallows_utils
import os
import pickle
from scipy.stats import sem
from matplotlib import pyplot as plt


def quadratic_score(r,p):
	return 2*r-np.sum(np.power(p,2))

def greedy_selection_score(r,p):
	return r

def correct_mode(r,p):
	return int(r == np.amax(p))
	
def logarithmic_score(r,p):
	return np.log(r)

def spherical_score(r,p):
	return r/np.linalg.norm(p,2)

def soi_score_plots(score):
	path = os.getcwd()+os.sep+'results'+os.sep+'soi'+os.sep
	PL = {};MMNL={};PCMC={};MG = {}
	m = 0
	count=0
	for filename in os.listdir(path):
		if filename.endswith('.p'):
			pl,mmnl,pcmc,mg = plot(filename,score,False)
			if pl is None:
				continue
			n = len(pl)
			count+=1
			m = max(m,n)
			for i in range(n):
				if i not in PL:
					PL[i]=[]
					MMNL[i]=[]
					PCMC[i]=[]
					MG[i]=[]
				PL[i].extend(pl[i])
				MMNL[i].extend(mmnl[i])
				PCMC[i].extend(pcmc[i])
				MG[i].extend(mg[i])
	print count
	for i in PL:
		print i, len(PL[i])
	assert False
	plot_scores_nomallows(PL,MMNL,PCMC,MG,'Preflib',score)

def plot(filename,score,p=True):
	path = os.getcwd()+os.sep+'results'+os.sep+'soi'+os.sep
	D = pickle.load(open(path+filename,'rb'))	
	PL=D['PL']
	MMNL=D['MMNL']
	PCMC=D['PCMC']
	N = D['N']
	if N>10:
		return None,None,None,None
	test_lists = D['test-lists']
	(sigma_greedy,theta_greedy) = D['Mallows-greedy']
	PL_RS_scores =  lib.PL_utils.greedy_sequential_prediction_error(PL,score,test_lists)
	MMNL_RS_scores =  lib.RS_mmnl_utils.greedy_sequential_prediction_error(MMNL,score,test_lists,N)
	PCMC_RS_scores =  lib.RS_pcmc_utils.greedy_sequential_prediction_error(PCMC,score,test_lists)
	Mallows_greedy_scores =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_greedy,theta_greedy,score,test_lists)
		
	ttl = filename[:-9]
	if p:
		plot_scores_nomallows(PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores,title = ttl,s=score)
			
	return PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores

	
def plot_scores_nomallows(PL,MMNL,PCMC,Mallows_greedy,title,s):
	"""
	plots errors
	"""
	n = len(PL)
	f=plt.figure(figsize=(5,4))
	obs = np.array([i for i in range(1,n) if len(PL[i])>0])
	PL_mean = [np.mean(PL[i]) for i in obs]
	MMNL_mean = [np.mean(MMNL[i]) for i in obs]
	PCMC_mean = [np.mean(PCMC[i]) for i in obs]
	Mallows_greedy_mean = [np.mean(Mallows_greedy[i]) for i in obs]
	PL_err= [sem(PL[i]) for i in obs]
	MMNL_err= [sem(MMNL[i]) for i in obs]
	PCMC_err= [sem(PCMC[i]) for i in obs]
	Mallows_greedy_err= [sem(Mallows_greedy[i]) for i in obs]

	plt.errorbar(x=obs+1,y=PL_mean,yerr=PL_err,label='PL',marker='o',linestyle='')
	plt.errorbar(x=obs+1,y=MMNL_mean,yerr=MMNL_err,label='RS-MMNL',marker='o',linestyle='')
	plt.errorbar(x=obs+1,y=PCMC_mean,yerr=PCMC_err,label='RS-PCMC',marker='o',linestyle='')
	plt.errorbar(x=obs+1,y=Mallows_greedy_mean,yerr=Mallows_greedy_err,label='Mallows approx',marker='o',linestyle='')	
	plt.xlabel('number of unranked elements')
	plt.ylabel('error guessing next element')
	scoring_name = s.__name__.replace('_','-')
	if s in [greedy_selection_score,correct_mode]:
		plt.ylim(0,1)
	plt.xlim(1.5,n+.5)
	plt.gca().invert_xaxis()

	plt.ylabel(scoring_name.replace('-',' '))	
	plt.legend(loc='best',prop={'size':10})
	plt.title(title)
	dir = os.getcwd()+os.sep+'pictures'+os.sep+'soi'+os.sep
	f.tight_layout()
	plt.savefig(dir+title+'-'+scoring_name+'.pdf')
	plt.clf()


if __name__=='__main__':
	for s in [greedy_selection_score,quadratic_score,logarithmic_score]:
		soi_score_plots(s)
