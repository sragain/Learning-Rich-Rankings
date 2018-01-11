import numpy as np
import lib.PL_utils,lib.RS_mmnl_utils,lib.RS_pcmc_utils,lib.RS_mallows_utils
import os,sys
import pickle
from scipy.stats import sem
from matplotlib import pyplot as plt

def greedy_score(r,p):
	return r

def quadratic_score(r,p):
	return 2*r-np.sum(np.power(p,2))

def soi_score_plots(path,dataset,score_fun):
	PL = {};MMNL={};PCMC={};MG = {}

	m = 0
	count=0
	for filename in os.listdir(path):
		if filename.endswith('.p'):
			pl,mmnl,pcmc,mg= score(path+filename,score_fun)
			if pl is None:
				continue
			n = len(pl)
			count+=1
			m = max(m,n)
			for i in range(n):
				if i not in PL:
					PL[i]=[]
					#MMNL[i]=[]
					PCMC[i]=[]
					MG[i]=[]

				PL[i].extend(pl[i])
				#MMNL[i].extend(mmnl[i])
				PCMC[i].extend(pcmc[i])
				MG[i].extend(mg[i])
	for i in PL:
		print i, len(PL[i])
	plot_scores(PL,MMNL,PCMC,MG,dataset,score_fun)

def score(filepath,score_fun):
	D = pickle.load(open(filepath,'rb'))
	N = D['N']
	PL=D['PL']
	MMNL=D['MMNL']
	PCMC=D['PCMC']
	(sigma_greedy,theta_greedy) = D['Mallows-greedy']
	test_lists = D['test-lists']
	PL_RS =  lib.PL_utils.greedy_sequential_prediction_error(PL,score_fun,test_lists)
	#MMNL_RS =  lib.RS_mmnl_utils.greedy_sequential_prediction_error(MMNL,score_fun,test_lists,N)
	PCMC_RS =  lib.RS_pcmc_utils.greedy_sequential_prediction_error(PCMC,score_fun,test_lists)
	Mallows_greedy =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_greedy,theta_greedy,score_fun,test_lists)
	return PL_RS,[],PCMC_RS,Mallows_greedy

def plot_scores(PL,MMNL,PCMC,Mallows_greedy,dataset,s):
	"""
	plots errors
	"""
	n = len(PL)
	f=plt.figure(figsize=(5,4))
	obs = np.array([i for i in range(1,n) if len(PL[i])>0])
	PL_mean = [np.mean(PL[i]) for i in obs]
	#MMNL_mean = [np.mean(MMNL[i]) for i in obs]
	PCMC_mean = [np.mean(PCMC[i]) for i in obs]
	Mallows_greedy_mean = [np.mean(Mallows_greedy[i]) for i in obs]

	PL_err= [sem(PL[i]) for i in obs]
	#MMNL_err= [sem(MMNL[i]) for i in obs]
	PCMC_err= [sem(PCMC[i]) for i in obs]
	Mallows_greedy_err= [sem(Mallows_greedy[i]) for i in obs]

	scoring_name = s.__name__.replace('_','-')
	save_dir = os.getcwd()+os.sep+'pictures'+os.sep+dataset+os.sep

	lnsty='-'
	plt.errorbar(x=obs+1,y=PL_mean,yerr=PL_err,label='PL',marker='o',linestyle=lnsty)
	plt.errorbar(x=obs+1,y=PCMC_mean,yerr=PCMC_err,label='RS-PCMC',marker='o',linestyle=lnsty)
	plt.errorbar(x=obs+1,y=Mallows_greedy_mean,yerr=Mallows_greedy_err,label='RS-Mallows',marker='o',linestyle=lnsty)
	plt.xlabel('k')
	plt.ylabel(scoring_name.replace('-',' ')+' for dist of sigma(k)')
	plt.title(dataset+' '+scoring_name+' RS')
	plt.legend(loc='best')
	plt.gca().invert_xaxis()
	f.tight_layout()


	plt.savefig(save_dir+dataset+'-'+scoring_name+'RS.pdf')




if __name__=='__main__':
    np.set_printoptions(suppress=True, precision=3)
    if sys.argv[1] not in ['soi','election']:
		print 'wrong data folder'
		assert False
    else:
        path = os.getcwd()+os.sep+'results'+os.sep+sys.argv[1]+os.sep
	for s in [quadratic_score,greedy_score]:
		soi_score_plots(path,sys.argv[1],s)
