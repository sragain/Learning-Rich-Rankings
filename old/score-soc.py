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

def soc_score_plots(score_fun):
	path = os.getcwd()+os.sep+'results'+os.sep+'soc'+os.sep
	PL = {};MMNL={};PCMC={};MG = {};PLRE={};MMNLRE={};PCMCRE={};MGRE={}

	m = 0
	count=0
	for filename in os.listdir(path):
		if filename.endswith('.p'):
			pl,mmnl,pcmc,mg,plre,mmnlre,pcmcre,mgre = score(filename,score_fun)
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
					PLRE[i]=[]
					MMNLRE[i]=[]
					PCMCRE[i]=[]
					MGRE[i]=[]
				PL[i].extend(pl[i])
				MMNL[i].extend(mmnl[i])
				PCMC[i].extend(pcmc[i])
				MG[i].extend(mg[i])
				PLRE[i].extend(plre[i])
				MMNLRE[i].extend(mmnlre[i])
				PCMCRE[i].extend(pcmcre[i])
				MGRE[i].extend(mgre[i])
	print count
	for i in PL:
		print i, len(PL[i])
	plot_scores(PL,MMNL,PCMC,MG,PLRE,MMNLRE,PCMCRE,MGRE,'Preflib',score_fun)

def score(filename,score_fun):
	path = os.getcwd()+os.sep+'results'+os.sep+'soc'+os.sep
	D = pickle.load(open(path+filename,'rb'))
	N = D['N']
	PL=D['PL']
	MMNL=D['MMNL']
	PCMC=D['PCMC']
	PL_RE=D['PL-RE']
	MMNL_RE=D['MMNL-RE']
	PCMC_RE=D['PCMC-RE']
	(sigma_greedy,theta_greedy) = D['Mallows-greedy']
	(sigma_greedy_RE,theta_greedy_RE) = D['Mallows-greedy-RE']

	if N>10:
		return None
	test_lists = D['test-lists']
	flipped_lists = map(lambda sigma: sigma[::-1],test_lists)
	PL_RS =  lib.PL_utils.greedy_sequential_prediction_error(PL,score_fun,test_lists)
	MMNL_RS =  lib.RS_mmnl_utils.greedy_sequential_prediction_error(MMNL,score_fun,test_lists,N)
	PCMC_RS =  lib.RS_pcmc_utils.greedy_sequential_prediction_error(PCMC,score_fun,test_lists)
	Mallows_greedy =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_greedy,theta_greedy,score_fun,test_lists)

	PL_RE =  lib.PL_utils.greedy_sequential_prediction_error(PL_RE,score_fun,flipped_lists)
	MMNL_RE =  lib.RS_mmnl_utils.greedy_sequential_prediction_error(MMNL_RE,score_fun,flipped_lists,N)
	PCMC_RE =  lib.RS_pcmc_utils.greedy_sequential_prediction_error(PCMC_RE,score_fun,flipped_lists)
	Mallows_greedy_RE =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_greedy_RE,theta_greedy_RE,score_fun,flipped_lists)
	ttl = filename[:-9]
	#plot_scores(PL_RS,MMNL_RS,PCMC_RS,Mallows_greedy,PL_RE,MMNL_RE,PCMC_RE,Mallows_greedy_RE,title = ttl,s=score)
	return PL_RS,MMNL_RS,PCMC_RS,Mallows_greedy,PL_RE,MMNL_RE,PCMC_RE,Mallows_greedy_RE

def plot_scores(PL,MMNL,PCMC,Mallows_greedy,PL_RE,MMNL_RE,PCMC_RE,Mallows_greedy_RE,title,s):
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
	PL_RE_mean = [np.mean(PL_RE[i]) for i in obs]
	MMNL_RE_mean = [np.mean(MMNL_RE[i]) for i in obs]
	PCMC_RE_mean = [np.mean(PCMC_RE[i]) for i in obs]
	Mallows_greedy_RE_mean = [np.mean(Mallows_greedy_RE[i]) for i in obs]

	PL_err= [sem(PL[i]) for i in obs]
	MMNL_err= [sem(MMNL[i]) for i in obs]
	PCMC_err= [sem(PCMC[i]) for i in obs]
	Mallows_greedy_err= [sem(Mallows_greedy[i]) for i in obs]
	PL_RE_err= [sem(PL_RE[i]) for i in obs]
	MMNL_RE_err= [sem(MMNL_RE[i]) for i in obs]
	PCMC_RE_err= [sem(PCMC_RE[i]) for i in obs]
	Mallows_greedy_RE_err= [sem(Mallows_greedy_RE[i]) for i in obs]


	lnsty='-'
	plt.errorbar(x=obs+1,y=PL_mean,yerr=PL_err,label='PL',marker='o',linestyle=lnsty)
	plt.errorbar(x=obs+1,y=MMNL_mean,yerr=MMNL_err,label='RS-MMNL',marker='o',linestyle=lnsty)
	plt.errorbar(x=obs+1,y=PCMC_mean,yerr=PCMC_err,label='RS-PCMC',marker='o',linestyle=lnsty)
	plt.errorbar(x=obs+1,y=Mallows_greedy_mean,yerr=Mallows_greedy_err,label='RS-Mallows',marker='o',linestyle=lnsty)

	plt.errorbar(x=obs+1,y=PL_RE_mean,yerr=PL_RE_err,label='RE-PL',marker='o',linestyle=lnsty)
	plt.errorbar(x=obs+1,y=MMNL_RE_mean,yerr=MMNL_RE_err,label='RE-MMNL',marker='o',linestyle=lnsty)
	plt.errorbar(x=obs+1,y=PCMC_RE_mean,yerr=PCMC_RE_err,label='RE-PCMC',marker='o',linestyle=lnsty)
	plt.errorbar(x=obs+1,y=Mallows_greedy_RE_mean,yerr=Mallows_greedy_RE_err,label='RE-Mallows',marker='o',linestyle=lnsty)


	plt.xlabel('number of unranked elements')
	plt.ylabel('error guessing next element')
	scoring_name = s.__name__.replace('_','-')
	if s in [greedy_selection_score,correct_mode]:
		plt.ylim(0,1)
	plt.xlim(1.5,n+.5)
	plt.gca().invert_xaxis()

	plt.ylabel(scoring_name.replace('-',' '))
	plt.legend(loc='best',prop={'size':4})
	#plt.legend(loc='best')
	plt.title(title)
	dir = os.getcwd()+os.sep+'pictures'+os.sep+'soc'+os.sep
	f.tight_layout()
	plt.savefig(dir+title+'-'+scoring_name+'.pdf')
	plt.clf()


if __name__=='__main__':

	for s in [greedy_selection_score,quadratic_score,logarithmic_score]:
		soc_score_plots(s)
