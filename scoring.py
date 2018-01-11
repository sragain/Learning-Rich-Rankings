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

def debian_score(score,year,N,plot=True):
	path = os.getcwd()+os.sep+'results'+os.sep+'debian'+os.sep
	test_lists = pickle.load(open(path+'debian-'+year+'-test-lists.p','rb'))	
	
	PL = np.load(path+'debian-'+year+'-PL.npy')
	MMNL = np.load(path+'debian-'+year+'-MMNL.npy')
	PCMC = np.load(path+'debian-'+year+'-PCMC.npy')
	sigma_greedy = np.load(path+'debian-'+year+'-sigma.npy')
	theta_greedy = np.load(path+'debian-'+year+'-theta.npy')
	
	
	PL_RS_scores =  lib.PL_utils.greedy_sequential_prediction_error(PL,score,test_lists)
	MMNL_RS_scores =  lib.RS_mmnl_utils.greedy_sequential_prediction_error(MMNL,score,test_lists,N)
	PCMC_RS_scores =  lib.RS_pcmc_utils.greedy_sequential_prediction_error(PCMC,score,test_lists)
	Mallows_greedy_scores =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_greedy,theta_greedy,score,test_lists)
		
	ttl = 'debian '+year
	if plot:
		plot_scores_nomallows(PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores,title = ttl,s=score)
			
	return PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores
	
def MQ_score(score,max_N=10):
	path = os.getcwd()+os.sep+'results'+os.sep+'MQ'+os.sep
	N = pickle.load(open(path+'MQ-N.p','rb'))
	test = pickle.load(open(path+'MQ-test-lists-'+str(max_N)+'.p','rb'))	
	
	PL_RS = pickle.load(open(path+'MQ-train-RS-PL-'+str(max_N)+'.p','rb'))	
	MMNL_RS = pickle.load(open(path+'MQ-train-RS-MMNL-'+str(max_N)+'.p','rb'))	
	PCMC_RS = pickle.load(open(path+'MQ-train-RS-PCMC-'+str(max_N)+'.p','rb'))
	if max_N<=8:
		Mallows = pickle.load(open(path+'MQ-train-mallows-'+str(max_N)+'.p'))
	Mallows_greedy = pickle.load(open(path+'MQ-train-greedy-mallows-'+str(max_N)+'.p'))
	
	#average errors
	PL_RS_scores = {}
	MMNL_RS_scores ={}
	PCMC_RS_scores = {}
	Mallows_scores = {}
	Mallows_greedy_scores = {}
	
	cnt = np.zeros(max_N+1)
	
	for i in range(2,max_N+1):
		PL_RS_scores[i]=[[] for _ in range(i)]
		PCMC_RS_scores[i]=[[] for _ in range(i)]
		MMNL_RS_scores[i]=[[] for _ in range(i)]
		Mallows_scores[i]=[[] for _ in range(i)]
		Mallows_greedy_scores[i]=[[] for _ in range(i)]
	
	for qid in PL_RS:
		cnt[N[qid]]+=1
		PL_RS_scr =  lib.PL_utils.greedy_sequential_prediction_error(PL_RS[qid],score,test[qid])
		MMNL_RS_scr =  lib.RS_mmnl_utils.greedy_sequential_prediction_error(MMNL_RS[qid],score,test[qid],N[qid])
		PCMC_RS_scr =  lib.RS_pcmc_utils.greedy_sequential_prediction_error(PCMC_RS[qid],score,test[qid])
		
		if max_N<=8:
			sigma_0,theta = Mallows[qid]
			Mallows_scr =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_0,theta,score,test[qid])
		sigma_greedy,theta_greedy = Mallows_greedy[qid]
		Mallows_greedy_scr =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_greedy,theta_greedy,score,test[qid])
		
		for i in range(N[qid]):
			PL_RS_scores[N[qid]][i].extend(PL_RS_scr[i])
			MMNL_RS_scores[N[qid]][i].extend(MMNL_RS_scr[i])
			PCMC_RS_scores[N[qid]][i].extend(PCMC_RS_scr[i])
			if max_N<=8:
				Mallows_scores[N[qid]][i].extend(Mallows_scr[i])
			Mallows_greedy_scores[N[qid]][i].extend(Mallows_greedy_scr[i])

		
	for i in range(max_N,max_N+1):
		if np.sum([len(x) for x in PL_RS_scores[i]])>0:
			ttl = 'LETOR queries with '+str(i)+' documents'
			plot_scores(PL_RS_scores[i],MMNL_RS_scores[i],PCMC_RS_scores[i],Mallows_scores[i],Mallows_greedy_scores[i],title = ttl,s=score)
			
	return None
	
def sushi_scores(score,plot=True):
	path = os.getcwd()+os.sep+'results'+os.sep+'sushi'+os.sep
	N=10
	test_lists = np.load(path+'test-lists.npy')
	
	PL = np.load(path+'sushi-PL.npy')
	MMNL = np.load(path+'sushi-MMNL.npy')
	PCMC = np.load(path+'sushi-PCMC.npy')
	sigma_greedy = np.load(path+'greedy-sigma.npy')
	theta_greedy = np.load(path+'greedy-theta.npy')
	
	PL_RS_scores =  lib.PL_utils.greedy_sequential_prediction_error(PL,score,test_lists)
	MMNL_RS_scores =  lib.RS_mmnl_utils.greedy_sequential_prediction_error(MMNL,score,test_lists,N)
	PCMC_RS_scores =  lib.RS_pcmc_utils.greedy_sequential_prediction_error(PCMC,score,test_lists)
	Mallows_greedy_scores =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_greedy,theta_greedy,score,test_lists)

	ttl = 'sushi'
	if plot:
		plot_scores_nomallows(PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores,title = ttl,s=score)
			
	return PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores

def west_scores(score,plot=True):
	path = os.getcwd()+os.sep+'results'+os.sep+'election'+os.sep
	N=9
	test_lists = pickle.load(open(path+'west-test-lists.p','rb'))	
	
	PL = np.load(path+'west-PL.npy')
	MMNL = np.load(path+'west-MMNL.npy')
	PCMC = np.load(path+'west-PCMC.npy')
	sigma_greedy = np.load(path+'west-greedy-sigma.npy')
	theta_greedy = np.load(path+'west-greedy-theta.npy')
	
	
	PL_RS_scores =  lib.PL_utils.greedy_sequential_prediction_error(PL,score,test_lists)
	MMNL_RS_scores =  lib.RS_mmnl_utils.greedy_sequential_prediction_error(MMNL,score,test_lists,N)
	PCMC_RS_scores =  lib.RS_pcmc_utils.greedy_sequential_prediction_error(PCMC,score,test_lists)
	Mallows_greedy_scores =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_greedy,theta_greedy,score,test_lists)
		
	ttl = 'west-election'
	if plot:
		plot_scores_nomallows(PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores,title = ttl,s=score)
			
	return PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores
	
def north_scores(score,plot=True):
	path = os.getcwd()+os.sep+'results'+os.sep+'election'+os.sep
	N=12
	test_lists = pickle.load(open(path+'north-test-lists.p','rb'))	
	
	PL = np.load(path+'north-PL.npy')
	MMNL = np.load(path+'north-MMNL.npy')
	PCMC = np.load(path+'north-PCMC.npy')
	sigma_greedy = np.load(path+'north-greedy-sigma.npy')
	theta_greedy = np.load(path+'north-greedy-theta.npy')
	
	PL_RS_scores =  lib.PL_utils.greedy_sequential_prediction_error(PL,score,test_lists)
	MMNL_RS_scores =  lib.RS_mmnl_utils.greedy_sequential_prediction_error(MMNL,score,test_lists,N)
	PCMC_RS_scores =  lib.RS_pcmc_utils.greedy_sequential_prediction_error(PCMC,score,test_lists)
	Mallows_greedy_scores =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_greedy,theta_greedy,score,test_lists)

	ttl = 'north-election'
	if plot:
		plot_scores_nomallows(PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores,title = ttl,s=score)
			
	return PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores

def meath_scores(score,plot=True):
	path = os.getcwd()+os.sep+'results'+os.sep+'election'+os.sep
	N=14
	test_lists = pickle.load(open(path+'meath-test-lists.p','rb'))	
	
	PL = np.load(path+'meath-PL.npy')
	MMNL = np.load(path+'meath-MMNL.npy')
	PCMC = np.load(path+'meath-PCMC.npy')
	sigma_greedy = np.load(path+'meath-greedy-sigma.npy')
	theta_greedy = np.load(path+'meath-greedy-theta.npy')
	
	
	PL_RS_scores =  lib.PL_utils.greedy_sequential_prediction_error(PL,score,test_lists)
	MMNL_RS_scores =  lib.RS_mmnl_utils.greedy_sequential_prediction_error(MMNL,score,test_lists,N)
	PCMC_RS_scores =  lib.RS_pcmc_utils.greedy_sequential_prediction_error(PCMC,score,test_lists)
	Mallows_greedy_scores =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_greedy,theta_greedy,score,test_lists)
		
	ttl = 'meath-election'
	if plot:
		plot_scores_nomallows(PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores,title = ttl,s=score)
	
	return PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores

def aspen_scores(score,plot=True):
	path = os.getcwd()+os.sep+'results'+os.sep+'aspen'+os.sep
	N=11
	test_lists = pickle.load(open(path+'aspen-test-lists.p','rb'))	
	
	PL = np.load(path+'aspen-PL.npy')
	MMNL = np.load(path+'aspen-MMNL.npy')
	PCMC = np.load(path+'aspen-PCMC.npy')
	sigma_greedy = np.load(path+'aspen-greedy-sigma.npy')
	theta_greedy = np.load(path+'aspen-greedy-theta.npy')
	
	
	PL_RS_scores =  lib.PL_utils.greedy_sequential_prediction_error(PL,score,test_lists)
	MMNL_RS_scores =  lib.RS_mmnl_utils.greedy_sequential_prediction_error(MMNL,score,test_lists,N)
	PCMC_RS_scores =  lib.RS_pcmc_utils.greedy_sequential_prediction_error(PCMC,score,test_lists)
	Mallows_greedy_scores =  lib.RS_mallows_utils.greedy_sequential_prediction_error(sigma_greedy,theta_greedy,score,test_lists)
		
	ttl = 'aspen-election'
	if plot:
		plot_scores_nomallows(PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores,title = ttl,s=score)
	
	return PL_RS_scores,MMNL_RS_scores,PCMC_RS_scores,Mallows_greedy_scores
		
def plot_scores(PL,MMNL,PCMC,Mallows,Mallows_greedy,title,s):
	"""
	plots errors
	"""
	f=plt.figure(figsize=(5,4))
	n = len(PL)
	obs = np.array([i for i in range(1,n) if len(PL[i])>0])
	PL_mean = [np.mean(PL[i]) for i in obs]
	MMNL_mean = [np.mean(MMNL[i]) for i in obs]
	PCMC_mean = [np.mean(PCMC[i]) for i in obs]
	Mallows_greedy_mean = [np.mean(Mallows_greedy[i]) for i in obs]
	PL_err= [sem(PL[i]) for i in obs]
	MMNL_err= [sem(MMNL[i]) for i in obs]
	PCMC_err= [sem(PCMC[i]) for i in obs]
	Mallows_greedy_err= [sem(Mallows_greedy[i]) for i in obs]
	
	if n<=8:
		Mallows_mean = [np.mean(Mallows[i]) for i in obs]
		Mallows_err= [sem(Mallows[i]) for i in obs]
			
		plt.errorbar(x=obs+1,y=PL_mean,yerr=PL_err,label='PL',marker='o',linestyle='')
		plt.errorbar(x=obs+1,y=MMNL_mean,yerr=MMNL_err,label='RS-MMNL',marker='o',linestyle='')
		plt.errorbar(x=obs+1,y=PCMC_mean,yerr=PCMC_err,label='RS-PCMC',marker='o',linestyle='')
		plt.errorbar(x=obs+1,y=Mallows_mean,yerr=Mallows_err,label='Mallows',marker='o',linestyle='')
		plt.errorbar(x=obs+1,y=Mallows_greedy_mean,yerr=Mallows_greedy_err,label='Mallows approx',marker='o',linestyle='')
	
	else:
		plt.errorbar(x=obs+1,y=PL_mean,yerr=PL_err,label='PL',marker='o',linestyle='')
		plt.errorbar(x=obs+1,y=MMNL_mean,yerr=MMNL_err,label='RS-MMNL',marker='o',linestyle='')
		plt.errorbar(x=obs+1,y=PCMC_mean,yerr=PCMC_err,label='RS-PCMC',marker='o',linestyle='')
		plt.errorbar(x=obs+1,y=Mallows_greedy_mean,yerr=PCMC_err,label='Mallows approx',marker='o',linestyle='')	
	plt.xlabel('number of unranked elements')
	scoring_name = s.__name__.replace('_','-')

	plt.ylabel(scoring_name.replace('-',' '))

	if s in [greedy_selection_score,correct_mode]:
		plt.ylim(0,1)
	plt.xlim(1.5,n+.5)
	plt.gca().invert_xaxis()
	
	plt.ylabel(scoring_name.replace('-',' '))	
	plt.legend(loc='best',prop={'size':8})
	plt.title(title)
	dir = os.getcwd()+os.sep+'pictures'+os.sep
	f.tight_layout()

	plt.savefig(dir+'LETOR-'+str(n)+'-docs-'+scoring_name+'.pdf')
	plt.clf()
	
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

	plt.errorbar(x=obs+.925,y=PL_mean,yerr=PL_err,label='PL',marker='o',linestyle='')
	plt.errorbar(x=obs+.975,y=MMNL_mean,yerr=MMNL_err,label='RS-MMNL',marker='o',linestyle='')
	plt.errorbar(x=obs+1.025,y=PCMC_mean,yerr=PCMC_err,label='RS-PCMC',marker='o',linestyle='')
	plt.errorbar(x=obs+1.075,y=Mallows_greedy_mean,yerr=Mallows_greedy_err,label='Mallows approx',marker='o',linestyle='')	
	plt.xlabel('number of unranked elements')
	plt.ylabel('error guessing next element')
	scoring_name = s.__name__.replace('_','-')
	if s in [greedy_selection_score,correct_mode]:
		plt.ylim(0,1)
	#plt.ylim(0,1.1)
	plt.xlim(1.5,n+.5)
	plt.gca().invert_xaxis()

	plt.ylabel(scoring_name.replace('-',' '))	
	plt.legend(loc='best',prop={'size':8})
	plt.title(title)
	dir = os.getcwd()+os.sep+'pictures'+os.sep
	f.tight_layout()
	plt.savefig(dir+title+'-'+scoring_name+'.png')
	plt.clf()


def plot_scores_nomallows_ax(PL,MMNL,PCMC,Mallows_greedy,s,ax,legend=False):
	"""
	plots errors
	"""
	n = len(PL)
	obs = np.array([i for i in range(1,n) if len(PL[i])>0])
	PL_mean = [np.mean(PL[i]) for i in obs]
	MMNL_mean = [np.mean(MMNL[i]) for i in obs]
	PCMC_mean = [np.mean(PCMC[i]) for i in obs]
	Mallows_greedy_mean = [np.mean(Mallows_greedy[i]) for i in obs]
	PL_err= [sem(PL[i]) for i in obs]
	MMNL_err= [sem(MMNL[i]) for i in obs]
	PCMC_err= [sem(PCMC[i]) for i in obs]
	Mallows_greedy_err= [sem(Mallows_greedy[i]) for i in obs]
	ax.errorbar(x=obs+1,y=PL_mean,yerr=PL_err,label='PL',marker='o',linestyle='')
	ax.errorbar(x=obs+1,y=MMNL_mean,yerr=MMNL_err,label='RS-MMNL',marker='o',linestyle='')
	ax.errorbar(x=obs+1,y=PCMC_mean,yerr=PCMC_err,label='RS-PCMC',marker='o',linestyle='')
	ax.errorbar(x=obs+1,y=Mallows_greedy_mean,yerr=Mallows_greedy_err,label='Mallows approx',marker='o',linestyle='')	
	ax.set_xlabel('remaining number of elements')
	scoring_name = s.__name__.replace('_','-')
	ax.set_ylabel(scoring_name.replace('-',' '))
	if s in [greedy_selection_score,correct_mode]:
		ax.set_ylim(0,1)

	ax.set_xlim(1.5,n+.5)
	ax.set_xlim(ax.get_xlim()[::-1])
	#ax.set_ylabel(scoring_name)	
	if legend:
		ax.legend(loc='best',prop={'size':10})
	return ax	
	
def big_plot():
	f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize = (20,4))
	PL,MMNL,PCMC,Mallows_greedy = sushi_scores(greedy_selection_score,False)
	plot_scores_nomallows_ax(PL,MMNL,PCMC,Mallows_greedy,greedy_selection_score,ax1,True)	
	ax1.set_title('Sushi greedy selection error')
	PL,MMNL,PCMC,Mallows_greedy = sushi_scores(quadratic_score,False)	
	plot_scores_nomallows_ax(PL,MMNL,PCMC,Mallows_greedy,quadratic_score,ax2)	
	ax2.set_title('Sushi quadratic score')
	PL,MMNL,PCMC,Mallows_greedy = meath_scores(greedy_selection_score,False)
	plot_scores_nomallows_ax(PL,MMNL,PCMC,Mallows_greedy,greedy_selection_score,ax3)
	ax3.set_title('Election greedy selection error')
	PL,MMNL,PCMC,Mallows_greedy = meath_scores(quadratic_score,False)	
	plot_scores_nomallows_ax(PL,MMNL,PCMC,Mallows_greedy,quadratic_score,ax4)	
	ax4.set_title('Election quadratic score')
	dir = os.getcwd()+os.sep+'pictures'+os.sep
	f.tight_layout()
	plt.savefig(dir+'bigplot.pdf')
	#plt.show()
	#plt.clf()
	
if __name__=='__main__':
	np.set_printoptions(suppress=True,precision=3)
	#for s in [correct_mode,greedy_selection_score,logarithmic_score,spherical_score,quadratic_score]:
	for s in [greedy_selection_score,logarithmic_score,quadratic_score]:
		aspen_scores(s)
		#map(lambda (y,n): debian_score(s,y,n),[('2002',4),('2003',5),('2005',7),('2006',8),('2007',9),('2010',5),('2012',4)])
		#plt.close('all')
		#MQ_score(s,16)
		#MQ_score(s,8)
		#west_scores(s)
		#north_scores(s)
		#meath_scores(s)
		#sushi_scores(s)
	#plt.clf()	
	#big_plot()

