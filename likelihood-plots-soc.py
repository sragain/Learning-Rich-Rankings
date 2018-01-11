import numpy as np
import lib.PL_utils,lib.RS_mmnl_utils,lib.RS_pcmc_utils,lib.RS_mallows_utils
import os,sys
import pickle
from scipy.stats import sem
from matplotlib import pyplot as plt

def plot(dataset,max_N=12):
    path = os.getcwd()+os.sep+'results'+os.sep+'soc'+os.sep
    PL = {};MMNL={};PCMC={};MG = {};PLRE={};MMNLRE={};PCMCRE={};MGRE={}

    m = 0
    count=0
    for filename in os.listdir(path):
        if filename.endswith('.p'):
            print filename

            n,pl,mmnl,pcmc,mg,plre,mmnlre,pcmcre,mgre = likelihoods(filename,max_N)
            if n>max_N:
                continue
            if n not in PL:
                PL[n]=[]
                #MMNL[n]=[]
                PCMC[n]=[]
                MG[n]=[]
                PLRE[n]=[]
                #MMNLRE[n]=[]
                PCMCRE[n]=[]
                MGRE[n]=[]
            count+=1
            PL[n].extend(pl)
            #MMNL[n].extend(mmnl)
            PCMC[n].extend(pcmc)
            MG[n].extend(mg)
            PLRE[n].extend(plre)
            #MMNLRE[n].extend(mmnlre)
            PCMCRE[n].extend(pcmcre)
            MGRE[n].extend(mgre)
    print count
    ct2=0
    for i in PL:
        print i, len(PL[i])
        ct2+=len(PL[i])*5
    print ct2
    plot_scores(PL,MMNL,PCMC,MG,PLRE,MMNLRE,PCMCRE,MGRE,dataset,max_N)

def likelihoods(filename,max_N):
    path = os.getcwd()+os.sep+'results'+os.sep+'soc'+os.sep
    D = pickle.load(open(path+filename,'rb'))
    N = D['N']
    print N
    if N>max_N:
        return [[] for _ in range(9)]

    test_lists = D['test-lists']
    flipped_lists = map(lambda sigma: sigma[::-1],test_lists)

    #RS models
    PL =  np.array(map(lambda sigma: lib.PL_utils.log_PL_prob(sigma,D['PL']),test_lists))
    #MMNL =  np.array(map(lambda sigma: lib.RS_mmnl_utils.log_RS_prob(D['MMNL'],sigma,N),test_lists))
    PCMC =  np.array(map(lambda sigma: lib.RS_pcmc_utils.log_RS_prob(D['PCMC'],sigma),test_lists))
    (sigma_greedy,theta_greedy) = D['Mallows-greedy']
    Mallows_greedy =  np.array(map(lambda sigma: lib.RS_mallows_utils.log_RS_prob(sigma_greedy,theta_greedy,sigma),test_lists))

    #RE models
    PL_RE =  np.array(map(lambda sigma: lib.PL_utils.log_PL_prob(sigma,D['PL-RE']),flipped_lists))
    #MMNL_RE =  np.array(map(lambda sigma: lib.RS_mmnl_utils.log_RS_prob(D['MMNL-RE'],sigma,N),flipped_lists))
    PCMC_RE =  np.array(map(lambda sigma: lib.RS_pcmc_utils.log_RS_prob(D['PCMC-RE'],sigma),flipped_lists))
    (sigma_greedy_RE,theta_greedy_RE) = D['Mallows-greedy-RE']
    Mallows_greedy_RE =  np.array(map(lambda sigma: lib.RS_mallows_utils.log_RS_prob(sigma_greedy_RE,theta_greedy_RE,sigma),flipped_lists))
    ttl = filename[:-9]

    return N,PL,[],PCMC,Mallows_greedy,PL_RE,[],PCMC_RE,Mallows_greedy_RE

def plot_scores(PL,MMNL,PCMC,Mallows_greedy,PL_RE,MMNL_RE,PCMC_RE,Mallows_greedy_RE,dataset,max_N):
    """
    plots errors
    """
    f=plt.figure(figsize=(5,4))
    obs = np.array(PL.keys())
    n=max_N
    PL_mean = [np.mean(PL[i]) for i in obs]
    #MMNL_mean = [np.mean(MMNL[i]) for i in obs]
    PCMC_mean = [np.mean(PCMC[i]) for i in obs]
    Mallows_greedy_mean = [np.mean(Mallows_greedy[i]) for i in obs]
    PL_RE_mean = [np.mean(PL_RE[i]) for i in obs]
    #MMNL_RE_mean = [np.mean(MMNL_RE[i]) for i in obs]
    PCMC_RE_mean = [np.mean(PCMC_RE[i]) for i in obs]
    Mallows_greedy_RE_mean = [np.mean(Mallows_greedy_RE[i]) for i in obs]

    PL_err= [sem(PL[i]) for i in obs]
    #MMNL_err= [sem(MMNL[i]) for i in obs]
    PCMC_err= [sem(PCMC[i]) for i in obs]
    Mallows_greedy_err= [sem(Mallows_greedy[i]) for i in obs]
    PL_RE_err= [sem(PL_RE[i]) for i in obs]
    #MMNL_RE_err= [sem(MMNL_RE[i]) for i in obs]
    PCMC_RE_err= [sem(PCMC_RE[i]) for i in obs]
    Mallows_greedy_RE_err= [sem(Mallows_greedy_RE[i]) for i in obs]

    lnsty=''
    mrkr = 'x'
    plt.errorbar(x=obs,y=PL_mean,yerr=PL_err,label='PL',marker=mrkr,linestyle=lnsty)
    #plt.errorbar(x=obs,y=MMNL_mean,yerr=MMNL_err,label='RS-MMNL',marker=mrkr,linestyle=lnsty)
    plt.errorbar(x=obs,y=PCMC_mean,yerr=PCMC_err,label='RS-PCMC',marker=mrkr,linestyle=lnsty)
    plt.errorbar(x=obs,y=Mallows_greedy_mean,yerr=Mallows_greedy_err,label='RS-Mallows',marker=mrkr,linestyle=lnsty)

    plt.errorbar(x=obs,y=PL_RE_mean,yerr=PL_RE_err,label='RE-PL',marker=mrkr,linestyle=lnsty)
    #plt.errorbar(x=obs,y=MMNL_RE_mean,yerr=MMNL_RE_err,label='RE-MMNL',marker=mrkr,linestyle=lnsty)
    plt.errorbar(x=obs,y=PCMC_RE_mean,yerr=PCMC_RE_err,label='RE-PCMC',marker=mrkr,linestyle=lnsty)
    plt.errorbar(x=obs,y=Mallows_greedy_RE_mean,yerr=Mallows_greedy_RE_err,label='RE-Mallows',marker=mrkr,linestyle=lnsty)
    np.set_printoptions(suppress=True,precision=3)

    plt.xlabel('length of rankings')
    plt.ylabel('log-likelihood')
    #plt.xlim(1.5,n+.5)

    plt.legend(loc='best',prop={'size':6})
    #plt.legend(loc='best')
    plt.title('preflib log likelihoods')
    save_dir = os.getcwd()+os.sep+'pictures'+os.sep+dataset
    f.tight_layout()
    plt.savefig(save_dir+os.sep+'loglikelihoods.pdf')
    plt.clf()


if __name__=='__main__':
    plot(sys.argv[1])
