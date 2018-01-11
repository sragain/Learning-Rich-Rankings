import numpy as np
import lib.PL_utils,lib.RS_mmnl_utils,lib.RS_pcmc_utils,lib.RS_mallows_utils
import os,sys
import pickle
from scipy.stats import sem
from matplotlib import pyplot as plt

def plot(directory,dataset,max_N=15):
    PL = {};MMNL={};PCMC={};MG = {}
    for n in range(max_N):
        PL[n]=[]
        MMNL[n]=[]
        PCMC[n]=[]
        MG[n]=[]
    m = 0
    count=0
    print directory
    for filename in os.listdir(directory):
        if filename.endswith('.p'):
            print filename

            n,pl,mmnl,pcmc,mg = likelihoods_soi(directory+os.sep+filename,max_N)
            if n>max_N:
                continue


            count+=1
            for (nelt,val) in pl:
                PL[nelt].append(val)
            for (nelt,val) in mmnl:
                MMNL[nelt].append(val)
            for (nelt,val) in pcmc:
                PCMC[nelt].append(val)
            for (nelt,val) in mg:
                MG[nelt].append(val)

    for i in PL:
        print i, len(PL[i])
    if directory == 'sushi':
        dir = os.getcwd()+os.sep+'pictures'+os.sep+dataset+os.sep
        with open(dir+'log-likelihoods-soi.txt','w') as g:
            g.write('n =    '+str(np.around(obs, decimals=3))+'\n')
            g.write('PL:    '+str(np.around(PL_mean, decimals=3))+'\n')
            g.write('MMNL:  '+str(np.around(MMNL_mean, decimals=3))+'\n')
            g.write('PCMC:  '+str(np.around(PCMC_mean, decimals=3))+'\n')
            g.write('Mall:  '+str(np.around(Mallows_greedy_mean, decimals=3))+'\n')
    else:
        plot_scores(PL,MMNL,PCMC,MG,max_N,dataset)

def likelihoods_soi(filepath,max_N):
    D = pickle.load(open(filepath,'rb'))
    N = D['N']
    print N
    if N>max_N:
        return []*4

    test_lists = D['test-lists']
    flipped_lists = map(lambda sigma: sigma[::-1],test_lists)

    #RS models
    PL =  np.array(map(lambda sigma: (len(sigma),lib.PL_utils.log_PL_prob(sigma,D['PL'])),test_lists))
    #MMNL =  np.array(map(lambda sigma: (len(sigma),lib.RS_mmnl_utils.log_RS_prob(D['MMNL'],sigma,N)),test_lists))
    PCMC =  np.array(map(lambda sigma: (len(sigma),lib.RS_pcmc_utils.log_RS_prob(D['PCMC'],sigma)),test_lists))
    (sigma_greedy,theta_greedy) = D['Mallows-greedy']
    Mallows_greedy =  np.array(map(lambda sigma: (len(sigma),lib.RS_mallows_utils.log_RS_prob(sigma_greedy,theta_greedy,sigma)),test_lists))

    return N,PL,[],PCMC,Mallows_greedy

def plot_scores(PL,MMNL,PCMC,Mallows_greedy,max_N,dataset):
    """
    plots errors
    """
    save_dir = os.getcwd()+os.sep+'pictures'+os.sep+dataset+os.sep
    f=plt.figure(figsize=(5,4))
    obs = []
    for x in PL:
        if len(PL[x])>0:
            obs.append(x)
    obs = np.array(obs)
    n=max_N
    PL_mean = -np.array([np.mean(PL[i]) for i in obs])
    #MMNL_mean = [np.mean(MMNL[i]) for i in obs]
    PCMC_mean = -np.array([np.mean(PCMC[i]) for i in obs])
    Mallows_greedy_mean = -np.array([np.mean(Mallows_greedy[i]) for i in obs])

    PL_err= [sem(PL[i]) for i in obs]
    #MMNL_err= [sem(MMNL[i]) for i in obs]
    PCMC_err= [sem(PCMC[i]) for i in obs]
    Mallows_greedy_err= [sem(Mallows_greedy[i]) for i in obs]

    lnsty=''
    mrkr = 'x'
    plt.errorbar(x=obs,y=PL_mean,yerr=PL_err,label='PL',marker=mrkr,linestyle=lnsty)
    #plt.errorbar(x=obs,y=MMNL_mean,yerr=MMNL_err,label='RS-MMNL',marker=mrkr,linestyle=lnsty)
    plt.errorbar(x=obs,y=PCMC_mean,yerr=PCMC_err,label='RS-PCMC',marker=mrkr,linestyle=lnsty)
    plt.errorbar(x=obs,y=Mallows_greedy_mean,yerr=Mallows_greedy_err,label='RS-Mallows',marker=mrkr,linestyle=lnsty)

    np.set_printoptions(suppress=True,precision=3)


    plt.xlabel('list length')
    plt.ylabel('average negative log-likelihood')
    #plt.xlim(1.5,np.amax(obs)+1)

    plt.legend(loc='best')
    plt.title(dataset+' average negative log-likelihood')
    f.tight_layout()
    plt.savefig(save_dir+dataset+'-loglikelihoods.pdf')
    plt.clf()


if __name__=='__main__':
    np.set_printoptions(suppress=True, precision=3)
    if sys.argv[1] not in ['soi','nascar','letor','election']:
		print 'wrong data folder'
		assert False
    path = os.getcwd()+os.sep+'results'+os.sep+sys.argv[1]
    plot(path,sys.argv[1])
