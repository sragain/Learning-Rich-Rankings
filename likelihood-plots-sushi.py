import numpy as np
import lib.PL_utils,lib.RS_mmnl_utils,lib.RS_pcmc_utils,lib.RS_mallows_utils
import os
import pickle
from scipy.stats import sem
from matplotlib import pyplot as plt


def autolabel(ax,rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')


def plot(directory,max_N=20):
    path = os.getcwd()+os.sep+'results'+os.sep+directory+os.sep
    m = 0
    count=0
    for filename in os.listdir(path):
        if filename.endswith('.p'):
            print filename
            n,pl,mmnl,pcmc,mg,plre,mmnlre,pcmcre,mgre = likelihoods_sushi(filename,max_N)

    #f=plt.figure(figsize=(5,4))
    N = 3
    rs_means =-np.array([np.mean(pl),np.mean(pcmc),np.mean(mg)])
    rs_vars =np.array([sem(pl),sem(pcmc),sem(mg)])
    re_means =-np.array([np.mean(plre),np.mean(pcmcre),np.mean(mgre)])
    re_vars =np.array([sem(plre),sem(pcmcre),sem(mgre)])

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    f, ax = plt.subplots(figsize=(5,4))
    #f.set_figheight(5)
    #f.set_figwidth(4)
    rects1 = ax.bar(ind, rs_means, width, color='r', yerr=rs_vars)
    rects2 = ax.bar(ind+width, re_means, width, color='b', yerr=re_vars)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('avg negative log-likelihood')
    ax.set_title('log-likelihood on SUSHI data')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('PL','PCMC','Mallows'))

    ax.legend((rects1[0], rects2[0]), ('RS', 'RE'))


    #autolabel(ax,rects1)
    #autolabel(ax,rects2)

    #plt.show()

    save_dir = os.getcwd()+os.sep+'pictures'+os.sep+directory+os.sep
    #plt.xlim(1.5,np.amax(obs)+1)

    plt.legend(loc='best')
    #plt.legend(loc='best')
    plt.title('Negative log-likelihood for SUSHI')
    f.tight_layout()
    plt.savefig(save_dir+os.sep+'sushi-loglikelihoods.pdf')
    plt.clf()


def likelihoods_sushi(filename,max_N):
    path = os.getcwd()+os.sep+'results'+os.sep+'sushi'+os.sep
    D = pickle.load(open(path+filename,'rb'))
    N = D['N']

    test_lists = D['test-lists']

    #RS models
    PL =  np.array(map(lambda sigma: lib.PL_utils.log_PL_prob(sigma,D['PL']),test_lists))
    MMNL =  np.array(map(lambda sigma: lib.RS_mmnl_utils.log_RS_prob(D['MMNL'],sigma,N),test_lists))
    PCMC =  np.array(map(lambda sigma: lib.RS_pcmc_utils.log_RS_prob(D['PCMC'],sigma),test_lists))
    (sigma_greedy,theta_greedy) = D['Mallows-greedy']
    Mallows_greedy =  np.array(map(lambda sigma: lib.RS_mallows_utils.log_RS_prob(sigma_greedy,theta_greedy,sigma),test_lists))


    flipped = map(lambda sigma: sigma[::-1],test_lists)
    PLRE =  np.array(map(lambda sigma: lib.PL_utils.log_PL_prob(sigma,D['PL-RE']),flipped))
    MMNLRE =  np.array(map(lambda sigma: lib.RS_mmnl_utils.log_RS_prob(D['MMNL-RE'],sigma,N),flipped))
    PCMCRE =  np.array(map(lambda sigma: lib.RS_pcmc_utils.log_RS_prob(D['PCMC-RE'],sigma),flipped))
    (sigma_greedy_re,theta_greedy_re) = D['Mallows-greedy-RE']
    Mallows_greedy_RE =  np.array(map(lambda sigma:lib.RS_mallows_utils.log_RS_prob(sigma_greedy,theta_greedy,sigma),flipped))


    return N,PL,MMNL,PCMC,Mallows_greedy,PLRE,MMNLRE,PCMCRE,Mallows_greedy_RE

if __name__=='__main__':
    plot('sushi')
