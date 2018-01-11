import lib.PL_utils as PL
import numpy as np
from numpy.random import dirichlet
from itertools import permutations
from matplotlib import pyplot as plt

def S_n_info(n):
    S_n = [sigma for sigma in permutations(range(n))]
    S_n_rev = []
    idx = {}; ct=0
    for sigma in S_n:
        S_n_rev.append(S_n.index(sigma[::-1]))
        idx[sigma]=ct
        ct+=1
    return S_n,S_n_rev,idx


def unif_rand_gamma(n):
    return dirichlet(np.ones(n))

def TV(gamma_RS,gamma_RE):
    d=0
    for sigma in S_n:
        d+=np.abs(PL.PL_prob(sigma,gamma_RS)-PL.PL_prob(sigma[::-1],gamma_RE))
    return d

def rand_RE_guessing(m=100000):
    TV_dists = np.empty(m)
    gamma_RE_list = np.empty((m,n))
    for i in range(m):
        gamma_RE = unif(n)
        TV_dists[i]=TV(gamma_RS,gamma_RE)
        gamma_RE_list[i,:] = gamma_RE
    print np.amin(TV_dists)
    plt.hist(TV_dists)
    plt.xlabel('TV distance')
    plt.ylabel('frequency')
    plt.title('Distribution of TV dists for RE from a fixed RS(MNL), n='+str(n))
    plt.savefig('RS_RE_dist_'+str(n)+'.png')
    print 'RS gamma:'
    print gamma_RS
    print 'RE gamma:'
    print gamma_RE_list[np.argmin(TV_dists),:]

def learn_RE(samples):
    return ILSR_perms(map(lambda s: s[::-1], samples),None,len(samples[0]))

def TV_plot(n=4,m=1000,points=100):
    S_n,S_n_rev,S_n_idxs = S_n_info(n)
    C_RE = {}; C_RS={}
    gamma = unif_rand_gamma(n)
    #gamma = np.ones(n).astype(float)
    #gamma/= n
    true_dist = PL.PL_dist(gamma)
    TV_emp = np.zeros(points)
    TV_RE = np.zeros(points)
    TV_RS = np.zeros(points)
    gamma_RS = unif_rand_gamma(n)
    gamma_RE = unif_rand_gamma(n)
    emp_counts = np.zeros(len(S_n))

    L = np.logspace(1,np.log10(m),num=points)
    cnt=0
    for i in range(points):
        for _ in range(cnt,int(L[i])):
            sigma = PL.sample_sigma(gamma)
            emp_counts[S_n_idxs[tuple(sigma)]]+=1.0
            flipped = tuple(sigma[::-1])
            C_RS = PL.add_choices(C_RS,sigma,n)
            C_RE = PL.add_choices(C_RE,flipped,n)
        print cnt
        cnt=int(L[i])
        gamma_RS = PL.ILSR(C_RS,gamma_RS,n)
        gamma_RE = PL.ILSR(C_RE,gamma_RE,n)
        RE_dist = PL.PL_dist(gamma_RE)[S_n_rev]
        RS_dist = PL.PL_dist(gamma_RS)
        TV_emp[i]=np.sum(np.abs(true_dist-emp_counts/np.sum(emp_counts)))
        TV_RE[i]=np.sum(np.abs(true_dist-RE_dist))
        TV_RS[i]=np.sum(np.abs(true_dist-RS_dist))
    #print emp_counts/np.sum(emp_counts)
    #print true_dist
    #print RE_dist
    #print gamma
    #print gamma_RE
    #plt.semilogx(L,TV_RS,label='learned RS dist TV')
    #plt.semilogx(L,TV_emp,label='empirical dist TV')
    #plt.semilogx(L,TV_RE,label='learned RE dist TV')
    plt.loglog(L,TV_RS,label='learned RS dist TV')
    plt.loglog(L,TV_emp,label='empirical dist TV')
    plt.loglog(L,TV_RE,label='learned RE dist TV')
    plt.title('Empirical vs. RE error')
    plt.xlabel('samples')
    plt.ylabel('TV dist')
    plt.legend()
    plt.savefig('EMPvsRE'+str(m)+'samplesn='+str(n)+'.png')


if __name__=='__main__':
    TV_plot(n=4,m=100   000,points=100)
