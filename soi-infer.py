import numpy as np
import lib.PL_utils
import lib.pcmc_utils_grad
import lib.mmnl_utils
from lib.kendall_utils import kt_min_parallel
from matplotlib import pyplot as plt
import pickle
from multiprocessing import Pool
import os
import lib.RS_mallows_utils
import sys

def infer_models(filepath,max_N=20):
    save_path = os.getcwd()+os.sep+'results'
    D = pickle.load(open(filepath,'rb'))
    Train = D['train-lists']
    C = D['train-choices']
    N = D['N']
    if N<=max_N and N>1:
        print N
        D_out = infer(Train,C,N)
        D_out['test-lists']=D['test-lists']
        D_out['N']=N
        filename = filepath[filepath.rfind(os.sep)+1:]
        print filename
        pickle.dump(D_out,open(save_path+os.sep+sys.argv[1]+os.sep+filename[:-9]+'inferred.p','wb'))

def infer_soi(directory,parallel=False):
    names = []
    for filename in os.listdir(directory):
        if filename.endswith('.p'):
            filepath = path+os.sep+filename
            names.append(filepath)
    if parallel:
        p = Pool(8)
        p.map(infer_models,names)
        p.close()
        p.join()
    else:
        map(infer_models,names)


def infer(train_lists,C,N,maxiter=25):
    sigma_greedy, theta_greedy = lib.RS_mallows_utils.infer_greedy(train_lists,N)
    print 'mallows done'
    PL=lib.PL_utils.ILSR(C,None,N)
    print 'PL done'
    MMNL=[]#lib.mmnl_utils.infer(C,N,None,maxiter,None)
    print 'MMNL done'
    PCMC=lib.pcmc_utils_grad.infer_grad(C,N,None,10**(-3),maxiter)
    print 'PCMC done'
    D_out = {'PL':PL,'MMNL':MMNL,'PCMC':PCMC,'Mallows-greedy':(sigma_greedy,theta_greedy)}
    return D_out

if __name__=='__main__':
    np.set_printoptions(suppress=True, precision=3)
    if sys.argv[1] not in ['soi','nascar','letor','election']:
		print 'wrong data folder'
		assert False
    if sys.argv[1] == 'soi':
        path = os.getcwd()+os.sep+'data'+os.sep+sys.argv[1]+os.sep+'filtered'
    else:
        path = os.getcwd()+os.sep+'data'+os.sep+sys.argv[1]
    infer_soi(path,parallel=False)
