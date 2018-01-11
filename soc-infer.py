import numpy as np
import lib.PL_utils
import lib.pcmc_utils_grad
import lib.mmnl_utils
from lib.kendall_utils import kt_min_parallel
from matplotlib import pyplot as plt
import pickle
from multiprocessing import Pool
import os,sys
import lib.RS_mallows_utils

def infer_models(filepath):
    save_path = os.getcwd()+os.sep+'results'
    D = pickle.load(open(filepath,'rb'))
    Train = D['train-lists']
    C = D['train-choices']
    C_RE = D['RE-train-choices']
    N = D['N']
    if N<=20 and N>2:
        print N
        D_out = infer(Train,C,C_RE,N)
        D_out['test-lists']=D['test-lists']
        D_out['N']=N
        filename = filepath[filepath.rfind(os.sep)+1:]
        print filename
        pickle.dump(D_out,open(save_path+os.sep+sys.argv[1]+os.sep+filename[:-9]+'inferred.p','wb'))

def infer_soc(filepath,parallel=False):
    names = []
    for filename in os.listdir(filepath):
        if filename.endswith('.p'):
            names.append(filepath+os.sep+filename)
    if parallel:
        p = Pool(8)
        p.map(infer_models,names)
        p.close()
        p.join()
    else:
        map(infer_models,names)

def infer(train_lists,C,C_RE,N,maxiter=500):
    sigma_greedy, theta_greedy = lib.RS_mallows_utils.infer_greedy(train_lists,N)
    PL=lib.PL_utils.ILSR(C,None,N)
    MMNL=lib.mmnl_utils.infer(C,N,None,maxiter,None)
    PCMC=lib.pcmc_utils_grad.infer_grad(C,N,None,10**(-3),maxiter)
    D_out = {'PL':PL,'MMNL':MMNL,'PCMC':PCMC,'Mallows-greedy':(sigma_greedy,theta_greedy)}
    PL_RE=lib.PL_utils.ILSR(C_RE,None,N)
    MMNL_RE=lib.mmnl_utils.infer(C_RE,N,None,maxiter,None)
    PCMC_RE=lib.pcmc_utils_grad.infer_grad(C_RE,N,None,10**(-3),maxiter)
    train_lists_RE = map(lambda sigma: sigma[::-1],train_lists)
    sigma_greedy_RE,theta_greedy_RE = lib.RS_mallows_utils.infer_greedy(train_lists_RE,N)

    D_out['PL-RE'] = PL_RE
    D_out['MMNL-RE'] = MMNL_RE
    D_out['PCMC-RE'] = PCMC_RE
    D_out['Mallows-greedy-RE']= (sigma_greedy_RE,theta_greedy_RE)
    return D_out

if __name__=='__main__':
    np.set_printoptions(suppress=True, precision=3)
    if sys.argv[1] not in ['soc','sushi']:
        print 'wrong data folder'
        assert False
    path = os.getcwd()+os.sep+'data'+os.sep+sys.argv[1]
    infer_soc(path,parallel=True)
