import numpy as np
import lib.PL_utils
import lib.pcmc_utils_grad
import lib.mmnl_utils
from lib.kendall_utils import kt_min,pairs_head,kt_head
from matplotlib import pyplot as plt
import pickle
from multiprocessing import Pool
import os
import lib.RS_mallows_utils

def greedy_mallows(a):
    return lib.RS_mallows_utils.infer_greedy(*a)
    
def infer_greedy_mallows(train_lists,N,max_N):
    #p = Pool(32)
    L = []
    keys = []
    for qid in train_lists:
        if N[qid]<=max_N:
            keys.append(qid)
            L.append((train_lists[qid],N[qid]))
   # g = p.map(greedy_mallows,L)
    g = map(greedy_mallows,L)
   # p.close()
    #p.join()
    return dict(zip(keys,g))    

def mallows(a):
    return kt_min(*a)

def infer_mallows(C,N,max_N):
    p = Pool(32)
    L = []
    keys = []
    for qid in C:
        if N[qid]<=max_N:
            assert max_N<=8 #or i'm wasting my time
            keys.append(qid)
            L.append((C[qid],N[qid]))
    g = p.map(mallows,L)
    p.close()
    p.join()
    return dict(zip(keys,g))
    
def MMNL(a):
    return lib.mmnl_utils.infer(*a)

def infer_MMNL(C,N,max_N):
    L=[]
    keys=[]
    for qid in C:
        if N[qid]<=max_N:
            keys.append(qid)
            L.append((C[qid],N[qid],None,100,None))
    p = Pool(32)
    g = p.map(MMNL,L)
    p.close()
    p.join()
    return dict(zip(keys,g))

def PCMC(a):
    return lib.pcmc_utils_grad.infer_grad(*a)

def infer_PCMC(C,N,max_N):
    L=[]
    keys=[]
    for qid in C:
        if N[qid]<=max_N:
            keys.append(qid)
            L.append((C[qid],N[qid],None,10**(-3),100))
    p = Pool(8)
    g = p.map(PCMC,L)
    p.close()
    p.join()
    return dict(zip(keys,g))

def PL(a):
    return lib.PL_utils.ILSR(*a)

def infer_PL(C,N,max_N):
    p = Pool(32)
    L = []
    keys = []
    for qid in C:
        if N[qid]<=max_N:
            keys.append(qid)
            L.append((C[qid],None,N[qid]))
    g = p.map(PL,L)
    p.close()
    p.join()
    return dict(zip(keys,g))

def infer_all(max_N=8,include_RE=False):
    rel_path = os.sep+'data'+os.sep+'MQ'+os.sep
    path = os.getcwd()+rel_path 
    C_RS=pickle.load(open(path+'MQ-all-RS.p','rb'))   
    N = pickle.load(open(path+'MQ-N.p','rb'))
    G = pickle.load(open(path+'MQ-ground-truth.p','rb'))
    np.set_printoptions(suppress=True,precision=3)

    save_path = os.getcwd()+os.sep+'results'+os.sep+'MQ'+os.sep

    E_RS_PL=infer_PL(C_RS,N,max_N)
    pickle.dump(E_RS_PL,open(save_path+'MQ-all-RS-PL-'+str(max_N)+'.p','wb'))
    print 'MNL done'
    
    E_RS_MMNL=infer_MMNL(C_RS,N,max_N)
    pickle.dump(E_RS_MMNL,open(save_path+'MQ-all-RS-MMNL-'+str(max_N)+'.p','wb'))
    print 'MMNL done'
    
    E_RS_PCMC=infer_PCMC(C_RS,N,max_N)
    
    pickle.dump(E_RS_PCMC,open(save_path+'MQ-all-RS-PCMC-'+str(max_N)+'.p','wb'))
    pickle.dump(N,open(save_path+'MQ-N.p','wb'))
    pickle.dump(G,open(save_path+'MQ-ground-truth.p','wb'))
    print 'PCMC done'
    if include_RE:
        'starting RE'
        C_RE=pickle.load(open(path+'MQ-all-RE.p','rb'))   
        E_RE_PL=infer_PL(C_RE,N,max_N)
        pickle.dump(E_RE_PL,open(save_path+'MQ-all-RE-PL-'+str(max_N)+'.p','wb'))
        E_RE_MMNL=infer_MMNL(C_RE,N,max_N)
        pickle.dump(E_RE_MMNL,open(save_path+'MQ-all-RE-MMNL-'+str(max_N)+'.p','wb'))
        E_RE_PCMC=infer_PCMC(C_RE,N,max_N)
        pickle.dump(E_RE_PCMC,open(save_path+'MQ-all-RE-PCMC-'+str(max_N)+'.p','wb'))
    
def infer_train(max_N=8,include_RE=False):  
    rel_path = os.sep+'data'+os.sep+'MQ'+os.sep
    path = os.getcwd()+rel_path 
    C_RS=pickle.load(open(path+'MQ-train-RS.p','rb'))   
    N = pickle.load(open(path+'MQ-N.p','rb'))
    np.set_printoptions(suppress=True,precision=3)
    #print 'number of problems with size at most'+str(max_N)+':'
    #print len(np.where(np.array(N.values())<=max_N)[0])

    save_path = os.getcwd()+os.sep+'results'+os.sep+'MQ'+os.sep

    E_RS_PL=infer_PL(C_RS,N,max_N)
    pickle.dump(E_RS_PL,open(save_path+'MQ-train-RS-PL-'+str(max_N)+'.p','wb'))
    print 'MNL done'
    

    E_RS_MMNL=infer_MMNL(C_RS,N,max_N)
    pickle.dump(E_RS_MMNL,open(save_path+'MQ-train-RS-MMNL-'+str(max_N)+'.p','wb'))
    print 'MMNL done'
    
    E_RS_PCMC=infer_PCMC(C_RS,N,max_N)
    pickle.dump(E_RS_PCMC,open(save_path+'MQ-train-RS-PCMC-'+str(max_N)+'.p','wb'))
    pickle.dump(pickle.load(open(path+'MQ-test-lists.p','rb')),open(save_path+'MQ-test-lists-'+str(max_N)+'.p','wb'))
    pickle.dump(N,open(save_path+'MQ-N.p','wb'))
    
    
    train_lists = pickle.load(open(path+'MQ-train-lists.p','rb'))
    greedy_mallows = infer_greedy_mallows(train_lists,N,max_N)
    pickle.dump(greedy_mallows,open(save_path+'MQ-train-greedy-mallows-'+str(max_N)+'.p','wb'))
        
    #mallows = infer_mallows(train_lists,N,max_N)
    #pickle.dump(mallows,open(save_path+'MQ-train-mallows-'+str(max_N)+'.p','wb'))

    if include_RE:
        C_RE=pickle.load(open(path+'MQ-train-RE.p','rb'))   
        E_RE_PL=infer_PL(C_RE,N,max_N)
        pickle.dump(E_RE_PL,open(save_path+'MQ-train-RE-PL-'+str(max_N)+'.p','wb'))
        E_RE_MMNL=infer_MMNL(C_RE,N,max_N)
        pickle.dump(E_RE_MMNL,open(save_path+'MQ-train-RE-MMNL-'+str(max_N)+'.p','wb'))
        E_RE_PCMC=infer_PCMC(C_RE,N,max_N)
        pickle.dump(E_RE_PCMC,open(save_path+'MQ-train-RE-PCMC-'+str(max_N)+'.p','wb'))
    
if __name__ == '__main__':
    
    #largest size input to use and whether we are training with all data
    #we train on all data when comparing to ground truth
    max_N=8; incl = False
    #infer_all(max_N,incl)
    infer_train(max_N,incl)

    

 
