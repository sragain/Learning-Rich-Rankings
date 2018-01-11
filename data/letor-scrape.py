import pickle
import numpy as np
import pickle

def RS_choices(L,n,alpha=.01):
    C = {tuple(range(n)):np.ones(n)*alpha}
    for sigma in L:
        S = range(n)
        for i in range(len(sigma)):
            if tuple(S) not in C:
                C[tuple(S)] = np.ones(len(S))*alpha
            C[tuple(S)][S.index(sigma[i])] += 1
            S.remove(sigma[i])

      
    return C

def RE_choices(L,n):
    return RS_choices(map(lambda sigma: sigma[::-1],L),n)

def split(C,frac=.2):
    Train = {}; Test = {}
    for qid in C:
        lists = C[qid]
        n = len(lists)
        testIdx = np.random.choice(range(n),size=int(frac*n),replace=False)
        trainIdx = [x for x in range(n) if x not in testIdx]
        train = [lists[idx] for idx in trainIdx]
        test = [lists[idx] for idx in testIdx]
        Train[qid] = train
        Test[qid] = test
    return Train,Test

def scrape_choices(C,N):
    C_RS={}
    C_RE={}
    for qid in C:
        C_RS[qid]=RS_choices(C[qid],N[qid])
        C_RE[qid]=RE_choices(C[qid],N[qid])
    return C_RS,C_RE

def scrape(file):
    """ scrapes rankings, counts from agg.txt file"""
    D={}
    G={}
    with open(file,'r') as f:
        for line in f:
            L = line.split(' ')
            qid = L[1][4:]
            if qid not in D:
                D[qid]=[]   
                G[qid]=[]
                
            #ground truth
            G[qid].append(int(L[0]))
            #extract ranks
            ranks=[]
            for i in range(2,27):
                [l,rank]=L[i].split(':')
                if rank != 'NULL':
                    ranks.append(int(rank))
                else:
                    ranks.append(0)
            D[qid].append(ranks)    
            

    C={};N={}
    for qid in D:
        C[qid]=[]
        N[qid] = len(D[qid])
        A= np.array(D[qid])
        assert A.shape[1] == 25
        for i in range(25):
            l = A[:,i]
            ranked = np.where(l>0)[0]
            ranking = ranked[np.argsort(l[ranked])]
            C[qid].append(ranking)
    #pickle.dump(C,open('MQ-lists.p','wb'))
    return C,N,G
    
if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    C,N,G = scrape('agg.txt')
    Train,Test = split(C)
    C_train_RS, C_train_RE = scrape_choices(Train,N) 
    C_all_RS, C_all_RE = scrape_choices(C,N)
	
    #save scraped data
    pickle.dump(C_train_RS,open('MQ-train-RS.p','wb'))    
    pickle.dump(C_train_RE,open('MQ-train-RE.p','wb'))    
    pickle.dump(C_all_RS,open('MQ-all-RS.p','wb'))    
    pickle.dump(C_all_RS,open('MQ-all-RE.p','wb'))    
	
    pickle.dump(N,open('MQ-N.p','wb'))    
    pickle.dump(Train,open('MQ-train-lists.p','wb'))  
    pickle.dump(Test,open('MQ-test-lists.p','wb'))   
    pickle.dump(C,open('MQ-all-lists.p','wb'))    
    pickle.dump(G,open('MQ-ground-truth.p','wb'))    
