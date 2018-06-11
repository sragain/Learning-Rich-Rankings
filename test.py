import numpy as np
import torch,os,argparse,pickle
import torch.nn as nn
from torch.autograd import Variable
from models import *
from scrape import *
import math

def evaluate_loss_setsizes(perms,n,models_k,criterion=nn.NLLLoss,Losses = None):
    """
    returns losses given by the model for the specified loss criterion
    grouped by input setsize

    Args:
    X- set indicator tensors
    Y- choice index tensors
    model- choice model
    criterion- loss function used
    Losses- losses already computed
    """
    if sum(map(len,perms))==0:
        return Losses
    X,Y = RS_choices(perms,n)
    X = torch.Tensor(X); Y = torch.Tensor(Y)
    dataset = torch.utils.data.TensorDataset(X,Y)
    testloader = torch.utils.data.DataLoader(dataset)
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    for data in testloader:
        x,y = data
        for model in models_k:
            output = model(Variable(x))
            target = Variable(y)
            loss = output.data.numpy()
            Losses[str(model)][int(torch.sum(x))].append(loss[0][int(y.numpy())])
    return Losses

def evaluate_log_loss(perms,n,models_k,Losses = None,Losses_unif = None):
    """
    returns losses given by the model for the specified loss criterion
    grouped by input setsize

    Args:
    X- set indicator tensors
    Y- choice index tensors
    model- choice model
    criterion- loss function used
    Losses- losses already computed
    """
    #unif_losses[k] is loss of uniform dist for top k list
    unif_losses = np.cumsum(map(np.log,range(1,n+1)[::-1]))
    ls = {}
    for model in models_k:
        ls[model]=0

    for sigma in perms:
        for model in models_k:
            ls[str(model)]=0
        if not sigma: #when sigma is empty we skip
            continue

        for model in models_k:
            X,Y= RS_choices([sigma],n)
            for idx in range(len(sigma)):
                S = torch.Tensor(X[idx,:])
                S = Variable(S[None,:])
                ch = Y[idx].astype(int)

                loss = model(S).data.numpy()
                ls[str(model)] -= loss[0,ch]
                if math.isnan(loss[0,ch]):

                    Q=model.parameters().next().data.numpy()
                    s = S.data.numpy().astype(int)
                    s = np.array([x for x in range(n) if s[0,x]==1])
                    assert False

        for model in models_k:
            improvement = unif_losses[len(sigma)-1]-ls[str(model)]
            #assert not math.isnan(improvement)
            #assert not math.isnan(ls[str(model)])
            Losses_unif[str(model)].append(improvement)
            Losses[str(model)].append(ls[str(model)])

    return Losses,Losses_unif

def max2(l,default=0):
    if l ==[]:
        return default
    return max(l)

def trawl(path,dset,dtype,setsize,RE):
    """
    trawls over a directory and fits models to all data files
    """
    save_path = os.getcwd()+os.sep+'errors'+os.sep+dset+os.sep
    Losses = {}; Losses_unif = {}

    for modelpath in os.listdir(path):
        if not modelpath.endswith('.p'):
            continue
        if RE != modelpath.endswith('RE.p'):
            continue

        cv_data = pickle.load(open(path+os.sep+modelpath,'rb'))
        models = [x for x in cv_data if x not in ['train','test','data']]
        L = cv_data['data']
        n = reduce(lambda x,y: max(x,y),map(max2,L))+1
        #if 'n' not in Losses:
        #    Losses['n']=n
        print modelpath,n,len(L),max(map(len,L))

        train = cv_data['train']; test = cv_data['test']; L = cv_data['data']
        Losses[modelpath]={}#{'n':n}
        Losses_unif[modelpath]={}
        assert modelpath in Losses
        if setsize:
            for model in models:
                Losses[modelpath][model]={}
                Losses_unif[modelpath][model]={}
                for i in range(n):
                    Losses[modelpath][model][i+int(setsize)]=[]
                    Losses_unif[modelpath][model][i+int(setsize)]=[]
        else:
            for model in models:
                Losses[modelpath][model]=[]
                Losses_unif[modelpath][model]=[]


        K = len(test)
        for k in range(K):
            test_perms  = [L[x] for x in test[k]]
            #for sigma in test_perms:
            #    print sigma
            models_k = [cv_data[model][k] for model in models]
            if setsize:
                Losses[modelpath] = evaluate_loss_setsizes(test_perms,n,models_k,Losses=Losses[modelpath])
            else:
                Losses[modelpath],Losses_unif[modelpath] = evaluate_log_loss(test_perms,n,models_k,Losses=Losses[modelpath],Losses_unif=Losses_unif[modelpath])


    s=''
    if RE:
        s+='-RE'
    print save_path
    if setsize:
        pickle.dump(Losses,open(save_path+dtype+'-setsize'+s+'.p','wb'))
    else:
        #for key in Losses:
            #for model in Losses[key]:
            #    if model == 'n':
            #        continue
            #    print key,model, np.mean(Losses[key][model]), np.mean(Losses_unif[key][model])
        #pickle.dump(Losses_unif,open(save_path+dtype+'-Lunif'+s+'.p','wb'))
        pickle.dump(Losses,open(save_path+dtype+'-Llog'+s+'.p','wb'))

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    parser = argparse.ArgumentParser(description='ctr data parser')
    parser.add_argument('-dset', help="dataset name", default=None)
    parser.add_argument('-dtype', help="dataset type", default='soi')
    parser.add_argument('-setsize', help = 'whether to compute losses by setsize', default='y')
    parser.add_argument('-re', help = 'whether to compute for RE models (y/n)', default='n')

    #parser.add_argument('-epochs', action="number of epochs to use", dest="c", type=int)

    args = parser.parse_args()

    if args.dset not in ['sushi','soi','nascar','letor','soc','election']:
        print 'invalid dataset'
        assert False
    if args.dtype not in ['soi','soc']:
        print 'invalid datatype'
        assert False
    if args.dset=='soc':
        args.dtype='soc'

    RE = args.re == 'y'
    setsize = args.setsize == 'y'
    path = os.getcwd()+os.sep+'learned'+os.sep+args.dset
    trawl(path,args.dset,args.dtype,setsize,RE)
