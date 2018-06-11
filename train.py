import sys,os,pickle,argparse
import numpy as np
from models import *
from utils import *
#from sklearn.model_selection import KFold
from sklearn.cross_validation import KFold
from scrape import *
import argparse
from torch.multiprocessing import Pool
from random import shuffle

def fit(X,Y,model,criterion= nn.NLLLoss(),epochs=20,batch_size=1,verbose=True,print_batches=1000,opt='Adam'):
    """
    Fits a choice model with pytorch's SGD

    X- Indicator vectors for choice sets
    Y- indices of choices
    model- choice model to fit
    epochs- number of times to loop over the training data
    batch- whether to batch samples
    """
    X = torch.Tensor(X)
    Y = torch.LongTensor(Y.astype(int))
    dataset = torch.utils.data.TensorDataset(X,Y)
    if batch_size>1:
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(dataset,shuffle=True)

    if opt=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    elif opt=='Adam':
        optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        print epoch
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if not verbose:
                continue
            if i % print_batches == print_batches-1:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, (i + 1)*batch_size, running_loss / print_batches))
                running_loss = 0.0
    return model

def cv(L,n,models,save_path,K=5,epochs=20,batch_size=1,opt='Adam',seed=True,RE=False):
    """
    trains and saves choosing to rank models with SGD via k-fold cv

    Args:
    L- list of data rankings
    n- number of items ranked
    model - choice models to fit
    save_path- folder to save to
    K- number of folds
    epochs- number of times to loop over the data
    """
    #kf = KFold(n_splits=5,shuffle=True)
    #splits = kf.split(L)
    kf = KFold(len(L),n_folds=K,shuffle=True)
    k = 0
    split_store = {'train':[],'test':[],'data':L}
    for model in models:
        split_store[str(model)]=[]
    for train,test in kf:#splits:

        print 'fold'+str(k)

        #scrape training choices and fit model
        X_train,Y_train = RS_choices([L[x] for x in train],n)
        for model in models:
            print model
            if seed and str(model) == 'PCMC':
                utils = models[0].parameters().next().data.numpy()
                #print utils
                g= np.exp(utils)
                g/= np.sum(g)
                model = PCMC(n,gamma=g)
            model = fit(X_train,Y_train,model,criterion=nn.NLLLoss(),epochs=epochs,batch_size=batch_size,opt=opt)
            split_store[str(model)].append(model)

        #store everything
        split_store['train'].append(train)
        split_store['test'].append(test)
        k+=1

    if not RE:
        pickle.dump(split_store,open(save_path+'.p','wb'))
    else:
        pickle.dump(split_store,open(save_path+'-RE.p','wb'))
    return 0

def parallel_helper(tup):
    L,n,models,save_path,epochs,batch_size,opt,seed,RE = tup
    return cv(L,n,models,save_path,epochs=epochs,batch_size=batch_size,opt=opt,seed=seed,RE=RE)

def trawl(path,dtype,epochs,parallel=True,batch_size=1,max_n=30,max_rankings=1000,opt='Adam',num_dsets=10,seed=True,RE=False):
    """
    trawls over a directory and fits models to all data files
    """
    job_list = []
    save_path = os.getcwd()+os.sep+'learned'+os.sep+args.dset+os.sep
    files = os.listdir(path)
    batch = (batch_size>1)
    shuffle(files)
    for filename in files:
        if filename.endswith(args.dtype):
            filepath = path+os.sep+filename
            if args.dtype=='soi':
                L,n = scrape_soi(filepath)
            else:
                L,n = scrape_soc(filepath)
            if len(L)<=10 or len(L)>max_rankings or n>max_n:#throw out more than 1000 data points
                print n,sum(map(len,L))
                print filename+' skipped'
                continue
            else:
                print n,sum(map(len,L))
                print filename+' added'
            #collect models
            models = []

            for d in [1,4,8]:
                if d>n:
                    continue
                models.append(CDM(n=n,d=d))

            models.append(MNL(n))
            models.append(PCMC(n,batch=batch))
            job_list.append((L,n,models,save_path+filename[:-4]+'-'+dtype,epochs,batch_size,opt,seed,False))
            if RE:
                job_list.append((map(lambda x:x[::-1],L),n,models,save_path+filename[:-4]+'-'+dtype,epochs,batch_size,opt,seed,True))
            if len(job_list)>=num_dsets:
                break

    print str(len(job_list))+' datasets total'
    print str(sum(map(lambda x: len(x[0]),job_list)))+ 'total rankings'
    sorted(job_list,key=lambda x: x[1]*len(x[0]))
    map(parallel_helper,job_list)

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    parser = argparse.ArgumentParser(description='ctr data parser')
    parser.add_argument('-dset', help="dataset name", default=None)
    parser.add_argument('-dtype', help="dataset type", default ='soi')
    parser.add_argument('-epochs', help="number of epochs to use", default='5')
    parser.add_argument('-batch_size', help='batch_size for training', default = '1')
    parser.add_argument('-max_n', help='maximum number of items ranked', default = '10')
    parser.add_argument('-max_rankings', help='maximum number of rankings', default = '1000')
    parser.add_argument('-opt', help='SGD or Adam', default='Adam')
    parser.add_argument('-num_dsets', help='how many datasets to use', default='100')
    parser.add_argument('-seed_pcmc', help='whether to seed pcmc with MNL (y/n)', default = 'n')
    parser.add_argument('-fit_re', help='whether to train RE models (y/n)', default = 'n')
    #parser.add_argument('-min_rankings', help='minimium number of rankings', default = '10')
    args = parser.parse_args()
    if args.dset not in ['sushi','soi','nascar','letor','soc','election']:
        print 'wrong data folder'
        assert False
    if args.dtype not in ['soi','soc']:
        print 'wrong data type'
        assert False
    if args.opt not in ['SGD','Adam']:
        print 'optmizer can be SGD or Adam'
        assert False
    if args.fit_re == 'y' and args.dtype=='soi':
        print 'cannot fit RE models to top-k rankings'
        assert False
    if args.dset=='soc':
        args.dtype='soc'
    path = os.getcwd()+os.sep+'data'+os.sep+args.dset
    if args.dset == 'soi':
        path += os.sep+'filtered'
    if args.seed_pcmc not in ['y','n']:
        print 'y or n required for -seed_pcmc'
    seed = (args.seed_pcmc=='y')
    re = (args.fit_re == 'y')
    trawl(path,args.dtype,epochs=int(args.epochs),batch_size=int(args.batch_size),
          max_n=int(args.max_n),max_rankings=int(args.max_rankings),opt=args.opt,
          num_dsets=int(args.num_dsets),seed=seed,RE=re)
