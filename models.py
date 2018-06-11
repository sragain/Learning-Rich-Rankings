import torch
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

class CDM(nn.Module):
    """
    cdm choice model
    """
    def __init__(self,n,d):
        super(CDM, self).__init__()
        self.fs = nn.Parameter(torch.nn.init.normal(torch.Tensor(n,d)))
        self.cs = nn.Parameter(torch.nn.init.normal(torch.Tensor(d,n)))
        self.m = nn.LogSoftmax()
        self.d = d
        self.n = n

    def forward(self,x):
        u = x*torch.sum(torch.mm(self.fs,x*self.cs),dim=1)+(1-x)*(-16)
        p = self.m(u)
        return p

    def __str__(self):
        return 'CDM-d='+str(self.d)#+',ue='+str(self.ue)

class PCMC(nn.Module):
    def __init__(self,n,batch=True,ep=10**(-3),gamma=None):
        super(PCMC, self).__init__()
        if gamma is None:
            Q = torch.nn.init.uniform(torch.Tensor(n,n),.4,.6)
        else:
            #TODO: this need to be figured out for gamma coming off of trained
            #models, the dimensions are funky between the __main__ code here
            #and train.py
            print 'code is deprecated, using random Q'
            #L = [torch.Tensor(gamma)]*n
            #Q = torch.cat(L)
            Q = torch.nn.init.uniform(torch.Tensor(n,n),.4,.6)
            

        self.Q = nn.Parameter(Q)
        self.epsilon = ep
        self.n = n
        self.b = torch.zeros(n)
        self.b[-1]=1
        self.batch = batch
        self.I = torch.eye(n)
        self.m = nn.LogSoftmax()

    def forward(self,x):
        if self.batch:
            L = []
            for S in x.split(1):
                S=x
                S_mat = torch.mm(torch.t(S),S)
                #print S_mat
                Q = torch.clamp(self.Q*S_mat,min=self.epsilon)
                #print Q
                for i in range(self.n-1):
                    Q[i,i]=-torch.sum(Q[i,:])+Q[i,i]
                Q[:,self.n-1]=S
                b = Variable(torch.zeros(self.n)+self.epsilon)
                b[-1]=1
                pi,LU = torch.gesv(b,torch.t(Q))
                pi = S*torch.t(pi)+1e-16
                L.append(torch.log(pi/torch.sum(pi)))
            return torch.cat(L)
        else:
            S=x
            S_mat = torch.mm(torch.t(S),S)
            Q = torch.clamp(self.Q*S_mat,min=self.epsilon)
            for i in range(self.n-1):
                Q[i,i]=-torch.sum(Q[i,:])+Q[i,i]
            Q[:,self.n-1]=S
            b = Variable(torch.zeros(self.n))
            b[-1]=1
            pi,LU = torch.gesv(b,torch.t(Q))
            #print 'S'
            #print S
            #print 'LU'
            #print LU
            #print 'Q'
            #print Q.data.numpy()
            #print 'pi raw'
            #print pi.data.numpy()

            pi = S*torch.t(pi)+1e-16
            #print pi.data.numpy()

            return torch.log(pi/torch.sum(pi))

    def __str__(self):
        return 'PCMC'

class MNL(nn.Module):
    """
    mnl choice model
    """
    def __init__(self,n):
        super(MNL, self).__init__()
        #self.u = nn.Linear(n,1,bias=False)
        self.u = nn.Parameter(torch.nn.init.normal(torch.Tensor(n)))
        self.n = n
        self.m = nn.Softmax()

    def forward(self, x):
        u = x*self.u+(1-x)*-16
        p = self.m(u)
        return torch.log(p/torch.sum(p))

    def __str__(self):
        return 'MNL'

if __name__ == '__main__':
    # some testing code
    np.set_printoptions(suppress=True,precision=3)
    n = 5;ns=200;d=2;mnl_data=True
    X = np.zeros((ns,n))
    Y = np.empty(ns).astype(int)
    gamma = np.random.rand(n)
    gamma/= np.sum(gamma)
    #print gamma
    for i in range(ns):
        S=np.random.choice(range(n),size=np.random.randint(2,n),replace=False)
        for j in S:
            X[i,j]=1
        if mnl_data:
            P = gamma[S]
            P/= np.sum(P)
        else:
            P = np.random.dirichlet(np.ones(len(S)))
        c = np.random.choice(S,p=P)
        Y[i]=int(c)

    X = torch.Tensor(X)
    Y = torch.LongTensor(Y)
    dataset = torch.utils.data.TensorDataset(X,Y)
    dataloader = torch.utils.data.DataLoader(dataset)#,batch_size=2)
    criterion = nn.NLLLoss()
    models = [MNL(n),PCMC(n),CDM(n,d)]
    #optimizer = optim.SGD(model.parameters(), lr=0.001)#, momentum=0.9)
    for model in models:
        if str(model)=='PCMC':
            #print models[0]
            utils = models[0].parameters().next().data.numpy()
            #print utils
            g= np.exp(utils)
            g/= np.sum(g)
            #print gamma
            #print g
            model = PCMC(n,gamma=g,batch=False)
        optimizer = optim.Adam(model.parameters())

        for epoch in range(20):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)
                #print 'choice set'
                #print inputs

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                #print 'NLL losses'
                #print outputs
                #print 'utils'
                #print model.parameters().next().data
                #assert np.random.rand()<.99
                loss = criterion(outputs, labels)

                #if np.isnan(loss.data[0]):
                #    print inputs
                #    print outputs
                #    print labels
                #    assert False
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0

            print('Finished Training')

    #for model in models:
    #    for x in model.parameters():
    #        print x.data
