import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem
import os,argparse,pickle
from matplotlib import rc


def plot_one_scores_setsizes_with_hist(Scores,dset,dsetnum,dtype):
    n = max([k for k in Scores[Scores.keys()[0]].keys()])

    print n
    re = ('RE' in dsetnum)
    s = ''
    if re:
        s+= '-RE'
    plt.figure(figsize=(9,7))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0), rowspan=1)
    for model in ['MNL','CDM-d=1','CDM-d=4','CDM-d=8','PCMC']:
        positions = [];means=[];sems=[];sizes=[]

        for i in Scores[model]:
            if len(Scores[model][i])==0:
                continue

            positions.append(n-i+1)
            #sizes.append(i)
            scores = np.array(Scores[model][i])
            means.append(np.mean(scores))
            sems.append(sem(scores))

        if n-1 in positions and n not in positions:
            positions = [n]+positions
            means =  [0] + means
            sems = [0] + sems
        positions = np.array(positions);means=-np.array(means);sems=np.array(sems)
        ax1.errorbar(positions,means,yerr=sems,label=model,marker='x')

    #get name for saving plot
    dashes = [pos for pos, char in enumerate(dsetnum) if char == '-']

    last_dash = dashes[-1-int(re)]
    dset_name = dsetnum[:last_dash]
    unif_losses = np.array(map(lambda pos: np.log(n-pos+1),positions))
    if re:
        unif_losses = unif_losses[::-1]
    ax1.plot(positions,unif_losses,label='uniform',linestyle='--')
    ax1.set_xlim(.5,np.amax(positions)+.5)
    #ax1.set_xlabel('k (position in ranking)')
    ax1.set_xticks(positions)
    ax1.set_xlabel('k (position in ranking)')
    ax1.set_ylim(0,ax1.get_ylim()[1])
    ax1.set_ylabel(r'$\ell(k;\hat \theta_{MLE},T)$')
    ax1.set_title(r'{\tt '+dset_name+s+r'}')#, $\ell_{\log}(\cdot,\hat \theta_{MLE})$ vs. position')
    ax1.legend(loc='best')



    #plt.savefig(os.getcwd()+os.sep+'plots'+os.sep+dset+os.sep+dset_name+s+'.pdf')
    n = max([k for k in Scores[Scores.keys()[0]].keys()])
    counts = np.zeros(n)
    m = Scores.keys()[0]
    for i in Scores[m]:
        pos = n-i+1
        counts[pos-1]+=len(Scores[m][i])
    counts[-1]=counts[-2]
    ax2.bar(range(1,n+1),counts,align='center')
    #get name for saving plot
    dashes = [pos for pos, char in enumerate(dsetnum) if char == '-']
    re = ('RE' in dsetnum)
    s = ''
    if re:
        s+= '-RE'
    last_dash = dashes[-1-int(re)]
    dset_name = dsetnum[:last_dash]
    ax2.set_xlabel('k (position in ranking)')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(positions)
    if n<30:
        ax2.set_xticks(range(1,n+1))
    ax2.set_ylabel(r'\# rankings with'+'\n'+r'$\geq k$ positions')
    #ax2.set_title(r'{\tt '+dset_name+s+'}, ranking lengths')

    if dset=='nascar':
        ax1.set_xticks([x for x in positions if x==1 or x%5==0])
        ax2.set_xticks([x for x in positions if x==1 or x%5==0])
    #ax2.legend(loc='best')
    #f.tight_layout()
    plt.tight_layout()
    plt.savefig(os.getcwd()+os.sep+'plots'+os.sep+dset+os.sep+dset_name+s+'-hist.pdf')
    plt.clf()

def plot_one_scores_setsizes(Scores,dset,dsetnum,dtype):
    n = max([k for k in Scores[Scores.keys()[0]].keys()])
    re = ('RE' in dsetnum)
    s = ''
    if re:
        s+= '-RE'

    for model in ['MNL','CDM-d=1','CDM-d=4','CDM-d=8','PCMC']:
        positions = [];means=[];sems=[];sizes=[]

        for i in Scores[model]:
            if len(Scores[model][i])==0:
                continue

            positions.append(n-i+1)
            #sizes.append(i)
            scores = np.array(Scores[model][i])
            means.append(np.mean(scores))
            sems.append(sem(scores))

        if n-1 in positions and n not in positions:
            positions = [n]+positions
            means =  [0] + means
            sems = [0] + sems

        positions = np.array(positions);means=-np.array(means);sems=np.array(sems)
        if re:
            positions = positions[::-1]

        plt.errorbar(positions,means,yerr=sems,label=model,marker='x')

    #get name for saving plot
    dashes = [pos for pos, char in enumerate(dsetnum) if char == '-']

    last_dash = dashes[-1-int(re)]
    dset_name = dsetnum[:last_dash]
    unif_losses = np.array(map(lambda pos: np.log(n-pos+1),positions))
    if re:
        unif_losses = unif_losses[::-1]
    plt.plot(positions,unif_losses,label='uniform',linestyle='--')
    plt.xlim(.9,np.amax(positions)+.1)
    plt.xlabel('k (position in ranking)')
    plt.xticks(positions)
    plt.ylabel(r'$\ell(k;\hat \theta_{MLE},T)$')
    plt.title(r'{\tt '+dset_name+s+r'}')#, $\ell_{log}(\cdot,\hat \theta_{MLE})$ vs. position')
    plt.legend(loc='best')
    plt.tight_layout()



    plt.savefig(os.getcwd()+os.sep+'plots'+os.sep+dset+os.sep+dset_name+s+'.pdf')
    plt.clf()

def print_one_losses(Scores,dset,dsetnum,dtype,unif=False,re=False):
    means = []; sems = [];labels = []
    model_list = ['MNL','CDM-d=1','CDM-d=4','CDM-d=8','PCMC']
    for model in model_list:
        if model not in Scores:
            continue
        labels.append(model)
        scrs = np.array(Scores[model])
        means.append(np.mean(scrs))
        sems.append(sem(scrs))

    means = np.array(means)
    sems = np.array(sems)
    if unif:
        s = 'unif'
    else:
        s = 'log'
    if re:
        s+= '-'

    with open(os.getcwd()+os.sep+'plots'+os.sep+dset+os.sep+dsetnum+'-'+dtype+'-L'+s+'.txt','w') as f:
        f.write('models:')
        for idx in range(len(labels)):
            model = labels[idx]
            f.write(model + ',')
        f.write('\nlosses:')
        for idx in range(len(labels)):
            log_loss = means[idx]
            f.write(("%.3f" % log_loss)+' & ')
        f.write('\nse:')
        for idx in range(len(labels)):
            se = sems[idx]
            f.write(("%.3f" % se)+ ' & ')

def print_all_losses(Scores,dset,dtype,unif=False,re=False):
    means = {}; sems = {}
    model_list = ['MNL','CDM-d=1','CDM-d=4','CDM-d=8','PCMC']
    rankings = 0
    for dsetid in Scores:
        for model in model_list:
            if model not in means:
                means[model]=[]
                sems[model]=[]

            if model in Scores[dsetid]:
                scrs = np.array(Scores[dsetid][model])
                rankings += int(model=='MNL')*len(scrs)
                means[model].append(np.mean(scrs))
                sems[model].append(sem(scrs))
            elif model=='CDM-d=8' and 'CDM-d=4' in Scores[dsetid]:
                scrs = np.array(Scores[dsetid]['CDM-d=4'])
                means[model].append(np.mean(scrs))
                sems[model].append(sem(scrs))
            else:
                scrs = np.array(Scores[dsetid]['CDM-d=1'])
                means[model].append(np.mean(scrs))
                sems[model].append(sem(scrs))

    means_list = []
    sems_list = []
    labels = []
    print 'datasets, rankings:'
    print len(means['MNL']),rankings
    for model in model_list:
        if model not in means:
            continue
        labels.append(str(model))
        means_list.append(np.mean(means[model]))
        sems_list.append(np.mean(sems[model]))
    means = np.array(means_list)
    sems = np.array(sems_list)

    if unif:
        s = 'unif'
    else:
        s = 'log'
    if re:
        s += '-RE'
    with open(os.getcwd()+os.sep+'plots'+os.sep+dset+os.sep+dtype+'-L'+s+'-all.txt','w') as f:
        f.write('models:')
        for idx in range(len(labels)):
            model = labels[idx]
            f.write(model + ' & ')
        f.write('\nlosses:')
        for idx in range(len(labels)):
            log_loss = means[idx]
            f.write(("%.3f" % log_loss)+' & ')
        f.write('\nse:')
        for idx in range(len(labels)):
            se = sems[idx]
            f.write(("%.3f" % se)+' & ')

if __name__ == '__main__':
    #get command line arguments
    np.set_printoptions(suppress=True, precision=3)
    plt.rcParams.update({'font.size': 14})

    rc('text', usetex=True)
    parser = argparse.ArgumentParser(description='ctr data parser')
    parser.add_argument('-dset', help="dataset name", default=None)
    parser.add_argument('-dtype', help="dataset type", default='soi')
    parser.add_argument('-setsize', help = 'whether to compute losses by setsize', default='y')
    parser.add_argument('-all', help='whether to aggregate over all datasets (y/n)', default='n')
    parser.add_argument('-re', help='whether to plot for re data (y/n)', default='n')
    parser.add_argument('-hist', help='whether to include a histogram of the ranking lengths(y/n)', default='n')
    args = parser.parse_args()
    if args.dset not in ['sushi','soi','nascar','letor','soc','election']:
        print 'invalid dataset'
        assert False
    if args.dtype not in ['soi','soc']:
        print 'invalid datatype'
        assert False

    #load in the already compute errors
    all = (args.all == 'y')
    setsize = (args.setsize=='y')
    re = (args.re=='y')
    hist = (args.hist=='y')
    s=''
    if re:
        s+='-RE'

    path = os.getcwd()+os.sep+'errors'+os.sep+args.dset+os.sep

    if setsize:
        Scores = pickle.load(open(path+args.dtype+'-setsize'+s+'.p'))
    else:
        Scores = pickle.load(open(path+args.dtype+'-Llog'+s+'.p','rb'))

    #call the appropriate plotting or printing function
    if all:
        assert not setsize
        print_all_losses(Scores,args.dset,args.dtype,re=re)
        #print_all_losses(Scores_unif,args.dset,dataset,args.dtype,unif=True)
    elif setsize:
        for dataset in Scores:
            print dataset
            if args.dtype == 'soi' and hist:
                plot_one_scores_setsizes_with_hist(Scores[dataset],args.dset,dataset,args.dtype)
            else:
                plot_one_scores_setsizes(Scores[dataset],args.dset,dataset,args.dtype)

    else:
        assert False
        for dataset in Scores:
            print dataset
            print_one_losses(Scores[dataset],args.dset,dataset,args.dtype,re=re)
