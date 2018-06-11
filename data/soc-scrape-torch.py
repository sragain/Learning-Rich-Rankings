import numpy as np
import pickle
import os,sys

def scrape(filename):
	L = []
	with open(filename,'r') as f:
		N = int(f.next())
		if N > 100:
			#these are too big
			return [],0
		#sometimes the first candidate is labeled '1', sometimes '0'
		offset = int(f.next()[0])
		for _ in range(N):
			f.next()
		for line in f:
			l = line.split(',')
			count = int(l[0])
			sig =[]
			i = 1
			sig = map(lambda k: int(k)-offset,l[1:])
			for _ in range(count):

				#some election data had repeated "write-in" markers
				L.append(list(sig))

	return L,N

def RS_choices(L,n,alpha=.01):
	#C = {tuple(range(n)):np.ones(n)*alpha}
	m = reduce(lambda x,y: x+y,map(len,L))
	X = np.zeros((m,n)); Y = np.empty(m)
	i = 0;j=0
	for sigma in L:
		S = range(n)
		for i in range(len(sigma)):
			X[j,S] = 1
			Y[j]=sigma[i]
			S.remove(sigma[i])
			j+=1
	return X,Y

def RE_choices(L,n,alpha=.01):
	return RS_choices(map(lambda s: s[::-1],L),n,alpha)

def scrape_dump(filename):
	L,N = scrape(filename)
	if N<=2:
		return 0
	X_RS,Y_RS = RS_choices(L,n=N)
	X_RE,Y_RE = RE_choices(L,n=N)

	np.savez(open(filename[:-4]+'-RS.npy','wb'),X_RS,Y_RS)
	#np.save(open(filename[:-4]+'-RS-choices.npy','wb'),Y_RS)

	np.save(open(filename[:-4]+'-RE.npy','wb'),X_RE,Y_RE)
	#np.save(open(filename[:-4]+'-RE-choices.npy','wb'),Y_RE)
	return N

if __name__ == '__main__':
	np.set_printoptions(suppress=True, precision=3)
	if sys.argv[1] not in ['soc','sushi']:
		print 'wrong data folder'
		assert False
	path = os.getcwd()+os.sep+sys.argv[1]
	for filename in os.listdir(path):
		if filename.endswith('soc'):
			filepath = path+os.sep+filename
			print filename
			a=scrape_dump(filepath)
