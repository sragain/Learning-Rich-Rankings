import numpy as np
import pickle
import os
import sys

def scrape(filename):
	L = []
	with open(filename,'r') as f:
		N = int(f.next())
		print N
		if N > 100 or N<=2:
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
			while l[i][0]!='{' and i<len(l)-1:
				sig.append(int(l[i])-offset)
				i+=1
			for _ in range(count):
				#some election data had repeated "write-in" markers
				L.append(list(sig))
	return L,N

def RS_choices(L,n,alpha=.01):
	C = {tuple(range(n)):np.ones(n)*alpha}
	for sigma in L:
		S = range(n)
		for i in range(len(sigma)):
			if tuple(S) not in C:
				C[tuple(S)] = np.zeros(len(S))#np.ones(len(S))*alpha
			C[tuple(S)][S.index(sigma[i])] += 1
			S.remove(sigma[i])
	return C

def split(lists,frac=.2):
	n = len(lists)
	testIdx = np.random.choice(range(n),size=int(frac*n),replace=False)
	trainIdx = [x for x in range(n) if x not in testIdx]
	train = [lists[idx] for idx in trainIdx]
	test = [lists[idx] for idx in testIdx]
	return train,test

def scrape_dump(filename):
	L,N = scrape(filename)
	if N<=2:
		return 0
	Train,Test = split(L)
	C_train_RS= RS_choices(Train,n=N)
	D = {'train-lists':Train,'test-lists':Test,'train-choices':C_train_RS,'N':N}
	pickle.dump(D,open(filename[:-4]+'-scraped.p','wb'))
	return N

if __name__ == '__main__':
	np.set_printoptions(suppress=True, precision=3)
	#path = os.getcwd()+os.sep+'soi'
	if sys.argv[1] not in ['sushi','soi','nascar','letor','election']:
		print 'wrong data folder'
		assert False
	if sys.argv[1] == 'soi':
		path = os.getcwd()+os.sep+'soi'+os.sep+'filtered'
	else:
		path = os.getcwd()+os.sep+sys.argv[1]
	for filename in os.listdir(path):
		if filename.endswith('soi'):
			filepath = path+os.sep+filename
			print filename
			a=scrape_dump(filepath)
