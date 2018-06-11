import numpy as np
import os,sys

def RS_choices(L,n):
	#C = {tuple(range(n)):np.ones(n)*alpha}
	m = reduce(lambda x,y: x+y,map(len,L))
	X = np.zeros((m,n)); Y = np.empty(m)
	i = 0; j = 0
	for sigma in L:
		S = range(n)
		for i in range(len(sigma)):
			X[j,S] = 1
			assert np.sum(X[j,:])>0
			Y[j]=sigma[i]
			S.remove(sigma[i])
			j+=1
	return X,Y.astype(int)

def RE_choices(L,n):
	return RS_choices(map(lambda s: s[::-1],L),n,alpha)

def scrape_soi(filename):
	L = []
	with open(filename,'r') as f:
		N = int(f.next())
		#print N
		if N<2:
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

def scrape_soc(filename):
	L = []
	with open(filename,'r') as f:
		N = int(f.next())
		#if N > 100:
		#	#these are too big
		#	return [],0
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
