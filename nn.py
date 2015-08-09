#!/usr/bin/env python
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import numpy

from scipy import *
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

N=4000

n_epoch=50
batchsize=10
n_units=1000

dataset = loadmat( 'data.mat' )
tmpX = dataset['X']    # image data from data.mat for X
tmpy = dataset['y']    # image data from data.mat for y
#dataX = dataset['X']    
#_datay = dataset['y']    
#print(dataX.shape) 
#print(_datay.shape) 

#ramdomize data 
data_len = len(tmpX)  #5000
print("data_len=", data_len)
my_perm=np.random.permutation(data_len)
dataX=[]
datay=[]
for i in range(data_len):
	no = my_perm[i]
	dataX.append(tmpX[no])

	# convert y=10 -> y=0,  
	# orignal data from cousera represents 0 as 10
	if tmpy[no] == 10:
		datay.append(0);
	else:
		datay.append(tmpy[no]);


#split training, test data
x_train, x_test = np.split(dataX,   [N])
y_train, y_test = np.split(datay, [N])
N_test = y_test.size

#set NN model
model = chainer.FunctionSet(l1=F.Linear(400, n_units),
                            l2=F.Linear(n_units, n_units),
							l3=F.Linear(n_units, 10))

if args.gpu >= 0:
	cuda.init(args.gpu)
	model.to_gpu()

def forward(x_data, y_data, train=True):
	#print("hoge", y_data)
	#x, t = chainer.Variable(x_data), chainer.Variable(y_data)
	x = chainer.Variable(x_data.reshape(batchsize,400).astype(numpy.float32), volatile=False)
	t = chainer.Variable(y_data.astype(numpy.int32), volatile=False)

	#h1 = F.dropout(F.relu(model.l1(x)), train=train)
	#y = F.dropout(F.relu(model.l2(h1)), train=train)
	#y = model.l2(h1)

	h1 = F.dropout(F.sigmoid(model.l1(x)), train=train)
	h2 = F.dropout(F.sigmoid(model.l2(h1)), train=train)
	y = F.dropout(F.sigmoid(model.l3(h2)), train=train)

	return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

for epoch in six.moves.range(1, n_epoch + 1):
	print('epoch', epoch)

	#training
	perm=np.random.permutation(N)
	sum_accuracy = 0
	sum_loss = 0
	print("=======training==========")
	for i in six.moves.range(0, N, batchsize):
		x_batch = x_train[perm[i:i + batchsize]]
		y_batch = y_train[perm[i:i + batchsize]]
		if args.gpu >= 0:
			x_batch = cuda.to_gpu(x_batch)
			y_batch = cuda.to_gpu(y_batch)

		optimizer.zero_grads()
		loss, acc = forward(x_batch, y_batch)
		print("training: loss=", loss.data, ":  acc=", acc.data)
		loss.backward()
		optimizer.update()

		sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
		sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

	print('epoch:', epoch,  ' train mean loss={}, accuracy={}'.format( sum_loss / N, sum_accuracy / N))

	# evaluation
	print("=======evalutation==========")
	sum_accuracy = 0
	sum_loss = 0
	for i in six.moves.range(0, N_test, batchsize):
		x_batch = x_test[i:i + batchsize]
		y_batch = y_test[i:i + batchsize]
		if args.gpu >= 0:
			x_batch = cuda.to_gpu(x_batch)
			y_batch = cuda.to_gpu(y_batch)

		loss, acc = forward(x_batch, y_batch, train=False)
		print("evaluation: loss=", loss.data, ":  acc=", acc.data)

		sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
		sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

	print('epoch:',epoch ,'  test  mean loss={}, accuracy={}'.format(
		sum_loss / N_test, sum_accuracy / N_test))



