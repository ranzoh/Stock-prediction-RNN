#!/usr/bin/python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split

# number of features
INPUT_SIZE = 19

# entire data size
DATA_SIZE = 9500

# opens the csv file with the stock data.
# creates train-test split of the stock data.
def getData():
	file = 'INTC.csv'
	data = pd.read_csv(file)
	close = data.Close[:DATA_SIZE]
	X = []
	y = []
	# we want to train each 20 available consecutive days
	for i in range(0, DATA_SIZE-INPUT_SIZE-1):
		X.append(close[i:i+INPUT_SIZE])
		y.append(close[i+INPUT_SIZE])

	return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == '__main__':
	# gets the data after the split
	# the X data contains stock prices of 19 consecutive days
	# the y data is the stock price of the 20th day
	
	X_train, X_test, y_train, y_test = getData()

	# create the numpy arrays
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	y_train = np.array(y_train)
	y_test = np.array(y_test)

	# reshape to a 3-dimensional tensor
	X_train = X_train.reshape(X_train.shape + tuple([1]))
	X_test = X_test.reshape(X_test.shape + tuple([1]))
	y_train = y_train.reshape(y_train.shape + tuple([1]))
	y_test = y_test.reshape(y_test.shape + tuple([1]))

	print 'X_train Size: ' + str(X_train.shape)
	print 'y_train Size: ' + str(y_train.shape)
	print 'X_test Size: ' + str(X_test.shape)
	print 'y_test Size: ' + str(y_test.shape)

	

	# rnn configuration
	input_size = INPUT_SIZE
	num_units = 1024
	output_size = 1
	epochs = 180
	batch_size = 79
	dropout = 0.8
	number_of_layers = 2
	learn_rate = 0.0001

	learning_rate = tf.constant(learn_rate)
	
	# placeholder for input and output
	X = tf.placeholder(tf.float32, shape=[batch_size, input_size, output_size])
	y = tf.placeholder(tf.float32, shape=[batch_size, output_size])
	
	'''
	cell = tf.contrib.rnn.LSTMCell(num_units)
	cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
	multiRNN = tf.contrib.rnn.MultiRNNCell([cell]*number_of_layers)
	output,_ = tf.nn.dynamic_rnn(multiRNN, X, dtype=tf.float32)
	'''
	
	# creates the hidden layers with the rnn cells
	cells = []
	for _ in range(number_of_layers):
	  cell = tf.contrib.rnn.LSTMCell(num_units)
	  # define the dropout
	  cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
	  cells.append(cell)
	multiRNN = tf.contrib.rnn.MultiRNNCell(cells)
	output,_ = tf.nn.dynamic_rnn(multiRNN, X, dtype=tf.float32)
	
	
	W = tf.Variable(tf.truncated_normal([num_units, output_size], stddev=0.05))
	b = tf.Variable(tf.zeros([output_size]))
	
	# ignore the middle dimension, we need two-dimensional tensor as output
	y_pred = tf.matmul(output[:, -1, :], W) + b

	# we want the loss of each batch to be reduced
	losses = []
	for i in range(y.get_shape()[0]):
	    losses.append([tf.reduce_mean(tf.square(y_pred[i] - y[i]))])
	
	loss = tf.reduce_sum(losses)/(2*batch_size)

	# creates the optimizer
	update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	# start learning
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	for e in range(epochs):
		# shuffles the train data (for diversed batches)
		shuffle_indices = np.random.permutation(np.arange(len(y_train)))
		X_train = X_train[shuffle_indices]
		y_train = y_train[shuffle_indices]
		for i in range(0, len(y_train) // batch_size):
			# creates the batches
			start = i * batch_size
			batch_x = X_train[start:start + batch_size]
			batch_y = y_train[start:start + batch_size]
			
			# Train!
			_, loss_ = sess.run([update, loss], feed_dict={X: batch_x, y: batch_y})
			if i%50==0:
				print 'Epoch: ' + str(e) + ', Training loss: ' + str(loss_)
		if e%3==0:
			losses = []
			for j in range(0, len(y_test) // batch_size):
				start = j * batch_size
				batch_x = X_test[start:start + batch_size]
				batch_y = y_test[start:start + batch_size]
				loss_ = sess.run(loss, feed_dict={X: batch_x, y: batch_y})
				losses.append(loss_)
				print '**Epoch: ' + str(e) + ', Test loss: ' + str(loss_)
			print '####Epoch: ' + str(e) + ':: MSE Test: ' + str(sum(losses) / len(losses))
	
	# prints the results
	losses = []
	pred = []
	real = []
	for j in range(0, len(y_test) // batch_size):
		start = j * batch_size
		batch_x = X_test[start:start + batch_size]
		batch_y = y_test[start:start + batch_size]
		loss_ = sess.run(loss, feed_dict={X: batch_x, y: batch_y})
		res = sess.run(y_pred, feed_dict={X: batch_x})
		losses.append(loss_)
		pred.append(res)
		real.append(batch_y.reshape(len(batch_y)))
	print 'MSE Test: ' + str(sum(losses) / len(losses))
	with open('results.txt', 'a') as f:
		f.write('PREDICTED: ' + str(pred) + '\n\n')
		f.write('REAL: ' + str(real))

	sess.close()

